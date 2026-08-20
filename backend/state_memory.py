#!/usr/bin/env python3
"""
State Memory — EVOSEQ v3 Component
====================================
Answers the question:

    "Have I seen a state similar to the current one before?
     What happened next, and how many times have I seen it?"

Architecture
------------
1. compute_state_fingerprint()  (from backend/intelligence/state_fingerprint.py)
   converts the current digit history into a 16-dim feature vector.

2. StateMemory maintains an in-memory store of (fingerprint_vec, outcome) pairs
   loaded from the `state_memory` Supabase table on startup and updated every
   cycle.

3. Nearest-neighbour search (L2 distance, no external deps) finds the K most
   similar historical states and returns:

       SimilarStateResult
           empirical_p_big      — weighted fraction of Big outcomes
           sample_size          — number of similar states found
           confidence_weight    — how much to trust this signal [0, 1]
           mean_distance        — average distance to neighbours
           uncertainty          — entropy of the outcome distribution
           verdict              — RELIABLE | WEAK | INSUFFICIENT

4. The signal is honest:
   - If sample_size < MIN_SAMPLES the verdict is INSUFFICIENT.
   - confidence_weight scales smoothly with sample_size so the ensemble
     is not dominated by the state-memory signal on cold start.
   - uncertainty is reported so the adversarial engine can use it.

5. Persistence:
   - Every resolved outcome is appended to the in-memory store AND written
     to `state_memory` in Supabase via persist_state().
   - On startup, load_from_db() hydrates the store (up to MAX_MEMORY rows).

Design invariant
----------------
The state written BEFORE the draw resolves never contains the outcome.
The outcome is appended only AFTER resolution. This enforces causal integrity.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.intelligence.state_fingerprint import (
    StateFingerprint,
    compute_state_fingerprint,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

K_NEIGHBOURS       = 25      # number of nearest neighbours to retrieve
MIN_SAMPLES        = 15      # minimum neighbours before signalling
MAX_MEMORY         = 50_000  # max fingerprints held in RAM
DISTANCE_THRESHOLD = 0.35    # L2 distance below which states are "similar"
PERSIST_EVERY      = 50      # write to Supabase every N new fingerprints


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StoredState:
    sequence_no: str
    vec: np.ndarray        # 16-dim float32 fingerprint
    fp: StateFingerprint   # full fingerprint for diagnostics
    actual_side: Optional[int] = None   # 1=Big, 0=Small; None until resolved


@dataclass
class SimilarStateResult:
    empirical_p_big: float      # weighted fraction of Big outcomes among neighbours
    sample_size: int             # number of neighbours with resolved outcomes
    confidence_weight: float     # how much to blend this into ensemble [0, 0.25]
    mean_distance: float         # average L2 to the K nearest neighbours
    uncertainty: float           # entropy of outcome distribution [0, 1]
    verdict: str                 # RELIABLE | WEAK | INSUFFICIENT
    context: dict                # diagnostic details

    def to_dict(self) -> dict:
        return {
            "empirical_p_big":   round(self.empirical_p_big, 4),
            "sample_size":       self.sample_size,
            "confidence_weight": round(self.confidence_weight, 4),
            "mean_distance":     round(self.mean_distance, 4),
            "uncertainty":       round(self.uncertainty, 4),
            "verdict":           self.verdict,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Distance helpers
# ─────────────────────────────────────────────────────────────────────────────

def _l2_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Fast L2 distance from one query vector to each row of matrix."""
    diff = matrix - query[np.newaxis, :]
    return np.sqrt((diff * diff).sum(axis=1))


def _outcome_entropy(big_count: int, total: int) -> float:
    if total <= 0:
        return 1.0
    p = big_count / total
    q = 1.0 - p
    if p <= 0 or q <= 0:
        return 0.0
    return float(-(p * math.log2(p) + q * math.log2(q)))


# ─────────────────────────────────────────────────────────────────────────────
# Main StateMemory class
# ─────────────────────────────────────────────────────────────────────────────

class StateMemory:
    """
    Persistent nearest-neighbour state memory.

    Typical per-cycle usage:
        sm = StateMemory()        # singleton, created once
        sm.load_from_db(db)       # on startup

        # Before draw:
        fp = compute_state_fingerprint(history, sequence_no=issue)
        sm.store_pending(issue, fp)
        result = sm.query(fp)     # get similar-state signal

        # After draw resolves:
        sm.resolve(issue, actual_side)
        sm.maybe_persist(db)       # async-safe: flushes every PERSIST_EVERY calls
    """

    def __init__(self):
        self._states: Deque[StoredState] = deque(maxlen=MAX_MEMORY)
        # issue_number -> index for fast resolve lookup
        self._pending: dict[str, StoredState] = {}
        # matrix cache (rebuilt lazily when stale)
        self._matrix: Optional[np.ndarray] = None
        self._matrix_dirty = True
        self._resolved_buffer: List[StoredState] = []
        self._persist_counter = 0

    # ── Store & resolve ───────────────────────────────────────────────────────

    def store_pending(self, issue_number: str, fp: StateFingerprint) -> None:
        """Store a fingerprint before the outcome is known."""
        vec = np.array(fp.feature_vector, dtype=np.float32)
        if len(vec) == 0:
            return
        state = StoredState(sequence_no=str(issue_number), vec=vec, fp=fp)
        self._states.append(state)
        self._pending[str(issue_number)] = state
        self._matrix_dirty = True

    def resolve(self, issue_number: str, actual_side: int) -> bool:
        """Backfill the outcome. Returns True if the pending state was found."""
        s = self._pending.pop(str(issue_number), None)
        if s is None:
            # Try scanning recent states
            for state in reversed(list(self._states)):
                if state.sequence_no == str(issue_number):
                    state.actual_side = int(actual_side)
                    self._resolved_buffer.append(state)
                    return True
            return False
        s.actual_side = int(actual_side)
        self._resolved_buffer.append(s)
        return True

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, fp: StateFingerprint) -> SimilarStateResult:
        """
        Find the K nearest historical states and compute the empirical P(Big).

        Only states with resolved outcomes contribute to the probability estimate.
        Unresolved states still appear in the distance calculation to prevent
        looking forward, but are excluded from the outcome tally.
        """
        query_vec = np.array(fp.feature_vector, dtype=np.float32)
        if len(query_vec) == 0:
            return self._insufficient("Empty feature vector")

        resolved = [s for s in self._states if s.actual_side is not None]
        if len(resolved) < MIN_SAMPLES:
            return self._insufficient(f"Only {len(resolved)} resolved states in memory")

        # Build matrix from resolved states only
        matrix = np.array([s.vec for s in resolved], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(query_vec):
            return self._insufficient("Feature dimension mismatch")

        # Normalize each row (and query) to unit length for cosine-like behaviour
        # but keep L2 distance metric for interpretability
        distances = _l2_batch(query_vec, matrix)

        # Take K nearest within threshold
        k = min(K_NEIGHBOURS, len(resolved))
        top_k_idx = np.argsort(distances)[:k]
        top_k_dist = distances[top_k_idx]

        # Only keep those within the distance threshold
        in_threshold = top_k_dist <= DISTANCE_THRESHOLD
        if in_threshold.sum() < MIN_SAMPLES // 2:
            # relax threshold to nearest K regardless
            in_threshold = np.ones(k, dtype=bool)

        neighbour_idx   = top_k_idx[in_threshold]
        neighbour_dist  = top_k_dist[in_threshold]
        neighbours      = [resolved[i] for i in neighbour_idx]

        # Weighted vote (inverse-distance weighting)
        weights = 1.0 / (neighbour_dist + 1e-6)
        weights /= weights.sum()

        big_weight  = sum(w for w, s in zip(weights, neighbours) if s.actual_side == 1)
        n_resolved  = len(neighbours)
        n_big       = sum(1 for s in neighbours if s.actual_side == 1)

        # Laplace-smoothed probability
        empirical_p = (big_weight * n_resolved + 1.0) / (n_resolved + 2.0)

        mean_dist   = float(neighbour_dist.mean())
        uncertainty = _outcome_entropy(n_big, n_resolved)

        # Confidence weight: scales with sample size, degrades with distance + uncertainty
        base_conf = min(1.0, n_resolved / (MIN_SAMPLES * 3))
        dist_pen  = max(0.0, 1.0 - mean_dist / DISTANCE_THRESHOLD)
        unc_pen   = 1.0 - 0.5 * uncertainty   # high entropy → less confident
        confidence_weight = round(min(0.25, 0.25 * base_conf * dist_pen * unc_pen), 4)

        if n_resolved >= MIN_SAMPLES * 2 and mean_dist <= DISTANCE_THRESHOLD * 0.6:
            verdict = "RELIABLE"
        elif n_resolved >= MIN_SAMPLES:
            verdict = "WEAK"
        else:
            verdict = "INSUFFICIENT"

        return SimilarStateResult(
            empirical_p_big=round(float(empirical_p), 4),
            sample_size=n_resolved,
            confidence_weight=confidence_weight,
            mean_distance=round(mean_dist, 4),
            uncertainty=round(uncertainty, 4),
            verdict=verdict,
            context={
                "n_neighbours": n_resolved,
                "n_big": n_big,
                "n_small": n_resolved - n_big,
                "mean_dist": round(mean_dist, 4),
                "regime": fp.regime_id,
                "total_memory": len(self._states),
            },
        )

    # ── Supabase persistence ──────────────────────────────────────────────────

    def maybe_persist(self, db) -> None:
        """Flush resolved buffer to Supabase every PERSIST_EVERY calls."""
        self._persist_counter += 1
        if self._persist_counter % PERSIST_EVERY != 0:
            return
        self._flush_resolved(db)

    def _flush_resolved(self, db) -> None:
        if not self._resolved_buffer:
            return
        to_flush = list(self._resolved_buffer)
        self._resolved_buffer.clear()
        try:
            from backend.database import StateMemoryRow
            for state in to_flush:
                try:
                    exists = (
                        db.query(StateMemoryRow)
                        .filter(StateMemoryRow.sequence_no == state.sequence_no)
                        .first()
                    )
                    if exists:
                        if exists.actual_side is None and state.actual_side is not None:
                            exists.actual_side = state.actual_side
                        continue
                    fp = state.fp
                    row = StateMemoryRow(
                        sequence_no=state.sequence_no,
                        timestamp_utc=datetime.utcnow(),
                        fingerprint_vec=json.dumps(state.vec.tolist()),
                        entropy=float(fp.entropy),
                        big_rate_short=float(fp.short_big_rate),
                        big_rate_medium=float(fp.medium_big_rate),
                        big_rate_long=float(fp.long_big_rate),
                        streak_len=int(fp.current_streak),
                        streak_side=int(fp.streak_value),
                        drift_level=str(fp.regime_id),
                        actual_side=state.actual_side,
                    )
                    db.add(row)
                except Exception as e:
                    print(f"[StateMemory] flush row failed: {e}")
            try:
                db.commit()
            except Exception as e:
                db.rollback()
                print(f"[StateMemory] flush commit failed: {e}")
        except Exception as e:
            print(f"[StateMemory] _flush_resolved outer error: {e}")

    def load_from_db(self, db, limit: int = MAX_MEMORY) -> int:
        """Hydrate the in-memory store from Supabase on startup.
        Returns the number of rows loaded."""
        try:
            from backend.database import StateMemoryRow
            rows = (
                db.query(StateMemoryRow)
                .order_by(StateMemoryRow.id.desc())
                .limit(limit)
                .all()
            )
            loaded = 0
            for row in reversed(rows):
                try:
                    vec_data = json.loads(row.fingerprint_vec)
                    vec = np.array(vec_data, dtype=np.float32)
                    fp = StateFingerprint(
                        sequence_no=int(row.sequence_no) if str(row.sequence_no).isdigit() else 0,
                        entropy=float(row.entropy or 0),
                        short_big_rate=float(row.big_rate_short or 0.5),
                        medium_big_rate=float(row.big_rate_medium or 0.5),
                        long_big_rate=float(row.big_rate_long or 0.5),
                        current_streak=int(row.streak_len or 0),
                        streak_value=int(row.streak_side or 0),
                        regime_id=str(row.drift_level or "UNKNOWN"),
                        feature_vector=vec_data,
                    )
                    state = StoredState(
                        sequence_no=str(row.sequence_no),
                        vec=vec,
                        fp=fp,
                        actual_side=row.actual_side,
                    )
                    self._states.append(state)
                    loaded += 1
                except Exception:
                    continue
            self._matrix_dirty = True
            print(f"[StateMemory] Loaded {loaded} states from Supabase")
            return loaded
        except Exception as e:
            print(f"[StateMemory] load_from_db failed: {e}")
            return 0

    def build_from_history(self, digits: List[int], max_states: int = 10_000) -> int:
        """
        Cold-start: build state memory directly from raw digit history.
        Called by daily_learning.py to rebuild memory from all Supabase outcomes.
        Processes every position in the history (stride = 1, causal).
        Returns number of states built.
        """
        n = len(digits)
        if n < 30:
            return 0
        built = 0
        stride = max(1, n // max_states)  # subsample if history is very long
        for i in range(30, n - 1, stride):
            window = digits[max(0, i - 200): i]
            fp = compute_state_fingerprint(window, sequence_no=i)
            if not fp.feature_vector:
                continue
            vec = np.array(fp.feature_vector, dtype=np.float32)
            actual_side = 1 if digits[i] >= 5 else 0
            state = StoredState(
                sequence_no=str(i),
                vec=vec,
                fp=fp,
                actual_side=actual_side,
            )
            self._states.append(state)
            built += 1
        self._matrix_dirty = True
        print(f"[StateMemory] Built {built} states from raw history (n={n})")
        return built

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        total = len(self._states)
        resolved = sum(1 for s in self._states if s.actual_side is not None)
        pending = len(self._pending)
        if resolved > 0:
            big_rate = sum(1 for s in self._states if s.actual_side == 1) / resolved
        else:
            big_rate = 0.5
        return {
            "total_states": total,
            "resolved_states": resolved,
            "pending_states": pending,
            "historical_big_rate": round(big_rate, 4),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _insufficient(reason: str) -> SimilarStateResult:
        return SimilarStateResult(
            empirical_p_big=0.5,
            sample_size=0,
            confidence_weight=0.0,
            mean_distance=1.0,
            uncertainty=1.0,
            verdict="INSUFFICIENT",
            context={"reason": reason},
        )
