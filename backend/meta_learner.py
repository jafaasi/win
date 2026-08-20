#!/usr/bin/env python3
"""
Meta-Learner — EVOSEQ v3 Component
====================================
Answers:  "Which models should I trust in the CURRENT regime?"

The meta-learner does NOT predict the next outcome directly.
Its sole job is to output a weight vector over the 12 sub-models that
reflects each model's recent reliability under the current regime.

Architecture
------------
1. Per-model, per-regime reliability tracking
   For every (model_index, regime_id) pair we maintain an online
   exponentially-weighted accuracy estimate and a Brier score.

2. Walk-forward micro-evaluation (fast)
   Every EVAL_WINDOW resolved rounds we re-score each model against its
   own per-round predictions stored in the last observation buffer, using
   the correct OOS discipline: the score window never overlaps the
   window used to set the current weights.

3. Model degradation detection
   If a model's rolling Brier exceeds its 30-day baseline by more than
   DEGRADATION_THRESHOLD, it is soft-penalised: its weight is halved and
   a degradation event is logged.

4. Output: meta_weights
   A 12-element numpy array normalised to sum=1. This is fed into the
   Hedge ensemble as a prior (blended with the Hedge's own online weights)
   to get the best of both — fast online adaptation (Hedge) and regime-aware
   calibration (MetaLearner).

5. Persistence
   The per-model reliability tables are serialised to JSON and stored in
   `ai_brain_state` under key `"MetaLearner_State"` so knowledge survives
   restarts and accumulates daily.

Design constraints
------------------
• Pure numpy, no torch/sklearn.
• Must complete in < 5 ms per call so the 30-second cycle is not impacted.
• Never modifies historical records.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

N_MODELS           = 12
EVAL_WINDOW        = 50          # re-score models every N resolved rounds
DEGRADATION_THRESHOLD = 0.03     # Brier increase vs baseline = degradation
EWA_DECAY          = 0.97        # exponential decay for online accuracy
MIN_OBS_FOR_REGIME = 20          # minimum observations in a regime before trusting per-regime weights
DEGRADATION_PENALTY = 0.5        # multiply weight by this on degradation
META_LEARNER_KEY   = "MetaLearner_State"

MODEL_NAMES = [
    "hip_ctw",            # 0
    "hip_ngram",          # 1
    "hip_streak",         # 2
    "hip_frequency",      # 3
    "evoseq_ensemble",    # 4
    "decay_markov",       # 5
    "session_bias",       # 6
    "exploit_detector",   # 7
    "pattern_intelligence", # 8
    "three_level_ml",     # 9
    "volatility_regime",  # 10
    "cross_round_corr",   # 11
]

# ─────────────────────────────────────────────────────────────────────────────
# Per-model reliability cell
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelReliability:
    """Online reliability statistics for one model in one regime."""
    model_idx: int
    regime: str

    # Exponentially-weighted Brier score (lower = better)
    ewa_brier: float = 0.25
    # Exponentially-weighted accuracy (higher = better)
    ewa_accuracy: float = 0.5
    # 30-day baseline Brier (set once we have MIN_OBS_FOR_REGIME observations)
    baseline_brier: Optional[float] = None
    # Total observations seen
    n_obs: int = 0
    # Degradation flag
    degraded: bool = False
    # Last update time
    last_update: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def update(self, p_big: float, actual_side: int) -> None:
        """Online update given model's P(Big) and the true outcome."""
        brier = (p_big - float(actual_side)) ** 2
        correct = int((p_big >= 0.5) == (actual_side == 1))
        self.ewa_brier    = EWA_DECAY * self.ewa_brier    + (1 - EWA_DECAY) * brier
        self.ewa_accuracy = EWA_DECAY * self.ewa_accuracy + (1 - EWA_DECAY) * correct
        self.n_obs += 1
        self.last_update = datetime.utcnow().isoformat()
        # Set baseline after first MIN_OBS_FOR_REGIME obs
        if self.baseline_brier is None and self.n_obs >= MIN_OBS_FOR_REGIME:
            self.baseline_brier = self.ewa_brier
        # Degradation check
        if self.baseline_brier is not None:
            self.degraded = (self.ewa_brier - self.baseline_brier) > DEGRADATION_THRESHOLD

    def reliability_score(self) -> float:
        """
        Combined reliability score in [0, 1].
        Higher = more reliable = gets more weight.
        Maps Brier [0, 0.5] → reliability score [0, 1]:
            perfect (Brier=0) → 1.0
            random  (Brier=0.25) → 0.5
            always_wrong (Brier=0.5) → 0.0
        """
        raw = max(0.0, 1.0 - 2.0 * self.ewa_brier)
        # Apply degradation penalty
        if self.degraded:
            raw *= DEGRADATION_PENALTY
        return max(0.01, min(1.0, raw))  # floor at 0.01 so no model is fully silenced

    def to_dict(self) -> dict:
        return {
            "model_idx": self.model_idx,
            "regime": self.regime,
            "ewa_brier": round(self.ewa_brier, 6),
            "ewa_accuracy": round(self.ewa_accuracy, 4),
            "baseline_brier": round(self.baseline_brier, 6) if self.baseline_brier else None,
            "n_obs": self.n_obs,
            "degraded": self.degraded,
            "last_update": self.last_update,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelReliability":
        r = cls(model_idx=int(d["model_idx"]), regime=str(d["regime"]))
        r.ewa_brier      = float(d.get("ewa_brier", 0.25))
        r.ewa_accuracy   = float(d.get("ewa_accuracy", 0.5))
        r.baseline_brier = float(d["baseline_brier"]) if d.get("baseline_brier") else None
        r.n_obs          = int(d.get("n_obs", 0))
        r.degraded       = bool(d.get("degraded", False))
        r.last_update    = str(d.get("last_update", ""))
        return r


# ─────────────────────────────────────────────────────────────────────────────
# Observation buffer for OOS micro-evaluation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionObservation:
    """One resolved round: model predictions + actual outcome."""
    model_p_big: List[float]    # len=12, each model's P(Big)
    actual_side: int             # 1=Big, 0=Small
    regime: str

    def brier_per_model(self) -> List[float]:
        return [(p - self.actual_side) ** 2 for p in self.model_p_big]

    def correct_per_model(self) -> List[int]:
        return [int((p >= 0.5) == (self.actual_side == 1)) for p in self.model_p_big]


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward OOS scorer
# ─────────────────────────────────────────────────────────────────────────────

def _oos_model_scores(
    observations: List[PredictionObservation],
    train_frac: float = 0.60,
) -> Dict[str, np.ndarray]:
    """
    Compute OOS accuracy and Brier per model using strict temporal split.
    Returns dict with keys 'accuracy' and 'brier', each a len=12 array.
    Train on the first train_frac; evaluate on the remainder.
    Never shuffle.
    """
    n = len(observations)
    if n < 20:
        return {
            "accuracy": np.full(N_MODELS, 0.5),
            "brier":    np.full(N_MODELS, 0.25),
        }

    split = max(1, int(n * train_frac))
    test  = observations[split:]
    if not test:
        test = observations

    acc_sum   = np.zeros(N_MODELS, dtype=np.float64)
    brier_sum = np.zeros(N_MODELS, dtype=np.float64)
    for obs in test:
        for i, p in enumerate(obs.model_p_big):
            brier_sum[i] += (p - obs.actual_side) ** 2
            acc_sum[i]   += int((p >= 0.5) == (obs.actual_side == 1))

    n_test = len(test)
    return {
        "accuracy": acc_sum / n_test,
        "brier":    brier_sum / n_test,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main MetaLearner class
# ─────────────────────────────────────────────────────────────────────────────

class MetaLearner:
    """
    Regime-aware model weight advisor.

    Usage (per cycle):
        ml = MetaLearner()
        ml.load_state(db)           # startup

        # After each resolved draw:
        ml.update(model_p_big_vector, actual_side, current_regime)

        # Before each prediction:
        weights = ml.get_meta_weights(current_regime)  # len-12 ndarray summing to 1

        ml.maybe_save(db)           # persists every EVAL_WINDOW updates
    """

    PERSIST_KEY = META_LEARNER_KEY

    def __init__(self):
        # (model_idx, regime) -> ModelReliability
        self._reliability: Dict[Tuple[int, str], ModelReliability] = {}
        # Global (regime-agnostic) reliability, used as fallback
        self._global: Dict[int, ModelReliability] = {
            i: ModelReliability(model_idx=i, regime="GLOBAL") for i in range(N_MODELS)
        }
        # Recent observation buffer for OOS micro-evaluation
        self._obs_buffer: Deque[PredictionObservation] = deque(maxlen=2000)
        # Degradation event log (last 50)
        self._degradation_log: List[dict] = []
        # Update counter — triggers persist/OOS-eval every EVAL_WINDOW updates
        self._update_count = 0
        # Cached OOS weights (recomputed every EVAL_WINDOW updates)
        self._oos_weights: Optional[np.ndarray] = None

    # ── Online update ─────────────────────────────────────────────────────────

    def update(
        self,
        model_p_big: List[float],
        actual_side: int,
        regime: str,
    ) -> None:
        """
        Feed one resolved round into the meta-learner.
        model_p_big: list/array of 12 P(Big) values (one per model).
        actual_side: 1=Big, 0=Small.
        regime: current drift/regime label string.
        """
        if len(model_p_big) != N_MODELS:
            return

        obs = PredictionObservation(
            model_p_big=[float(p) for p in model_p_big],
            actual_side=int(actual_side),
            regime=str(regime),
        )
        self._obs_buffer.append(obs)

        for i, p in enumerate(model_p_big):
            # Update per-regime cell
            key = (i, regime)
            if key not in self._reliability:
                self._reliability[key] = ModelReliability(model_idx=i, regime=regime)
            self._reliability[key].update(float(p), int(actual_side))
            # Update global cell
            self._global[i].update(float(p), int(actual_side))

            # Log degradation events
            if self._reliability[key].degraded and not any(
                e["model"] == MODEL_NAMES[i] and e["regime"] == regime
                for e in self._degradation_log[-5:]
            ):
                self._degradation_log.append({
                    "model": MODEL_NAMES[i],
                    "regime": regime,
                    "ewa_brier": round(self._reliability[key].ewa_brier, 4),
                    "baseline": round(self._reliability[key].baseline_brier or 0.25, 4),
                    "detected_at": datetime.utcnow().isoformat(),
                })
                if len(self._degradation_log) > 50:
                    self._degradation_log = self._degradation_log[-50:]

        self._update_count += 1
        # Re-run OOS evaluation every EVAL_WINDOW updates
        if self._update_count % EVAL_WINDOW == 0:
            self._refresh_oos_weights()

    # ── Weight output ─────────────────────────────────────────────────────────

    def get_meta_weights(self, regime: str) -> np.ndarray:
        """
        Return a len-12 weight vector for the given regime.

        Blend of three sources (priority order):
          1. OOS-evaluated weights (most objective, 50%)
          2. Per-regime reliability scores (35%)
          3. Global reliability fallback (15%)

        All normalised to sum=1.
        """
        global_w  = self._global_weights()
        regime_w  = self._regime_weights(regime)
        oos_w     = self._oos_weights if self._oos_weights is not None else np.ones(N_MODELS) / N_MODELS

        blended = 0.50 * oos_w + 0.35 * regime_w + 0.15 * global_w
        blended = np.clip(blended, 1e-4, None)
        return blended / blended.sum()

    def _global_weights(self) -> np.ndarray:
        w = np.array([self._global[i].reliability_score() for i in range(N_MODELS)], dtype=np.float64)
        w = np.clip(w, 1e-4, None)
        return w / w.sum()

    def _regime_weights(self, regime: str) -> np.ndarray:
        w = np.zeros(N_MODELS, dtype=np.float64)
        for i in range(N_MODELS):
            key = (i, regime)
            if key in self._reliability and self._reliability[key].n_obs >= MIN_OBS_FOR_REGIME:
                w[i] = self._reliability[key].reliability_score()
            else:
                # Fall back to global if insufficient regime data
                w[i] = self._global[i].reliability_score()
        w = np.clip(w, 1e-4, None)
        return w / w.sum()

    def _refresh_oos_weights(self) -> None:
        """Recompute OOS weights from the observation buffer."""
        obs = list(self._obs_buffer)
        if len(obs) < 20:
            return
        scores = _oos_model_scores(obs)
        # Convert Brier to weight: lower Brier = higher weight
        # Map brier [0, 0.5] → score [1, 0], then normalise
        brier = scores["brier"]
        w = np.clip(1.0 - 2.0 * brier, 0.01, 1.0)
        self._oos_weights = w / w.sum()

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def model_report(self, regime: str = "GLOBAL") -> List[dict]:
        """Return per-model reliability report for the given regime."""
        report = []
        for i, name in enumerate(MODEL_NAMES):
            key = (i, regime)
            cell = self._reliability.get(key, self._global[i])
            oos_w = float(self._oos_weights[i]) if self._oos_weights is not None else 1 / N_MODELS
            report.append({
                "model": name,
                "index": i,
                "regime": regime,
                "ewa_brier": round(cell.ewa_brier, 4),
                "ewa_accuracy": round(cell.ewa_accuracy, 4),
                "n_obs": cell.n_obs,
                "degraded": cell.degraded,
                "reliability_score": round(cell.reliability_score(), 4),
                "oos_weight": round(oos_w, 4),
                "meta_weight": round(float(self.get_meta_weights(regime)[i]), 4),
            })
        return report

    def degradation_events(self, last_n: int = 10) -> List[dict]:
        return self._degradation_log[-last_n:]

    def strongest_model(self, regime: str) -> str:
        w = self.get_meta_weights(regime)
        return MODEL_NAMES[int(np.argmax(w))]

    def weakest_model(self, regime: str) -> str:
        w = self.get_meta_weights(regime)
        return MODEL_NAMES[int(np.argmin(w))]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_state(self, db) -> None:
        """Persist reliability tables to Supabase `ai_brain_state`."""
        try:
            from backend.database import save_ai_brain_state
            payload = {
                "reliability": {
                    f"{i}_{r}": v.to_dict()
                    for (i, r), v in self._reliability.items()
                },
                "global": {str(i): v.to_dict() for i, v in self._global.items()},
                "degradation_log": self._degradation_log,
                "update_count": self._update_count,
                "saved_at": datetime.utcnow().isoformat(),
            }
            save_ai_brain_state(
                db=db,
                model_name=self.PERSIST_KEY,
                generation=self._update_count,
                total_samples=self._update_count,
                weights_json=json.dumps(payload),
                win_rate=0.0,
            )
        except Exception as e:
            print(f"[MetaLearner] save_state failed: {e}")

    def load_state(self, db) -> bool:
        """Load reliability tables from Supabase. Returns True on success."""
        try:
            from backend.database import load_ai_brain_state
            brain = load_ai_brain_state(db, model_name=self.PERSIST_KEY)
            if not brain or not brain.synaptic_weights:
                return False
            payload = json.loads(brain.synaptic_weights)

            # Reliability
            for composite_key, v in payload.get("reliability", {}).items():
                parts = composite_key.split("_", 1)
                if len(parts) != 2:
                    continue
                try:
                    i = int(parts[0])
                    r = parts[1]
                    self._reliability[(i, r)] = ModelReliability.from_dict(v)
                except Exception:
                    continue

            # Global
            for i_str, v in payload.get("global", {}).items():
                try:
                    i = int(i_str)
                    self._global[i] = ModelReliability.from_dict(v)
                except Exception:
                    continue

            self._degradation_log = payload.get("degradation_log", [])
            self._update_count = int(payload.get("update_count", 0))
            print(
                f"[MetaLearner] Loaded state: {len(self._reliability)} regime cells, "
                f"{self._update_count} updates"
            )
            return True
        except Exception as e:
            print(f"[MetaLearner] load_state failed: {e}")
            return False

    def maybe_save(self, db) -> None:
        """Persist every EVAL_WINDOW updates."""
        if self._update_count > 0 and self._update_count % EVAL_WINDOW == 0:
            self.save_state(db)
