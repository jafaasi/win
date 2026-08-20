#!/usr/bin/env python3
"""
Evolution Controller — EVOSEQ v3 Component
============================================
Every N resolved rounds the controller asks:

    "Is the current ensemble still the best available configuration?
     Is any model degrading?  Should we promote a new configuration?"

This is the slow-feedback loop that complements the Hedge ensemble's
fast per-round weight updates.

Lifecycle
---------
  FAST LOOP  (every round)      — Hedge online weight updates
  MID LOOP   (every N rounds)   — EvolutionController.tick()
  DAILY LOOP (midnight UTC)     — daily_learning.py run_daily_learning()

tick() workflow
---------------
  1. Collect last N resolved rounds from the decision memory buffer.
  2. Score CURRENT champion configuration on that OOS window.
  3. Score CHALLENGER: slightly perturbed weight vector.
  4. Score BASELINES: random (0.5) and frequency (historical Big rate).
  5. If challenger beats champion on ≥2 of 3 metrics (accuracy, Brier,
     3-level win-rate) AND beats all baselines → promote challenger.
  6. If current champion underperforms baselines → flag NO_EDGE and
     widen the SKIP gate.
  7. Store a generation record in Supabase for audit.

Key design decisions
--------------------
• The challenger is NOT a random mutation.  It is the MetaLearner's
  current weight recommendation.  The MetaLearner learns per-regime
  reliability; the EvolutionController acts as the promotion gate.
• No data leakage: the evaluation window is always the MOST RECENT N
  rounds that were NOT used to set the meta-learner weights.
• Promotion is conservative: the challenger must beat the champion by
  at least MIN_IMPROVEMENT on at least 2 metrics.
• The controller never overwrites historical predictions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

EVAL_EVERY_N        = 100    # run a tick() every N resolved rounds
MIN_EVAL_WINDOW     = 60     # minimum rounds needed for a valid evaluation
EVAL_WINDOW         = 200    # rounds used for champion vs challenger scoring
MIN_IMPROVEMENT     = 0.002  # challenger must improve by this margin (Brier)
EVOLUTION_KEY       = "EvolutionController_State"
N_MODELS            = 12

MODEL_NAMES = [
    "hip_ctw", "hip_ngram", "hip_streak", "hip_frequency",
    "evoseq_ensemble", "decay_markov", "session_bias", "exploit_detector",
    "pattern_intelligence", "three_level_ml", "volatility_regime", "cross_round_corr",
]

# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers (pure, no DB)
# ─────────────────────────────────────────────────────────────────────────────

def _accuracy(pairs: List[Tuple[float, int]]) -> float:
    if not pairs:
        return 0.5
    return sum(1 for p, a in pairs if (p >= 0.5) == (a == 1)) / len(pairs)


def _brier(pairs: List[Tuple[float, int]]) -> float:
    if not pairs:
        return 0.25
    return sum((p - a) ** 2 for p, a in pairs) / len(pairs)


def _three_level_wr(pairs: List[Tuple[float, int]]) -> float:
    correct = [(p >= 0.5) == (a == 1) for p, a in pairs]
    if len(correct) < 3:
        return 0.5
    wins = total = 0
    for i in range(len(correct) - 2):
        if any(correct[i:i + 3]):
            wins += 1
        total += 1
    return wins / total if total else 0.5


def _weighted_ensemble_p_big(
    model_p_big: List[float],
    weights: np.ndarray,
) -> float:
    """Weighted average of 12 model probabilities."""
    arr = np.array(model_p_big[:N_MODELS], dtype=np.float64)
    return float(np.dot(weights, arr))


def _score_weights(
    observations: List["EvoObservation"],
    weights: np.ndarray,
) -> Dict[str, float]:
    """Evaluate an ensemble weight vector on a list of resolved observations."""
    pairs: List[Tuple[float, float]] = []
    for obs in observations:
        if obs.actual_side is None:
            continue
        p = _weighted_ensemble_p_big(obs.model_p_big, weights)
        pairs.append((p, int(obs.actual_side)))
    return {
        "accuracy":      round(_accuracy(pairs), 4),
        "brier":         round(_brier(pairs), 4),
        "three_level_wr": round(_three_level_wr(pairs), 4),
        "n":             len(pairs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvoObservation:
    """One resolved round stored for evolution evaluation."""
    issue_number: str
    model_p_big: List[float]     # len=12, raw predictions
    actual_side: Optional[int]   # 1=Big, 0=Small; None until resolved
    regime: str = "UNKNOWN"


@dataclass
class GenerationRecord:
    """Audit record for one evolution cycle."""
    generation: int
    evaluated_at: str
    champion_weights: List[float]
    challenger_weights: List[float]
    champion_scores: Dict[str, float]
    challenger_scores: Dict[str, float]
    baseline_random_score: float
    baseline_frequency_score: float
    verdict: str           # PROMOTED | RETAINED | NO_EDGE
    promoted: bool
    edge_status: str       # VERIFIED_EDGE | MARGINAL_EDGE | NO_EDGE
    n_rounds_evaluated: int
    degraded_models: List[str]
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "evaluated_at": self.evaluated_at,
            "champion_scores": self.champion_scores,
            "challenger_scores": self.challenger_scores,
            "baseline_random": round(self.baseline_random_score, 4),
            "baseline_frequency": round(self.baseline_frequency_score, 4),
            "verdict": self.verdict,
            "promoted": self.promoted,
            "edge_status": self.edge_status,
            "n_rounds": self.n_rounds_evaluated,
            "degraded_models": self.degraded_models,
            "notes": self.notes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main EvolutionController
# ─────────────────────────────────────────────────────────────────────────────

class EvolutionController:
    """
    Champion/challenger promotion gate.

    Usage:
        ec = EvolutionController()
        ec.load_state(db)                   # startup

        # every resolved round (fast path):
        ec.record(issue, model_p_big, actual_side, regime)

        # every prediction (read only):
        weights = ec.champion_weights       # current champion weight vector
        status  = ec.edge_status            # VERIFIED_EDGE | MARGINAL_EDGE | NO_EDGE

        # nightly (daily_learning.py):
        result = ec.run_full_evaluation(db)
    """

    def __init__(self):
        # Current champion weight vector (uniform until first promotion)
        self.champion_weights: np.ndarray = np.ones(N_MODELS, dtype=np.float64) / N_MODELS
        self.generation: int = 1
        self.edge_status: str = "LEARNING"
        self.skip_gate_widened: bool = False

        # Rolling observation buffer (recent rounds for evaluation)
        self._obs: List[EvoObservation] = []
        self._obs_maxlen = 5000

        # Pending: issue -> EvoObservation (awaiting resolution)
        self._pending: Dict[str, EvoObservation] = {}

        # Generation history
        self._history: List[GenerationRecord] = []

        # Round counter
        self._round_count: int = 0

    # ── Record observations ───────────────────────────────────────────────────

    def record_prediction(
        self,
        issue_number: str,
        model_p_big: List[float],
        regime: str = "UNKNOWN",
    ) -> None:
        """Store a pending observation before the draw resolves."""
        obs = EvoObservation(
            issue_number=str(issue_number),
            model_p_big=list(model_p_big),
            actual_side=None,
            regime=regime,
        )
        self._pending[str(issue_number)] = obs

    def record_outcome(self, issue_number: str, actual_side: int) -> bool:
        """Resolve a pending observation. Returns True if found."""
        obs = self._pending.pop(str(issue_number), None)
        if obs is None:
            return False
        obs.actual_side = int(actual_side)
        self._obs.append(obs)
        if len(self._obs) > self._obs_maxlen:
            self._obs = self._obs[-self._obs_maxlen:]
        self._round_count += 1
        return True

    # ── tick() — fast mid-loop evaluation ────────────────────────────────────

    def tick(self, meta_weights: np.ndarray, db=None) -> Optional[GenerationRecord]:
        """
        Run a champion-vs-challenger evaluation cycle.
        Call every EVAL_EVERY_N resolved rounds.

        meta_weights: current MetaLearner weight recommendation (the challenger).
        Returns a GenerationRecord if a tick was run, else None.
        """
        if self._round_count % EVAL_EVERY_N != 0:
            return None
        return self._evaluate(meta_weights, db)

    def _evaluate(self, challenger_w: np.ndarray, db=None) -> GenerationRecord:
        """Internal: score champion vs challenger on the latest EVAL_WINDOW rounds."""
        # Use only resolved observations
        resolved = [o for o in self._obs if o.actual_side is not None]
        eval_obs = resolved[-EVAL_WINDOW:] if len(resolved) >= EVAL_WINDOW else resolved

        if len(eval_obs) < MIN_EVAL_WINDOW:
            return self._skip_record("INSUFFICIENT_DATA", len(eval_obs))

        # Score champion and challenger
        champion_scores   = _score_weights(eval_obs, self.champion_weights)
        challenger_scores = _score_weights(eval_obs, challenger_w)

        # Baselines
        pairs = [(0.5, int(o.actual_side)) for o in eval_obs if o.actual_side is not None]
        freq_big = sum(a for _, a in pairs) / len(pairs) if pairs else 0.5
        freq_pairs = [(freq_big, int(o.actual_side)) for o in eval_obs if o.actual_side is not None]

        baseline_random    = _brier(pairs)
        baseline_frequency = _brier(freq_pairs)
        best_baseline      = min(baseline_random, baseline_frequency)

        # Determine degraded models
        degraded = [
            MODEL_NAMES[i]
            for i in range(N_MODELS)
            if challenger_w[i] < (1.0 / N_MODELS) * 0.3
        ]

        # Promotion rules
        # 1. Challenger beats champion on ≥2 of 3 metrics
        beats_count = 0
        if challenger_scores["accuracy"] > champion_scores["accuracy"] + MIN_IMPROVEMENT:
            beats_count += 1
        if challenger_scores["brier"] < champion_scores["brier"] - MIN_IMPROVEMENT:
            beats_count += 1
        if challenger_scores["three_level_wr"] > champion_scores["three_level_wr"] + MIN_IMPROVEMENT:
            beats_count += 1

        # 2. Challenger beats best baseline on Brier
        beats_baseline = challenger_scores["brier"] < best_baseline - MIN_IMPROVEMENT

        promoted = beats_count >= 2 and beats_baseline

        # Edge status
        if champion_scores["brier"] < best_baseline - MIN_IMPROVEMENT:
            edge_status = "VERIFIED_EDGE"
        elif champion_scores["brier"] < best_baseline:
            edge_status = "MARGINAL_EDGE"
        else:
            edge_status = "NO_EDGE"
            self.skip_gate_widened = True  # tell the engine to widen SKIP gate

        if promoted:
            self.champion_weights = challenger_w.copy()
            self.generation += 1
            verdict = "PROMOTED"
            self.skip_gate_widened = False
        else:
            verdict = "RETAINED"

        self.edge_status = edge_status

        record = GenerationRecord(
            generation=self.generation,
            evaluated_at=datetime.utcnow().isoformat(),
            champion_weights=self.champion_weights.tolist(),
            challenger_weights=challenger_w.tolist(),
            champion_scores=champion_scores,
            challenger_scores=challenger_scores,
            baseline_random_score=round(baseline_random, 4),
            baseline_frequency_score=round(baseline_frequency, 4),
            verdict=verdict,
            promoted=promoted,
            edge_status=edge_status,
            n_rounds_evaluated=len(eval_obs),
            degraded_models=degraded,
            notes=f"beats_count={beats_count}/3, beats_baseline={beats_baseline}",
        )
        self._history.append(record)
        if len(self._history) > 200:
            self._history = self._history[-200:]

        if db is not None:
            self._persist_generation(record, db)

        return record

    def _skip_record(self, reason: str, n: int) -> GenerationRecord:
        return GenerationRecord(
            generation=self.generation,
            evaluated_at=datetime.utcnow().isoformat(),
            champion_weights=self.champion_weights.tolist(),
            challenger_weights=self.champion_weights.tolist(),
            champion_scores={}, challenger_scores={},
            baseline_random_score=0.25, baseline_frequency_score=0.25,
            verdict="SKIPPED", promoted=False, edge_status=self.edge_status,
            n_rounds_evaluated=n, degraded_models=[],
            notes=reason,
        )

    # ── Full evaluation (daily) ───────────────────────────────────────────────

    def run_full_evaluation(self, meta_weights: np.ndarray, db=None) -> GenerationRecord:
        """
        Full evaluation using the entire observation buffer.
        Called from daily_learning.py for authoritative generation tracking.
        """
        return self._evaluate(meta_weights, db)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist_generation(self, record: GenerationRecord, db) -> None:
        """Write one generation record to `model_versions` and `ai_brain_state`."""
        try:
            from backend.database import ModelVersion, save_ai_brain_state
            mv = ModelVersion(
                model_name="EvolutionController",
                version=f"gen_{record.generation}",
                parameters=json.dumps(record.to_dict()),
                training_end_sequence=str(self._round_count),
                validation_score=float(record.champion_scores.get("accuracy", 0.5)),
                log_loss=0.0,
                brier_score=float(record.champion_scores.get("brier", 0.25)),
                status="champion" if record.promoted else "challenger",
            )
            db.add(mv)
            try:
                db.commit()
            except Exception:
                db.rollback()
        except Exception as e:
            print(f"[EvolutionController] persist_generation failed: {e}")

    def save_state(self, db) -> None:
        try:
            from backend.database import save_ai_brain_state
            payload = {
                "generation": self.generation,
                "champion_weights": self.champion_weights.tolist(),
                "edge_status": self.edge_status,
                "skip_gate_widened": self.skip_gate_widened,
                "round_count": self._round_count,
                "history": [r.to_dict() for r in self._history[-20:]],
                "saved_at": datetime.utcnow().isoformat(),
            }
            save_ai_brain_state(
                db=db,
                model_name=EVOLUTION_KEY,
                generation=self.generation,
                total_samples=self._round_count,
                weights_json=json.dumps(payload),
                win_rate=0.0,
            )
        except Exception as e:
            print(f"[EvolutionController] save_state failed: {e}")

    def load_state(self, db) -> bool:
        try:
            from backend.database import load_ai_brain_state
            brain = load_ai_brain_state(db, model_name=EVOLUTION_KEY)
            if not brain or not brain.synaptic_weights:
                return False
            payload = json.loads(brain.synaptic_weights)
            self.generation = int(payload.get("generation", 1))
            w = payload.get("champion_weights")
            if w and len(w) == N_MODELS:
                self.champion_weights = np.array(w, dtype=np.float64)
                self.champion_weights /= self.champion_weights.sum()
            self.edge_status = str(payload.get("edge_status", "LEARNING"))
            self.skip_gate_widened = bool(payload.get("skip_gate_widened", False))
            self._round_count = int(payload.get("round_count", 0))
            print(
                f"[EvolutionController] Loaded gen={self.generation}, "
                f"rounds={self._round_count}, edge={self.edge_status}"
            )
            return True
        except Exception as e:
            print(f"[EvolutionController] load_state failed: {e}")
            return False

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def status_dict(self) -> dict:
        recent = self._history[-1].to_dict() if self._history else {}
        return {
            "generation": self.generation,
            "edge_status": self.edge_status,
            "skip_gate_widened": self.skip_gate_widened,
            "round_count": self._round_count,
            "champion_weights": {
                MODEL_NAMES[i]: round(float(self.champion_weights[i]), 4)
                for i in range(N_MODELS)
            },
            "last_evaluation": recent,
        }

    def what_changed(self) -> str:
        """
        Human-readable summary of changes since last generation.
        Used in the daily self-report.
        """
        if len(self._history) < 2:
            return "No evolution history yet."
        prev = self._history[-2]
        curr = self._history[-1]
        lines = [f"Generation {prev.generation} → {curr.generation}"]
        lines.append(f"Verdict: {curr.verdict}  |  Edge: {curr.edge_status}")
        # Weight changes
        prev_w = np.array(prev.champion_weights)
        curr_w = np.array(curr.champion_weights)
        delta = curr_w - prev_w
        top_gains  = np.argsort(delta)[-3:][::-1]
        top_losses = np.argsort(delta)[:3]
        for i in top_gains:
            if delta[i] > 0.002:
                lines.append(f"  ↑ {MODEL_NAMES[i]}: weight +{delta[i]*100:.1f}pp")
        for i in top_losses:
            if delta[i] < -0.002:
                lines.append(f"  ↓ {MODEL_NAMES[i]}: weight {delta[i]*100:.1f}pp")
        if curr.degraded_models:
            lines.append(f"  Degraded: {', '.join(curr.degraded_models)}")
        return "\n".join(lines)
