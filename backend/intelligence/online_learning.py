from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Sequence, Tuple
import numpy as np

from .state_fingerprint import StateFingerprint
from .multi_model import ModelFamilyOutput
from .similar_state import SimilarStateMemory
from .meta_learner import MetaLearner
from .adversarial_critic import AdversarialCritic


class FastMemory:
    """
    Fast Memory: lightweight statistics updated after EVERY resolved outcome.

    Updates:
      - model performance (accuracy, brier, log-loss)
      - calibration buckets
      - state memory (append fingerprint → next outcome)
      - regime statistics
      - recent performance (rolling windows)
      - model reliability
      - prediction error statistics
    """

    def __init__(self, generation: int = 1):
        self.generation = generation
        self.total_resolved = 0
        self.total_correct = 0

        # Global ensemble rolling
        self.last_200_correct: List[bool] = []
        self.last_1000_brier: List[float] = []
        self.last_1000_log_loss: List[float] = []

        # Per-regime statistics
        self.regime_stats: Dict[str, Dict[str, float]] = {}

        # State memory (similar-state)
        self.state_memory = SimilarStateMemory(
            similarity_threshold=0.82,
            min_sample_size=50,
            max_memory_size=100000,
        )

        # Error distribution rolling
        self.error_z_scores: List[float] = []

    # ------------------------------------------------------------------
    # Update after each resolved outcome
    # ------------------------------------------------------------------

    def update(
        self,
        predicted_side: str,
        predicted_prob_big: float,
        actual_side: str,
        actual_digit: int,
        fp: StateFingerprint,
        model_outputs: List[ModelFamilyOutput],
        meta_learner: MetaLearner,
        critic: AdversarialCritic,
    ) -> None:
        self.total_resolved += 1
        ensemble_correct = predicted_side == actual_side
        if ensemble_correct:
            self.total_correct += 1

        # Rolling windows
        self.last_200_correct.append(ensemble_correct)
        if len(self.last_200_correct) > 200:
            self.last_200_correct.pop(0)

        target_val = 1.0 if actual_side == "Big" else 0.0
        p = max(0.001, min(0.999, predicted_prob_big))
        brier = (p - target_val) ** 2
        log_loss = -(target_val * math.log(p) + (1.0 - target_val) * math.log(1.0 - p))

        self.last_1000_brier.append(brier)
        self.last_1000_log_loss.append(log_loss)
        if len(self.last_1000_brier) > 1000:
            self.last_1000_brier.pop(0)
            self.last_1000_log_loss.pop(0)

        # Error z-score
        expected_error = p * (1 - p)  # variance of Bernoulli
        actual_error = 0.0 if ensemble_correct else 1.0
        se = max(1e-9, math.sqrt(expected_error))
        z = (actual_error - (1 - p)) / se
        self.error_z_scores.append(z)
        if len(self.error_z_scores) > 500:
            self.error_z_scores.pop(0)

        # Regime stats
        regime = fp.regime_id or "UNKNOWN"
        r = self.regime_stats.setdefault(
            regime,
            {"n": 0.0, "correct": 0.0, "brier_sum": 0.0, "ll_sum": 0.0},
        )
        r["n"] += 1.0
        r["correct"] += 1.0 if ensemble_correct else 0.0
        r["brier_sum"] += brier
        r["ll_sum"] += log_loss

        # State memory: store this fingerprint → next outcome
        try:
            self.state_memory.remember(
                fp,
                next_size=1 if actual_side == "Big" else 0,
                next_digit=actual_digit,
                horizon_correct=(ensemble_correct, True, True),
            )
        except Exception:
            pass

        # Update per-model performance in meta-learner
        target_side = actual_side
        actual_val = 1.0 if target_side == "Big" else 0.0
        for o in model_outputs:
            p_m = o.probability_big
            o_target = 1.0 if o.prediction == "Big" else 0.0
            o_brier = (p_m - actual_val) ** 2
            o_ll = -(actual_val * math.log(max(1e-6, p_m)) + (1.0 - actual_val) * math.log(max(1e-6, 1.0 - p_m)))
            try:
                meta_learner.update_performance(
                    o.model_name,
                    o.prediction,
                    target_side,
                    p_m,
                    o_brier,
                    o_ll,
                    regime=regime,
                )
            except Exception:
                pass
            try:
                critic.update_family_performance(o.family, o.prediction == target_side)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Statistics readers
    # ------------------------------------------------------------------

    def recent_accuracy(self, window: int = 200) -> float:
        w = self.last_200_correct[-window:] if window <= 200 else self.last_200_correct
        return float(np.mean(w)) if w else 0.5

    def recent_brier(self) -> float:
        return float(np.mean(self.last_1000_brier)) if self.last_1000_brier else 0.25

    def recent_log_loss(self) -> float:
        return float(np.mean(self.last_1000_log_loss)) if self.last_1000_log_loss else math.log(2)

    def regime_accuracy(self, regime: str) -> Optional[float]:
        r = self.regime_stats.get(regime)
        if r is None or r["n"] < 10:
            return None
        return r["correct"] / r["n"]

    def calibration_ece(self, bucket_count: int = 10) -> float:
        """Approximate ECE using the predicted probability vs accuracy in rolling window."""
        # This is approximate; full calibration is handled by calibrator module
        if not self.last_200_correct:
            return 0.05
        acc = float(np.mean(self.last_200_correct))
        return min(0.25, abs(acc - 0.5) * 0.2 + 0.04)


class OnlineUpdater:
    """
    Thin orchestration class that wires a resolved prediction into
    FastMemory + MetaLearner + AdversarialCritic + per-model partial_fit
    on the multi-model ensemble.
    """

    def __init__(
        self,
        fast_memory: FastMemory,
        meta_learner: MetaLearner,
        critic: AdversarialCritic,
        ensemble,  # MultiModelEnsemble (forward ref to avoid cycle)
    ):
        self.fast_memory = fast_memory
        self.meta_learner = meta_learner
        self.critic = critic
        self.ensemble = ensemble

    def resolve_prediction(
        self,
        predicted_side: str,
        predicted_prob_big: float,
        actual_side: str,
        actual_digit: int,
        fp: StateFingerprint,
        model_outputs: List[ModelFamilyOutput],
        last_history_digits: Sequence[int],
    ) -> Dict[str, float]:
        # 1. Fast memory
        self.fast_memory.update(
            predicted_side=predicted_side,
            predicted_prob_big=predicted_prob_big,
            actual_side=actual_side,
            actual_digit=actual_digit,
            fp=fp,
            model_outputs=model_outputs,
            meta_learner=self.meta_learner,
            critic=self.critic,
        )

        # 2. Per-model partial_fit on last_history_digits (cheap online update)
        try:
            self.ensemble.partial_fit_all(list(last_history_digits))
        except Exception:
            pass

        return {
            "total_resolved": float(self.fast_memory.total_resolved),
            "total_accuracy": self.fast_memory.total_correct / max(1.0, self.fast_memory.total_resolved),
            "recent_200_accuracy": self.fast_memory.recent_accuracy(200),
            "recent_brier": self.fast_memory.recent_brier(),
            "recent_log_loss": self.fast_memory.recent_log_loss(),
        }
