from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Sequence, Tuple
import numpy as np

from .state_fingerprint import StateFingerprint
from .multi_model import ModelFamilyOutput
from .similar_state import SimilarStateResult


@dataclass
class MetaLearnerOutput:
    model_weights: Dict[str, float]
    ensemble_probability_big: float
    ensemble_probability_small: float
    ensemble_digit_vector: List[float]
    uncertainty: float
    reliability: float
    edge_status: str
    learning_status: str
    ensemble_target_digit: int
    ensemble_hedge_digit: int

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class MetaLearner:
    """
    The Meta-Learner does NOT predict directly.
    It learns WHICH models to trust in the current state.

    Inputs:
      - model probabilities
      - model agreement / disagreement
      - state fingerprint
      - regime
      - model historical accuracy (fast memory)
      - model sample sizes
      - calibration state
      - similar-state memory evidence

    Outputs:
      - model weights
      - ensemble probability
      - uncertainty
      - reliability
    """

    def __init__(self, generation: int = 1):
        self.generation = generation

        # Fast-memory performance tracking: model_name -> dict of metrics
        self.model_performance: Dict[str, Dict[str, float]] = {}

        # Regime-specific weighting multipliers learned online
        self.regime_model_boost: Dict[str, Dict[str, float]] = {}

        # Calibration trackers per-model
        self.model_calibration: Dict[str, float] = {}

        # Decay factor for historical performance
        self.decay = 0.992

        # How much of each model's own confidence to trust
        self.confidence_trust = 0.6

    # ------------------------------------------------------------------
    # Fast-memory updates (Online Learning)
    # ------------------------------------------------------------------

    def update_performance(
        self,
        model_name: str,
        predicted_side: str,
        actual_side: str,
        predicted_prob_big: float,
        brier: float,
        log_loss: float,
        regime: str = "UNKNOWN",
    ) -> None:
        perf = self.model_performance.setdefault(
            model_name,
            {
                "n": 0.0,
                "correct": 0.0,
                "recent_50_correct": 0.0,
                "recent_50_n": 0.0,
                "brier_sum": 0.0,
                "log_loss_sum": 0.0,
                "per_regime_n": {},
                "per_regime_correct": {},
            },
        )
        correct = 1.0 if predicted_side == actual_side else 0.0

        # Decay existing
        perf["n"] *= self.decay
        perf["correct"] *= self.decay
        perf["brier_sum"] *= self.decay
        perf["log_loss_sum"] *= self.decay

        # Add new sample
        perf["n"] += 1.0
        perf["correct"] += correct
        perf["brier_sum"] += brier
        perf["log_loss_sum"] += log_loss

        # Rolling recent-50 via exponential decay approximation
        perf["recent_50_correct"] = perf["recent_50_correct"] * 0.98 + correct
        perf["recent_50_n"] = perf["recent_50_n"] * 0.98 + 1.0

        # Per-regime tracking
        rkey = regime or "UNKNOWN"
        perf["per_regime_n"][rkey] = perf["per_regime_n"].get(rkey, 0.0) * self.decay + 1.0
        perf["per_regime_correct"][rkey] = perf["per_regime_correct"].get(rkey, 0.0) * self.decay + correct

        # Update calibration: ECE-like single-bucket estimate
        cal = self.model_calibration.setdefault(model_name, 0.5)
        predicted_conf = max(predicted_prob_big, 1.0 - predicted_prob_big)
        self.model_calibration[model_name] = 0.97 * cal + 0.03 * abs(predicted_conf - correct)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_model_accuracy(self, model_name: str, regime: str) -> float:
        perf = self.model_performance.get(model_name)
        if perf is None or perf["n"] < 3:
            return 0.5  # No evidence → treat as coin flip baseline
        # Prefer regime-specific accuracy if enough samples
        r_n = perf["per_regime_n"].get(regime, 0.0)
        if r_n >= 10:
            r_c = perf["per_regime_correct"].get(regime, 0.0)
            return r_c / r_n if r_n > 0 else 0.5
        return perf["correct"] / perf["n"] if perf["n"] > 0 else 0.5

    def _get_model_recent_accuracy(self, model_name: str) -> float:
        perf = self.model_performance.get(model_name)
        if perf is None:
            return 0.5
        if perf["recent_50_n"] < 3:
            return 0.5
        return perf["recent_50_correct"] / perf["recent_50_n"]

    def _get_calibration_quality(self, model_name: str) -> float:
        ece = self.model_calibration.get(model_name, 0.1)
        # 0.0 = perfectly calibrated → 1.0 quality
        return max(0.0, 1.0 - ece * 3.0)

    def _model_disagreement(self, outputs: List[ModelFamilyOutput]) -> Tuple[float, float, float]:
        """Return (mean_JSD, fraction_predict_BIG, std_dev_confidences)."""
        if len(outputs) <= 1:
            return 0.0, 0.5, 0.0
        p_bigs = np.array([o.probability_big for o in outputs])
        # Pairwise JS divergence using the binary probabilities
        divergences = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                p = np.clip(p_bigs[i], 1e-9, 1 - 1e-9)
                q = np.clip(p_bigs[j], 1e-9, 1 - 1e-9)
                pv = np.array([p, 1 - p])
                qv = np.array([q, 1 - q])
                m = 0.5 * (pv + qv)
                js = 0.5 * np.sum(pv * np.log(pv / m)) + 0.5 * np.sum(qv * np.log(qv / m))
                divergences.append(max(0.0, float(js)))
        mean_js = float(np.mean(divergences)) if divergences else 0.0
        frac_big = float(np.mean(p_bigs >= 0.5))
        std_conf = float(np.std([o.confidence for o in outputs]))
        return mean_js, frac_big, std_conf

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(
        self,
        model_outputs: List[ModelFamilyOutput],
        fp: Optional[StateFingerprint] = None,
        sim_state: Optional[SimilarStateResult] = None,
        baseline_accuracy: float = 0.5,
    ) -> MetaLearnerOutput:
        if not model_outputs:
            return MetaLearnerOutput(
                model_weights={},
                ensemble_probability_big=0.5,
                ensemble_probability_small=0.5,
                ensemble_digit_vector=[0.1] * 10,
                uncertainty=1.0,
                reliability=0.0,
                edge_status="NO_EDGE",
                learning_status="NO_MODELS",
                ensemble_target_digit=5,
                ensemble_hedge_digit=4,
            )

        regime = fp.regime_id if fp is not None else "UNKNOWN"

        # 1. Compute raw model scores
        raw_scores = []
        names = []
        for o in model_outputs:
            names.append(o.model_name)
            hist_acc = self._get_model_accuracy(o.model_name, regime)
            recent_acc = self._get_model_recent_accuracy(o.model_name)
            cal_quality = self._get_calibration_quality(o.model_name)

            # Sample-size-aware bonus: models with more evidence get slightly higher weight
            sample_term = min(1.0, math.log1p(max(0, o.sample_size)) / max(1.0, math.log1p(200)))

            # Confidence agreement: if model says its confident AND historically accurate, boost
            own_conf = o.confidence
            conf_term = self.confidence_trust * own_conf + (1 - self.confidence_trust) * hist_acc

            # Combined score: accuracy history + recent + calibration + sample
            score = (
                0.35 * hist_acc
                + 0.25 * recent_acc
                + 0.15 * cal_quality
                + 0.10 * sample_term
                + 0.15 * conf_term
            )
            # Shift so 0.5 is baseline (uniform)
            baseline = 0.5
            score_above = max(0.001, score - baseline)
            raw_scores.append(score_above)

        # 2. Softmax weights with temperature
        scores_arr = np.array(raw_scores, dtype=np.float64)
        # Numerical stability
        scores_arr = scores_arr - scores_arr.max()
        temperature = 0.05
        w = np.exp(scores_arr / temperature)
        if w.sum() <= 0:
            w = np.ones(len(raw_scores), dtype=np.float64)
        w = w / w.sum()
        # Baseline models (RandomBaseline / MajorityBaseline) are intentionally downweighted
        for i, o in enumerate(model_outputs):
            if o.family == "baseline":
                w[i] *= 0.2
        w = w / max(1e-12, w.sum())

        weights = {name: float(w[i]) for i, name in enumerate(names)}

        # 3. Ensemble probability
        p_big = float(sum(w[i] * o.probability_big for i, o in enumerate(model_outputs)))
        p_small = float(sum(w[i] * o.probability_small for i, o in enumerate(model_outputs)))
        total = p_big + p_small
        if total > 0:
            p_big /= total
            p_small /= total

        # 4. Similar-state evidence blending (15% weight if sufficient)
        if sim_state is not None and sim_state.sample_size >= 30:
            state_weight = min(0.25, sim_state.sample_size / 4000.0)
            # Only blend if similar-state evidence is not flat
            state_edge = abs(sim_state.empirical_p_big - 0.5)
            if state_edge > 0.002:
                state_weight *= min(1.0, state_edge / 0.05)
            else:
                state_weight *= 0.2
            p_big = (1 - state_weight) * p_big + state_weight * sim_state.empirical_p_big
            p_small = 1.0 - p_big

        # 5. Digit-level ensemble vector
        digit_vec = np.zeros(10, dtype=np.float64)
        for i, o in enumerate(model_outputs):
            if len(o.probability_vector) == 10:
                vec = np.array(o.probability_vector, dtype=np.float64)
                digit_vec += w[i] * (vec / vec.sum())
        digit_vec = digit_vec / max(1e-9, digit_vec.sum())
        target_digit = int(np.argmax(digit_vec))
        sorted_idx = digit_vec.argsort()[::-1]
        hedge_digit = int(sorted_idx[1]) if len(sorted_idx) > 1 else 0

        # 6. Disagreement and uncertainty
        mean_js, frac_big, std_conf = self._model_disagreement(model_outputs)
        # Uncertainty = high disagreement, low sample size, close to 0.5
        ensemble_edge = abs(p_big - 0.5)
        uncertainty = min(
            1.0,
            0.35 * mean_js + 0.35 * (1.0 - 2 * ensemble_edge) + 0.15 * (1 - min(1.0, len(model_outputs) / 10.0)),
        )
        # Reliability: inverse of uncertainty, modulated by calibration quality
        avg_cal = float(np.mean([self._get_calibration_quality(n) for n in names]))
        reliability = max(0.0, min(1.0, (1.0 - uncertainty) * (0.5 + 0.5 * avg_cal)))

        # 7. Edge status
        # Compare ensemble accuracy estimate vs baseline
        avg_acc_est = sum(w[i] * self._get_model_accuracy(o.model_name, regime) for i, o in enumerate(model_outputs))
        edge = avg_acc_est - baseline_accuracy
        if sim_state is not None and sim_state.sample_size >= 100:
            sim_edge = abs(sim_state.empirical_p_big - 0.5)
            if sim_state.sufficient_evidence:
                edge = max(edge, sim_edge)
        if edge >= 0.015 and uncertainty < 0.55:
            edge_status = "MODEST_EDGE"
        elif edge >= 0.03 and uncertainty < 0.45:
            edge_status = "VERIFIED_EDGE"
        elif ensemble_edge > 0.02 and reliability > 0.55:
            edge_status = "TENTATIVE_EDGE"
        else:
            edge_status = "NO_EDGE"

        # 8. Learning status
        n_trained = sum(1 for n in names if self.model_performance.get(n, {}).get("n", 0) >= 10)
        if n_trained < len(names) // 2:
            learning_status = "WARMING_UP"
        elif mean_js < 0.01:
            learning_status = "CONVERGED"
        elif mean_js > 0.10:
            learning_status = "EXPLORING"
        else:
            learning_status = "ACTIVE"

        return MetaLearnerOutput(
            model_weights=weights,
            ensemble_probability_big=float(p_big),
            ensemble_probability_small=float(p_small),
            ensemble_digit_vector=digit_vec.tolist(),
            uncertainty=float(uncertainty),
            reliability=float(reliability),
            edge_status=edge_status,
            learning_status=learning_status,
            ensemble_target_digit=target_digit,
            ensemble_hedge_digit=hedge_digit,
        )
