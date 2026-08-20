from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Sequence, Tuple
import numpy as np

from .state_fingerprint import StateFingerprint
from .multi_model import ModelFamilyOutput
from .similar_state import SimilarStateResult


@dataclass
class CriticOutput:
    support_score: float
    contradiction_score: float
    uncertainty_score: float
    final_recommended_action: str
    final_confidence_multiplier: float
    rejection_reason: Optional[str]
    contradictory_models: List[str]
    supporting_models: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


class AdversarialCritic:
    """
    Adversarial Self-Critic: before finalizing a prediction,
    ask: "What evidence suggests the main prediction may be WRONG?"

    Evaluates:
      - contradictory models (predict opposite side)
      - high entropy / near-50/50 state
      - weak similar-state evidence
      - recent model performance degradation
      - regime change / high drift
      - poor calibration history
      - insufficient sample sizes
      - high disagreement between families

    Produces support/contradiction/uncertainty scores,
    and is allowed to REJECT the prediction (action = SKIP)
    or reduce the confidence multiplier.
    """

    def __init__(self):
        # Configurable thresholds
        self.high_entropy_threshold = 0.95  # binary entropy
        self.high_drift_threshold = 0.18
        self.min_state_samples = 30
        self.high_disagreement_threshold = 0.08
        self.min_families_support = 2
        self.contradiction_fraction_threshold = 0.35  # if >= 35% of models contradict, warn
        self.recent_performance_window = 50

        # History: running estimates of model family performance degradation
        self.family_recent_perf: Dict[str, List[bool]] = {}
        self.family_long_perf: Dict[str, List[bool]] = {}

    # ------------------------------------------------------------------
    # Online updates
    # ------------------------------------------------------------------

    def update_family_performance(self, family: str, correct: bool) -> None:
        r = self.family_recent_perf.setdefault(family, [])
        l = self.family_long_perf.setdefault(family, [])
        r.append(correct)
        l.append(correct)
        if len(r) > 200:
            r[:] = r[-200:]
        if len(l) > 2000:
            l[:] = l[-2000:]

    def family_degradation(self, family: str) -> float:
        r = self.family_recent_perf.get(family, [])
        l = self.family_long_perf.get(family, [])
        if len(r) < 20 or len(l) < 100:
            return 0.0
        recent_acc = sum(r) / len(r)
        long_acc = sum(l) / len(l)
        # Positive = degradation (recent worse than long-term)
        return max(0.0, long_acc - recent_acc)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def review(
        self,
        proposed_prediction: str,
        proposed_prob_big: float,
        model_outputs: List[ModelFamilyOutput],
        fp: Optional[StateFingerprint] = None,
        sim_state: Optional[SimilarStateResult] = None,
        meta_reliability: float = 0.5,
        calibration_error: float = 0.05,
    ) -> CriticOutput:
        # 1. Find supporting vs contradicting models
        supporting = []
        contradicting = []
        for o in model_outputs:
            if o.family == "baseline":
                continue
            if o.prediction == proposed_prediction:
                supporting.append(o.model_name)
            else:
                contradicting.append(o.model_name)

        n_non_baseline = len([o for o in model_outputs if o.family != "baseline"])
        contradiction_fraction = (len(contradicting) / n_non_baseline) if n_non_baseline > 0 else 0.0
        support_fraction = (len(supporting) / n_non_baseline) if n_non_baseline > 0 else 0.0

        # 2. Family diversity check: how many different families support?
        supporting_families = set()
        contradicting_families = set()
        for o in model_outputs:
            if o.family == "baseline":
                continue
            if o.prediction == proposed_prediction:
                supporting_families.add(o.family)
            else:
                contradicting_families.add(o.family)
        family_support_count = len(supporting_families)

        # 3. Compute components
        # --- Support score (1.0 = strong supporting evidence)
        # Average confidence of supporting models weighted by sample size
        support_confidences = []
        for o in model_outputs:
            if o.model_name in supporting:
                sample_w = min(1.0, math.log1p(max(0, o.sample_size)) / max(1.0, math.log1p(200)))
                support_confidences.append(o.confidence * (0.5 + 0.5 * sample_w))
        avg_support_conf = float(np.mean(support_confidences)) if support_confidences else 0.5
        family_bonus = min(1.0, family_support_count / 4.0)
        support_score = min(
            1.0,
            0.40 * support_fraction + 0.35 * avg_support_conf + 0.25 * family_bonus,
        )

        # --- Contradiction score (1.0 = strong evidence against)
        contradiction_confidences = []
        for o in model_outputs:
            if o.model_name in contradicting:
                sample_w = min(1.0, math.log1p(max(0, o.sample_size)) / max(1.0, math.log1p(200)))
                contradiction_confidences.append(o.confidence * (0.5 + 0.5 * sample_w))
        avg_contradict_conf = float(np.mean(contradiction_confidences)) if contradiction_confidences else 0.0

        # Family degradation: how much have supporting families been underperforming lately?
        degradation = 0.0
        for fam in supporting_families:
            degradation = max(degradation, self.family_degradation(fam))

        contradiction_score = min(
            1.0,
            0.40 * contradiction_fraction
            + 0.30 * avg_contradict_conf
            + 0.15 * degradation
            + 0.15 * (1.0 if len(contradicting_families) >= 3 else 0.0),
        )

        # --- Uncertainty score (1.0 = high uncertainty)
        entropy = fp.entropy if fp is not None else 1.0
        drift = fp.drift_score if fp is not None else 0.0
        state_samples = sim_state.sample_size if sim_state is not None else 0
        state_evidence_ratio = sim_state.evidence_ratio if sim_state is not None else 0.0

        prob_edge = abs(proposed_prob_big - 0.5)
        near_flip = 1.0 - min(1.0, prob_edge / 0.08)  # close to 50/50 → high

        ent_component = 1.0 if entropy >= self.high_entropy_threshold else entropy / self.high_entropy_threshold
        drift_component = 1.0 if drift >= self.high_drift_threshold else drift / self.high_drift_threshold

        sample_component = 0.0
        if state_samples > 0:
            sample_component = max(0.0, 1.0 - state_samples / max(1, self.min_state_samples * 3))
        else:
            sample_component = 0.7  # missing state-memory evidence

        cal_component = min(1.0, calibration_error * 6.0)

        reliability_component = 1.0 - min(1.0, meta_reliability)

        uncertainty_score = min(
            1.0,
            0.20 * near_flip
            + 0.15 * ent_component
            + 0.15 * drift_component
            + 0.15 * sample_component
            + 0.15 * cal_component
            + 0.20 * reliability_component,
        )

        # 4. Recommended action and confidence multiplier
        # Start with multiplier; strong contradictions reduce it; strong evidence increases up to 1.0
        multiplier = 1.0
        reason = None
        action = "PREDICT"

        # If uncertainty is very high → reduce
        if uncertainty_score > 0.80:
            multiplier *= 0.60
        elif uncertainty_score > 0.65:
            multiplier *= 0.80
        elif uncertainty_score > 0.50:
            multiplier *= 0.92

        # If contradiction dominates → reduce
        if contradiction_score - support_score > 0.30:
            multiplier *= 0.65
            reason = "HIGH_CONTRADICTION"
        elif contradiction_score - support_score > 0.15:
            multiplier *= 0.85

        # Family support count < threshold → be conservative
        if family_support_count < self.min_families_support and n_non_baseline >= 4:
            multiplier *= 0.85
            if reason is None:
                reason = "LOW_FAMILY_CONSENSUS"

        # High drift + high uncertainty → skip
        if drift >= self.high_drift_threshold and uncertainty_score > 0.70:
            reason = "REGIME_CHANGE_HIGH_UNCERTAINTY"
            action = "SKIP"

        # Very close to 0.5 AND high uncertainty AND weak state evidence
        if prob_edge < 0.02 and uncertainty_score > 0.55:
            if state_samples < self.min_state_samples or state_evidence_ratio < 0.8:
                reason = "NO_STATISTICALLY_DEFENSIBLE_EDGE"
                action = "SKIP"

        # If calibration is very poor and multiplier already low → skip
        if calibration_error > 0.15 and multiplier < 0.75:
            reason = "POOR_CALIBRATION"
            action = "SKIP"

        # Contradiction fraction too high with high confidence from opponents
        if (
            contradiction_fraction >= self.contradiction_fraction_threshold
            and avg_contradict_conf > 0.65
        ):
            if action != "SKIP":
                action = "REDUCE_CONFIDENCE"
                reason = reason or "CONTRADICTING_MODELS"

        return CriticOutput(
            support_score=float(support_score),
            contradiction_score=float(contradiction_score),
            uncertainty_score=float(uncertainty_score),
            final_recommended_action=action,
            final_confidence_multiplier=float(max(0.1, min(1.0, multiplier))),
            rejection_reason=reason,
            contradictory_models=list(contradicting),
            supporting_models=list(supporting),
        )
