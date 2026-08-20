from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Sequence, Tuple
import numpy as np

from .state_fingerprint import StateFingerprint
from .multi_model import ModelFamilyOutput
from .similar_state import SimilarStateResult
from .concept_drift import DriftResult
from .adversarial_critic import CriticOutput
from .calibration import CalibrationResult
from .meta_learner import MetaLearnerOutput


@dataclass
class AbstentionResult:
    action: str  # "PREDICT" | "SKIP" | "REDUCE_CONFIDENCE"
    edge_status: str  # "NO_EDGE" | "TENTATIVE_EDGE" | "MODEST_EDGE" | "VERIFIED_EDGE"
    reason: Optional[str]
    abstention_score: float  # Higher = more likely to abstain
    baseline_beaten: bool
    skip_reasons: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


class AbstentionEngine:
    """
    The engine must know when NOT to predict confidently.

    Conditions for SKIP / REDUCE_CONFIDENCE:
      - High entropy
      - High model disagreement
      - Similar-state sample size too low
      - Calibration poor
      - Drift detected
      - Cannot beat simple baselines (Random / Majority / Markov)
      - High adversarial contradiction
    """

    def __init__(self):
        # Thresholds
        self.entropy_skip_threshold = 0.985
        self.disagreement_skip_threshold = 0.12
        self.state_sample_skip_threshold = 25
        self.ece_skip_threshold = 0.14
        self.composite_drift_skip = 0.35
        self.contradiction_skip_threshold = 0.55
        self.uncertainty_skip_threshold = 0.70
        self.reliability_min_threshold = 0.35

        # Counters
        self.total_opportunities = 0
        self.total_predictions = 0
        self.total_skips = 0
        self.skip_reason_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Decide
    # ------------------------------------------------------------------

    def decide(
        self,
        fp: StateFingerprint,
        model_outputs: List[ModelFamilyOutput],
        meta_out: MetaLearnerOutput,
        critic_out: CriticOutput,
        calib_result: CalibrationResult,
        drift_result: Optional[DriftResult],
        sim_result: Optional[SimilarStateResult],
        baseline_accuracy: float,
        ensemble_oos_accuracy: Optional[float] = None,
    ) -> AbstentionResult:
        self.total_opportunities += 1
        skip_reasons: List[str] = []

        # 1. Entropy
        entropy_score = 0.0
        if fp.entropy >= self.entropy_skip_threshold:
            entropy_score = 0.25
            skip_reasons.append("HIGH_ENTROPY")
        elif fp.entropy >= 0.96:
            entropy_score = 0.10

        # 2. Model disagreement (across families)
        family_predictions: Dict[str, List[str]] = {}
        for o in model_outputs:
            family_predictions.setdefault(o.family, []).append(o.prediction)
        family_disagree = 0
        total_families = 0
        for fam, preds in family_predictions.items():
            if fam == "baseline":
                continue
            total_families += 1
            bigs = sum(1 for p in preds if p == "Big")
            if bigs != 0 and bigs != len(preds):
                family_disagree += 1
        if total_families >= 3:
            disagree_ratio = family_disagree / total_families
        else:
            disagree_ratio = 0.0
        disagreement_score = 0.0
        if disagree_ratio >= self.disagreement_skip_threshold:
            disagreement_score = 0.25
            skip_reasons.append("FAMILY_DISAGREEMENT")
        elif meta_out.uncertainty > 0.55:
            disagreement_score = 0.15

        # 3. State memory evidence
        state_score = 0.0
        sample_size = sim_result.sample_size if sim_result is not None else 0
        if sample_size > 0 and sample_size < self.state_sample_skip_threshold:
            state_score = 0.15
            skip_reasons.append("LOW_STATE_SAMPLE")

        # 4. Calibration quality
        cal_score = 0.0
        ece = calib_result.expected_calibration_error
        if ece >= self.ece_skip_threshold:
            cal_score = 0.18
            skip_reasons.append("POOR_CALIBRATION")
        elif ece >= 0.09:
            cal_score = 0.08

        # 5. Drift
        drift_score = 0.0
        if drift_result is not None and drift_result.composite >= self.composite_drift_skip:
            drift_score = 0.25
            skip_reasons.append("CONCEPT_DRIFT")
        elif drift_result is not None and drift_result.composite >= 0.22:
            drift_score = 0.10

        # 6. Adversarial contradiction
        contra_score = 0.0
        if critic_out.contradiction_score >= self.contradiction_skip_threshold:
            contra_score = 0.20
            skip_reasons.append("CONTRADICTION_HIGH")
        elif critic_out.contradiction_score - critic_out.support_score > 0.10:
            contra_score = 0.08

        # 7. Uncertainty from meta-learner
        uncert_score = 0.0
        if meta_out.uncertainty >= self.uncertainty_skip_threshold:
            uncert_score = 0.20
            skip_reasons.append("HIGH_UNCERTAINTY")
        elif meta_out.uncertainty >= 0.55:
            uncert_score = 0.08

        # 8. Reliability (meta)
        rel_score = 0.0
        if meta_out.reliability < self.reliability_min_threshold:
            rel_score = 0.15
            skip_reasons.append("LOW_RELIABILITY")

        # 9. Baseline beat
        baseline_beaten = True
        if ensemble_oos_accuracy is not None:
            baseline_beaten = ensemble_oos_accuracy > baseline_accuracy + 0.002
        baseline_score = 0.0
        if not baseline_beaten:
            baseline_score = 0.22
            skip_reasons.append("CANNOT_BEAT_BASELINE")

        # 10. Edge signal from meta
        edge_score = 0.0
        if meta_out.edge_status == "NO_EDGE":
            edge_score = 0.15
            if "CANNOT_BEAT_BASELINE" not in skip_reasons:
                skip_reasons.append("NO_STATISTICAL_EDGE")

        abstention_score = min(
            1.0,
            entropy_score
            + disagreement_score
            + state_score
            + cal_score
            + drift_score
            + contra_score
            + uncert_score
            + rel_score
            + baseline_score
            + edge_score,
        )

        # Action
        action = "PREDICT"
        if abstention_score >= 0.60:
            action = "SKIP"
            self.total_skips += 1
            for r in skip_reasons:
                self.skip_reason_counts[r] = self.skip_reason_counts.get(r, 0) + 1
        elif abstention_score >= 0.35:
            action = "REDUCE_CONFIDENCE"
            self.total_predictions += 1
        else:
            self.total_predictions += 1

        edge_status = meta_out.edge_status
        if action == "SKIP":
            edge_status = "NO_EDGE"
        elif action == "REDUCE_CONFIDENCE" and edge_status == "VERIFIED_EDGE":
            edge_status = "MODEST_EDGE"

        reason = skip_reasons[0] if skip_reasons and action != "PREDICT" else None

        return AbstentionResult(
            action=action,
            edge_status=edge_status,
            reason=reason,
            abstention_score=float(abstention_score),
            baseline_beaten=baseline_beaten,
            skip_reasons=skip_reasons,
        )
