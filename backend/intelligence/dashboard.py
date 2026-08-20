from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Sequence, Tuple
import numpy as np

from .state_fingerprint import StateFingerprint
from .online_learning import FastMemory
from .meta_learner import MetaLearner
from .daily_evolution import GenerationRecord
from .concept_drift import ConceptDriftDetector
from .calibration import ConfidenceCalibrator
from .abstention import AbstentionEngine


@dataclass
class DashboardData:
    current_generation: int
    previous_generation: Optional[int]
    training_samples: int
    validation_samples: int
    oos_samples: int
    champion_model: str
    champion_accuracy: Optional[float]
    model_weights: Dict[str, float]
    model_reliability: Dict[str, float]
    current_regime: str
    state_similarity: Optional[float]
    state_sample_size: int
    entropy: float
    model_disagreement: float
    adversarial_support_score: float
    adversarial_contradiction_score: float
    adversarial_uncertainty_score: float
    calibration_ece: float
    recent_accuracy_200: Optional[float]
    long_term_accuracy: Optional[float]
    drift_status: str
    drift_composite: float
    num_abstentions: int
    skip_reasons: Dict[str, int]
    baseline_performance: Dict[str, float]
    generation_change_summary: Dict[str, Any]
    edge_status: str
    learning_status: str
    action: str

    def to_dict(self) -> dict:
        return asdict(self)


class IntelligenceDashboard:
    """
    Builds the Intelligence Dashboard snapshot used by the API endpoints
    and the daily self-report.
    """

    def build(
        self,
        generation_record: Optional[GenerationRecord],
        fast_memory: FastMemory,
        meta_learner: MetaLearner,
        calibrator: ConfidenceCalibrator,
        drift_detector: ConceptDriftDetector,
        abstention: AbstentionEngine,
        current_fp: Optional[StateFingerprint],
        current_sim_sample_size: int = 0,
        current_sim_similarity: Optional[float] = None,
        current_disagreement: float = 0.0,
        current_critic_support: float = 0.0,
        current_critic_contradiction: float = 0.0,
        current_critic_uncertainty: float = 0.0,
        current_edge: str = "NO_EDGE",
        current_learning: str = "ACTIVE",
        current_action: str = "PREDICT",
        current_regime_override: Optional[str] = None,
    ) -> DashboardData:
        # Generation info
        gen = generation_record.generation if generation_record else 1
        prev_gen = max(1, gen - 1) if generation_record and gen > 1 else None
        train_n = generation_record.training_cutoff if generation_record else 0
        val_n = (
            (generation_record.validation_cutoff - generation_record.training_cutoff)
            if generation_record
            else 0
        )
        test_n = (
            (generation_record.test_cutoff - generation_record.validation_cutoff)
            if generation_record
            else 0
        )

        champion_model = generation_record.champion_model if generation_record else "FrequencyModel"
        champion_acc = generation_record.champion_oos_accuracy if generation_record else None

        weights = generation_record.model_weights if generation_record else {}

        # Model reliability: from meta learner calibration
        reliability: Dict[str, float] = {}
        for name in meta_learner.model_performance:
            perf = meta_learner.model_performance[name]
            if perf["n"] >= 10:
                reliability[name] = round(perf["correct"] / perf["n"], 4)
            else:
                reliability[name] = 0.5

        regime = current_regime_override or (current_fp.regime_id if current_fp else "UNKNOWN")
        entropy_val = current_fp.entropy if current_fp else 1.0
        drift_composite = drift_detector.mean_drift(50)
        if drift_composite >= 0.35:
            drift_status = "DRIFT_DETECTED"
        elif drift_composite >= 0.18:
            drift_status = "ELEVATED"
        else:
            drift_status = "STABLE"

        recent_200 = fast_memory.recent_accuracy(200) if fast_memory.total_resolved >= 10 else None
        long_term = (
            fast_memory.total_correct / fast_memory.total_resolved
            if fast_memory.total_resolved >= 1
            else None
        )

        ece = calibrator.expected_calibration_error()

        baselines = generation_record.baselines if generation_record else {"random": 0.5, "majority": 0.5, "recent_frequency": 0.5, "simple_markov": 0.5}

        # Generation change summary
        change_summary: Dict[str, Any] = {
            "new_generation": gen,
            "previous_generation": prev_gen,
            "champion_change": None,
            "accuracy_delta": generation_record.accuracy_delta if generation_record else 0.0,
            "status": generation_record.status if generation_record else "CANDIDATE",
            "promoted": generation_record.promoted_models if generation_record else [],
            "rejected": generation_record.rejected_models if generation_record else [],
        }

        return DashboardData(
            current_generation=gen,
            previous_generation=prev_gen,
            training_samples=int(train_n),
            validation_samples=int(val_n),
            oos_samples=int(test_n),
            champion_model=champion_model,
            champion_accuracy=champion_acc,
            model_weights={k: round(float(v), 6) for k, v in weights.items()},
            model_reliability=reliability,
            current_regime=regime,
            state_similarity=current_sim_similarity,
            state_sample_size=int(current_sim_sample_size),
            entropy=float(entropy_val),
            model_disagreement=float(current_disagreement),
            adversarial_support_score=float(current_critic_support),
            adversarial_contradiction_score=float(current_critic_contradiction),
            adversarial_uncertainty_score=float(current_critic_uncertainty),
            calibration_ece=float(ece),
            recent_accuracy_200=recent_200,
            long_term_accuracy=long_term,
            drift_status=drift_status,
            drift_composite=float(drift_composite),
            num_abstentions=int(abstention.total_skips),
            skip_reasons=dict(abstention.skip_reason_counts),
            baseline_performance={k: round(float(v), 6) for k, v in baselines.items()},
            generation_change_summary=change_summary,
            edge_status=current_edge,
            learning_status=current_learning,
            action=current_action,
        )
