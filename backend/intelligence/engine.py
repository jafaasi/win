from __future__ import annotations

import json
import math
import os
import sys
from typing import List, Dict, Any, Optional, Sequence, Tuple
import numpy as np

from .state_fingerprint import StateFingerprint, compute_state_fingerprint
from .similar_state import SimilarStateMemory, SimilarStateResult
from .multi_model import MultiModelEnsemble, ModelFamilyOutput
from .meta_learner import MetaLearner, MetaLearnerOutput
from .adversarial_critic import AdversarialCritic, CriticOutput
from .three_level import ThreeLevelAnalysis, ThreeLevelProbabilities
from .online_learning import FastMemory, OnlineUpdater
from .concept_drift import ConceptDriftDetector, DriftResult
from .calibration import ConfidenceCalibrator, CalibrationResult
from .abstention import AbstentionEngine, AbstentionResult
from .daily_evolution import DailyEvolution, GenerationRecord, BaselineEvaluator
from .dashboard import IntelligenceDashboard, DashboardData
from .daily_report import DailyIntelligenceReport, DailyReport


class AdaptiveIntelligenceEngine:
    """
    EVOSEQ Adaptive Intelligence Engine v3.

    Orchestrates the full pipeline:
      DATA → MEMORY → FEATURES → MULTIPLE HYPOTHESES → ENSEMBLE →
      ADVERSARIAL CRITIC → META-LEARNING → CALIBRATION → DECISION →
      OUTCOME → AUDIT → LEARNING → GENERATION → VALIDATION → PROMOTION/REJECTION →
      NEW KNOWLEDGE → NEXT DAY → REPEAT FOREVER
    """

    def __init__(self, generation: int = 1):
        self.generation = generation

        # Core modules
        self.ensemble = MultiModelEnsemble(generation=generation)
        self.meta_learner = MetaLearner(generation=generation)
        self.critic = AdversarialCritic()
        self.three_level = ThreeLevelAnalysis()
        self.fast_memory = FastMemory(generation=generation)
        self.drift_detector = ConceptDriftDetector()
        self.calibrator = ConfidenceCalibrator()
        self.abstention = AbstentionEngine()
        self.daily_evolution = DailyEvolution(generation=generation)
        self.dashboard_builder = IntelligenceDashboard()
        self.report_builder = DailyIntelligenceReport()

        # Wire up online updater
        self.online_updater = OnlineUpdater(
            fast_memory=self.fast_memory,
            meta_learner=self.meta_learner,
            critic=self.critic,
            ensemble=self.ensemble,
        )

        # Runtime state
        self.last_fp: Optional[StateFingerprint] = None
        self.last_model_outputs: List[ModelFamilyOutput] = []
        self.last_meta: Optional[MetaLearnerOutput] = None
        self.last_critic: Optional[CriticOutput] = None
        self.last_sim_result: Optional[SimilarStateResult] = None
        self.last_calib: Optional[CalibrationResult] = None
        self.last_abstention: Optional[AbstentionResult] = None
        self.last_three_level: Optional[ThreeLevelProbabilities] = None
        self.last_generation_record: Optional[GenerationRecord] = None
        self.last_prediction_side: Optional[str] = None
        self.last_prediction_prob_big: float = 0.5
        self.initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_from_history(self, history_digits: Sequence[int]) -> Dict[str, Any]:
        """
        One-time cold-start: fit all models on historical data,
        populate similar-state memory, prime calibrator from audit records.
        """
        history = [int(d) for d in history_digits]
        report: Dict[str, Any] = {}

        # 1. Fit multi-model ensemble
        self.ensemble.fit_all(history)
        report["ensemble_fit_samples"] = len(history)

        # 2. Populate similar-state memory via FastMemory bulk
        n_remembered = self.fast_memory.state_memory.bulk_remember_from_history(
            history, start_offset=min(60, len(history) // 10), stride=max(1, len(history) // 20000 + 1)
        )
        report["similar_state_memory_size"] = n_remembered

        # 3. Warm up drift detector
        self.drift_detector.detect(history)

        self.initialized = True
        report["generation"] = self.generation
        report["status"] = "INITIALIZED"
        return report

    def prime_calibrator_from_audit(self, audit_records: Sequence[Dict[str, Any]]) -> int:
        """Feed historical prediction audits into the calibrator."""
        return self.calibrator.bulk_update(audit_records)

    # ------------------------------------------------------------------
    # Prediction (real-time path — fast)
    # ------------------------------------------------------------------

    def predict(
        self,
        recent_history_digits: Sequence[int],
        next_issue_number: Optional[str] = None,
        next_sequence_no: int = 0,
    ) -> Dict[str, Any]:
        """
        Real-time prediction path. Must be fast: no expensive training here.

        Flow:
          1. State fingerprint
          2. Query similar-state memory
          3. Run all models in ensemble → model outputs
          4. Meta-learner produces weights + ensemble probability
          5. 3-level analysis (L1/L2/L3)
          6. Adversarial critic review
          7. Confidence calibration
          8. Abstention/no-edge gate
          9. Build final output with backward-compatible fields
        """
        history = [int(d) for d in recent_history_digits]

        # 1. State fingerprint
        fp = compute_state_fingerprint(history, sequence_no=next_sequence_no)
        self.last_fp = fp

        # 2. Similar-state memory
        sim_result = self.fast_memory.state_memory.query(fp)
        self.last_sim_result = sim_result

        # 3. Multi-model ensemble predictions
        model_outputs = self.ensemble.predict_all(history, fp)
        self.last_model_outputs = model_outputs

        # Baseline accuracy estimate (majority baseline)
        baseline_eval = BaselineEvaluator()
        baseline_acc = baseline_eval.majority(history[-min(1000, len(history)):]) if len(history) >= 30 else 0.5

        # 4. Meta-learner
        meta_out = self.meta_learner.infer(
            model_outputs=model_outputs,
            fp=fp,
            sim_state=sim_result,
            baseline_accuracy=baseline_acc,
        )
        self.last_meta = meta_out

        # 5. 3-level analysis
        proposed_side = "Big" if meta_out.ensemble_probability_big >= 0.5 else "Small"
        tl_model = self.three_level.model_based(meta_out.ensemble_probability_big)
        tl_empirical = self.three_level.evaluate_empirically(
            history_digits=history,
            proposed_side=proposed_side,
            p_big_l1=meta_out.ensemble_probability_big,
            window=min(200, len(history) // 5),
            min_samples=50,
        )
        # Prefer empirical if it has enough samples; otherwise model-based
        if tl_empirical.empirical:
            three_level = tl_empirical
        else:
            three_level = tl_model
        self.last_three_level = three_level

        # 6. Adversarial critic review
        critic_out = self.critic.review(
            proposed_prediction=proposed_side,
            proposed_prob_big=meta_out.ensemble_probability_big,
            model_outputs=model_outputs,
            fp=fp,
            sim_state=sim_result,
            meta_reliability=meta_out.reliability,
            calibration_error=self.calibrator.expected_calibration_error(),
        )
        self.last_critic = critic_out

        # 7. Confidence calibration
        calib_result = self.calibrator.calibrate(meta_out.ensemble_probability_big)
        self.last_calib = calib_result

        # Use calibrated probability with the critic's confidence multiplier
        calibrated_p_big_raw = calib_result.calibrated_probability_big
        # Blend meta + calibration, respecting critic multiplier
        raw_p = meta_out.ensemble_probability_big
        blend = 0.5 * raw_p + 0.5 * calibrated_p_big_raw
        # Apply critic multiplier only on the distance-from-chance (never flip side)
        side = 1.0 if blend >= 0.5 else -1.0
        distance = abs(blend - 0.5)
        adjusted_distance = distance * critic_out.final_confidence_multiplier
        final_p_big = 0.5 + side * adjusted_distance
        final_p_big = max(0.01, min(0.99, final_p_big))
        final_side = "Big" if final_p_big >= 0.5 else "Small"

        # 8. Drift context (lightweight, no training)
        drift_result = self.drift_detector.detect(
            history,
            long_accuracy=(self.fast_memory.total_correct / max(1, self.fast_memory.total_resolved)) if self.fast_memory.total_resolved >= 30 else None,
            recent_accuracy=self.fast_memory.recent_accuracy(50) if self.fast_memory.total_resolved >= 50 else None,
            calibration_error_before=calib_result.expected_calibration_error,
        )

        # 9. Abstention / No-Edge
        abstention_out = self.abstention.decide(
            fp=fp,
            model_outputs=model_outputs,
            meta_out=meta_out,
            critic_out=critic_out,
            calib_result=calib_result,
            drift_result=drift_result,
            sim_result=sim_result,
            baseline_accuracy=baseline_acc,
            ensemble_oos_accuracy=(
                self.last_generation_record.champion_oos_accuracy if self.last_generation_record else None
            ),
        )
        self.last_abstention = abstention_out

        # Final action
        if abstention_out.action == "SKIP":
            action = "SKIP"
            final_confidence = 0.5
        elif abstention_out.action == "REDUCE_CONFIDENCE":
            action = "REDUCE_CONFIDENCE"
            # Final confidence is pct of max side
            final_confidence = max(final_p_big, 1 - final_p_big)
            # Further reduce the stated confidence
            final_confidence = 0.5 + (final_confidence - 0.5) * 0.85
        else:
            action = "PREDICT"
            final_confidence = max(final_p_big, 1 - final_p_big)

        # Store runtime state for next-round online update
        self.last_prediction_side = final_side
        self.last_prediction_prob_big = final_p_big

        # Digit-level: use meta's target/hedge
        target_digit = int(meta_out.ensemble_target_digit)
        hedge_digit = int(meta_out.ensemble_hedge_digit)

        # Consensus: fraction of models predicting the same side
        if len(model_outputs) > 0:
            agree = sum(1 for o in model_outputs if o.prediction == final_side and o.family != "baseline")
            non_bl = sum(1 for o in model_outputs if o.family != "baseline")
            model_consensus = agree / non_bl if non_bl > 0 else 0.5
        else:
            model_consensus = 0.5

        # Martingale / drift level (preserved for backward compat)
        drift_level = drift_result.severity

        # Pattern name (for backward compat)
        champ = max(meta_out.model_weights, key=meta_out.model_weights.get) if meta_out.model_weights else "Ensemble"
        pattern_name = f"{champ}::{fp.regime_id}"

        # modelPBigVector: probability-big per model
        model_pbig_vector = [round(float(o.probability_big), 4) for o in model_outputs]

        # Build result with ALL required fields (backward-compatible + new)
        result = {
            # --- Backward compatible fields ---
            "prediction": final_side,
            "confidence": round(final_confidence * 100.0, 1),
            "targetNum": target_digit,
            "hedgeNum": hedge_digit,
            "nextIssue": str(next_issue_number) if next_issue_number is not None else "",
            "action": action,
            "strikeQuality": self._strike_quality(final_confidence, action, meta_out.reliability),
            "modelConsensus": round(model_consensus, 4),
            "martingaleLevel": self._martingale_level(final_confidence, action),
            "driftLevel": drift_level,
            "patternName": pattern_name,
            "totalSamplesTrained": int(self.fast_memory.total_resolved),
            "ensembleWeights": {k: round(v, 6) for k, v in meta_out.model_weights.items()},
            "modelPBigVector": model_pbig_vector,
            # --- New fields (spec 24) ---
            "generation": int(self.generation),
            "stateFingerprint": fp.to_dict(),
            "stateSimilarity": round(sim_result.mean_similarity, 4),
            "stateSampleSize": int(sim_result.sample_size),
            "entropy": round(fp.entropy, 6),
            "regime": fp.regime_id,
            "adversarialScore": round(critic_out.support_score - critic_out.contradiction_score, 4),
            "contradictionScore": round(critic_out.contradiction_score, 4),
            "calibratedProbability": round(final_p_big, 6),
            "calibrationError": round(calib_result.expected_calibration_error, 6),
            "oosScore": round(self.last_generation_record.champion_oos_accuracy, 6) if self.last_generation_record else None,
            "baselineScore": round(baseline_acc, 6),
            "edgeStatus": abstention_out.edge_status,
            "learningStatus": meta_out.learning_status,
            "modelReliability": {
                o.model_name: round(self.meta_learner._get_model_accuracy(o.model_name, fp.regime_id), 4)
                for o in model_outputs
            },
            "knowledgeVersion": f"gen{self.generation}",
            # --- Internals (useful for audit / dashboard) ---
            "probability_big": round(final_p_big, 6),
            "probability_small": round(1.0 - final_p_big, 6),
            "metaLearner": meta_out.to_dict(),
            "critic": critic_out.to_dict(),
            "abstention": abstention_out.to_dict(),
            "calibration": calib_result.to_dict(),
            "threeLevel": three_level.to_dict(),
            "drift": drift_result.to_dict(),
            "similarState": sim_result.to_dict(),
            "ensembleTargetDigit": target_digit,
            "ensembleHedgeDigit": hedge_digit,
        }
        return result

    # ------------------------------------------------------------------
    # Learning path: resolve a prediction (called AFTER outcome arrives)
    # ------------------------------------------------------------------

    def resolve_outcome(
        self,
        actual_digit: int,
        history_suffix_for_partial_fit: Sequence[int],
        three_level_outcomes: Optional[Tuple[bool, bool, bool]] = None,
    ) -> Dict[str, Any]:
        """
        Called after the actual outcome for the prediction round is known.
        Never modify the stored prediction retroactively; only UPDATE LEARNING.
        """
        actual_side = "Big" if int(actual_digit) >= 5 else "Small"

        # Feed calibration
        if self.last_calib is not None:
            self.calibrator.update(self.last_calib.raw_probability_big, actual_side)

        # Feed three-level (if provided)
        tl_provided = (
            three_level_outcomes is not None
            and isinstance(three_level_outcomes, (list, tuple))
            and len(three_level_outcomes) >= 3
        )
        if tl_provided:
            self.three_level.update_empirical(
                self.last_three_level.p_success_l1 if self.last_three_level else 0.5,
                self.last_three_level.p_success_l2 if self.last_three_level else 0.75,
                self.last_three_level.p_success_l3 if self.last_three_level else 0.88,
                bool(three_level_outcomes[0]),
                bool(three_level_outcomes[1]),
                bool(three_level_outcomes[2]),
            )

        # Feed fast memory + online updater
        if self.last_fp is not None:
            stats = self.online_updater.resolve_prediction(
                predicted_side=self.last_prediction_side or "Big",
                predicted_prob_big=self.last_prediction_prob_big,
                actual_side=actual_side,
                actual_digit=int(actual_digit),
                fp=self.last_fp,
                model_outputs=self.last_model_outputs,
                last_history_digits=history_suffix_for_partial_fit,
            )
        else:
            stats = {}

        return {
            "resolved": True,
            "predicted": self.last_prediction_side,
            "actual": actual_side,
            "correct": self.last_prediction_side == actual_side,
            "stats": stats,
        }

    # ------------------------------------------------------------------
    # Daily evolution path (runs periodically)
    # ------------------------------------------------------------------

    def run_daily_evolution(
        self,
        full_history_digits: Sequence[int],
    ) -> Dict[str, Any]:
        """
        Daily (slow) learning path. Runs walk-forward evaluation.
        Produces a new generation record. Only promotes if OOS improvement
        with statistical evidence.
        """
        prev_acc = None
        if self.last_generation_record is not None:
            prev_acc = self.last_generation_record.champion_oos_accuracy

        gen_record = self.daily_evolution.run_daily_evolution(
            full_history_digits,
            previous_accuracy=prev_acc,
        )
        self.last_generation_record = gen_record
        if gen_record.status == "PROMOTED":
            self.generation = gen_record.generation
            # Increment next generation for tomorrow's run
            self.daily_evolution.current_generation = self.generation + 1
            self.ensemble.generation = self.generation
            self.meta_learner.generation = self.generation
        return gen_record.to_dict()

    # ------------------------------------------------------------------
    # Dashboard + Daily Report
    # ------------------------------------------------------------------

    def build_dashboard(self) -> DashboardData:
        return self.dashboard_builder.build(
            generation_record=self.last_generation_record,
            fast_memory=self.fast_memory,
            meta_learner=self.meta_learner,
            calibrator=self.calibrator,
            drift_detector=self.drift_detector,
            abstention=self.abstention,
            current_fp=self.last_fp,
            current_sim_sample_size=(self.last_sim_result.sample_size if self.last_sim_result else 0),
            current_sim_similarity=(self.last_sim_result.mean_similarity if self.last_sim_result else None),
            current_disagreement=self._mean_disagreement(),
            current_critic_support=(self.last_critic.support_score if self.last_critic else 0.0),
            current_critic_contradiction=(self.last_critic.contradiction_score if self.last_critic else 0.0),
            current_critic_uncertainty=(self.last_critic.uncertainty_score if self.last_critic else 0.0),
            current_edge=(self.last_abstention.edge_status if self.last_abstention else "NO_EDGE"),
            current_learning=(self.last_meta.learning_status if self.last_meta else "ACTIVE"),
            current_action=(self.last_abstention.action if self.last_abstention else "PREDICT"),
        )

    def build_daily_report(
        self,
        historical_samples_total: int,
        new_samples_today: int,
    ) -> DailyReport:
        dash = self.build_dashboard()
        # Strongest/weakest model from reliability dict
        rel = dash.model_reliability
        if rel:
            strongest = max(rel, key=rel.get)
            weakest = min(rel, key=rel.get)
        else:
            strongest, weakest = None, None

        drift_today = dash.drift_status != "STABLE"

        return self.report_builder.generate(
            dashboard=dash,
            generation_record=self.last_generation_record,
            historical_samples_total=historical_samples_total,
            new_samples_today=new_samples_today,
            mean_brier_recent=self.fast_memory.recent_brier(),
            mean_log_loss_recent=self.fast_memory.recent_log_loss(),
            strongest_model=strongest,
            weakest_model=weakest,
            drift_detected_today=drift_today,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mean_disagreement(self) -> float:
        if not self.last_model_outputs:
            return 0.0
        p_bigs = np.array([o.probability_big for o in self.last_model_outputs if o.family != "baseline"])
        if len(p_bigs) < 2:
            return 0.0
        return float(np.std(p_bigs))

    def _strike_quality(self, final_confidence: float, action: str, reliability: float) -> str:
        if action == "SKIP":
            return "NO_EDGE"
        score = final_confidence * (0.6 + 0.4 * reliability)
        if score >= 0.92:
            return "EXCELLENT"
        if score >= 0.88:
            return "HIGH"
        if score >= 0.82:
            return "GOOD"
        return "MODERATE"

    def _martingale_level(self, final_confidence: float, action: str) -> int:
        if action == "SKIP":
            return 0
        if final_confidence >= 0.94:
            return 1
        if final_confidence >= 0.90:
            return 2
        if final_confidence >= 0.86:
            return 3
        return 1
