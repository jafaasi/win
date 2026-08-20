from __future__ import annotations

import json
import os
import sys
import math
import hashlib
from typing import Dict, Any, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import (
    SessionLocal,
    Draw,
    Outcome,
    PredictionAudit,
    save_ai_brain_state,
    save_full_prediction_audit,
    save_prediction,
    migrate_prediction_audit_columns,
    to_big_small,
)
from backend.intelligence import (
    AdaptiveIntelligenceEngine,
)


class EVOSEQBridge:
    """
    Bridge between the AdaptiveIntelligenceEngine and the existing
    backend database/server pipeline.

    Responsibilities:
      * Load historical data from Supabase (Draws + Outcomes)
      * Initialize the engine on cold-start
      * Run a prediction round and persist:
          - AIBrainState (Live_UI_State) for the server/telegram bot
          - PredictionAudit (extended v3) for the ledger
          - PredictionLog (legacy) for backward compat
      * After outcome arrives, run online-learning resolve step
      * Daily evolution trigger
      * Dashboard / report builders
    """

    LIVE_UI_MODEL_NAME = "Live_UI_State"
    HISTORY_MAX = int(os.environ.get("EVOSEQ_HISTORY_MAX", "80000"))
    MIN_HISTORY_TO_ENGAGE = int(os.environ.get("EVOSEQ_MIN_HISTORY", "100"))

    def __init__(self):
        self.engine = AdaptiveIntelligenceEngine(generation=1)
        self.initialized = False
        self._last_processed_sequence_no: Optional[int] = None
        self._migrated = False

    # ------------------------------------------------------------------
    # Boot
    # ------------------------------------------------------------------

    def _ensure_migrations(self, db):
        if self._migrated:
            return
        try:
            migrate_prediction_audit_columns(db)
        except Exception:
            pass
        # Also ensure the intelligence memory tables exist (ORM create_all for intelligence/models.py Base)
        try:
            from backend.intelligence.models import ensure_intelligence_tables
            from backend.database import engine as db_engine
            ensure_intelligence_tables(db_engine)
        except Exception:
            pass
        self._migrated = True

    def cold_start(self, db=None) -> Dict[str, Any]:
        """
        Load history from Supabase → initialize engine.
        Idempotent: returns immediately if already initialized.
        """
        if self.initialized:
            return {"status": "ALREADY_INITIALIZED", "generation": self.engine.generation}
        if db is None:
            db = SessionLocal()
            own_db = True
        else:
            own_db = False
        try:
            self._ensure_migrations(db)
            history_digits = self._load_history_digits(db)
            n = len(history_digits)
            if n < self.MIN_HISTORY_TO_ENGAGE:
                return {
                    "status": "INSUFFICIENT_HISTORY",
                    "available": n,
                    "required": self.MIN_HISTORY_TO_ENGAGE,
                }
            init_report = self.engine.initialize_from_history(history_digits)
            # Prime calibrator from existing audits
            audits = self._load_recent_audits(db, limit=5000)
            if audits:
                n_cal = self.engine.prime_calibrator_from_audit(audits)
                init_report["calibration_primed_samples"] = n_cal
            # Restore generation number from DB if possible
            try:
                from backend.database import AIBrainState as Brain
                gen_row = db.query(Brain).filter(Brain.model_name == "generation_tracker").first()
                if gen_row and gen_row.generation:
                    self.engine.generation = max(1, int(gen_row.generation))
                    self.engine.daily_evolution.current_generation = self.engine.generation + 1
                    init_report["restored_generation"] = self.engine.generation
            except Exception:
                pass
            self.initialized = True
            init_report["history_used"] = n
            return init_report
        finally:
            if own_db:
                db.close()

    # ------------------------------------------------------------------
    # History loaders
    # ------------------------------------------------------------------

    def _load_history_digits(self, db) -> List[int]:
        """
        Prefer Outcome (digit, ordered by sequence_no). Fall back to Draw.
        """
        digits: List[int] = []
        try:
            q = (
                db.query(Outcome.digit)
                .order_by(Outcome.sequence_no.asc())
                .limit(self.HISTORY_MAX)
            )
            digits = [int(r[0]) for r in q.all()]
        except Exception:
            digits = []
        if not digits:
            try:
                q = (
                    db.query(Draw.number)
                    .order_by(Draw.created_at.asc())
                    .limit(self.HISTORY_MAX)
                )
                digits = [int(r[0]) for r in q.all() if r[0] is not None]
            except Exception:
                digits = []
        return digits

    def _load_recent_audits(self, db, limit: int = 5000) -> List[Dict[str, Any]]:
        try:
            rows = (
                db.query(PredictionAudit)
                .filter(PredictionAudit.actual_size.isnot(None))
                .order_by(PredictionAudit.id.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "probability_big": r.probability_big,
                    "actual_size": r.actual_size,
                }
                for r in rows
                if r.probability_big is not None and r.actual_size is not None
            ]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Prediction round
    # ------------------------------------------------------------------

    def run_prediction_round(self, next_issue: Optional[str] = None, current_issue: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete real-time prediction pipeline and persist to DB.
        Returns the final prediction dict for AIBrainState / API consumption.
        
        CRITICAL: Includes current_issue to ensure proper issue tracking in stored state.
        """
        db = SessionLocal()
        try:
            self._ensure_migrations(db)
            if not self.initialized:
                boot = self.cold_start(db)
                if boot.get("status") == "INSUFFICIENT_HISTORY":
                    return self._fallback_prediction(next_issue, boot)
            history = self._load_history_digits(db)
            if len(history) < 10:
                return self._fallback_prediction(next_issue, {"reason": "history_too_short"})

            seq_no = 0
            if next_issue and str(next_issue).isdigit():
                seq_no = int(next_issue)
            else:
                # Guess next issue number from max known
                try:
                    last_outcome = db.query(Outcome).order_by(Outcome.sequence_no.desc()).first()
                    if last_outcome:
                        seq_no = int(last_outcome.sequence_no) + 1
                except Exception:
                    seq_no = 0

            # Run the engine
            pred = self.engine.predict(
                recent_history_digits=history,
                next_issue_number=str(next_issue) if next_issue else str(seq_no),
                next_sequence_no=seq_no,
            )
            
            # CRITICAL: Store current_issue in prediction for API validation
            if current_issue:
                pred["currentIssue"] = str(current_issue)

            # Resolve prior unresolved prediction if any
            self._resolve_previous_if_available(db, history)

            # Save prediction audit (v3 extended)
            fp = self.engine.last_fp
            critic = self.engine.last_critic
            calib = self.engine.last_calib
            meta = self.engine.last_meta
            tl = self.engine.last_three_level
            try:
                model_preds_serial = [o.to_dict() for o in self.engine.last_model_outputs]
            except Exception:
                model_preds_serial = []
            # input_window_hash (causal)
            window_hex = hashlib.sha256(
                repr(history[-min(64, len(history)):]).encode()
            ).hexdigest()

            save_full_prediction_audit(
                db=db,
                sequence_no=str(seq_no),
                model_version=f"evoseq_v3_gen{self.engine.generation}",
                probability_big=pred.get("probability_big", 0.5),
                predicted_digit=pred.get("targetNum"),
                entropy=pred.get("entropy"),
                regime_id=pred.get("regime"),
                drift_score=(critic.uncertainty_score if critic else None),
                null_advantage=pred.get("oosScore"),
                # v3 extended
                generation=self.engine.generation,
                state_fingerprint=(fp.to_dict() if fp else None),
                input_window_hash=window_hex,
                state_similarity=pred.get("stateSimilarity"),
                state_sample_size=pred.get("stateSampleSize"),
                regime=pred.get("regime"),
                adversarial_score=pred.get("adversarialScore"),
                support_score=(critic.support_score if critic else None),
                contradiction_score=(critic.contradiction_score if critic else None),
                uncertainty_score=(critic.uncertainty_score if critic else None),
                calibrated_probability=pred.get("calibratedProbability"),
                calibration_error=pred.get("calibrationError"),
                expected_calibration_error=(calib.expected_calibration_error if calib else None),
                oos_score=pred.get("oosScore"),
                baseline_score=pred.get("baselineScore"),
                edge_status=pred.get("edgeStatus"),
                learning_status=pred.get("learningStatus"),
                model_reliability=pred.get("modelReliability"),
                knowledge_version=pred.get("knowledgeVersion"),
                action=pred.get("action"),
                model_predictions=model_preds_serial,
                model_weights=(meta.model_weights if meta else None),
                ensemble_probability=pred.get("probability_big"),
                meta_probability=(meta.ensemble_probability_big if meta else None),
                p_success_l1=(tl.p_success_l1 if tl else None),
                p_success_l2=(tl.p_success_l2 if tl else None),
                p_success_l3=(tl.p_success_l3 if tl else None),
                model_consensus=pred.get("modelConsensus"),
                strike_quality=pred.get("strikeQuality"),
                target_num=pred.get("targetNum"),
                hedge_num=pred.get("hedgeNum"),
            )

            # Also save legacy PredictionLog for compatibility
            try:
                save_prediction(
                    db=db,
                    issue_number=str(seq_no) if seq_no else (next_issue or "unknown"),
                    prediction=pred.get("prediction", "Big"),
                    confidence=pred.get("confidence", 50.0),
                    pattern_name=pred.get("patternName", ""),
                )
            except Exception:
                pass

            # Save to AIBrainState in the same format the server/telegram expects
            # Server reads AIBrainState where model_name == Live_UI_State, parses synaptic_weights JSON
            self._persist_live_ui_state(db, pred, generation=self.engine.generation)

            # Track last sequence we predicted
            self._last_processed_sequence_no = seq_no
            return pred
        finally:
            db.close()

    def _fallback_prediction(self, next_issue, reason) -> Dict[str, Any]:
        db = SessionLocal()
        try:
            fallback = {
                "prediction": "Big",
                "confidence": 50.0,
                "targetNum": 5,
                "hedgeNum": 4,
                "nextIssue": str(next_issue) if next_issue else "",
                "action": "SKIP",
                "strikeQuality": "NO_EDGE",
                "modelConsensus": 0.5,
                "martingaleLevel": 0,
                "driftLevel": "NONE",
                "patternName": "Bootstrapping",
                "totalSamplesTrained": 0,
                "ensembleWeights": {},
                "modelPBigVector": [],
                "generation": 1,
                "stateFingerprint": {},
                "stateSimilarity": 0.0,
                "stateSampleSize": 0,
                "entropy": 1.0,
                "regime": "UNKNOWN",
                "adversarialScore": 0.0,
                "contradictionScore": 0.0,
                "calibratedProbability": 0.5,
                "calibrationError": 0.05,
                "oosScore": None,
                "baselineScore": 0.5,
                "edgeStatus": "NO_EDGE",
                "learningStatus": "WARMING_UP",
                "modelReliability": {},
                "knowledgeVersion": "gen1",
                "fallbackReason": reason,
                "probability_big": 0.5,
                "probability_small": 0.5,
            }
            self._persist_live_ui_state(db, fallback, 1)
            return fallback
        finally:
            db.close()

    def _persist_live_ui_state(self, db, prediction: Dict[str, Any], generation: int) -> None:
        try:
            weights_json = json.dumps(prediction, default=str)
            save_ai_brain_state(
                db=db,
                model_name=self.LIVE_UI_MODEL_NAME,
                generation=int(generation),
                total_samples=int(prediction.get("totalSamplesTrained") or 0),
                weights_json=weights_json,
                win_rate=0.5,  # Fast memory tracks accuracy; placeholder here
            )
            # Also track generation in its own row
            try:
                save_ai_brain_state(
                    db=db,
                    model_name="generation_tracker",
                    generation=int(generation),
                    total_samples=int(prediction.get("totalSamplesTrained") or 0),
                    weights_json=json.dumps({"generation": generation}),
                    win_rate=0.5,
                )
            except Exception:
                pass
        except Exception as e:
            print(f"[Bridge] Persist note: {e}")

    # ------------------------------------------------------------------
    # Resolve previous prediction when outcome becomes known
    # ------------------------------------------------------------------

    def _resolve_previous_if_available(self, db, history_digits: List[int]) -> None:
        """
        If the engine made a prediction last round and history has now advanced
        by at least one digit, feed the resolved outcome back for online learning.
        """
        try:
            if self.engine.last_prediction_side is None or len(history_digits) < 2:
                return
            actual_digit = history_digits[-1]
            self.engine.resolve_outcome(
                actual_digit=actual_digit,
                history_suffix_for_partial_fit=history_digits,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Daily evolution trigger
    # ------------------------------------------------------------------

    def run_daily_evolution(self) -> Dict[str, Any]:
        db = SessionLocal()
        try:
            self._ensure_migrations(db)
            if not self.initialized:
                self.cold_start(db)
            history = self._load_history_digits(db)
            gen_record_dict = self.engine.run_daily_evolution(history)
            # Persist generation number to DB
            try:
                save_ai_brain_state(
                    db=db,
                    model_name="generation_tracker",
                    generation=int(self.engine.generation),
                    total_samples=len(history),
                    weights_json=json.dumps(gen_record_dict, default=str),
                    win_rate=float(gen_record_dict.get("champion_oos_accuracy") or 0.5),
                )
            except Exception:
                pass
            return gen_record_dict
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Dashboard / report helpers
    # ------------------------------------------------------------------

    def get_dashboard(self) -> Dict[str, Any]:
        if not self.initialized:
            db = SessionLocal()
            try:
                self.cold_start(db)
            finally:
                db.close()
        try:
            return self.engine.build_dashboard().to_dict()
        except Exception as e:
            return {"error": str(e)}

    def get_daily_report(self, new_samples_today: int = 0) -> Dict[str, Any]:
        db = SessionLocal()
        try:
            if not self.initialized:
                self.cold_start(db)
            total_n = len(self._load_history_digits(db))
            report = self.engine.build_daily_report(
                historical_samples_total=total_n,
                new_samples_today=new_samples_today,
            )
            return report.to_dict()
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Global singleton (use this from server or evoseq_loop)
# ---------------------------------------------------------------------------

_BRIDGE: Optional[EVOSEQBridge] = None


def get_bridge() -> EVOSEQBridge:
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = EVOSEQBridge()
    return _BRIDGE


def run_prediction(next_issue: Optional[str] = None) -> Dict[str, Any]:
    return get_bridge().run_prediction_round(next_issue)


def run_evolution() -> Dict[str, Any]:
    return get_bridge().run_daily_evolution()


def get_dashboard_data() -> Dict[str, Any]:
    return get_bridge().get_dashboard()


def get_daily_report_data(new_today: int = 0) -> Dict[str, Any]:
    return get_bridge().get_daily_report(new_samples_today=new_today)
