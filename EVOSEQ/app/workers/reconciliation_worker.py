from typing import Optional
import numpy as np
from .base import BaseWorker
from ..database import SessionLocal
from ..schemas import SystemEventRecord, Outcome, PredictionRecord, ChampionHealthRecord
from ..evaluation.metrics import log_loss, brier_score

class ReconciliationWorker(BaseWorker):
    """
    Reconciliation Worker: Matches actual outcomes with previous predictions, computes log-loss,
    Brier score, records champion health, and publishes OUTCOME_RECONCILED.
    """

    def __init__(self, poll_interval: float = 1.0):
        super().__init__(worker_name="reconciliation_worker", poll_interval=poll_interval)

    def process_cycle(self) -> int:
        state = self.get_state()
        last_seq = state.last_processed_sequence or 0
        
        with SessionLocal() as session:
            # Look for outcomes that have arrived
            outcomes = session.query(Outcome).filter(Outcome.sequence_no > last_seq).order_by(Outcome.sequence_no.asc()).limit(50).all()
            if not outcomes:
                self.update_state(last_seq, status="idle")
                return 0
                
            for out in outcomes:
                seq_no = out.sequence_no
                # Find predictions for this target sequence_no
                preds = session.query(PredictionRecord).filter(PredictionRecord.sequence_no == seq_no).all()
                for p in preds:
                    p.actual_class = out.digit
                    probs = np.array(p.probability_vector, dtype=np.float64)
                    ll = log_loss(probs, out.digit)
                    bs = brier_score(probs, out.digit)
                    p.log_loss = float(ll)
                    p.brier_score = float(bs)
                    
                    # Track health
                    health = ChampionHealthRecord(
                        model_version_id=p.model_version_id or 1,
                        sequence_no=seq_no,
                        health_score=round(1.0 - bs, 4),
                        drift_score=0.0,
                        calibration_error=0.0,
                        disagreement=0.0
                    )
                    session.add(health)
                    
                self.publish_event("OUTCOME_RECONCILED", seq_no, {"reconciled_predictions": len(preds)})
                last_seq = seq_no
                
            session.commit()
            self.update_state(last_seq, status="running")
            return len(outcomes)

if __name__ == "__main__":
    ReconciliationWorker().run()
