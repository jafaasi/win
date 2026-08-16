from typing import Optional
from .base import BaseWorker
from ..database import SessionLocal
from ..schemas import SystemEventRecord, Outcome
from ..evolution.drift import calculate_multidimensional_drift

class DriftWorker(BaseWorker):
    """
    Drift Worker: Evaluates multi-dimensional drift across recent vs reference windows,
    and publishes DRIFT_EVALUATED events.
    """

    def __init__(self, poll_interval: float = 2.0):
        super().__init__(worker_name="drift_worker", poll_interval=poll_interval)

    def process_cycle(self) -> int:
        state = self.get_state()
        last_seq = state.last_processed_sequence or 0
        
        with SessionLocal() as session:
            events = session.query(SystemEventRecord).filter(
                SystemEventRecord.event_type == "OUTCOME_RECONCILED",
                SystemEventRecord.sequence_no > last_seq
            ).order_by(SystemEventRecord.sequence_no.asc()).limit(20).all()
            
            if not events:
                self.update_state(last_seq, status="idle")
                return 0
                
            latest_evt = events[-1]
            seq_no = latest_evt.sequence_no
            
            rows = session.query(Outcome).filter(Outcome.sequence_no <= seq_no).order_by(Outcome.sequence_no.desc()).limit(150).all()
            rows.reverse()
            
            if len(rows) >= 50:
                digits = [r.digit for r in rows]
                sizes = [r.size for r in rows]
                colors = [r.color for r in rows]
                parities = [r.parity for r in rows]
                
                drift_res = calculate_multidimensional_drift(
                    digits=digits,
                    sizes=sizes,
                    colors=colors,
                    parities=parities,
                    recent_window=25,
                    historical_window=50
                )
                self.publish_event("DRIFT_EVALUATED", seq_no, {
                    "composite_drift": drift_res.composite_drift,
                    "state": drift_res.state.value
                })
                
            last_seq = seq_no
            self.update_state(last_seq, status="running")
            return len(events)

if __name__ == "__main__":
    DriftWorker().run()
