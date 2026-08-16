from typing import Optional, List, Dict, Any
from .base import BaseWorker
from ..database import SessionLocal
from ..schemas import Outcome

class IngestionWorker(BaseWorker):
    """
    Ingestion Worker: Monitors database stream cursor and publishes OUTCOME_ARRIVED events.
    """

    def __init__(self, poll_interval: float = 1.0):
        super().__init__(worker_name="ingestion_worker", poll_interval=poll_interval)

    def process_cycle(self) -> int:
        state = self.get_state()
        last_seq = state.last_processed_sequence or 0
        
        with SessionLocal() as session:
            rows = session.query(Outcome).filter(Outcome.sequence_no > last_seq).order_by(Outcome.sequence_no.asc()).limit(50).all()
            if not rows:
                self.update_state(last_seq, status="idle")
                return 0
                
            for r in rows:
                payload = {
                    "sequence_no": r.sequence_no,
                    "digit": r.digit,
                    "size": r.size,
                    "color": r.color,
                    "parity": r.parity
                }
                self.publish_event("OUTCOME_ARRIVED", r.sequence_no, payload)
                last_seq = r.sequence_no
                
            self.update_state(last_seq, status="running")
            return len(rows)

if __name__ == "__main__":
    IngestionWorker().run()
