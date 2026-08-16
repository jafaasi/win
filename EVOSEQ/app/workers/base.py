import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any
from ..database import SessionLocal
from ..schemas import WorkerStateRecord, SystemEventRecord

class BaseWorker(ABC):
    """
    Abstract base class for decoupled, idempotent event workers.
    Guarantees state tracking in worker_state and crash recovery from last sequence cursor.
    """

    def __init__(self, worker_name: str, poll_interval: float = 1.0):
        self.worker_name = worker_name
        self.poll_interval = poll_interval
        self._running = True

    def get_state(self) -> WorkerStateRecord:
        with SessionLocal() as session:
            state = session.query(WorkerStateRecord).filter(WorkerStateRecord.worker_name == self.worker_name).first()
            if not state:
                state = WorkerStateRecord(worker_name=self.worker_name, last_processed_sequence=0, status="idle")
                session.add(state)
                session.commit()
                session.refresh(state)
            return state

    def update_state(self, last_seq: int, status: str = "running", error: Optional[str] = None) -> None:
        with SessionLocal() as session:
            state = session.query(WorkerStateRecord).filter(WorkerStateRecord.worker_name == self.worker_name).first()
            if state:
                state.last_processed_sequence = max(state.last_processed_sequence or 0, last_seq)
                state.status = status
                state.records_processed = (state.records_processed or 0) + 1
                if error:
                    state.last_failure_at = datetime.utcnow()
                    state.last_error = error
                else:
                    state.last_success_at = datetime.utcnow()
                    state.last_error = None
                session.commit()

    def publish_event(self, event_type: str, sequence_no: int, payload: Dict[str, Any], model_version_id: Optional[int] = None) -> bool:
        """Publishes system event with guaranteed unique idempotency."""
        with SessionLocal() as session:
            existing = session.query(SystemEventRecord).filter(
                SystemEventRecord.event_type == event_type,
                SystemEventRecord.sequence_no == sequence_no,
                SystemEventRecord.model_version_id == model_version_id
            ).first()
            if existing:
                return False # Idempotent skip
                
            evt = SystemEventRecord(
                event_type=event_type,
                sequence_no=sequence_no,
                model_version_id=model_version_id,
                payload=payload,
                status="completed"
            )
            session.add(evt)
            session.commit()
            return True

    @abstractmethod
    def process_cycle(self) -> int:
        """Executes a single processing step. Returns number of items processed."""
        pass

    def run(self, max_cycles: Optional[int] = None) -> None:
        cycles = 0
        while self._running:
            try:
                processed = self.process_cycle()
                if processed == 0:
                    time.sleep(self.poll_interval)
            except Exception as e:
                self.update_state(0, status="error", error=str(e))
                time.sleep(self.poll_interval)
                
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
                
    def stop(self) -> None:
        self._running = False
