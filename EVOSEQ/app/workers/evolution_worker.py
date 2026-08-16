from typing import Optional
from .base import BaseWorker
from ..evolution.orchestrator import autonomous_evolution_cycle

class EvolutionWorker(BaseWorker):
    """
    Evolution Worker: Evaluates model degradation, triggers challenger tournaments,
    and runs continuous evolutionary cycles when justified.
    """

    def __init__(self, step_threshold: int = 50, poll_interval: float = 3.0):
        super().__init__(worker_name="evolution_worker", poll_interval=poll_interval)
        self.step_threshold = step_threshold

    def process_cycle(self) -> int:
        state = self.get_state()
        last_seq = state.last_processed_sequence or 0
        
        report = autonomous_evolution_cycle(last_seq_cursor=last_seq)
        if report.get("status") == "SUCCESS":
            new_cursor = report.get("latest_sequence", last_seq)
            if new_cursor > last_seq:
                self.publish_event("EVOLUTION_DECISION", new_cursor, report)
                self.update_state(new_cursor, status="idle")
                return 1
                
        return 0

if __name__ == "__main__":
    EvolutionWorker().run()
