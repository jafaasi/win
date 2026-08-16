import time
from typing import Optional
from .ingestion_worker import IngestionWorker
from .feature_worker import FeatureWorker
from .inference_worker import InferenceWorker
from .reconciliation_worker import ReconciliationWorker
from .drift_worker import DriftWorker
from .evolution_worker import EvolutionWorker
from .research_worker import ResearchWorker
from .dashboard import render_system_intelligence_hud

class OrchestratorDaemon:
    """
    Unified Orchestrator Daemon: Coordinates all decoupled workers in sequential event order.
    """

    def __init__(self):
        self.ingestion = IngestionWorker()
        self.features = FeatureWorker()
        self.inference = InferenceWorker()
        self.reconciliation = ReconciliationWorker()
        self.drift = DriftWorker()
        self.evolution = EvolutionWorker()
        self.research = ResearchWorker()

    def run_step(self) -> int:
        """Runs one full cycle across all workers."""
        n_ing = self.ingestion.process_cycle()
        n_feat = self.features.process_cycle()
        n_inf = self.inference.process_cycle()
        n_rec = self.reconciliation.process_cycle()
        n_drf = self.drift.process_cycle()
        n_evo = self.evolution.process_cycle()
        n_res = self.research.process_cycle()
        return n_ing + n_feat + n_inf + n_rec + n_drf + n_evo + n_res

    def run_forever(self, poll_interval: float = 2.0, display_hud: bool = False):
        print("=== 🚀 EVOSEQ PRODUCTION ORCHESTRATOR RUNNING ===")
        while True:
            try:
                processed = self.run_step()
                if display_hud:
                    print(render_system_intelligence_hud())
                if processed == 0:
                    time.sleep(poll_interval)
            except Exception as e:
                print(f"⚠️ Orchestrator Daemon error: {e}")
                time.sleep(poll_interval)

if __name__ == "__main__":
    OrchestratorDaemon().run_forever(display_hud=True)
