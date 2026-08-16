import sys
import os
import unittest
from fastapi.testclient import TestClient

# Add EVOSEQ root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.workers.ingestion_worker import IngestionWorker
from app.workers.feature_worker import FeatureWorker
from app.workers.inference_worker import InferenceWorker
from app.workers.reconciliation_worker import ReconciliationWorker
from app.workers.drift_worker import DriftWorker
from app.workers.orchestrator_daemon import OrchestratorDaemon
from app.checkpoints.checkpoint_manager import CheckpointManager
from app.evolution.registry import ModelRegistry
from app.models.markov import MarkovModel
from app.ingestion.stream import ingest_outcomes_batch
from app.database import SessionLocal
from app.schemas import WorkerStateRecord, SystemEventRecord
from app.api import app

class TestProductionOrchestration(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.registry = ModelRegistry()

    def test_worker_idempotency(self):
        worker = IngestionWorker()
        pub1 = worker.publish_event("TEST_EVENT", 12345, {"data": 1}, model_version_id=1)
        self.assertTrue(pub1)
        
        # Second identical publish must be rejected / skipped
        pub2 = worker.publish_event("TEST_EVENT", 12345, {"data": 1}, model_version_id=1)
        self.assertFalse(pub2)

    def test_checkpoint_manager(self):
        chk = CheckpointManager(base_dir="/tmp/test_evoseq_chk")
        model = MarkovModel(order=2, version="markov-test-v1")
        model.fit([1, 2, 3, 4, 1, 2, 3, 4])
        
        d = chk.save_checkpoint(model)
        self.assertTrue(os.path.isdir(d))
        self.assertTrue(chk.verify_integrity("Markov", "markov-test-v1"))

    def test_model_probation_and_rollback(self):
        m1_id = self.registry.register_candidate("Markov", "mkv-champ-1", parameters={"order": 1})
        m2_id = self.registry.register_candidate("Markov", "mkv-challenger-2", parameters={"order": 2})
        
        self.registry.promote(m1_id, reason="Initial champion")
        champ = self.registry.get_champion()
        self.assertEqual(champ["id"], m1_id)
        
        # Put m2 on probation
        self.registry.put_on_probation(m2_id, reason="Won tournament")
        
        # Trigger rollback back to m1
        rolled = self.registry.rollback(m2_id, m1_id, reason="Probation degraded")
        self.assertTrue(rolled)
        champ_after = self.registry.get_champion()
        self.assertEqual(champ_after["id"], m1_id)

    def test_end_to_end_worker_pipeline(self):
        # 1. Ingest synthetic outcomes
        batch = [
            {"sequence_no": 880000 + i, "digit": (i % 5) * 2}
            for i in range(25)
        ]
        ingest_outcomes_batch(batch)
        
        # 2. Run orchestrator daemon step
        daemon = OrchestratorDaemon()
        daemon.ingestion.get_state()
        daemon.ingestion.update_state(879999) # reset cursor for test
        
        step_processed = daemon.run_step()
        self.assertGreater(step_processed, 0)
        
        with SessionLocal() as session:
            evts = session.query(SystemEventRecord).filter(SystemEventRecord.sequence_no >= 880000).all()
            self.assertGreater(len(evts), 0)

    def test_fastapi_endpoints(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "healthy")
        
        res_sys = self.client.get("/system/status")
        self.assertEqual(res_sys.status_code, 200)
        self.assertIn("observations", res_sys.json())
        
        res_models = self.client.get("/models")
        self.assertEqual(res_models.status_code, 200)
        self.assertIsInstance(res_models.json(), list)
        
        res_events = self.client.get("/events")
        self.assertEqual(res_events.status_code, 200)

if __name__ == "__main__":
    unittest.main()
