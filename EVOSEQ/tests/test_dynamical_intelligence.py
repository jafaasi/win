import sys
import os
import unittest
import numpy as np
import torch
from fastapi.testclient import TestClient

# Add EVOSEQ root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.dynamical.change_point import OnlineChangeDetector
from app.dynamical.recurrence import recurrence_matrix, recurrence_quantification_analysis
from app.dynamical.symbolic import symbolic_words, symbolic_complexity_curve
from app.dynamical.latent import LatentStateEncoder, LatentStabilityTester
from app.dynamical.synthetic_generators import ControlledGenerators
from app.dynamical.smt_solver import AnalyticalLCGSolver
from app.dynamical.bottleneck import MemoryDepthEstimator
from app.ingestion.stream import ingest_outcomes_batch
from app.database import SessionLocal
from app.schemas import HiddenStateExperimentRecord, EnvironmentFingerprintRecord
from app.api import app

class TestDynamicalIntelligence(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_online_change_detector(self):
        detector = OnlineChangeDetector(reference_size=50, recent_size=20, threshold=0.05)
        # Identical sequence -> near zero change
        uniform_seq = list(np.random.default_rng(42).integers(0, 10, size=100))
        score_low = detector.score(uniform_seq)
        self.assertLess(score_low, 0.10)
        
        # Injected step change from 0-4 to 5-9
        shifted_seq = [1, 2, 3, 2, 1] * 12 + [8, 9, 8, 9, 8] * 6
        score_high = detector.score(shifted_seq)
        self.assertGreater(score_high, 0.05)
        
        state = detector.classify_state(score_high, info_gain=0.08)
        self.assertEqual(state, "HIGH_DRIFT_HIGH_INFO")

    def test_recurrence_quantification_analysis(self):
        # Periodic sequence creates clear diagonal recurrence structures
        periodic_seq = np.array([0, 5, 0, 5, 0, 5, 0, 5, 0, 5] * 3)
        R = recurrence_matrix(periodic_seq, epsilon=0.1)
        self.assertEqual(R.shape, (30, 30))
        self.assertEqual(R[0, 0], 1)
        
        rqa = recurrence_quantification_analysis(R)
        self.assertIn("recurrence_rate", rqa)
        self.assertIn("determinism", rqa)
        self.assertGreater(rqa["determinism"], 0.0)

    def test_symbolic_dynamics(self):
        seq = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3] * 5
        words = symbolic_words(seq, k=3)
        self.assertEqual(len(words), len(seq) - 2)
        
        curve = symbolic_complexity_curve(seq, max_k=3)
        self.assertIn("C_1", curve)
        self.assertIn("C_2", curve)
        self.assertIn("C_3", curve)
        # Repeating pattern has very low normalized complexity for k=3
        self.assertLess(curve["C_3"], 0.5)

    def test_latent_state_encoder_and_stability(self):
        encoder = LatentStateEncoder(input_size=10, hidden_size=16, output_size=10)
        x = torch.randn(2, 15, 10)
        logits, z_t = encoder(x)
        self.assertEqual(logits.shape, (2, 10))
        self.assertEqual(z_t.shape, (2, 16))
        
        seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 3
        stability = LatentStabilityTester.measure_stability(encoder, seq)
        self.assertIsInstance(stability, float)

    def test_controlled_synthetic_generators(self):
        obs_iid, st_iid = ControlledGenerators.generator_iid(length=50, seed=42)
        self.assertEqual(len(obs_iid), 50)
        
        obs_mkv, st_mkv = ControlledGenerators.generator_markov(length=50, seed=42)
        self.assertEqual(len(obs_mkv), 50)
        
        obs_hmm, st_hmm = ControlledGenerators.generator_hmm(length=50, n_states=3, seed=42)
        self.assertEqual(len(obs_hmm), 50)
        self.assertEqual(len(st_hmm), 50)
        
        obs_fsm, st_fsm = ControlledGenerators.generator_fsm(length=50, seed=42)
        self.assertEqual(len(obs_fsm), 50)
        
        obs_lcg, st_lcg = ControlledGenerators.generator_lcg(length=50, a=17, c=43, m=10007, seed=42)
        self.assertEqual(len(obs_lcg), 50)
        self.assertEqual(st_lcg[0], 42)

    def test_analytical_smt_solver_recovery(self):
        obs_lcg, st_lcg = ControlledGenerators.generator_lcg(length=30, a=13, c=7, m=10007, seed=23)
        res = AnalyticalLCGSolver.attempt_recovery(obs_lcg, known_m=10007, max_search_a=20, max_search_c=20)
        self.assertIn("recovered", res)
        self.assertIn("runtime_seconds", res)
        self.assertGreater(res["confidence"], 0.0)

    def test_memory_depth_estimator(self):
        seq = [0, 1, 2, 0, 1, 2, 0, 1, 2] * 10
        depth_res = MemoryDepthEstimator.estimate_depth_curve(seq, context_horizons=(1, 2, 3))
        self.assertIn("saturation_order", depth_res)
        self.assertIn("scores", depth_res)
        self.assertIn("deltas", depth_res)

    def test_database_records_persistence(self):
        with SessionLocal() as session:
            exp = HiddenStateExperimentRecord(
                generator_type="LCG_TOY",
                observation_count=100,
                state_dimension=1,
                state_recovery_score=1.0,
                parameter_recovery_score=1.0,
                runtime_seconds=0.02
            )
            session.add(exp)
            
            fp = EnvironmentFingerprintRecord(
                sequence_no=999999,
                entropy=3.32,
                conditional_entropy_1=3.10,
                conditional_entropy_2=3.00,
                information_gain_1=0.22,
                information_gain_2=0.32,
                lz_complexity=45.0,
                lz_zscore=0.1,
                autocorrelation_1=0.01,
                autocorrelation_2=0.00,
                drift_score=0.02,
                recurrence_rate=0.12
            )
            session.add(fp)
            session.commit()
            
            saved_exp = session.query(HiddenStateExperimentRecord).filter(HiddenStateExperimentRecord.generator_type == "LCG_TOY").first()
            self.assertIsNotNone(saved_exp)
            
            saved_fp = session.query(EnvironmentFingerprintRecord).filter(EnvironmentFingerprintRecord.sequence_no == 999999).first()
            self.assertIsNotNone(saved_fp)

    def test_fastapi_dynamical_endpoints(self):
        # Ingest samples so endpoints have data
        batch = [{"sequence_no": 770000 + i, "digit": (i % 4) * 2} for i in range(40)]
        ingest_outcomes_batch(batch)
        
        res_cp = self.client.get("/dynamical/change-points")
        self.assertEqual(res_cp.status_code, 200)
        
        res_rec = self.client.get("/dynamical/recurrence")
        self.assertEqual(res_rec.status_code, 200)
        
        res_sym = self.client.get("/dynamical/symbolic")
        self.assertEqual(res_sym.status_code, 200)
        
        res_md = self.client.get("/dynamical/memory-depth")
        self.assertEqual(res_md.status_code, 200)

if __name__ == "__main__":
    unittest.main()
