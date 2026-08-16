import sys
import os
import unittest
import torch
import numpy as np

# Add EVOSEQ root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.ensemble import (
    OnlineLossTracker,
    softmax_weights,
    combine_predictions,
    MetaGate,
    jensen_shannon_divergence,
    pairwise_diversity_matrix,
    model_diversity_score,
    diversity_adjusted_weights,
    HierarchicalEnsemble,
    TemperatureScaler,
    decompose_uncertainty,
    evolution_pressure,
    evolution_state,
    compute_model_contributions,
    MetaReplayBuffer
)
from app.database import SessionLocal, engine, Base
from app.schemas import EnsembleObservationRecord
from scripts.run_dynamic_ensemble import run_dynamic_ensemble

class TestDynamicEnsemble(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)

    def test_tracker_and_softmax_weights(self):
        tracker = OnlineLossTracker(decay=0.1)
        tracker.update("m1", 2.30)
        tracker.update("m1", 2.20)
        self.assertAlmostEqual(tracker.losses["m1"], 0.1 * 2.20 + 0.9 * 2.30, places=4)
        
        w = softmax_weights([2.30, 2.10, 2.50], beta=2.0)
        self.assertEqual(len(w), 3)
        self.assertAlmostEqual(float(np.sum(w)), 1.0, places=5)
        self.assertGreater(w[1], w[0]) # Lower loss gets higher weight
        self.assertGreater(w[0], w[2])

    def test_meta_gate(self):
        gate = MetaGate(feature_size=6, models=4, hidden=16)
        env = torch.randn(2, 6)
        w = gate(env)
        self.assertEqual(w.shape, (2, 4))
        for i in range(2):
            self.assertAlmostEqual(float(torch.sum(w[i]).item()), 1.0, places=4)

    def test_diversity_and_js_divergence(self):
        p1 = np.array([0.9] + [0.0111]*9)
        p2 = np.array([0.0111]*9 + [0.9])
        js_val = jensen_shannon_divergence(p1, p2)
        self.assertGreater(js_val, 0.5)
        
        preds = [p1, p2, np.full(10, 0.1)]
        mat = pairwise_diversity_matrix(preds)
        self.assertEqual(mat.shape, (3, 3))
        self.assertEqual(mat[0, 0], 0.0)
        
        d_weights = diversity_adjusted_weights(losses=[2.30, 2.30, 2.30], predictions=preds, gamma=0.2)
        self.assertEqual(len(d_weights), 3)
        self.assertAlmostEqual(float(np.sum(d_weights)), 1.0, places=4)

    def test_hierarchical_ensemble(self):
        ensemble = HierarchicalEnsemble()
        preds = {
            "frequency": np.full((5, 10), 0.1),
            "markov1": np.full((5, 10), 0.1),
            "esn": np.full((5, 10), 0.1),
            "transformer": np.full((5, 10), 0.1)
        }
        res = ensemble.combine_hierarchical(preds)
        self.assertIn("statistical", res["family_distributions"])
        self.assertIn("recurrent", res["family_distributions"])
        self.assertIn("neural", res["family_distributions"])
        self.assertEqual(res["ensemble_prediction"].shape, (5, 10))

    def test_temperature_scaler(self):
        scaler = TemperatureScaler(temperature=1.0)
        p = np.array([0.7, 0.1, 0.1, 0.1])
        p_soft = scaler.scale(p, temperature=2.0)
        self.assertLess(p_soft[0], 0.7) # Softened
        
        p_sharp = scaler.scale(p, temperature=0.5)
        self.assertGreater(p_sharp[0], 0.7) # Sharpened

    def test_uncertainty_decomposition_and_evolution_pressure(self):
        p1 = np.array([0.8] + [0.0222]*9)
        p2 = np.array([0.0222]*9 + [0.8])
        unc = decompose_uncertainty([p1, p2])
        self.assertGreater(unc["disagreement"], 0.5)
        
        pressure = evolution_pressure(loss_degradation=0.1, drift=0.05, disagreement=unc["disagreement"], calibration_error=0.02)
        self.assertGreater(pressure, 0.1)
        
        state = evolution_state(pressure)
        self.assertIn(state, ["STABLE", "INVESTIGATE", "EVOLVE"])

    def test_ablation_contribution(self):
        preds = {
            "m1": np.array([[0.9] + [0.0111]*9, [0.9] + [0.0111]*9]),
            "m2": np.array([[0.1]*10, [0.1]*10])
        }
        targets = [0, 0]
        contrib = compute_model_contributions(preds, targets)
        self.assertIn("m1", contrib)
        self.assertIn("m2", contrib)
        self.assertGreater(contrib["m1"]["contribution"], 0.0) # Removing m1 increases loss

    def test_meta_replay_buffer_and_db_record(self):
        buf = MetaReplayBuffer(max_size=50)
        for i in range(20):
            buf.add({"step": i, "data": np.random.rand(5)})
        self.assertEqual(len(buf), 20)
        samples = buf.sample(5)
        self.assertEqual(len(samples), 5)
        
        with SessionLocal() as session:
            rec = EnsembleObservationRecord(
                sequence_no=99999,
                environment={"entropy": 3.3},
                model_predictions={"m1": [0.1]*10},
                model_weights={"statistical": 0.5},
                ensemble_prediction=[0.1]*10,
                actual_digit=3,
                ensemble_log_loss=2.3025,
                disagreement=0.02
            )
            session.add(rec)
            session.commit()
            
            saved = session.query(EnsembleObservationRecord).filter(EnsembleObservationRecord.sequence_no == 99999).first()
            self.assertIsNotNone(saved)
            self.assertEqual(saved.actual_digit, 3)

    def test_run_dynamic_ensemble_script(self):
        report = run_dynamic_ensemble()
        self.assertIn("EVOSEQ — DYNAMIC ENSEMBLE INTELLIGENCE", report)
        self.assertIn("FAMILY WEIGHTS", report)

if __name__ == "__main__":
    unittest.main()
