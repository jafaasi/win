import sys
import os
import unittest
from datetime import datetime
import numpy as np
import torch

# Add EVOSEQ root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.types import Outcome, validate_outcome, validate_probability_vector
from app.models.multi_horizon import MultiHorizonHead
from app.ensemble.weighting import adaptive_weights, ExponentialLossTracker
from app.ensemble.calibrator import TemperatureCalibrator
from app.ensemble.disagreement import calculate_ensemble_disagreement
from scripts.bootstrap_db import bootstrap
from scripts.daily_pipeline import run_daily_pipeline

class TestEngineAssembly(unittest.TestCase):

    def test_outcome_data_contract(self):
        valid_out = Outcome(sequence_no=100, timestamp=datetime.utcnow(), digit=7, size=1, color=0, parity=1)
        validate_outcome(valid_out) # should pass cleanly
        
        invalid_digit = Outcome(sequence_no=101, timestamp=datetime.utcnow(), digit=15, size=1, color=0, parity=1)
        with self.assertRaises(ValueError):
            validate_outcome(invalid_digit)
            
        invalid_color = Outcome(sequence_no=102, timestamp=datetime.utcnow(), digit=3, size=0, color=4, parity=1)
        with self.assertRaises(ValueError):
            validate_outcome(invalid_color)

    def test_validate_probability_vector(self):
        # Valid simplex
        p_valid = np.array([0.1] * 10)
        norm_p = validate_probability_vector(p_valid)
        self.assertEqual(len(norm_p), 10)
        self.assertAlmostEqual(float(np.sum(norm_p)), 1.0, places=5)
        
        # Invalid class count
        with self.assertRaises(ValueError):
            validate_probability_vector([0.5, 0.5])
            
        # Invalid sum
        with self.assertRaises(ValueError):
            validate_probability_vector([0.2] * 10)

    def test_multi_horizon_neural_head(self):
        head = MultiHorizonHead(hidden_size=32, horizons=3, classes=10)
        rep = torch.randn(4, 32)
        outputs = head(rep)
        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs[0].shape, (4, 10))
        self.assertEqual(outputs[1].shape, (4, 10))
        self.assertEqual(outputs[2].shape, (4, 10))
        
        targets = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2]], dtype=torch.long)
        loss = head.compute_loss(outputs, targets)
        self.assertGreater(loss.item(), 0.0)

    def test_adaptive_ensemble_weighting(self):
        losses = [2.25, 2.40, 2.10]
        weights = adaptive_weights(losses, beta=2.0)
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=5)
        # Lowest loss model (2.10) must receive largest mixture weight
        self.assertEqual(np.argmax(weights), 2)
        
        tracker = ExponentialLossTracker(alpha=0.1)
        tracker.update("m1", 2.30)
        tracker.update("m2", 2.15)
        mixture = tracker.get_mixture_weights()
        self.assertGreater(mixture["m2"], mixture["m1"])

    def test_temperature_calibrator(self):
        calibrator = TemperatureCalibrator()
        probs = np.array([[0.7, 0.3] + [0.0]*8, [0.1, 0.9] + [0.0]*8])
        labels = [0, 1]
        best_t = calibrator.fit_from_probabilities(probs, labels)
        self.assertIsInstance(best_t, float)
        calibrated = calibrator.calibrate(probs[0])
        self.assertEqual(len(calibrated), 10)

    def test_ensemble_disagreement(self):
        p1 = np.array([0.9] + [0.0111]*9)
        p2 = np.array([0.0111]*9 + [0.9])
        disagree = calculate_ensemble_disagreement([p1, p2])
        self.assertGreater(disagree, 0.5)

    def test_operational_scripts(self):
        # Test bootstrap
        bootstrap()
        # Test daily pipeline HUD report generation
        report = run_daily_pipeline()
        self.assertIn("EVOSEQ DAILY RESEARCH REPORT", report)
        self.assertIn("CURRENT CHAMPION", report)

if __name__ == "__main__":
    unittest.main()
