import sys
import os
import unittest
import numpy as np

# Add EVOSEQ root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.features.encoding import one_hot, encode_outcome
from app.features.entropy import categorical_entropy, entropy, shannon_entropy
from app.features.conditional_entropy import conditional_entropy
from app.features.information import information_gain
from app.features.autocorrelation import autocorrelation
from app.features.runs import current_run_length, run_lengths, run_statistics
from app.features.transitions import transition_matrix, transition_entropy
from app.features.lz import lz_complexity
from app.features.builder import build_features, build_temporal_tensor
from app.features.monitor import FeatureHealthMonitor

class TestFeatureIntelligenceEngine(unittest.TestCase):

    def test_categorical_encoding(self):
        oh = one_hot(3, 10)
        self.assertEqual(len(oh), 10)
        self.assertEqual(oh[3], 1.0)
        self.assertEqual(oh.sum(), 1.0)
        
        encoded = encode_outcome(digit=7, size=1, color=1, parity=1)
        self.assertEqual(len(encoded), 17)
        self.assertEqual(encoded.dtype, np.float32)
        # Check digit 7 is 1.0
        self.assertEqual(encoded[7], 1.0)
        # Check size 1 is 1.0 (offset 10 + 1 = 11)
        self.assertEqual(encoded[11], 1.0)

    def test_entropy_properties(self):
        # Uniform 10-class distribution
        uniform_seq = list(range(10)) * 100
        h_uniform = categorical_entropy(uniform_seq, cardinality=10)
        self.assertAlmostEqual(h_uniform, np.log2(10), places=3)
        
        # Degenerate single class
        constant_seq = [5] * 100
        h_const = categorical_entropy(constant_seq, cardinality=10)
        self.assertEqual(h_const, 0.0)

    def test_conditional_entropy_and_information_gain(self):
        # Deterministic cycle: 0 -> 1 -> 2 -> 0 -> 1 -> 2
        cyclic_seq = [0, 1, 2] * 50
        h_cond_1 = conditional_entropy(cyclic_seq, order=1, cardinality=10)
        self.assertAlmostEqual(h_cond_1, 0.0, places=3)
        
        # Information Gain should equal H(X) since H(X|X_{-1}) = 0
        ig_1 = information_gain(cyclic_seq, order=1, cardinality=10)
        h_marg = categorical_entropy(cyclic_seq, cardinality=10)
        self.assertAlmostEqual(ig_1, h_marg, places=3)

    def test_autocorrelation(self):
        # Alternating series [1, -1, 1, -1] -> lag-1 should be -1.0
        alt_seq = [1, 0, 1, 0, 1, 0, 1, 0] * 20
        acf_1 = autocorrelation(alt_seq, lag=1)
        self.assertLess(acf_1, -0.9)
        
        acf_2 = autocorrelation(alt_seq, lag=2)
        self.assertGreater(acf_2, 0.9)

    def test_run_statistics(self):
        seq = [1, 1, 1, 2, 2, 3, 3, 3, 3]
        self.assertEqual(current_run_length(seq), 4) # four 3s
        
        stats = run_statistics(seq)
        self.assertEqual(stats["max_run"], 4.0)
        self.assertEqual(stats["current_run"], 4.0)
        self.assertGreater(stats["alternation_rate"], 0.0)

    def test_transitions(self):
        seq = [0, 1, 0, 1, 0, 1, 0, 1]
        mat = transition_matrix(seq, cardinality=10)
        self.assertEqual(mat.shape, (10, 10))
        self.assertAlmostEqual(mat[0, 1], 1.0)
        self.assertAlmostEqual(mat[1, 0], 1.0)
        
        t_ent = transition_entropy(mat)
        self.assertAlmostEqual(t_ent, 0.0, places=3)

    def test_lz_complexity(self):
        simple = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        complex_seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 4, 1, 7, 3, 9, 8]
        self.assertLess(lz_complexity(simple), lz_complexity(complex_seq))

    def test_feature_builder_and_temporal_tensor(self):
        seq = [3, 8, 2, 7, 1, 9, 4, 6, 0, 5] * 20
        features = build_features(seq)
        
        self.assertEqual(features.digit, seq[-1])
        self.assertEqual(features.vector.ndim, 1)
        self.assertEqual(len(features.vector), 36)
        
        # Test temporal tensor shape [128, 36]
        tensor = build_temporal_tensor(seq, context_length=128, window_size=32)
        self.assertEqual(tensor.shape, (128, 36))
        self.assertEqual(tensor.dtype, np.float32)

    def test_feature_health_monitor(self):
        monitor = FeatureHealthMonitor(history_window=100)
        for _ in range(50):
            monitor.update({"entropy": 3.32, "lz": 15.0})
            
        report = monitor.compute_health_report({"entropy": 3.32, "lz": 15.0})
        self.assertEqual(report["entropy"]["drift_status"], "NORMAL")
        
        # Test anomaly triggering
        abnormal_report = monitor.compute_health_report({"entropy": 0.10, "lz": 15.0})
        self.assertIn(abnormal_report["entropy"]["drift_status"], ["MODERATE", "CRITICAL"])

if __name__ == "__main__":
    unittest.main()
