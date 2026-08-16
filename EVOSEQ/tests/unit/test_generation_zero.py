import sys
import os
import unittest
import numpy as np

# Add EVOSEQ root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core.randomness import seed_everything
from app.research.audit.data_quality import audit_dataframe
from app.features.builder import digit_features, build_feature_matrix, create_supervised_dataset
from app.models.uniform import UniformBaseline
from app.models.frequency import FrequencyBaseline
from app.models.markov import MarkovModel
from app.models.hmm import HMMModel
from app.models.esn import ESN
from app.research.evaluator import evaluate, ModelEvaluation, rank_models
from app.features.online_histogram import OnlineHistogram, entropy
from scripts.run_generation_zero import run_generation_zero

class TestGenerationZero(unittest.TestCase):

    def test_deterministic_seeding(self):
        seed_everything(42)
        r1 = np.random.rand(5)
        seed_everything(42)
        r2 = np.random.rand(5)
        np.testing.assert_allclose(r1, r2)

    def test_audit_dataframe(self):
        # Valid dataset
        valid_df = [
            {"sequence_no": 1, "digit": 3},
            {"sequence_no": 2, "digit": 7},
            {"sequence_no": 3, "digit": 0}
        ]
        errors = audit_dataframe(valid_df)
        self.assertEqual(len(errors), 0)
        
        # Invalid dataset with gaps and duplicates
        bad_df = [
            {"sequence_no": 1, "digit": 3},
            {"sequence_no": 1, "digit": 12}, # dup & invalid digit
            {"sequence_no": 5, "digit": 4}   # gap
        ]
        bad_errors = audit_dataframe(bad_df)
        self.assertGreater(len(bad_errors), 0)

    def test_feature_matrix_and_supervised_dataset(self):
        rows = [
            {"digit": 3, "size": 0, "color": 0, "parity": 1},
            {"digit": 8, "size": 1, "color": 1, "parity": 0},
            {"digit": 0, "size": 0, "color": 2, "parity": 0}
        ]
        mat = build_feature_matrix(rows)
        self.assertEqual(mat.shape, (3, 17))
        
        # Supervised dataset
        feat_stream = [digit_features(i % 10, (i % 10) >= 5, 0, (i % 10) % 2) for i in range(20)]
        digits = [i % 10 for i in range(20)]
        X, y = create_supervised_dataset(feat_stream, digits, context_length=5)
        self.assertEqual(X.shape, (15, 5, 17))
        self.assertEqual(y.shape, (15,))

    def test_generation_zero_models(self):
        seq = [1, 2, 3, 4, 1, 2, 3, 4] * 4
        
        # 1. Uniform
        u = UniformBaseline()
        p_u = u.predict_proba(5)
        self.assertEqual(p_u.shape, (5, 10))
        self.assertAlmostEqual(float(p_u[0, 0]), 0.1, places=5)
        
        # 2. Frequency
        f = FrequencyBaseline()
        f.fit(seq)
        p_f = f.predict_proba(5)
        self.assertEqual(p_f.shape, (5, 10))
        
        # 3. Markov
        m1 = MarkovModel(order=1)
        m1.fit(seq)
        p_m1 = m1.predict_sequence(seq[:5])
        self.assertEqual(p_m1.shape, (5, 10))
        
        # 4. HMM
        hmm = HMMModel(states=2)
        hmm.fit(seq)
        p_hmm = hmm.predict_proba(seq[:5])
        self.assertEqual(p_hmm.shape, (10,))
        
        # 5. ESN
        esn = ESN(input_size=10, reservoir_size=16)
        esn.fit(seq[:10])
        p_esn = esn.predict_proba(seq[:5])
        self.assertEqual(len(p_esn), 10)

    def test_evaluator_and_model_ranking(self):
        m1 = ModelEvaluation("M1", mean_log_loss=2.28, std_log_loss=0.01, mean_accuracy=0.12, mean_brier=0.088, calibration_error=0.01, robustness=0.9, stability_score=0.99)
        m2 = ModelEvaluation("M2", mean_log_loss=2.30, std_log_loss=0.02, mean_accuracy=0.10, mean_brier=0.090, calibration_error=0.02, robustness=0.8, stability_score=0.98)
        
        ranked = rank_models([m2, m1])
        self.assertEqual(ranked[0].model_name, "M1")

    def test_online_histogram_and_entropy(self):
        hist = OnlineHistogram(num_classes=10)
        for d in [0, 0, 0, 5, 5, 9]:
            hist.update(d)
        probs = hist.probabilities()
        self.assertEqual(len(probs), 10)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)
        
        h = entropy(probs)
        self.assertGreater(h, 0.0)
        self.assertLess(h, 3.33)

    def test_generation_zero_script_execution(self):
        report = run_generation_zero()
        self.assertIn("EVOSEQ — GENERATION 0 BASELINE", report)
        self.assertIn("BASELINE ESTABLISHED", report)

if __name__ == "__main__":
    unittest.main()
