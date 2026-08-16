import sys
import os
import unittest
import numpy as np

# Add EVOSEQ root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.features.basic import digit_distribution, digit_entropy
from app.features.entropy import shannon_entropy
from app.features.conditional_entropy import conditional_entropy
from app.features.ngrams import ngram_counts, normalized_ngram_counts
from app.features.runs import run_lengths, run_statistics
from app.models.uniform import UniformModel
from app.models.markov import MarkovModel
from app.evaluation.metrics import log_loss, brier_score, calibration_error, calculate_null_advantage
from app.evaluation.walk_forward import walk_forward, evaluate_model_walk_forward
from app.evolution.drift import calculate_drift
from app.evolution.registry import ModelRegistry
from app.ingestion.stream import ingest_outcomes_batch, get_latest_sequence
from app.evolution.orchestrator import daily_evolution

class TestEvoseqCore(unittest.TestCase):

    def test_features(self):
        sequence = [1, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        dist = digit_distribution(sequence)
        self.assertEqual(len(dist), 10)
        self.assertAlmostEqual(dist.sum(), 1.0)
        
        ent = digit_entropy(sequence)
        self.assertGreater(ent, 0.0)
        
        ng = ngram_counts(sequence, 2)
        self.assertIn((1, 1), ng)
        self.assertIn((2, 2), ng)
        
        runs = run_lengths(sequence)
        self.assertEqual(runs[0], (1, 2)) # two 1s
        self.assertEqual(runs[1], (2, 3)) # three 2s
        
        stat = run_statistics(sequence)
        self.assertEqual(stat["max_run"], 3.0)

    def test_models_and_metrics(self):
        train_seq = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3] * 10
        uniform = UniformModel()
        probs_u = uniform.predict_proba([1, 2])
        self.assertEqual(len(probs_u), 10)
        self.assertAlmostEqual(probs_u[0], 0.1)
        
        markov = MarkovModel(order=2, smoothing=0.1)
        markov.fit(train_seq)
        probs_m = markov.predict_proba([1, 2])
        # After [1, 2], digit 3 should have highest probability
        self.assertEqual(int(np.argmax(probs_m)), 3)
        self.assertGreater(probs_m[3], 0.8)
        
        ll = log_loss(probs_m, 3)
        brier = brier_score(probs_m, 3)
        self.assertLess(ll, 0.5)
        self.assertLess(brier, 0.1)

    def test_walk_forward(self):
        seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 30
        model = MarkovModel(order=1, smoothing=0.5)
        results = walk_forward(model, seq, initial_train_size=50)
        self.assertEqual(len(results), len(seq) - 50)
        
        summary = evaluate_model_walk_forward(model, seq, initial_train_size=50)
        self.assertIn("accuracy", summary)
        self.assertIn("mean_brier_score", summary)
        self.assertIn("null_advantage", summary)

    def test_drift_detection(self):
        stable_seq = [1, 2, 3, 4, 5] * 200
        drift_res = calculate_drift(stable_seq, recent_window=50, historical_window=200)
        self.assertFalse(drift_res.is_significant)
        
        shifted_seq = ([1, 2] * 200) + ([8, 9] * 100)
        drift_shifted = calculate_drift(shifted_seq, recent_window=100, historical_window=200)
        self.assertTrue(drift_shifted.is_significant)
        self.assertIn(drift_shifted.level, ["MODERATE", "CRITICAL"])

    def test_end_to_end_orchestrator(self):
        synthetic_batch = []
        for i in range(120):
            synthetic_batch.append({
                "sequence_no": 900000 + i,
                "digit": i % 10
            })
        ingested = ingest_outcomes_batch(synthetic_batch)
        self.assertGreaterEqual(ingested, 0)
        
        res = daily_evolution(last_seq_cursor=899999)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("champion", res)

if __name__ == "__main__":
    unittest.main()
