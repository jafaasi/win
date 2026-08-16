import sys
import os
import unittest
from datetime import datetime
import numpy as np

# Add EVOSEQ root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.mapping import map_digit
from app.repositories.outcomes import find_gaps, validate_sequence
from app.features.vector import one_hot, encode_observation, make_windows
from app.research.walk_forward import Fold, create_folds
from app.research.metrics import log_loss, accuracy, brier_score
from app.models.frequency import FrequencyBaseline
from app.evolution.promotion_gate import promotion_gate
from app.schemas import Outcome, ModelScoreRecord
from app.database import SessionLocal

class TestWorkingCore(unittest.TestCase):

    def test_deterministic_mapping(self):
        # Digit 0 -> Violet, Small, Even
        m0 = map_digit(0)
        self.assertEqual(m0, {"digit": 0, "size": 0, "color": 2, "parity": 0})
        
        # Digit 1 -> Green, Small, Odd
        m1 = map_digit(1)
        self.assertEqual(m1, {"digit": 1, "size": 0, "color": 0, "parity": 1})
        
        # Digit 2 -> Red, Small, Even
        m2 = map_digit(2)
        self.assertEqual(m2, {"digit": 2, "size": 0, "color": 1, "parity": 0})
        
        # Digit 5 -> Violet, Big, Odd
        m5 = map_digit(5)
        self.assertEqual(m5, {"digit": 5, "size": 1, "color": 2, "parity": 1})
        
        # Digit 8 -> Red, Big, Even
        m8 = map_digit(8)
        self.assertEqual(m8, {"digit": 8, "size": 1, "color": 1, "parity": 0})
        
        with self.assertRaises(ValueError):
            map_digit(10)

    def test_find_gaps_and_sequence_validation(self):
        seq = [1001, 1002, 1003, 1005, 1006, 1010]
        gaps = find_gaps(seq)
        self.assertEqual(gaps, [(1004, 1004), (1007, 1009)])
        
        now = datetime.now()
        outcomes_valid = [
            Outcome(sequence_no=1, timestamp_utc=now, digit=3, size=0, color=0, parity=1),
            Outcome(sequence_no=2, timestamp_utc=now, digit=7, size=1, color=0, parity=1)
        ]
        validate_sequence(outcomes_valid) # must pass cleanly
        
        outcomes_duplicate = [
            Outcome(sequence_no=1, timestamp_utc=now, digit=3, size=0, color=0, parity=1),
            Outcome(sequence_no=1, timestamp_utc=now, digit=7, size=1, color=0, parity=1)
        ]
        with self.assertRaises(ValueError):
            validate_sequence(outcomes_duplicate)

    def test_17_dim_feature_vector_and_causal_windows(self):
        vec = encode_observation(digit=7, size=1, color=0, parity=1)
        self.assertEqual(len(vec), 17)
        self.assertEqual(vec[7], 1.0)  # digit 7
        self.assertEqual(vec[11], 1.0) # size 1 (10 + 1)
        self.assertEqual(vec[12], 1.0) # color 0 (10 + 2 + 0)
        self.assertEqual(vec[16], 1.0) # parity 1 (10 + 2 + 3 + 1)
        
        feat_mat = [encode_observation(i % 10, (i % 10) >= 5, 0, (i % 10) % 2) for i in range(25)]
        digits = [i % 10 for i in range(25)]
        
        X, y = make_windows(feat_mat, digits, context_length=8)
        self.assertEqual(X.shape, (17, 8, 17))
        self.assertEqual(y.shape, (17,))
        self.assertEqual(y[0], digits[8])

    def test_walk_forward_folds(self):
        folds = create_folds(n=1000, initial_train=500, validation_size=100, test_size=50, step=100)
        self.assertGreater(len(folds), 3)
        self.assertEqual(folds[0].train_end, 500)
        self.assertEqual(folds[0].validation_end, 600)
        self.assertEqual(folds[0].test_end, 650)
        self.assertEqual(folds[1].train_end, 600)

    def test_research_metrics(self):
        probs = np.array([[0.9] + [0.0111]*9, [0.0111]*9 + [0.9]])
        targets = np.array([0, 9])
        
        acc = accuracy(probs, targets)
        self.assertEqual(acc, 1.0)
        
        ll = log_loss(probs, targets)
        self.assertLess(ll, 0.20)
        
        bs = brier_score(probs, targets)
        self.assertLess(bs, 0.05)

    def test_frequency_baseline_model(self):
        freq = FrequencyBaseline()
        # Biased distribution: mostly 7s
        seq = [7, 7, 7, 7, 3, 2, 7, 7, 7, 1]
        freq.fit(seq)
        
        p = freq.predict_proba()
        self.assertEqual(len(p), 10)
        self.assertAlmostEqual(float(np.sum(p)), 1.0, places=5)
        self.assertGreater(p[7], p[3])
        self.assertGreater(p[7], 0.5)

    def test_promotion_gate(self):
        champ = {"temporal_score": -0.35, "calibration_error": 0.02}
        
        # Valid challenger: better score, good calibration, positive null advantage
        good_challenger = {
            "temporal_score": -0.30,
            "calibration_error": 0.022,
            "null_advantage": 0.05,
            "robustness_score": 0.8
        }
        self.assertTrue(promotion_gate(good_challenger, champ, min_robustness=0.5))
        
        # Bad challenger: worse score
        bad_score_challenger = dict(good_challenger, temporal_score=-0.40)
        self.assertFalse(promotion_gate(bad_score_challenger, champ))
        
        # Bad challenger: poor calibration
        bad_calib_challenger = dict(good_challenger, calibration_error=0.08)
        self.assertFalse(promotion_gate(bad_calib_challenger, champ))
        
        # Bad challenger: negative null advantage
        bad_null_challenger = dict(good_challenger, null_advantage=-0.01)
        self.assertFalse(promotion_gate(bad_null_challenger, champ))

    def test_model_score_persistence(self):
        with SessionLocal() as session:
            rec = ModelScoreRecord(
                model_version_id=1,
                fold_id=1,
                horizon=1,
                accuracy=0.12,
                log_loss=2.29,
                brier_score=0.088,
                calibration_error=0.014
            )
            session.add(rec)
            session.commit()
            
            saved = session.query(ModelScoreRecord).filter(ModelScoreRecord.fold_id == 1).first()
            self.assertIsNotNone(saved)
            self.assertEqual(saved.horizon, 1)

if __name__ == "__main__":
    unittest.main()
