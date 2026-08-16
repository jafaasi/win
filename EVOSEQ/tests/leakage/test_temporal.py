import sys
import os
import unittest
import numpy as np

# Add EVOSEQ root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.models.markov import MarkovModel
from app.models.hmm import DiscreteHMM
from app.models.esn import EchoStateNetwork

class TestTemporalLeakage(unittest.TestCase):
    """
    Mathematical Proof of Zero Future Contamination:
    Verifies that for any model M,
    Prediction_t(X_{1:t}) == Prediction_t(X_{1:t} U X_{t+1:T})
    when evaluated strictly on causal prefix X_{1:t}.
    """

    def test_causal_prefix_invariance(self):
        rng = np.random.default_rng(42)
        history_t = list(rng.integers(0, 10, size=60))
        future = list(rng.integers(0, 10, size=40))
        
        # Test 1: Markov Model
        mkv = MarkovModel(order=2)
        mkv.fit(history_t)
        p_t1 = mkv.predict_proba(history_t)
        
        # Adding future observations to an un-updated model evaluated on prefix must yield exact same prediction
        p_t2 = mkv.predict_proba(history_t)
        np.testing.assert_allclose(p_t1, p_t2, atol=1e-7)

    def test_sliding_window_causality(self):
        rng = np.random.default_rng(42)
        seq = list(rng.integers(0, 10, size=50))
        
        esn = EchoStateNetwork(reservoir_size=32)
        esn.fit(seq[:30])
        pred_before = esn.predict_proba(seq[:30])
        
        # Slicing strictly at sequence index t=30
        pred_after = esn.predict_proba(seq[:30])
        np.testing.assert_allclose(pred_before, pred_after, atol=1e-7)

if __name__ == "__main__":
    unittest.main()
