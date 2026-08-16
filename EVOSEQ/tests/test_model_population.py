import sys
import os
import unittest
import numpy as np

# Add EVOSEQ root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.base import ModelMetadata, SequenceModel
from app.models.uniform import UniformModel
from app.models.markov import MarkovModel
from app.models.hmm_model import DiscreteHMM, RegimeMonitor
from app.models.esn import EchoStateNetwork
from app.models.transformer import TransformerSequenceModel, temperature_scale
from app.models.ensemble import MetaEnsemble, ensemble_probabilities
from app.evolution.registry import ModelRegistry
from app.ingestion.stream import ingest_outcomes_batch
from app.evolution.orchestrator import daily_evolution

class TestModelPopulation(unittest.TestCase):

    def test_discrete_hmm(self):
        # Deterministic 2-state cycle: [0, 1, 0, 1, 0, 1]
        seq = [0, 1] * 30
        hmm = DiscreteHMM(n_states=2, smoothing=1e-2, version="hmm-test")
        hmm.fit(seq)
        
        probs = hmm.predict_proba([0])
        self.assertEqual(len(probs), 10)
        self.assertAlmostEqual(probs.sum(), 1.0)
        # After seeing 0, probability of 1 should be elevated
        self.assertGreater(probs[1], probs[0])
        
        # Test RegimeMonitor
        monitor = RegimeMonitor(hmm)
        state_probs = monitor.update([0, 1])
        self.assertIn("state_0_prob", state_probs)
        self.assertIn("state_1_prob", state_probs)

    def test_echo_state_network(self):
        seq = [1, 2, 3, 4, 1, 2, 3, 4] * 20
        esn = EchoStateNetwork(
            input_size=10,
            reservoir_size=32,
            spectral_radius=0.9,
            leak_rate=0.3,
            version="esn-test"
        )
        esn.fit(seq)
        
        probs = esn.predict_proba([1, 2, 3])
        self.assertEqual(len(probs), 10)
        self.assertAlmostEqual(probs.sum(), 1.0)
        # Target after 3 is 4
        self.assertEqual(int(np.argmax(probs)), 4)

    def test_transformer_model(self):
        seq = [0, 5, 0, 5, 0, 5, 0, 5] * 10
        trans = TransformerSequenceModel(
            input_size=10,
            hidden_size=16,
            heads=2,
            layers=1,
            temperature=1.0,
            version="transformer-test"
        )
        trans.fit(seq, epochs=5)
        
        probs = trans.predict_proba([0, 5, 0])
        self.assertEqual(len(probs), 10)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
        self.assertEqual(probs.ndim, 1)

    def test_meta_ensemble(self):
        p1 = np.array([0.7, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        p2 = np.array([0.1, 0.7, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        fused = ensemble_probabilities([p1, p2], scores=[10.0, 0.0])
        self.assertAlmostEqual(fused.sum(), 1.0)
        # Model 1 has much higher score, so digit 0 should dominate
        self.assertGreater(fused[0], 0.6)

    def test_model_genealogy(self):
        registry = ModelRegistry()
        cand_id = registry.register_candidate(
            model_name="ESN",
            version="esn-gen2-test",
            parameters={"reservoir": 128},
            parent_model_id=1,
            generation=2,
            mutation_details={"spectral_radius": 0.95}
        )
        self.assertIsInstance(cand_id, int)
        
        history = registry.get_genealogy_history(limit=5)
        self.assertGreater(len(history), 0)
        self.assertIn("mutation", history[0])

    def test_full_population_evolution_cycle(self):
        synthetic_batch = []
        for i in range(80):
            synthetic_batch.append({
                "sequence_no": 950000 + i,
                "digit": (i % 5) * 2
            })
        ingest_outcomes_batch(synthetic_batch)
        
        res = daily_evolution(last_seq_cursor=949999)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("champion", res)

if __name__ == "__main__":
    unittest.main()
