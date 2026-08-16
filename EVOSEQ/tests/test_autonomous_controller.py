import sys
import os
import unittest
import numpy as np

# Add EVOSEQ root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.evolution.performance import PerformanceMonitor, EWMA
from app.evolution.drift import calculate_multidimensional_drift, DriftState, DriftController, js_divergence
from app.evolution.uncertainty import calculate_prediction_uncertainty, calculate_model_disagreement
from app.evolution.mutation import mutate_model_parameters, compute_adaptive_exploration
from app.evolution.controller import EvolutionController, Action
from app.evolution.orchestrator import autonomous_evolution_cycle
from app.ingestion.stream import ingest_outcomes_batch

class TestAutonomousController(unittest.TestCase):

    def test_performance_monitor_and_ewma(self):
        ewma = EWMA(alpha=0.1)
        for _ in range(10):
            ewma.update(1.0)
        self.assertGreater(ewma.value, 0.5)
        
        monitor = PerformanceMonitor(window_size=100)
        for i in range(80):
            monitor.update(correct=(i % 2 == 0), log_loss=2.302, brier=0.09)
            
        snap = monitor.snapshot()
        self.assertEqual(snap.sample_count, 80)
        self.assertAlmostEqual(snap.accuracy, 0.5, places=2)
        
        deltas = monitor.compute_multi_horizon_deltas()
        self.assertIn("delta_50", deltas)
        self.assertIn("delta_250", deltas)

    def test_multidimensional_drift(self):
        # Stable repeating sequence
        stable_seq = [1, 2, 3, 4, 5] * 200
        res_stable = calculate_multidimensional_drift(stable_seq, recent_window=50, historical_window=200)
        self.assertEqual(res_stable.state, DriftState.STABLE)
        self.assertFalse(res_stable.is_significant)
        
        # Shifted sequence
        shifted_seq = ([1, 2] * 200) + ([8, 9] * 100)
        res_shifted = calculate_multidimensional_drift(shifted_seq, recent_window=80, historical_window=200)
        self.assertIn(res_shifted.state, [DriftState.WARNING, DriftState.EVOLVE])
        self.assertGreater(res_shifted.composite_drift, 0.05)
        self.assertIn("digit", res_shifted.dimension_drifts)

    def test_uncertainty_and_disagreement(self):
        p_uniform = np.full(10, 0.1)
        p_sharp = np.array([0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0])
        
        u_high = calculate_prediction_uncertainty(p_uniform)
        u_low = calculate_prediction_uncertainty(p_sharp)
        self.assertGreater(u_high, u_low)
        
        disagreement = calculate_model_disagreement([p_uniform, p_sharp])
        self.assertGreater(disagreement, 0.0)

    def test_mutation_engine(self):
        esn_params = {"reservoir_size": 64, "spectral_radius": 0.9, "leak_rate": 0.3, "ridge": 1e-4}
        mutated = mutate_model_parameters("EchoStateNetwork", esn_params, mutation_rate=1.0)
        self.assertIn(mutated["reservoir_size"], [32, 64, 128, 256, 512])
        
        exp_factor = compute_adaptive_exploration(uncertainty=3.32, drift_score=0.15, recent_improvement=-0.05)
        self.assertGreater(exp_factor, 0.20)

    def test_evolution_controller_policy(self):
        controller = EvolutionController(min_observations=50)
        
        # 1. Under observation count -> WAIT
        self.assertEqual(controller.decide(observations=20, drift_score=0.01, performance_delta=0.0, uncertainty=2.0, model_disagreement=0.01), Action.WAIT)
        
        # 2. High drift -> EVOLVE
        self.assertEqual(controller.decide(observations=100, drift_score=0.15, performance_delta=0.0, uncertainty=2.0, model_disagreement=0.01), Action.EVOLVE)
        
        # 3. Deteriorating performance -> ADAPT
        self.assertEqual(controller.decide(observations=100, drift_score=0.02, performance_delta=-0.06, uncertainty=2.0, model_disagreement=0.01), Action.ADAPT)
        
        # 4. High model disagreement -> INVESTIGATE
        self.assertEqual(controller.decide(observations=100, drift_score=0.02, performance_delta=0.0, uncertainty=2.0, model_disagreement=0.20), Action.INVESTIGATE)
        
        # 5. Normal healthy stream -> UPDATE
        self.assertEqual(controller.decide(observations=100, drift_score=0.01, performance_delta=0.01, uncertainty=2.0, model_disagreement=0.02), Action.UPDATE)

    def test_autonomous_evolution_cycle(self):
        synthetic_batch = []
        for i in range(100):
            synthetic_batch.append({
                "sequence_no": 980000 + i,
                "digit": i % 10
            })
        ingest_outcomes_batch(synthetic_batch)
        
        report = autonomous_evolution_cycle(last_seq_cursor=979999)
        self.assertIn("action", report)
        self.assertIn("drift", report)
        self.assertIn("performance", report)

if __name__ == "__main__":
    unittest.main()
