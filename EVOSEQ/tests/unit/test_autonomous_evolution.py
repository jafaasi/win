import sys
import os
import unittest
import numpy as np

# Add EVOSEQ root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.evolution import (
    ResearchHypothesis,
    generate_hypotheses,
    CandidateFactory,
    SEARCH_SPACE,
    mutate_single_variable,
    cost_aware_objective,
    FeatureAblationTester,
    ResearchBudgetController,
    experiment_priority,
    TemporalValidationLab,
    SequentialPromotionGate,
    PromotionStage,
    ArchitectureSurvivalAnalyzer,
    AutonomousResearchDirector
)
from app.database import SessionLocal, engine, Base
from app.schemas import ResearchHypothesisRecord, ModelCandidateRecord, ExperimentResultRecord
from scripts.run_autonomous_evolution import run_autonomous_evolution

class TestAutonomousEvolution(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)

    def test_hypothesis_generation(self):
        class MockEnv:
            drift_score = 0.08
            model_disagreement = 0.25
            entropy = 3.31
            
        hypotheses = generate_hypotheses(MockEnv(), budget_limit=4)
        self.assertGreater(len(hypotheses), 0)
        categories = [h.category for h in hypotheses]
        self.assertIn("context", categories)
        self.assertIn("null_referee", categories)

    def test_candidate_factory_and_mutation(self):
        hyp = ResearchHypothesis(
            id="H-TEST",
            category="architecture",
            parent_model="mamba",
            description="Test larger state",
            configuration={"d_model": 64, "d_state": 32},
            expected_effect="expand capacity",
            priority=0.8
        )
        factory = CandidateFactory()
        cand = factory.instantiate_candidate(hyp, generation=2)
        self.assertEqual(cand["family"], "mamba")
        self.assertEqual(cand["configuration"]["d_model"], 64)
        
        mutated = mutate_single_variable(cand["configuration"], "d_model", 128)
        self.assertEqual(mutated["d_model"], 128)
        self.assertEqual(cand["configuration"]["d_model"], 64) # Immutability preserved

    def test_cost_aware_objective(self):
        score_light = cost_aware_objective(log_loss=2.30, latency_ms=0.2, param_count=5000)
        score_heavy = cost_aware_objective(log_loss=2.30, latency_ms=5.0, param_count=500000)
        self.assertGreater(score_light, score_heavy)

    def test_feature_ablation(self):
        tester = FeatureAblationTester()
        full_mat = np.random.randn(20, 10, 17)
        targets = np.random.randint(0, 10, 20)
        
        def dummy_eval(mat):
            return np.full((len(mat), 10), 0.1)
            
        res = tester.ablate_features(full_mat, targets, dummy_eval)
        self.assertIn("size", res)
        self.assertIn("color", res)
        self.assertIn("parity", res)

    def test_budget_controller_and_bayesian_update(self):
        controller = ResearchBudgetController({"architecture": 2, "null_tests": 1})
        self.assertTrue(controller.allocate("architecture", 1))
        self.assertTrue(controller.allocate("architecture", 1))
        self.assertFalse(controller.allocate("architecture", 1)) # Exceeded budget
        
        prior = controller.beliefs["architecture"]
        new_belief = controller.update_belief("architecture", success=True)
        self.assertGreater(new_belief, prior)

    def test_temporal_validation_lab(self):
        lab = TemporalValidationLab(seeds=[1, 7, 42])
        train_data = np.random.randint(0, 10, 100)
        test_data = np.random.randint(0, 10, 30)
        
        from app.models.markov import MarkovModel
        builder = lambda seed: MarkovModel(order=1)
        
        metrics = lab.evaluate_multi_seed(builder, train_data, test_data)
        self.assertIn("mean_loss", metrics)
        self.assertIn("std_loss", metrics)
        
        ci_res = lab.bootstrap_paired_delta([2.20]*30, [2.30]*30)
        self.assertTrue(ci_res["significant_advantage"])
        
        null_res = lab.null_significance_test(lambda seq: np.full((len(seq), 10), 0.1), test_data, n_surrogates=10)
        self.assertIn("null_p_value", null_res)

    def test_sequential_promotion_gate(self):
        gate = SequentialPromotionGate(min_improvement_delta=0.01, max_null_p_value=0.10)
        
        # Test rejection on high p-value
        res_fail = gate.evaluate_promotion(
            candidate_metrics={"mean_loss": 2.25, "std_loss": 0.01},
            champion_metrics={"mean_loss": 2.30},
            null_test_result={"null_p_value": 0.50}
        )
        self.assertFalse(res_fail["promoted"])
        
        # Test promotion
        res_pass = gate.evaluate_promotion(
            candidate_metrics={"mean_loss": 2.25, "std_loss": 0.01},
            champion_metrics={"mean_loss": 2.30},
            null_test_result={"null_p_value": 0.04}
        )
        self.assertTrue(res_pass["promoted"])
        self.assertEqual(res_pass["stage"], PromotionStage.CHAMPION)

    def test_genealogy_survival_analysis(self):
        analyzer = ArchitectureSurvivalAnalyzer()
        analyzer.register_candidate("M1", generation=1, family="mamba", parent_code=None, log_loss=2.30, promoted=True)
        analyzer.register_candidate("M2", generation=1, family="mamba", parent_code="M1", log_loss=2.35, promoted=False)
        analyzer.register_candidate("T1", generation=1, family="transformer", parent_code=None, log_loss=2.32, promoted=False)
        
        rates = analyzer.get_survival_rates()
        self.assertEqual(rates["mamba"], 0.5)
        self.assertEqual(rates["transformer"], 0.0)

    def test_autonomous_research_director_cycle(self):
        class MockEnv:
            drift_score = 0.05
            model_disagreement = 0.20
            entropy = 3.28
            
        director = AutonomousResearchDirector()
        train_data = np.random.randint(0, 10, 100)
        test_data = np.random.randint(0, 10, 30)
        
        result = director.run_research_cycle(
            environment_state=MockEnv(),
            data_train=train_data,
            data_test=test_data,
            champion_metrics={"mean_loss": 2.3026, "std_loss": 0.005},
            generation=2
        )
        self.assertIn("candidates_evaluated", result)
        self.assertIn("survival_rates", result)

    def test_run_autonomous_evolution_script(self):
        report = run_autonomous_evolution()
        self.assertIn("EVOSEQ — AUTONOMOUS EVOLUTION & META-SEARCH REPORT", report)
        self.assertIn("CANDIDATES & NULL REFEREE AUDIT", report)

if __name__ == "__main__":
    unittest.main()
