import sys
import os
import unittest
from fastapi.testclient import TestClient

# Add EVOSEQ root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.meta.types import EnvironmentState, ModelDescriptor, ParetoPoint
from app.meta.meta_model import MetaModel
from app.meta.planner import ExperimentPlanner
from app.meta.questions import ResearchQuestionManager
from app.meta.knowledge_graph import ModelKnowledgeGraph
from app.meta.director import ResearchDirector
from app.api import app
from app.database import SessionLocal
from app.schemas import MetaExperimentRecord

class TestMetaLearning(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_environment_and_model_descriptors(self):
        env = EnvironmentState(
            entropy=3.32,
            conditional_entropy_1=3.20,
            conditional_entropy_2=3.10,
            information_gain_1=0.12,
            information_gain_2=0.22,
            autocorrelation_1=0.01,
            autocorrelation_2=-0.02,
            autocorrelation_3=0.00,
            lz_zscore=0.15,
            drift_score=0.02,
            model_disagreement=0.25,
            regime_entropy=1.2
        )
        vec = env.to_vector()
        self.assertEqual(len(vec), 12)
        
        desc = ModelDescriptor(family="Mamba", context_length=128, parameter_count=15000, hidden_size=48, layers=2)
        d_vec = desc.to_vector()
        self.assertEqual(len(d_vec), 8)

    def test_meta_model_ucb_prediction(self):
        meta_model = MetaModel()
        env = EnvironmentState(3.32, 3.20, 3.10, 0.12, 0.22, 0.01, -0.02, 0.0, 0.15, 0.02, 0.25, 1.2)
        desc = ModelDescriptor("Transformer", 64, 5000)
        
        mu, sigma, ucb = meta_model.predict_ucb(env, desc, kappa=2.0)
        self.assertGreaterEqual(ucb, mu)

    def test_pareto_frontier_filter(self):
        # Point A: lower log_loss, slightly higher complexity
        pA = ParetoPoint("pA", ModelDescriptor("Markov", 32, 100), log_loss=2.20, brier_score=0.08, calibration_error=0.02, complexity=4.0, latency=0.001, robustness=0.10)
        # Point B: higher log_loss, worse in everything -> dominated by A
        pB = ParetoPoint("pB", ModelDescriptor("Markov", 32, 200), log_loss=2.30, brier_score=0.09, calibration_error=0.03, complexity=5.0, latency=0.002, robustness=0.05)
        # Point C: very low complexity, higher log loss -> non-dominated
        pC = ParetoPoint("pC", ModelDescriptor("Uniform", 0, 0), log_loss=2.30, brier_score=0.09, calibration_error=0.01, complexity=0.0, latency=0.0001, robustness=0.00)
        
        frontier = ExperimentPlanner.compute_pareto_frontier([pA, pB, pC])
        f_ids = [p.candidate_id for p in frontier]
        self.assertIn("pA", f_ids)
        self.assertIn("pC", f_ids)
        self.assertNotIn("pB", f_ids)

    def test_hypothesis_lifecycle(self):
        qm = ResearchQuestionManager()
        hypo_id = qm.register_hypothesis(
            question_code="RQ-001",
            title="Context L=256 vs L=64",
            description="Testing whether context 256 improves out-of-sample log loss"
        )
        self.assertIsInstance(hypo_id, int)
        
        # Update evidence with high significance
        status = qm.update_evidence(hypo_id, p_value=0.002, delta_null=0.08, summary={"trials": 50})
        self.assertEqual(status, "SUPPORTED")
        
        agenda = qm.get_agenda()
        self.assertGreaterEqual(len(agenda), 4)

    def test_research_director_and_meta_recording(self):
        director = ResearchDirector()
        digits = [1, 2, 3, 4, 1, 2, 3, 4] * 10
        env = director.analyze_environment(digits, drift_score=0.01, disagreement=0.15)
        self.assertEqual(env.entropy, round(env.entropy, 4))
        
        # Test candidate prioritization
        cands = [
            {"model_name": "Markov", "parameters": {"order": 2}},
            {"model_name": "Transformer", "parameters": {"context_length": 64}}
        ]
        planned = director.plan_candidate_evaluation(cands, env, budget=2)
        self.assertEqual(len(planned), 2)
        self.assertIn("meta_ucb", planned[0])
        
        # Record meta experiment
        director.record_meta_experiment(
            model_version_id=1,
            env=env,
            desc=ModelDescriptor("Markov", 2, 100),
            eval_metrics={"mean_log_loss": 2.25, "mean_brier_score": 0.085, "null_advantage": 0.05}
        )
        
        with SessionLocal() as session:
            saved = session.query(MetaExperimentRecord).first()
            self.assertIsNotNone(saved)

    def test_meta_fastapi_endpoints(self):
        res_q = self.client.get("/meta/questions")
        self.assertEqual(res_q.status_code, 200)
        self.assertIsInstance(res_q.json(), list)
        
        res_ins = self.client.get("/meta/insights")
        self.assertEqual(res_ins.status_code, 200)
        self.assertIn("robust_families_under_drift", res_ins.json())

if __name__ == "__main__":
    unittest.main()
