from typing import List, Dict, Any, Optional
import numpy as np
from .types import EnvironmentState, ModelDescriptor, ParetoPoint
from .meta_model import MetaModel
from .planner import ExperimentPlanner
from .questions import ResearchQuestionManager
from .knowledge_graph import ModelKnowledgeGraph
from ..features.entropy import categorical_entropy
from ..features.conditional_entropy import conditional_entropy
from ..features.information import information_gain
from ..features.autocorrelation import autocorrelation
from ..features.lz import lz_complexity
from ..database import SessionLocal
from ..schemas import MetaExperimentRecord

class ResearchDirector:
    """
    High-Level Scientific Director:
    Orchestrates statistical environment analysis, Bayesian experiment planning,
    Pareto architecture filtering, and institutional hypothesis memory.
    """

    def __init__(self):
        self.meta_model = MetaModel()
        self.planner = ExperimentPlanner(self.meta_model)
        self.questions = ResearchQuestionManager()
        self.knowledge_graph = ModelKnowledgeGraph()

    def analyze_environment(
        self,
        digits: List[int],
        drift_score: float = 0.0,
        disagreement: float = 0.0,
        regime_entropy: float = 1.0
    ) -> EnvironmentState:
        """Extracts the 12-dimensional canonical EnvironmentState."""
        h_marg = categorical_entropy(digits)
        h_cond1 = conditional_entropy(digits, order=1)
        h_cond2 = conditional_entropy(digits, order=2)
        ig1 = information_gain(digits, order=1)
        ig2 = information_gain(digits, order=2)
        acf1 = autocorrelation(digits, lag=1)
        acf2 = autocorrelation(digits, lag=2)
        acf3 = autocorrelation(digits, lag=3)
        lz_z = float(lz_complexity(digits) / max(1.0, len(digits))) # normalized ratio
        
        return EnvironmentState(
            entropy=round(h_marg, 4),
            conditional_entropy_1=round(h_cond1, 4),
            conditional_entropy_2=round(h_cond2, 4),
            information_gain_1=round(ig1, 4),
            information_gain_2=round(ig2, 4),
            autocorrelation_1=round(acf1, 4),
            autocorrelation_2=round(acf2, 4),
            autocorrelation_3=round(acf3, 4),
            lz_zscore=round(lz_z, 4),
            drift_score=round(drift_score, 4),
            model_disagreement=round(disagreement, 4),
            regime_entropy=round(regime_entropy, 4)
        )

    def plan_candidate_evaluation(
        self,
        candidates: List[Dict[str, Any]],
        env: EnvironmentState,
        budget: int = 8
    ) -> List[Dict[str, Any]]:
        """Updates meta-model and uses Bayesian UCB to select top experiment budget."""
        self.meta_model.fit_from_database()
        return self.planner.prioritize_by_ucb(candidates, env, budget=budget)

    def record_meta_experiment(
        self,
        model_version_id: int,
        env: EnvironmentState,
        desc: ModelDescriptor,
        eval_metrics: Dict[str, Any]
    ) -> None:
        """Persists meta-experiment observation to database."""
        with SessionLocal() as session:
            rec = MetaExperimentRecord(
                model_version_id=model_version_id,
                environment=env.to_dict(),
                model_descriptor=desc.to_dict(),
                log_loss=eval_metrics.get("mean_log_loss"),
                brier_score=eval_metrics.get("mean_brier_score"),
                calibration_error=eval_metrics.get("calibration_error"),
                null_advantage=eval_metrics.get("null_advantage"),
                inference_latency=eval_metrics.get("inference_latency", 0.001)
            )
            session.add(rec)
            session.commit()
