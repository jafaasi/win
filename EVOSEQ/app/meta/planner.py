from typing import List, Dict, Any, Tuple
import numpy as np
from .types import EnvironmentState, ModelDescriptor, ParetoPoint
from .meta_model import MetaModel

class ExperimentPlanner:
    """
    Pareto & Bayesian Acquisition Experiment Planner:
    Allocates experiment budget by ranking candidates according to Upper Confidence Bound (UCB)
    and filtering along the multi-objective Pareto efficiency frontier.
    """

    def __init__(self, meta_model: MetaModel):
        self.meta_model = meta_model

    def prioritize_by_ucb(
        self,
        candidates: List[Dict[str, Any]],
        env: EnvironmentState,
        budget: int = 10,
        kappa: float = 1.96
    ) -> List[Dict[str, Any]]:
        """Ranks candidate architectures by Expected Value + Information Value (UCB)."""
        scored_cands = []
        for c in candidates:
            desc = c.get("descriptor")
            if isinstance(desc, dict):
                desc = ModelDescriptor(**desc)
            elif not isinstance(desc, ModelDescriptor):
                # Fallback descriptor from model metadata
                mod = c.get("model_instance")
                fam = c.get("model_name", "Markov")
                ctx = getattr(mod, "context_length", getattr(mod, "order", 32))
                params = getattr(mod, "hidden_size", 32)
                desc = ModelDescriptor(family=fam, context_length=ctx, parameter_count=params)
                
            mu, sigma, ucb = self.meta_model.predict_ucb(env, desc, kappa=kappa)
            c_copy = dict(c)
            c_copy["meta_mu"] = mu
            c_copy["meta_sigma"] = sigma
            c_copy["meta_ucb"] = ucb
            scored_cands.append(c_copy)
            
        ranked = sorted(scored_cands, key=lambda x: x["meta_ucb"], reverse=True)
        return ranked[:budget]

    @staticmethod
    def compute_pareto_frontier(points: List[ParetoPoint]) -> List[ParetoPoint]:
        """
        Filters non-dominated points along 5 objectives:
        - min LogLoss
        - min CalibrationError
        - min Complexity (log params)
        - min Latency
        - max Robustness (higher is better)
        """
        frontier = []
        for p1 in points:
            dominated = False
            for p2 in points:
                if p1.candidate_id == p2.candidate_id:
                    continue
                # p2 dominates p1 if p2 is >= p1 in all criteria and strictly > in at least one
                is_better_or_equal = (
                    p2.log_loss <= p1.log_loss and
                    p2.calibration_error <= p1.calibration_error and
                    p2.complexity <= p1.complexity and
                    p2.latency <= p1.latency and
                    p2.robustness >= p1.robustness
                )
                strictly_better = (
                    p2.log_loss < p1.log_loss or
                    p2.calibration_error < p1.calibration_error or
                    p2.complexity < p1.complexity or
                    p2.latency < p1.latency or
                    p2.robustness > p1.robustness
                )
                if is_better_or_equal and strictly_better:
                    dominated = True
                    break
            if not dominated:
                frontier.append(p1)
        return frontier
