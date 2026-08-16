from typing import Dict, Any, List, Optional
import numpy as np

DAILY_BUDGET = {
    "architecture": 10,
    "feature": 5,
    "ensemble": 5,
    "null_tests": 5,
    "robustness": 5
}

def experiment_priority(
    expected_gain: float,
    uncertainty: float = 0.5,
    information_value: float = 0.8
) -> float:
    """
    Calculates Bayesian-informed experiment priority:
    Priority = ExpectedGain * Uncertainty * InformationValue
    Prioritizes experiments expected to maximize information gain rather than only easy gains.
    """
    return float(max(1e-4, expected_gain) * max(1e-4, uncertainty) * max(1e-4, information_value))

class ResearchBudgetController:
    """
    Autonomous Experimentation Budget & Direction Belief Controller:
    - Enforces hard category constraints.
    - Maintains Bayesian posterior beliefs P(useful | evidence) per research category.
    """

    def __init__(self, limits: Optional[Dict[str, int]] = None):
        self.limits = limits or DAILY_BUDGET.copy()
        self.usage = {k: 0 for k in self.limits}
        # Bayesian prior beliefs P(category is useful)
        self.beliefs = {
            "context": 0.60,
            "architecture": 0.45,
            "regularization": 0.50,
            "features": 0.35,
            "null_referee": 0.90
        }

    def can_allocate(self, category: str, cost: int = 1) -> bool:
        cat_key = "architecture" if category not in self.limits else category
        return (self.usage.get(cat_key, 0) + cost) <= self.limits.get(cat_key, 10)

    def allocate(self, category: str, cost: int = 1) -> bool:
        cat_key = "architecture" if category not in self.limits else category
        if self.can_allocate(cat_key, cost):
            self.usage[cat_key] += cost
            return True
        return False

    def update_belief(self, category: str, success: bool) -> float:
        """Bayesian update on research direction effectiveness."""
        prior = self.beliefs.get(category, 0.5)
        # Likelihood P(success | useful) = 0.8, P(success | not useful) = 0.2
        if success:
            posterior = (0.8 * prior) / ((0.8 * prior) + (0.2 * (1.0 - prior)))
        else:
            posterior = (0.2 * prior) / ((0.2 * prior) + (0.8 * (1.0 - prior)))
        self.beliefs[category] = round(float(posterior), 4)
        return self.beliefs[category]

    def reset_daily_budget(self) -> None:
        self.usage = {k: 0 for k in self.limits}
