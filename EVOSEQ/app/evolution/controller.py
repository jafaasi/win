from enum import Enum
from typing import Dict, Any, Optional

class Action(Enum):
    WAIT = "wait"
    UPDATE = "update"
    ADAPT = "adapt"
    INVESTIGATE = "investigate"
    EVOLVE = "evolve"

class EvolutionController:
    """
    Autonomous meta-controller deciding whether to WAIT, UPDATE, ADAPT, INVESTIGATE, or EVOLVE
    based on multi-dimensional drift, multi-horizon degradation, prediction entropy, and model disagreement.
    """

    def __init__(
        self,
        min_observations: int = 40,
        drift_evolve_threshold: float = 0.08,
        drift_adapt_threshold: float = 0.04,
        performance_drop_threshold: float = -0.03,
        disagreement_investigate_threshold: float = 0.12,
    ):
        self.min_observations = min_observations
        self.drift_evolve_threshold = drift_evolve_threshold
        self.drift_adapt_threshold = drift_adapt_threshold
        self.performance_drop_threshold = performance_drop_threshold
        self.disagreement_investigate_threshold = disagreement_investigate_threshold

    def decide(
        self,
        observations: int,
        drift_score: Optional[float] = None,
        performance_delta: float = 0.0,
        uncertainty: float = 0.0,
        model_disagreement: Optional[float] = None,
        drift: Optional[float] = None,
        performance_change: Optional[float] = None,
        disagreement: Optional[float] = None,
    ) -> Action:
        d_score = drift_score if drift_score is not None else (drift or 0.0)
        p_delta = performance_delta if performance_change is None else performance_change
        disagree = model_disagreement if model_disagreement is not None else (disagreement or 0.0)

        # 1. If insufficient observation evidence, do nothing
        if observations < self.min_observations:
            return Action.WAIT

        # 2. If significant distribution drift detected, trigger full evolution
        if d_score >= self.drift_evolve_threshold:
            return Action.EVOLVE

        # 3. If recent performance drops below multi-horizon baseline, trigger adaptation
        if p_delta <= self.performance_drop_threshold or d_score >= self.drift_adapt_threshold:
            return Action.ADAPT

        # 4. If population models strongly disagree on structural regime, trigger forensic investigation
        if disagree >= self.disagreement_investigate_threshold:
            return Action.INVESTIGATE

        # 5. Default healthy regime: apply lightweight online updates
        return Action.UPDATE

