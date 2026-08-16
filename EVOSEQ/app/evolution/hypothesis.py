from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import uuid

@dataclass
class ResearchHypothesis:
    """
    Structured scientific hypothesis formulated by the Autonomous Research Director:
    Encapsulates specific targeted mutations or experimental interventions.
    """
    id: str
    category: str
    parent_model: Optional[str]
    description: str
    configuration: Dict[str, Any]
    expected_effect: str
    priority: float = 0.0
    budget: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

def generate_hypotheses(
    environment_state: Any,
    population_metrics: Optional[Dict[str, Any]] = None,
    budget_limit: int = 5
) -> List[ResearchHypothesis]:
    """
    Generates prioritized research hypotheses based on statistical environment indicators:
    - High Drift: Tests shorter causal context and higher recency weighting.
    - High Disagreement: Tests diverse hybrid architectures (e.g. ESN + S4D).
    - High Entropy / Instability: Tests stronger regularization & null validation.
    - Performance Degradation: Tests feature ablations.
    """
    hypotheses = []
    drift = getattr(environment_state, "drift_score", None) or getattr(environment_state, "drift", 0.0)
    disagreement = getattr(environment_state, "model_disagreement", None) or getattr(environment_state, "disagreement", 0.0)
    entropy = getattr(environment_state, "entropy", 3.25)
    
    # 1. Drift-Driven Hypotheses
    if float(drift) > 0.03:
        hypotheses.append(
            ResearchHypothesis(
                id=f"H-DRIFT-{uuid.uuid4().hex[:6]}",
                category="context",
                parent_model="mamba",
                description="Test shorter context to adapt quickly under detected drift",
                configuration={"context_length": 64, "learning_rate": 3e-4},
                expected_effect="reduce lag during non-stationary regime shift",
                priority=0.85,
                budget=1
            )
        )
        hypotheses.append(
            ResearchHypothesis(
                id=f"H-RECENCY-{uuid.uuid4().hex[:6]}",
                category="regularization",
                parent_model="transformer",
                description="Test stronger recency weighting in replay buffer",
                configuration={"recency_decay": 0.1},
                expected_effect="prioritize recent observations under drift",
                priority=0.75,
                budget=1
            )
        )
        
    # 2. Disagreement-Driven Hypotheses
    if float(disagreement) > 0.15:
        hypotheses.append(
            ResearchHypothesis(
                id=f"H-HYBRID-{uuid.uuid4().hex[:6]}",
                category="architecture",
                parent_model="s4d",
                description="Test hybrid S4D + ESN state-space representation",
                configuration={"d_model": 64, "state_size": 32},
                expected_effect="reconcile long-range and short-range divergence",
                priority=0.80,
                budget=1
            )
        )
        
    # 3. High Entropy / Null Referee Hypothesis
    if float(entropy) > 3.20:
        hypotheses.append(
            ResearchHypothesis(
                id=f"H-NULL-{uuid.uuid4().hex[:6]}",
                category="null_referee",
                parent_model=None,
                description="Verify apparent model gains against surrogate Markov/IID null sequences",
                configuration={"null_trials": 50},
                expected_effect="filter out spurious overfitting under high entropy",
                priority=0.90,
                budget=1
            )
        )
        
    # 4. Standard Continuous Exploration Hypotheses
    if len(hypotheses) == 0:
        hypotheses.append(
            ResearchHypothesis(
                id=f"H-EXPLORE-{uuid.uuid4().hex[:6]}",
                category="architecture",
                parent_model="transformer",
                description="Test wider transformer feedforward dimension",
                configuration={"feedforward": 256, "n_heads": 4},
                expected_effect="probe representation capacity",
                priority=0.60,
                budget=1
            )
        )
        
    hypotheses.sort(key=lambda h: h.priority, reverse=True)
    return hypotheses[:budget_limit]
