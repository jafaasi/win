from typing import Dict, Any, List, Optional
import uuid
from .hypothesis import ResearchHypothesis

SEARCH_SPACE = {
    "context_length": [32, 64, 128, 256, 512],
    "d_model": [32, 64, 128, 256],
    "dropout": [0.0, 0.05, 0.10, 0.20],
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "feature_version": ["v1", "v2", "v3", "v4", "v5", "v6", "v7"]
}

def mutate_single_variable(configuration: Dict[str, Any], parameter: str, value: Any) -> Dict[str, Any]:
    """
    Performs controlled, single-variable mutation:
    Ensures that empirical performance changes remain strictly interpretable.
    """
    child = configuration.copy()
    child[parameter] = value
    return child

def cost_aware_objective(
    log_loss: float,
    latency_ms: float = 0.5,
    param_count: int = 10000,
    lambda_latency: float = 1e-3,
    lambda_params: float = 1e-7
) -> float:
    """
    Cost-Aware Evaluation Objective:
    Score = -LogLoss - lambda_latency * Latency - lambda_params * Params
    Prevents promotion of excessively heavy architectures with negligible predictive delta.
    """
    cost = (lambda_latency * latency_ms) + (lambda_params * param_count)
    return float(-log_loss - cost)

class CandidateFactory:
    """
    Translates research hypotheses into concrete, instantiable candidate model specifications.
    """

    def __init__(self, search_space: Optional[Dict[str, List[Any]]] = None):
        self.search_space = search_space or SEARCH_SPACE

    def instantiate_candidate(
        self,
        hypothesis: ResearchHypothesis,
        generation: int = 1,
        parent_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        base_config = {
            "family": hypothesis.parent_model or "transformer",
            "context_length": 128,
            "d_model": 64,
            "dropout": 0.1,
            "learning_rate": 1e-3,
            "feature_version": "v17",
            "seed": 42
        }
        if parent_config:
            base_config.update(parent_config)
            
        # Apply hypothesis specific configurations
        base_config.update(hypothesis.configuration)
        
        candidate_code = f"CAND-GEN{generation}-{base_config['family']}-{uuid.uuid4().hex[:6]}"
        
        return {
            "candidate_code": candidate_code,
            "hypothesis_id": hypothesis.id,
            "generation": generation,
            "family": base_config["family"],
            "configuration": base_config,
            "status": "CANDIDATE"
        }
