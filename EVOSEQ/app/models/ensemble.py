from typing import Sequence, List, Dict, Any, Optional, Union
import numpy as np
from .base import SequenceModel, ModelMetadata

def ensemble_probabilities(
    probabilities_list: Sequence[np.ndarray],
    scores: Sequence[float],
) -> np.ndarray:
    """
    Computes performance-weighted Softmax Bayesian model combination:
    w_i = exp(q_i) / sum_j exp(q_j)
    P_fused = sum_i w_i * P_i
    """
    probs_array = np.asarray(probabilities_list, dtype=np.float64) # [N_models, 10]
    scores_array = np.asarray(scores, dtype=np.float64)
    
    if len(probs_array) == 0:
        return np.full(10, 0.1, dtype=np.float64)
        
    if len(probs_array) == 1:
        return probs_array[0]
        
    scores_stable = scores_array - np.max(scores_array)
    weights = np.exp(scores_stable)
    weights_sum = weights.sum()
    if weights_sum == 0:
        weights = np.full(len(scores_array), 1.0 / len(scores_array))
    else:
        weights /= weights_sum
        
    fused_probs = np.sum(probs_array * weights[:, None], axis=0)
    total = fused_probs.sum()
    if total == 0:
        return np.full(10, 0.1, dtype=np.float64)
    return fused_probs / total

class MetaEnsemble(SequenceModel):
    """
    Dynamic Meta-Ensemble combining a heterogeneous population of sequence models
    (Markov, DiscreteHMM, EchoStateNetwork, Transformer).
    """

    def __init__(
        self,
        models: Optional[List[SequenceModel]] = None,
        scores: Optional[List[float]] = None,
        version: str = "ensemble-v1"
    ):
        self.models = models or []
        self.scores = scores or ([1.0] * len(self.models))
        self.metadata = ModelMetadata(
            name="MetaEnsemble",
            version=version,
            parameters={"n_models": len(self.models)}
        )

    def fit(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None) -> "MetaEnsemble":
        for m in self.models:
            m.fit(X, y)
        return self

    def update(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None) -> "MetaEnsemble":
        for m in self.models:
            m.update(X, y)
        return self

    def predict_proba(self, X: Union[np.ndarray, list]) -> np.ndarray:
        if not self.models:
            return np.full(10, 0.1, dtype=np.float64)
            
        prob_predictions = [m.predict_proba(X) for m in self.models]
        return ensemble_probabilities(prob_predictions, self.scores)

    def save(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, path: str) -> "MetaEnsemble":
        return cls()
