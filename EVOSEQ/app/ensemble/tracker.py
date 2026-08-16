from typing import Dict, Sequence, Union, Any, List
import numpy as np

def softmax_weights(losses: Union[Sequence[float], np.ndarray], beta: float = 2.0) -> np.ndarray:
    """
    Converts model losses into normalized Gibbs / softmax mixture weights:
    w_m = exp(-beta * L_m) / sum_j exp(-beta * L_j)
    """
    losses_arr = np.asarray(losses, dtype=np.float64)
    if len(losses_arr) == 0:
        return np.empty(0, dtype=np.float64)
        
    logits = -beta * losses_arr
    logits -= np.max(logits) # Stability subtraction
    weights = np.exp(logits)
    return weights / np.sum(weights)

def combine_predictions(predictions: Union[Sequence[np.ndarray], np.ndarray], weights: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """
    Computes linearly weighted combination of multiple categorical probability matrices:
    P_t = sum_m w_m P_{m, t}
    """
    preds = np.asarray(predictions, dtype=np.float64) # [M, N, K] or [M, K]
    w = np.asarray(weights, dtype=np.float64)          # [M]
    
    if preds.ndim == 2: # [M, K]
        combined = np.sum(preds * w[:, None], axis=0)
        total = np.sum(combined)
        return combined / (total if total > 0 else 1.0)
    elif preds.ndim == 3: # [M, N, K]
        combined = np.sum(preds * w[:, None, None], axis=0)
        total = np.sum(combined, axis=-1, keepdims=True)
        return combined / np.where(total > 0, total, 1.0)
    else:
        return np.asarray(preds[0]) if len(preds) > 0 else np.full(10, 0.1)

class OnlineLossTracker:
    """
    Maintains running exponentially-weighted historical losses for models:
    L_{m, t} = decay * loss_{m, t} + (1 - decay) * L_{m, t-1}
    """

    def __init__(self, decay: float = 0.05):
        self.decay = decay
        self.losses: Dict[str, float] = {}

    def update(self, model_id: str, loss: float) -> float:
        if model_id not in self.losses:
            self.losses[model_id] = float(loss)
        else:
            prev = self.losses[model_id]
            self.losses[model_id] = self.decay * float(loss) + (1.0 - self.decay) * prev
        return self.losses[model_id]

    def get_weights(self, beta: float = 2.0) -> Dict[str, float]:
        if not self.losses:
            return {}
        keys = list(self.losses.keys())
        w = softmax_weights([self.losses[k] for k in keys], beta=beta)
        return {k: float(w[i]) for i, k in enumerate(keys)}
