from typing import Sequence, Dict, List
import numpy as np

def adaptive_weights(losses: Sequence[float], beta: float = 2.0) -> np.ndarray:
    """
    Computes Gibbs / Softmax mixture weights from out-of-sample losses:
    w_i = exp(-beta * L_i) / sum(exp(-beta * L_j))
    """
    losses_arr = np.asarray(losses, dtype=np.float64)
    scores = -beta * losses_arr
    scores -= np.max(scores) # numerical stability
    weights = np.exp(scores)
    return weights / np.sum(weights)

class ExponentialLossTracker:
    """
    Maintains online exponentially-decayed historical loss for model i:
    L_{i,t} = alpha * loss_{i,t} + (1 - alpha) * L_{i,t-1}
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.running_losses: Dict[str, float] = {}

    def update(self, model_id: str, current_loss: float) -> float:
        if model_id not in self.running_losses:
            self.running_losses[model_id] = current_loss
        else:
            self.running_losses[model_id] = (
                self.alpha * current_loss + (1.0 - self.alpha) * self.running_losses[model_id]
            )
        return self.running_losses[model_id]

    def get_mixture_weights(self, beta: float = 2.0) -> Dict[str, float]:
        if not self.running_losses:
            return {}
        model_keys = list(self.running_losses.keys())
        losses = [self.running_losses[k] for k in model_keys]
        w = adaptive_weights(losses, beta=beta)
        return {k: round(float(w[i]), 4) for i, k in enumerate(model_keys)}
