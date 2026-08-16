from typing import Union, Sequence, Optional
import numpy as np

class TemperatureScaler:
    """
    Post-Hoc Probability Calibration via Temperature Scaling:
    p'_k = exp(log(p_k) / T) / sum_j exp(log(p_j) / T)
    T > 1 softens overconfident predictions; T < 1 sharpens confident ones.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = max(1e-3, float(temperature))

    def fit_temperature(
        self,
        probabilities: Sequence[np.ndarray],
        targets: Sequence[int],
        t_range: Sequence[float] = (0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5)
    ) -> float:
        """Finds optimal temperature minimizing validation negative log-likelihood."""
        best_t = 1.0
        best_ll = float("inf")
        
        for t in t_range:
            scaled_probs = [self.scale(p, temperature=t) for p in probabilities]
            ll = -np.mean([np.log(np.clip(p[int(y)], 1e-15, 1.0)) for p, y in zip(scaled_probs, targets)])
            if ll < best_ll:
                best_ll = ll
                best_t = t
                
        self.temperature = best_t
        return self.temperature

    def scale(self, probabilities: Union[Sequence[float], np.ndarray], temperature: Optional[float] = None) -> np.ndarray:
        t = temperature if temperature is not None else self.temperature
        p = np.asarray(probabilities, dtype=np.float64)
        p = np.clip(p, 1e-15, 1.0)
        log_p = np.log(p) / t
        log_p -= np.max(log_p, axis=-1, keepdims=True)
        exp_p = np.exp(log_p)
        total = np.sum(exp_p, axis=-1, keepdims=True)
        return exp_p / total
