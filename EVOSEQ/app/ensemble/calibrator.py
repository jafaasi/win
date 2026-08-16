from typing import Sequence, Tuple
import numpy as np

class TemperatureCalibrator:
    """
    Post-Hoc Probability Calibrator:
    Calibrates raw logits z_i by learning optimal scalar temperature T*:
    p_i = softmax(z_i / T*).
    Strict rule: T* is optimized on validation splits, never on test holdouts.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = max(0.1, float(temperature))

    def fit_from_probabilities(
        self,
        probabilities: Sequence[np.ndarray],
        actual_labels: Sequence[int],
        search_grid: Sequence[float] = (0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0)
    ) -> float:
        best_t = 1.0
        best_nll = float("inf")
        
        probs_arr = np.asarray(probabilities)
        # Approximate logits z = log(p + eps)
        eps = 1e-7
        logits = np.log(probs_arr + eps)
        
        for t in search_grid:
            scaled_logits = logits / t
            scaled_logits -= np.max(scaled_logits, axis=-1, keepdims=True)
            p_cal = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits), axis=-1, keepdims=True)
            
            # Cross entropy
            nll = 0.0
            for i, y in enumerate(actual_labels):
                nll -= np.log(max(eps, p_cal[i, int(y)]))
            nll /= max(1, len(actual_labels))
            
            if nll < best_nll:
                best_nll = nll
                best_t = t
                
        self.temperature = best_t
        return self.temperature

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        if abs(self.temperature - 1.0) < 1e-4:
            return probs
        logits = np.log(np.maximum(probs, 1e-7)) / self.temperature
        logits -= np.max(logits, axis=-1, keepdims=True)
        exp_z = np.exp(logits)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
