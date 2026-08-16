import numpy as np
from typing import Sequence, Union

def log_loss(probabilities: Union[np.ndarray, Sequence[float]], actual: int) -> float:
    """Calculates cross-entropy loss on a multiclass probability vector."""
    probs = np.asarray(probabilities, dtype=np.float64)
    p = float(np.clip(probs[actual], 1e-15, 1.0))
    return float(-np.log(p))

def brier_score(probabilities: Union[np.ndarray, Sequence[float]], actual: int) -> float:
    """Calculates multiclass Brier score (mean squared error of probability vector)."""
    probs = np.asarray(probabilities, dtype=np.float64)
    target = np.zeros(len(probs), dtype=np.float64)
    target[actual] = 1.0
    return float(np.mean((probs - target) ** 2))

def entropy(probabilities: Union[np.ndarray, Sequence[float]]) -> float:
    """Calculates Shannon entropy in bits."""
    probs = np.asarray(probabilities, dtype=np.float64)
    p = probs[probs > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))

def calibration_error(predicted_probs: Sequence[np.ndarray], actuals: Sequence[int], n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) for sequence probability predictions."""
    if not len(predicted_probs) or not len(actuals):
        return 0.0
        
    bin_total = np.zeros(n_bins)
    bin_correct = np.zeros(n_bins)
    bin_conf_sum = np.zeros(n_bins)
    
    for probs, actual in zip(predicted_probs, actuals):
        pred_label = int(np.argmax(probs))
        conf = float(probs[pred_label])
        bin_idx = min(n_bins - 1, int(conf * n_bins))
        
        bin_total[bin_idx] += 1
        if pred_label == actual:
            bin_correct[bin_idx] += 1
        bin_conf_sum[bin_idx] += conf
        
    ece = 0.0
    total_n = float(len(actuals))
    for i in range(n_bins):
        if bin_total[i] > 0:
            acc = bin_correct[i] / bin_total[i]
            avg_conf = bin_conf_sum[i] / bin_total[i]
            ece += (bin_total[i] / total_n) * abs(acc - avg_conf)
            
    return float(ece)

def calculate_null_advantage(model_accuracy: float, null_accuracy: float = 0.10) -> float:
    """Differential out-of-sample edge over random/null model."""
    return float(model_accuracy - null_accuracy)
