from typing import Sequence, Union
import numpy as np

def log_loss(probabilities: Union[Sequence[np.ndarray], np.ndarray], targets: Union[Sequence[int], np.ndarray]) -> float:
    """
    Computes categorical cross-entropy / log-loss:
    LL = -1/N * sum(log(p_t(y_t)))
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64)
    if len(probs) == 0 or len(y) == 0:
        return 0.0
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
        y = y.reshape(1)
        
    selected = probs[np.arange(len(y)), y]
    selected = np.clip(selected, 1e-15, 1.0)
    return float(-np.mean(np.log(selected)))

def accuracy(probabilities: Union[Sequence[np.ndarray], np.ndarray], targets: Union[Sequence[int], np.ndarray]) -> float:
    """Computes argmax classification accuracy."""
    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64)
    if len(probs) == 0 or len(y) == 0:
        return 0.0
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
        y = y.reshape(1)
        
    preds = np.argmax(probs, axis=-1)
    return float(np.mean(preds == y))

def brier_score(probabilities: Union[Sequence[np.ndarray], np.ndarray], targets: Union[Sequence[int], np.ndarray]) -> float:
    """
    Computes multi-class Brier score:
    BS = 1/N * sum(sum((p_{t,k} - y_{t,k})^2))
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64)
    if len(probs) == 0 or len(y) == 0:
        return 0.0
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
        y = y.reshape(1)
        
    one_hot_mat = np.zeros_like(probs)
    one_hot_mat[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - one_hot_mat) ** 2, axis=-1)))
