import numpy as np
from typing import Sequence

def categorical_entropy(values: Sequence[int], cardinality: int = 10) -> float:
    """
    Computes empirical Shannon entropy in bits for a discrete categorical sequence:
    H(X) = - sum(p_i * log2(p_i)).
    """
    values = np.asarray(values)
    if len(values) == 0:
        return 0.0
    counts = np.bincount(values, minlength=cardinality)
    total = counts.sum()
    if total == 0:
        return 0.0
    probabilities = counts / float(total)
    probabilities = probabilities[probabilities > 0]
    if len(probabilities) == 0:
        return 0.0
    return float(-np.sum(probabilities * np.log2(probabilities)))

def entropy(probabilities: Sequence[float]) -> float:
    """Calculates Shannon entropy in bits for an explicit probability vector."""
    probs = np.asarray(probabilities, dtype=float)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))

def shannon_entropy(sequence: Sequence[int], alphabet_size: int = 10) -> float:
    """Alias for categorical entropy over an integer sequence."""
    return categorical_entropy(sequence, cardinality=alphabet_size)
