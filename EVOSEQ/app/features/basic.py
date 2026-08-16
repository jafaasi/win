import numpy as np

def digit_distribution(values):
    """Computes the empirical 10-class probability distribution of digits."""
    if not len(values):
        return np.zeros(10, dtype=np.float64)
    counts = np.bincount(values, minlength=10)
    total = counts.sum()
    if total == 0:
        return np.zeros(10, dtype=np.float64)
    return (counts / total).astype(np.float64)

def entropy(probabilities):
    """Calculates Shannon entropy in bits for a given probability vector."""
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities[probabilities > 0]
    if len(probabilities) == 0:
        return 0.0
    return float(-np.sum(probabilities * np.log2(probabilities)))

def digit_entropy(values):
    """Calculates empirical Shannon entropy of a digit sequence."""
    return entropy(digit_distribution(values))
