import numpy as np
from typing import Sequence, Optional

def estimate_transition(sequence: Sequence[int], n_symbols: int = 10, smoothing: float = 1.0) -> np.ndarray:
    """Estimates the 1st-order empirical transition matrix P(X_t | X_{t-1})."""
    counts = np.full((n_symbols, n_symbols), float(smoothing), dtype=np.float64)
    sequence = list(sequence)
    for a, b in zip(sequence[:-1], sequence[1:]):
        if 0 <= a < n_symbols and 0 <= b < n_symbols:
            counts[a, b] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)

def generate_markov(
    transition: np.ndarray,
    length: int,
    initial: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generates a synthetic Markov sequence following a transition probability matrix.
    Tests Null Hypothesis D: X_t ~ Markov(P(X_t | X_{t-1})).
    """
    rng = np.random.default_rng(seed)
    n = transition.shape[0]
    if initial is None:
        initial = np.full(n, 1.0 / n, dtype=np.float64)
        
    result = np.empty(length, dtype=np.int64)
    result[0] = rng.choice(n, p=initial)
    
    for t in range(1, length):
        prev = result[t - 1]
        p_row = transition[prev]
        result[t] = rng.choice(n, p=p_row)
        
    return result
