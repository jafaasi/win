import numpy as np
from typing import Sequence, Optional, Union

def generate_iid(
    probabilities_or_sequence: Union[Sequence[float], Sequence[int], np.ndarray],
    length: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generates an IID synthetic sequence drawn from an empirical categorical distribution or historical sequence.
    Tests Null Hypothesis A: X_t ~ IID P(X).
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(probabilities_or_sequence)
    
    if length is None:
        # A historical sequence of digits was passed
        length = len(arr)
        counts = np.bincount(arr.astype(int), minlength=10)
        probs = counts / np.sum(counts)
    else:
        probs = np.asarray(probabilities_or_sequence, dtype=np.float64)
        total = probs.sum()
        if total == 0:
            probs = np.full(len(probs), 1.0 / len(probs), dtype=np.float64)
        else:
            probs = probs / total
            
    return rng.choice(len(probs), size=length, p=probs)
