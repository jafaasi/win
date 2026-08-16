import numpy as np
from typing import Sequence, Optional

def generate_iid(
    probabilities: Sequence[float],
    length: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generates an IID synthetic sequence drawn from an empirical categorical distribution.
    Tests Null Hypothesis A: X_t ~ IID P(X).
    """
    rng = np.random.default_rng(seed)
    probs = np.asarray(probabilities, dtype=np.float64)
    total = probs.sum()
    if total == 0:
        probs = np.full(len(probs), 1.0 / len(probs), dtype=np.float64)
    else:
        probs = probs / total
        
    return rng.choice(len(probs), size=length, p=probs)
