from dataclasses import dataclass
from datetime import datetime
from typing import Sequence, Union
import numpy as np

@dataclass(frozen=True)
class Outcome:
    """Canonical single observation record."""
    sequence_no: int
    timestamp: datetime
    digit: int
    size: int    # 0 = Small (0-4), 1 = Big (5-9)
    color: int   # 0 = Green (1,3,7,9), 1 = Red (2,4,6,8), 2 = Violet (0,5)
    parity: int  # 0 = Even, 1 = Odd

def validate_outcome(outcome: Outcome) -> None:
    """Validates domain constraints of an outcome observation."""
    if not isinstance(outcome.digit, int) or not (0 <= outcome.digit <= 9):
        raise ValueError(f"digit must be integer in 0..9, got {outcome.digit}")
    if outcome.size not in (0, 1):
        raise ValueError(f"size must be in (0, 1), got {outcome.size}")
    if outcome.color not in (0, 1, 2):
        raise ValueError(f"color must be in (0, 1, 2), got {outcome.color}")
    if outcome.parity not in (0, 1):
        raise ValueError(f"parity must be in (0, 1), got {outcome.parity}")

def validate_probability_vector(probabilities: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Validates that a probability vector is non-negative and sums exactly to 1."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape[-1] != 10:
        raise ValueError(f"Expected 10 classes in probability vector, got shape {probs.shape}")
    if np.any(probs < -1e-7):
        raise ValueError("Probabilities cannot be negative")
    # Clip negative precision noise
    probs = np.maximum(probs, 0.0)
    sums = probs.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=1e-4):
        raise ValueError(f"Probabilities must sum to 1.0, got sum {sums}")
    # Normalize exact simplex
    return probs / sums[..., None] if probs.ndim > 1 else probs / sums
