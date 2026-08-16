from typing import Sequence
import numpy as np

class OnlineHistogram:
    """Incremental online histogram tracking marginal digit frequencies without full scan."""

    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self.counts = np.zeros(num_classes, dtype=np.int64)
        self.total = 0

    def update(self, digit: int) -> None:
        if 0 <= digit < self.num_classes:
            self.counts[digit] += 1
            self.total += 1

    def probabilities(self) -> np.ndarray:
        if self.total == 0:
            return np.full(self.num_classes, 1.0 / self.num_classes, dtype=np.float64)
        return self.counts.astype(np.float64) / self.total

def entropy(probabilities: Sequence[float]) -> float:
    """Computes Shannon entropy H(X) in bits (base 2)."""
    p = np.asarray(probabilities, dtype=np.float64)
    mask = p > 0
    if not np.any(mask):
        return 0.0
    return float(-np.sum(p[mask] * np.log2(p[mask])))
