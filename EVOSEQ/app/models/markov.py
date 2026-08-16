import numpy as np
from typing import Sequence, Dict, Tuple
from .base import SequenceModel

class MarkovModel(SequenceModel):
    """Discrete Order-N Markov Chain Predictor with Laplace additive smoothing."""

    def __init__(self, order: int = 3, smoothing: float = 1.0):
        self.order = order
        self.smoothing = smoothing
        self.counts: Dict[Tuple[int, ...], np.ndarray] = {}

    def fit(self, sequence: Sequence[int]) -> "MarkovModel":
        sequence = list(sequence)
        self.counts = {}
        if len(sequence) <= self.order:
            return self
            
        for i in range(self.order, len(sequence)):
            context = tuple(sequence[i-self.order:i])
            target = sequence[i]
            if context not in self.counts:
                self.counts[context] = np.zeros(10, dtype=np.float64)
            self.counts[context][target] += 1.0
        return self

    def update(self, sequence: Sequence[int]) -> "MarkovModel":
        sequence = list(sequence)
        if len(sequence) <= self.order:
            return self
            
        for i in range(self.order, len(sequence)):
            context = tuple(sequence[i-self.order:i])
            target = sequence[i]
            if context not in self.counts:
                self.counts[context] = np.zeros(10, dtype=np.float64)
            self.counts[context][target] += 1.0
        return self

    def predict_proba(self, context: Sequence[int]) -> np.ndarray:
        if len(context) < self.order:
            return np.full(10, 0.1, dtype=np.float64)
            
        ctx = tuple(context[-self.order:])
        counts = self.counts.get(ctx, np.zeros(10, dtype=np.float64))
        probabilities = counts + self.smoothing
        total = probabilities.sum()
        if total == 0:
            return np.full(10, 0.1, dtype=np.float64)
        return probabilities / total

    def save(self, path: str) -> None:
        np.save(path, self.counts)

    def load(self, path: str) -> "MarkovModel":
        self.counts = np.load(path, allow_pickle=True).item()
        return self
