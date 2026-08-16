import numpy as np
from typing import Sequence
from .base import SequenceModel

class UniformModel(SequenceModel):
    """Memoryless Null Hypothesis Model (P(d) = 0.10 for all digits 0-9)."""

    def fit(self, sequence: Sequence[int]) -> "UniformModel":
        return self

    def update(self, sequence: Sequence[int]) -> "UniformModel":
        return self

    def predict_proba(self, context: Sequence[int]) -> np.ndarray:
        return np.full(10, 0.1, dtype=np.float64)

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> "UniformModel":
        return self
