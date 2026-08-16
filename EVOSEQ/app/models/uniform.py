from typing import Optional, Union, Sequence
import numpy as np
from .base import SequenceModel, ModelMetadata

class UniformModel(SequenceModel):
    """Memoryless Null Hypothesis Model (P(d) = 0.10 for all digits 0-9)."""

    def __init__(self, version: str = "null-v1"):
        self.metadata = ModelMetadata(
            name="UniformNull",
            version=version,
            parameters={"cardinality": 10}
        )

    def fit(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None) -> "UniformModel":
        return self

    def update(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None) -> "UniformModel":
        return self

    def predict_proba(self, X: Optional[Union[np.ndarray, Sequence, int]] = None) -> np.ndarray:
        if isinstance(X, int):
            return np.full((X, 10), 0.1, dtype=np.float64)
        elif isinstance(X, np.ndarray) and X.ndim == 2:
            return np.full((len(X), 10), 0.1, dtype=np.float64)
        # 1D context or None -> single 10-class probability distribution
        return np.full(10, 0.1, dtype=np.float64)

    def save(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, path: str) -> "UniformModel":
        return cls()

class UniformBaseline(UniformModel):
    name = "uniform"
