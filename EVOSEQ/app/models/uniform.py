from typing import Optional, Union
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

    def predict_proba(self, X: Union[np.ndarray, list]) -> np.ndarray:
        return np.full(10, 0.1, dtype=np.float64)

    def save(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, path: str) -> "UniformModel":
        return cls()
