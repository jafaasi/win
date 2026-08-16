from abc import ABC, abstractmethod
import numpy as np
from typing import Sequence, Dict, Any

class SequenceModel(ABC):
    """Abstract Base Class for all EVOSEQ Sequence Intelligence Models."""

    @abstractmethod
    def fit(self, sequence: Sequence[int]) -> "SequenceModel":
        """Fits the model parameters on a batch sequence history."""
        pass

    @abstractmethod
    def update(self, sequence: Sequence[int]) -> "SequenceModel":
        """Incrementally adapts model parameters on incoming observations."""
        pass

    @abstractmethod
    def predict_proba(self, context: Sequence[int]) -> np.ndarray:
        """Returns the 10-class probability distribution vector for the next step."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Serializes model parameters to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> "SequenceModel":
        """Deserializes model parameters from disk."""
        pass
