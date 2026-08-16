from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
import numpy as np

@dataclass
class ModelMetadata:
    name: str
    version: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    parent_version: Optional[str] = None
    generation: int = 1

class SequenceModel(ABC):
    """Universal contract for all EVOSEQ Sequence Intelligence models."""
    metadata: ModelMetadata

    @abstractmethod
    def fit(
        self,
        X: Union[np.ndarray, list],
        y: Optional[Union[np.ndarray, list]] = None,
    ) -> "SequenceModel":
        """Fits model parameters on input features/sequence X and optional targets y."""
        pass

    @abstractmethod
    def update(
        self,
        X: Union[np.ndarray, list],
        y: Optional[Union[np.ndarray, list]] = None,
    ) -> "SequenceModel":
        """Incrementally updates model parameters on new observations."""
        pass

    @abstractmethod
    def predict_proba(
        self,
        X: Union[np.ndarray, list],
    ) -> np.ndarray:
        """Returns 10-class probability distribution vector for the next step [P(0), ..., P(9)]."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Serializes model weights and parameters to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "SequenceModel":
        """Deserializes model instance from disk."""
        pass
