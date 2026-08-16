from typing import Sequence, Dict, Tuple, Optional, Union
import numpy as np
from .base import SequenceModel, ModelMetadata

class MarkovModel(SequenceModel):
    """Discrete Order-N Markov Chain Predictor with Laplace additive smoothing."""

    def __init__(
        self,
        order: int = 1,
        smoothing: float = 1.0,
        version: str = "markov-v1"
    ):
        self.order = order
        self.smoothing = smoothing
        self.counts: Dict[Tuple[int, ...], np.ndarray] = {}
        self.metadata = ModelMetadata(
            name=f"Markov-{order}",
            version=version,
            parameters={"order": order, "smoothing": smoothing}
        )

    def fit(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None) -> "MarkovModel":
        sequence = list(X)
        self.counts = {}
        if len(sequence) <= self.order:
            return self
            
        for i in range(self.order, len(sequence)):
            context = tuple(sequence[i - self.order: i])
            target = sequence[i]
            if context not in self.counts:
                self.counts[context] = np.zeros(10, dtype=np.float64)
            self.counts[context][target] += 1.0
        return self

    def update(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None) -> "MarkovModel":
        sequence = list(X)
        if len(sequence) <= self.order:
            return self
            
        for i in range(self.order, len(sequence)):
            context = tuple(sequence[i - self.order: i])
            target = sequence[i]
            if context not in self.counts:
                self.counts[context] = np.zeros(10, dtype=np.float64)
            self.counts[context][target] += 1.0
        return self

    def predict_one_step(self, context: Sequence[int]) -> np.ndarray:
        ctx_list = list(context)
        if len(ctx_list) < self.order:
            return np.full(10, 0.1, dtype=np.float64)
            
        ctx = tuple(ctx_list[-self.order:])
        counts = self.counts.get(ctx, np.zeros(10, dtype=np.float64))
        probabilities = counts + self.smoothing
        total = probabilities.sum()
        if total == 0:
            return np.full(10, 0.1, dtype=np.float64)
        return probabilities / total

    def predict_proba(self, X: Union[np.ndarray, list]) -> np.ndarray:
        seq = list(X)
        if len(seq) == 0:
            return np.full(10, 0.1, dtype=np.float64)
        # If single context
        return self.predict_one_step(seq)

    def predict_sequence(self, sequence: Sequence[int]) -> np.ndarray:
        """Generates probability distribution for every observation in the sequence."""
        seq = list(sequence)
        preds = []
        for i in range(len(seq)):
            start = max(0, i - self.order + 1)
            ctx = seq[start: i + 1]
            preds.append(self.predict_one_step(ctx))
        return np.asarray(preds)

    def save(self, path: str) -> None:
        save_data = {
            "order": self.order,
            "smoothing": self.smoothing,
            "counts": self.counts,
            "metadata": self.metadata
        }
        np.save(path, save_data)

    @classmethod
    def load(cls, path: str) -> "MarkovModel":
        data = np.load(path, allow_pickle=True).item()
        model = cls(order=data["order"], smoothing=data["smoothing"])
        model.counts = data["counts"]
        model.metadata = data.get("metadata", model.metadata)
        return model
