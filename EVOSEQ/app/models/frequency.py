from typing import Sequence, Optional, Union
import numpy as np
from .base import SequenceModel

class FrequencyBaseline(SequenceModel):
    """
    Empirical Marginal Frequency Baseline (Null Model #0):
    P(X = k) = counts(k) / N.
    Every higher-order Markov, HMM, ESN, Transformer, or SSM candidate must
    statistically outperform this memoryless marginal baseline under identical future holdouts.
    """

    def __init__(self, smoothing: float = 1e-6):
        self.smoothing = smoothing
        self.probabilities: np.ndarray = np.full(10, 0.1, dtype=np.float64)

    def fit(self, sequence: Sequence[int]) -> "FrequencyBaseline":
        seq = np.asarray(sequence, dtype=np.int64)
        if len(seq) == 0:
            self.probabilities = np.full(10, 0.1, dtype=np.float64)
            return self
            
        counts = np.bincount(seq, minlength=10).astype(np.float64)
        counts += self.smoothing
        self.probabilities = counts / counts.sum()
        return self

    def update(self, new_observations: Sequence[int]) -> "FrequencyBaseline":
        # Incremental update
        return self.fit(new_observations)

    def predict_proba(self, context: Optional[Sequence[int]] = None) -> np.ndarray:
        return self.probabilities.copy()

    def save(self, path: str) -> None:
        np.save(path, self.probabilities)

    @classmethod
    def load(cls, path: str) -> "FrequencyBaseline":
        inst = cls()
        inst.probabilities = np.load(path if path.endswith(".npy") else f"{path}.npy")
        return inst
