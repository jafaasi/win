from typing import List, Dict, Any, Optional
import numpy as np

class MetaReplayBuffer:
    """
    Experience replay buffer for environment-conditioned meta-gating:
    Stores (environment_vector, model_predictions_matrix, target) tuples.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer: List[Dict[str, Any]] = []

    def add(self, item: Dict[str, Any]) -> None:
        self.buffer.append(item)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, n: int) -> List[Dict[str, Any]]:
        sample_size = min(n, len(self.buffer))
        if sample_size == 0:
            return []
        indices = np.random.choice(len(self.buffer), sample_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)
