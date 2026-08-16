from typing import List, Any, Optional
import numpy as np

class ReplayBuffer:
    """
    Experience replay buffer combining recent observations with stratified historical samples:
    B = B_recent U B_historical
    Mitigates catastrophic forgetting during online neural parameter fine-tuning.
    """

    def __init__(self, recent_capacity: int = 1000, historical_capacity: int = 1000):
        self.recent_capacity = recent_capacity
        self.historical_capacity = historical_capacity
        self.recent_buffer: List[Any] = []
        self.historical_buffer: List[Any] = []

    def add_recent(self, item: Any) -> None:
        self.recent_buffer.append(item)
        if len(self.recent_buffer) > self.recent_capacity:
            oldest = self.recent_buffer.pop(0)
            if len(self.historical_buffer) < self.historical_capacity:
                self.historical_buffer.append(oldest)
            else:
                idx = np.random.randint(0, self.historical_capacity)
                self.historical_buffer[idx] = oldest

    def sample(self, batch_size: int = 32, recent_ratio: float = 0.7) -> List[Any]:
        n_recent = min(int(batch_size * recent_ratio), len(self.recent_buffer))
        n_hist = min(batch_size - n_recent, len(self.historical_buffer))
        
        sampled = []
        if n_recent > 0 and len(self.recent_buffer) > 0:
            indices = np.random.choice(len(self.recent_buffer), size=n_recent, replace=False)
            sampled.extend([self.recent_buffer[i] for i in indices])
            
        if n_hist > 0 and len(self.historical_buffer) > 0:
            indices = np.random.choice(len(self.historical_buffer), size=n_hist, replace=False)
            sampled.extend([self.historical_buffer[i] for i in indices])
            
        return sampled

def recency_weights(length: int, decay_lambda: float = 0.001) -> np.ndarray:
    """Calculates exponential recency sample weights: w_t = exp(-lambda * (T - t))."""
    t = np.arange(length, dtype=np.float64)
    weights = np.exp(-decay_lambda * (length - 1 - t))
    return weights / weights.sum()
