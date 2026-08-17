import numpy as np
from typing import List

class BaseModel:
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes

    def partial_fit(self, sequence: np.ndarray):
        pass

    def predict_proba(self, recent_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def evaluate(self, sequence: np.ndarray, val_window: int = 30) -> float:
        """
        Evaluate model on recent slice of sequence. Returns average negative log-likelihood (log loss).
        Lower is better.
        """
        if len(sequence) < 2:
            return math.log(self.num_classes)
            
        start_idx = max(1, len(sequence) - val_window)
        loss = 0.0
        count = 0
        for i in range(start_idx, len(sequence)):
            ctx = sequence[:i]
            target = sequence[i]
            probs = self.predict_proba(ctx)
            p = max(1e-9, probs[target])
            loss -= np.log(p)
            count += 1
            
        return float(loss / max(1, count))

class UniformModel(BaseModel):
    """
    Always predicts P(x) = 1 / N. The baseline to beat.
    """
    def predict_proba(self, recent_values: np.ndarray) -> np.ndarray:
        return np.ones(self.num_classes) / self.num_classes


class FrequencyModel(BaseModel):
    """
    Predicts based on historical frequency.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__(num_classes)
        self.counts = np.ones(num_classes)  # Laplace smoothing

    def partial_fit(self, sequence: np.ndarray):
        new_counts = np.bincount(sequence, minlength=self.num_classes)
        # Exponential decay to adapt to changing behavior
        decay_lambda = 0.99
        self.counts = (self.counts * decay_lambda) + new_counts

    def predict_proba(self, recent_values: np.ndarray) -> np.ndarray:
        return self.counts / np.sum(self.counts)


class MarkovModel(BaseModel):
    """
    Predicts based on previous value transitions.
    Order=1 means P(x_t | x_{t-1}).
    """
    def __init__(self, order: int = 1, num_classes: int = 10):
        super().__init__(num_classes)
        self.order = order
        # State shape: [num_classes, num_classes] for order 1
        shape = tuple([num_classes] * (order + 1))
        self.transitions = np.ones(shape) # Laplace smoothing

    def partial_fit(self, sequence: np.ndarray):
        if len(sequence) <= self.order:
            return
            
        decay_lambda = 0.99
        self.transitions *= decay_lambda
        
        for i in range(self.order, len(sequence)):
            idx = tuple(sequence[i - self.order : i + 1])
            self.transitions[idx] += 1.0

    def predict_proba(self, recent_values: np.ndarray) -> np.ndarray:
        if len(recent_values) < self.order:
            return np.ones(self.num_classes) / self.num_classes
            
        ctx = tuple(recent_values[-self.order:])
        # The transition slice for the given context
        probs = self.transitions[ctx]
        return probs / np.sum(probs)

class BitwiseModel(BaseModel):
    """
    Predicts Big/Small probabilities and maps them back to the 0-9 distribution.
    Big (5-9) vs Small (0-4).
    """
    def __init__(self):
        super().__init__(10)
        # We track Big(1) and Small(0) frequencies
        self.bit_counts = np.ones(2)

    def partial_fit(self, sequence: np.ndarray):
        bitwise = np.where(sequence >= 5, 1, 0)
        new_counts = np.bincount(bitwise, minlength=2)
        decay_lambda = 0.99
        self.bit_counts = (self.bit_counts * decay_lambda) + new_counts

    def predict_proba(self, recent_values: np.ndarray) -> np.ndarray:
        p_big = self.bit_counts[1] / np.sum(self.bit_counts)
        p_small = self.bit_counts[0] / np.sum(self.bit_counts)
        
        probs = np.zeros(10)
        # Distribute P(small) evenly among 0-4
        probs[0:5] = p_small / 5.0
        # Distribute P(big) evenly among 5-9
        probs[5:10] = p_big / 5.0
        
        return probs

import math
