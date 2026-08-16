from dataclasses import dataclass
from collections import deque
from typing import Dict, Any, Optional, List
import numpy as np

@dataclass
class PerformanceSnapshot:
    accuracy: float
    log_loss: float
    brier: float
    entropy: float
    sample_count: int

class EWMA:
    """Exponentially Weighted Moving Average filter for smooth trend tracking."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = float(x)
        else:
            self.value = float(self.alpha * x + (1.0 - self.alpha) * self.value)
        return self.value

class PerformanceMonitor:
    """
    Monitors online accuracy, log loss, and Brier score across multiple horizons (50, 250, 1000 observations)
    to detect true structural deterioration versus local noise.
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.losses: deque = deque(maxlen=window_size)
        self.correct: deque = deque(maxlen=window_size)
        self.brier: deque = deque(maxlen=window_size)
        self.ewma_accuracy = EWMA(alpha=0.05)
        self.ewma_loss = EWMA(alpha=0.05)

    def update(self, correct: bool, log_loss: float, brier: float) -> None:
        c_val = 1.0 if correct else 0.0
        self.correct.append(c_val)
        self.losses.append(float(log_loss))
        self.brier.append(float(brier))
        self.ewma_accuracy.update(c_val)
        self.ewma_loss.update(log_loss)

    def snapshot(self, window: Optional[int] = None) -> PerformanceSnapshot:
        if not self.correct:
            return PerformanceSnapshot(accuracy=0.0, log_loss=0.0, brier=0.0, entropy=0.0, sample_count=0)
            
        c_slice = list(self.correct) if window is None else list(self.correct)[-window:]
        l_slice = list(self.losses) if window is None else list(self.losses)[-window:]
        b_slice = list(self.brier) if window is None else list(self.brier)[-window:]
        
        if not c_slice:
            return PerformanceSnapshot(accuracy=0.0, log_loss=0.0, brier=0.0, entropy=0.0, sample_count=0)
            
        return PerformanceSnapshot(
            accuracy=round(float(np.mean(c_slice)), 4),
            log_loss=round(float(np.mean(l_slice)), 4),
            brier=round(float(np.mean(b_slice)), 4),
            entropy=0.0,
            sample_count=len(c_slice)
        )

    def compute_multi_horizon_deltas(self) -> Dict[str, float]:
        """
        Computes delta in accuracy between recent short window (50), medium window (250),
        and overall historical window (1000).
        """
        total = len(self.correct)
        if total < 60:
            return {"delta_50": 0.0, "delta_250": 0.0, "ewma_accuracy": self.ewma_accuracy.value or 0.0}
            
        acc_recent_50 = float(np.mean(list(self.correct)[-50:]))
        acc_recent_250 = float(np.mean(list(self.correct)[-min(250, total):]))
        acc_hist = float(np.mean(self.correct))
        
        return {
            "delta_50": round(acc_recent_50 - acc_hist, 4),
            "delta_250": round(acc_recent_250 - acc_hist, 4),
            "ewma_accuracy": round(self.ewma_accuracy.value or acc_hist, 4)
        }
