from typing import Sequence
import numpy as np

def autocorrelation(values: Sequence[float], lag: int = 1) -> float:
    """
    Computes sample autocorrelation at a specific lag:
    rho_k = sum((x_t - mu)*(x_{t-k} - mu)) / sum((x_t - mu)^2).
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= lag:
        return 0.0
        
    x = values[:-lag]
    y = values[lag:]
    
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    
    denominator = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
    if denominator == 0:
        return 0.0
        
    corr = float(np.sum(x_centered * y_centered) / denominator)
    return float(np.clip(corr, -1.0, 1.0))
