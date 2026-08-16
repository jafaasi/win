from dataclasses import dataclass
from typing import Sequence
import numpy as np
from ..features.basic import digit_distribution

@dataclass
class DriftResult:
    is_significant: bool
    js_divergence: float
    level: str # 'LOW', 'MODERATE', 'CRITICAL'

def calculate_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Calculates Jensen-Shannon Divergence between two probability distributions."""
    def kl_divergence(dist1, dist2):
        mask = dist1 > 0
        return np.sum(dist1[mask] * np.log2(dist1[mask] / (dist2[mask] + 1e-15)))
        
    m = 0.5 * (p + q)
    js = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return float(np.clip(js, 0.0, 1.0))

def calculate_drift(
    history: Sequence[int],
    recent_window: int = 500,
    historical_window: int = 2500,
    threshold: float = 0.08
) -> DriftResult:
    """
    Compares recent probability distribution against historical window using JS Divergence.
    """
    history = list(history)
    if len(history) < recent_window * 2:
        return DriftResult(is_significant=False, js_divergence=0.0, level="LOW")
        
    recent_slice = history[-recent_window:]
    hist_slice = history[-min(len(history), historical_window + recent_window):-recent_window]
    
    p_recent = digit_distribution(recent_slice)
    p_hist = digit_distribution(hist_slice)
    
    jsd = calculate_js_divergence(p_recent, p_hist)
    is_sig = jsd >= threshold
    level = "CRITICAL" if jsd >= 0.18 else "MODERATE" if jsd >= 0.08 else "LOW"
    
    return DriftResult(
        is_significant=is_sig,
        js_divergence=round(jsd, 4),
        level=level
    )
