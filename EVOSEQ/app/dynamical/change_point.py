from typing import Sequence, Dict, Any, Tuple
import numpy as np

class OnlineChangeDetector:
    """
    Online Bayesian Change-Point / Distribution Drift Score:
    Computes Jensen-Shannon divergence between reference and recent historical windows.
    Explicitly separates distribution drift D_t from measured predictive information P_t.
    """

    def __init__(self, reference_size: int = 200, recent_size: int = 40, threshold: float = 0.05):
        self.reference_size = reference_size
        self.recent_size = recent_size
        self.threshold = threshold

    def score(self, values: Sequence[int]) -> float:
        values = np.asarray(values)
        if len(values) < (self.reference_size + self.recent_size):
            return 0.0
            
        reference = values[-(self.reference_size + self.recent_size): -self.recent_size]
        recent = values[-self.recent_size:]
        
        p_ref = np.bincount(reference, minlength=10).astype(np.float64)
        p_recent = np.bincount(recent, minlength=10).astype(np.float64)
        
        sum_ref = p_ref.sum()
        sum_recent = p_recent.sum()
        if sum_ref == 0 or sum_recent == 0:
            return 0.0
            
        p_ref /= sum_ref
        p_recent /= sum_recent
        m = 0.5 * (p_ref + p_recent)
        
        # Masked entropy computation preventing zero division
        mask_ref = (p_ref > 0) & (m > 0)
        mask_recent = (p_recent > 0) & (m > 0)
        
        js_ref = np.sum(p_ref[mask_ref] * np.log2(p_ref[mask_ref] / m[mask_ref])) if np.any(mask_ref) else 0.0
        js_recent = np.sum(p_recent[mask_recent] * np.log2(p_recent[mask_recent] / m[mask_recent])) if np.any(mask_recent) else 0.0
        
        js = 0.5 * js_ref + 0.5 * js_recent
        return float(np.clip(js, 0.0, 1.0))

    def classify_state(self, drift_score: float, info_gain: float) -> str:
        """
        Classifies current regime into one of 4 fundamental states:
        - LOW_DRIFT_LOW_INFO
        - HIGH_DRIFT_LOW_INFO
        - LOW_DRIFT_HIGH_INFO (High investigation priority)
        - HIGH_DRIFT_HIGH_INFO (High evolution priority)
        """
        is_drift = drift_score >= self.threshold
        is_info = info_gain >= 0.04
        
        if is_drift and is_info:
            return "HIGH_DRIFT_HIGH_INFO"
        elif not is_drift and is_info:
            return "LOW_DRIFT_HIGH_INFO"
        elif is_drift and not is_info:
            return "HIGH_DRIFT_LOW_INFO"
        else:
            return "LOW_DRIFT_LOW_INFO"
