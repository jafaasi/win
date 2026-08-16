from typing import Sequence, List, Tuple
import numpy as np

def bonferroni_correction(p_values: Sequence[float], alpha: float = 0.05) -> Tuple[List[bool], float]:
    """
    Applies Bonferroni multiple-testing family-wise error rate control:
    alpha' = alpha / m.
    """
    m = len(p_values)
    if m == 0:
        return [], alpha
    adjusted_alpha = alpha / float(m)
    significant_flags = [float(p) < adjusted_alpha for p in p_values]
    return significant_flags, adjusted_alpha

def benjamini_hochberg_fdr(p_values: Sequence[float], q: float = 0.05) -> List[bool]:
    """
    Applies Benjamini-Hochberg False Discovery Rate (FDR) control:
    Finds largest k where p_(k) <= (k/m) * q.
    """
    p_arr = np.asarray(p_values, dtype=np.float64)
    m = len(p_arr)
    if m == 0:
        return []
        
    sorted_indices = np.argsort(p_arr)
    sorted_p = p_arr[sorted_indices]
    
    thresholds = (np.arange(1, m + 1) / float(m)) * q
    below_thresh = np.where(sorted_p <= thresholds)[0]
    
    significant = np.zeros(m, dtype=bool)
    if len(below_thresh) > 0:
        max_k = below_thresh[-1]
        significant[sorted_indices[:max_k + 1]] = True
        
    return significant.tolist()
