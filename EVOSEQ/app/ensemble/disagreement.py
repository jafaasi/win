from typing import Sequence
import numpy as np

def calculate_ensemble_disagreement(probability_vectors: Sequence[np.ndarray]) -> float:
    """
    Computes average pairwise Jensen-Shannon divergence across model population predictions.
    Quantifies epistemic disagreement between competing inductive hypotheses.
    """
    vectors = [np.asarray(p, dtype=np.float64) for p in probability_vectors]
    if len(vectors) < 2:
        return 0.0
        
    M = len(vectors)
    mean_vec = np.mean(vectors, axis=0)
    mean_vec = np.maximum(mean_vec, 1e-12)
    
    js_sum = 0.0
    for p in vectors:
        p_safe = np.maximum(p, 1e-12)
        kl = np.sum(np.where(p_safe > 1e-10, p_safe * np.log2(p_safe / mean_vec), 0.0))
        js_sum += float(kl)
        
    js_divergence = js_sum / M
    return float(np.clip(js_divergence, 0.0, 1.0))
