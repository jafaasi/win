from typing import Sequence, List
import numpy as np
from .drift import js_divergence

def calculate_prediction_uncertainty(probabilities: Sequence[float]) -> float:
    """
    Computes instantaneous Shannon entropy of a model prediction:
    U_t = - sum(p_i * log2(p_i)).
    Higher values = diffuse uncertainty; lower values = concentrated confidence.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))

def calculate_model_disagreement(model_probabilities_list: Sequence[np.ndarray]) -> float:
    """
    Computes average pairwise Jensen-Shannon divergence across a population of model predictions:
    D_{models} = (1 / (N*(N-1))) * sum_{i != j} D_JS(p_i, p_j).
    Low value = models in consensus; High value = models detecting divergent structures.
    """
    probs_list = [np.asarray(p, dtype=np.float64) for p in model_probabilities_list if len(p) > 0]
    N = len(probs_list)
    if N < 2:
        return 0.0
        
    pairwise_js_sum = 0.0
    pair_count = 0
    
    for i in range(N):
        for j in range(i + 1, N):
            js = js_divergence(probs_list[i], probs_list[j])
            pairwise_js_sum += js
            pair_count += 1
            
    if pair_count == 0:
        return 0.0
    return float(pairwise_js_sum / pair_count)
