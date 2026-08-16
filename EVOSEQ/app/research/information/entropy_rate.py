from typing import Sequence, Dict, List
import numpy as np
from ...features.entropy import categorical_entropy
from ...features.conditional_entropy import conditional_entropy

def calculate_entropy_rate_profile(sequence: Sequence[int], max_order: int = 5, cardinality: int = 10) -> Dict[str, float]:
    """
    Estimates the conditional entropy rate curve H_k = H(X_t | X_{t-k:t-1}) for k in [1, max_order].
    If H_k plateaus early, additional history provides diminishing predictive information.
    """
    sequence = list(sequence)
    h_marg = categorical_entropy(sequence, cardinality=cardinality)
    
    profile = {"H_0_marginal": round(h_marg, 4)}
    for k in range(1, max_order + 1):
        if len(sequence) > k + 5:
            hk = conditional_entropy(sequence, order=k, cardinality=cardinality)
            profile[f"H_{k}_conditional"] = round(hk, 4)
        else:
            profile[f"H_{k}_conditional"] = round(h_marg, 4)
            
    return profile

def information_gain_curve(sequence: Sequence[int], max_order: int = 5, cardinality: int = 10) -> Dict[str, float]:
    """
    Estimates Information Gain curve IG_k = H(X_t) - H(X_t | X_{t-k:t-1}).
    Quantifies effective memory horizon of the discrete sequence stream.
    """
    sequence = list(sequence)
    h_marg = categorical_entropy(sequence, cardinality=cardinality)
    
    curve = {}
    for k in range(1, max_order + 1):
        if len(sequence) > k + 5:
            hk = conditional_entropy(sequence, order=k, cardinality=cardinality)
            ig = max(0.0, h_marg - hk)
            curve[f"IG_{k}"] = round(ig, 4)
        else:
            curve[f"IG_{k}"] = 0.0
            
    return curve
