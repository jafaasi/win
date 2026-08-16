import numpy as np
from typing import Sequence, Callable, Dict, Any, Optional

def feature_permutation_test(
    features: np.ndarray,
    targets: Sequence[int],
    eval_fn: Callable[[np.ndarray, Sequence[int]], float],
    repetitions: int = 50,
    seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Tests feature significance against null hypothesis H_0: Feature F_t is independent of target y_t.
    Permutes target vector relative to feature matrix to construct empirical null score distribution.
    """
    rng = np.random.default_rng(seed)
    targets_arr = np.asarray(targets)
    
    observed_score = eval_fn(features, targets_arr)
    null_scores = []
    
    for _ in range(repetitions):
        shuffled_targets = targets_arr.copy()
        rng.shuffle(shuffled_targets)
        null_s = eval_fn(features, shuffled_targets)
        null_scores.append(null_s)
        
    null_arr = np.asarray(null_scores)
    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr) + 1e-9)
    p_value = float((1 + np.sum(null_arr >= observed_score)) / (len(null_arr) + 1))
    
    return {
        "observed_score": round(observed_score, 4),
        "null_mean": round(null_mean, 4),
        "null_std": round(null_std, 4),
        "p_value": round(p_value, 4),
        "z_score": round((observed_score - null_mean) / null_std, 2)
    }
