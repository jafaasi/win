from typing import Sequence, Dict, Any, Optional
import numpy as np
from ...features.lz import lz_complexity
from ..null_models.iid import generate_iid
from ...features.basic import digit_distribution

def lz_null_z_score(sequence: Sequence[int], repetitions: int = 30, seed: Optional[int] = None) -> Dict[str, float]:
    """
    Computes normalized LZ complexity relative to empirical IID null distribution:
    Z = (C_obs - mu_null) / sigma_null.
    """
    sequence = list(sequence)
    c_obs = float(lz_complexity(sequence))
    
    probs = digit_distribution(sequence)
    L = len(sequence)
    
    null_complexities = []
    for s in range(repetitions):
        curr_seed = (seed + s) if seed is not None else None
        syn_seq = generate_iid(probs, length=L, seed=curr_seed)
        c_null = float(lz_complexity(syn_seq))
        null_complexities.append(c_null)
        
    null_arr = np.asarray(null_complexities)
    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr) + 1e-9)
    z_score = float((c_obs - null_mean) / null_std)
    
    return {
        "lz_observed": c_obs,
        "null_mean": round(null_mean, 2),
        "null_std": round(null_std, 2),
        "z_score": round(z_score, 2),
        "algorithmic_compression_ratio": round(c_obs / max(1.0, null_mean), 4)
    }
