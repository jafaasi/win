import numpy as np
from typing import Sequence, List, Optional, Tuple

def block_bootstrap(
    sequence: Sequence[int],
    block_size: int = 16,
    samples: int = 50,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Generates B resampled surrogate sequences via Moving Block Bootstrap.
    Preserves temporal dependency within blocks for honest uncertainty quantification.
    """
    rng = np.random.default_rng(seed)
    seq = np.asarray(sequence)
    if len(seq) <= block_size:
        return [seq.copy() for _ in range(samples)]
        
    blocks = [seq[i:i + block_size] for i in range(0, len(seq) - block_size + 1)]
    num_blocks_needed = int(np.ceil(len(seq) / float(block_size)))
    
    result = []
    for _ in range(samples):
        selected_indices = rng.integers(0, len(blocks), size=num_blocks_needed)
        sample = np.concatenate([blocks[idx] for idx in selected_indices])[:len(seq)]
        result.append(sample)
        
    return result

def compute_bootstrap_ci(
    scores: Sequence[float],
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """Computes median, lower percentile, and upper percentile confidence interval."""
    arr = np.asarray(scores, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    lower = float(np.percentile(arr, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
    median = float(np.median(arr))
    return round(median, 4), round(lower, 4), round(upper, 4)
