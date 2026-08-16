import numpy as np
from typing import Sequence, Optional

def block_shuffle(
    sequence: Sequence[int],
    block_size: int = 16,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Randomly reorders contiguous blocks of length B.
    Preserves short-range autocorrelation and n-gram structure while destroying long-range dependencies.
    Tests Null Hypothesis C: Block-Shuffled Temporal Structure.
    """
    rng = np.random.default_rng(seed)
    seq = np.asarray(sequence)
    if len(seq) <= block_size:
        return seq.copy()
        
    blocks = [seq[i:i + block_size] for i in range(0, len(seq), block_size)]
    perm = rng.permutation(len(blocks))
    shuffled_blocks = [blocks[idx] for idx in perm]
    return np.concatenate(shuffled_blocks)

def marginal_preserving_shuffle(
    sequence: Sequence[int],
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Full random permutation preserving exact marginal token frequencies.
    Tests Null Hypothesis B: Memoryless Marginal-Preserving Shuffle.
    """
    rng = np.random.default_rng(seed)
    seq = np.asarray(sequence).copy()
    rng.shuffle(seq)
    return seq
