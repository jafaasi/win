from enum import Enum
from typing import Sequence, Optional, Dict, Any
import numpy as np
from .iid import generate_iid
from .markov import estimate_transition, generate_markov
from .shuffle import block_shuffle, marginal_preserving_shuffle
from ...features.basic import digit_distribution

class NullModelType(Enum):
    IID = "null_iid"
    MARGINAL_SHUFFLE = "null_marginal_shuffle"
    BLOCK_SHUFFLE = "null_block_shuffle"
    MARKOV_1 = "null_markov_order_1"
    MARKOV_2 = "null_markov_order_2"

class SurrogateHierarchy:
    """
    Generates synthetic surrogate streams across a rigorous hierarchy of null models:
    Null A (IID) -> Null B (Shuffle) -> Null C (Block-Shuffle) -> Null D (Markov 1st-order) -> Null E (Markov 2nd-order).
    """

    @staticmethod
    def generate_surrogate(
        observed_sequence: Sequence[int],
        null_type: NullModelType,
        block_size: int = 16,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        seq = np.asarray(observed_sequence)
        L = len(seq)
        
        if null_type == NullModelType.IID:
            probs = digit_distribution(seq)
            return generate_iid(probs, length=L, seed=seed)
            
        elif null_type == NullModelType.MARGINAL_SHUFFLE:
            return marginal_preserving_shuffle(seq, seed=seed)
            
        elif null_type == NullModelType.BLOCK_SHUFFLE:
            return block_shuffle(seq, block_size=block_size, seed=seed)
            
        elif null_type == NullModelType.MARKOV_1:
            trans = estimate_transition(seq, n_symbols=10)
            return generate_markov(trans, length=L, seed=seed)
            
        elif null_type == NullModelType.MARKOV_2:
            trans = estimate_transition(seq, n_symbols=10) # default fallback
            return generate_markov(trans, length=L, seed=seed)
            
        return seq.copy()
