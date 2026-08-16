from typing import Sequence, Dict, List, Set, Tuple
import numpy as np

def symbolic_words(sequence: Sequence[int], k: int = 3) -> List[Tuple[int, ...]]:
    """Extracts sliding symbolic words w_t = (x_t, x_{t+1}, ..., x_{t+k-1})."""
    seq = list(sequence)
    if len(seq) < k:
        return []
    return [tuple(seq[i:i + k]) for i in range(len(seq) - k + 1)]

def symbolic_complexity_curve(sequence: Sequence[int], max_k: int = 4, alphabet_size: int = 10) -> Dict[str, float]:
    """
    Computes normalized symbolic word complexity growth:
    C_k = N_k / min(L - k + 1, alphabet_size^k).
    Deterministic or low-dimensional systems show rapid decay in C_k compared to memoryless streams.
    """
    seq = list(sequence)
    L = len(seq)
    curve = {}
    
    for k in range(1, max_k + 1):
        if L < k:
            curve[f"C_{k}"] = 0.0
            continue
            
        words = symbolic_words(seq, k=k)
        unique_words = len(set(words))
        max_possible = min(len(words), alphabet_size ** k)
        c_k = float(unique_words / max(1, max_possible))
        curve[f"C_{k}"] = round(c_k, 4)
        
    return curve
