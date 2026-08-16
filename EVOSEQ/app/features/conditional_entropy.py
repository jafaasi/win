from collections import defaultdict
from typing import Sequence
import numpy as np

def conditional_entropy(
    sequence: Sequence[int],
    order: int = 1,
    cardinality: int = 10,
) -> float:
    """
    Computes order-K conditional entropy:
    H(X_t | X_{t-1:t-order}) = sum_c P(context=c) * H(X_t | context=c).
    """
    sequence = list(sequence)
    if len(sequence) <= order:
        return 0.0
        
    contexts = defaultdict(list)
    for i in range(order, len(sequence)):
        context = tuple(sequence[i-order:i])
        contexts[context].append(sequence[i])
        
    total = sum(len(v) for v in contexts.values())
    if total == 0:
        return 0.0
        
    result = 0.0
    for values in contexts.values():
        counts = np.bincount(values, minlength=cardinality)
        total_ctx = counts.sum()
        if total_ctx == 0:
            continue
        probabilities = counts / float(total_ctx)
        probabilities = probabilities[probabilities > 0]
        h = -np.sum(probabilities * np.log2(probabilities))
        weight = len(values) / float(total)
        result += weight * h
        
    return float(result)
