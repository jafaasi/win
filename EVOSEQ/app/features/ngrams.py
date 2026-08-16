from collections import Counter
from typing import Sequence, Dict, Tuple, Any

def ngram_counts(values: Sequence[Any], n: int) -> Counter:
    """Extracts raw frequency counts of n-grams from a sequence."""
    values = list(values)
    if len(values) < n:
        return Counter()
    return Counter(
        tuple(values[i:i+n])
        for i in range(len(values) - n + 1)
    )

def normalized_ngram_counts(values: Sequence[Any], n: int) -> Dict[Tuple[Any, ...], float]:
    """Extracts relative frequency probability distribution of n-grams."""
    counts = ngram_counts(values, n)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {
        key: value / total
        for key, value in counts.items()
    }
