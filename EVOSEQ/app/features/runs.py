from typing import Sequence, List, Tuple, Dict, Any
import numpy as np

def run_lengths(values: Sequence[Any]) -> List[Tuple[Any, int]]:
    """Extracts run lengths (symbol, length) from an arbitrary sequence."""
    values = list(values)
    if not values:
        return []
    runs = []
    current = values[0]
    length = 1
    for value in values[1:]:
        if value == current:
            length += 1
        else:
            runs.append((current, length))
            current = value
            length = 1
    runs.append((current, length))
    return runs

def run_statistics(values: Sequence[Any]) -> Dict[str, float]:
    """Computes max streak length, mean run length, and alternation frequency."""
    runs = run_lengths(values)
    if not runs:
        return {"max_run": 0.0, "mean_run": 0.0, "total_runs": 0.0, "alternation_rate": 0.0}
    lengths = [r[1] for r in runs]
    alternation_rate = float(len(runs) - 1) / float(max(1, len(values) - 1))
    return {
        "max_run": float(max(lengths)),
        "mean_run": float(np.mean(lengths)),
        "total_runs": float(len(runs)),
        "alternation_rate": round(alternation_rate, 4)
    }
