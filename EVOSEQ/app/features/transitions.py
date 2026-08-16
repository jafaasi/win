from typing import Sequence
import numpy as np

def transition_matrix(
    sequence: Sequence[int],
    cardinality: int = 10,
) -> np.ndarray:
    """Computes the 1-step empirical transition probability matrix T_ij = P(X_{t+1}=j | X_t=i)."""
    matrix = np.zeros((cardinality, cardinality), dtype=np.float64)
    sequence = list(sequence)
    if len(sequence) < 2:
        return np.full((cardinality, cardinality), 1.0 / cardinality, dtype=np.float64)
        
    for a, b in zip(sequence[:-1], sequence[1:]):
        if 0 <= a < cardinality and 0 <= b < cardinality:
            matrix[a, b] += 1.0
            
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums

def transition_entropy(matrix: np.ndarray) -> float:
    """Computes average row conditional transition entropy: H_T = (1/N) * sum_i H(T_i)."""
    if matrix.size == 0:
        return 0.0
    entropy_values = []
    for row in matrix:
        p = row[row > 0]
        if len(p) == 0:
            entropy_values.append(0.0)
            continue
        entropy_values.append(float(-np.sum(p * np.log2(p))))
    return float(np.mean(entropy_values))
