from typing import Sequence, Tuple
import numpy as np

def one_hot(value: int, cardinality: int) -> np.ndarray:
    """Generates 1D float32 one-hot encoding array."""
    res = np.zeros(cardinality, dtype=np.float32)
    if 0 <= value < cardinality:
        res[value] = 1.0
    return res

def encode_observation(digit: int, size: int, color: int, parity: int) -> np.ndarray:
    """
    Encodes single observation into a 17-dimensional compact categorical vector:
    - digit: 10 dimensions
    - size: 2 dimensions
    - color: 3 dimensions
    - parity: 2 dimensions
    Total: 10 + 2 + 3 + 2 = 17 dimensions.
    """
    return np.concatenate([
        one_hot(digit, 10),
        one_hot(size, 2),
        one_hot(color, 3),
        one_hot(parity, 2)
    ])

def make_windows(
    feature_matrix: Sequence[np.ndarray],
    digits: Sequence[int],
    context_length: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs strictly causal temporal training windows:
    X_t = [x_{t-L}, ..., x_{t-1}]
    y_t = x_t
    Zero future observation leakage is mathematically guaranteed.
    """
    X_list = []
    y_list = []
    features_arr = np.asarray(feature_matrix)
    digits_arr = np.asarray(digits)
    
    for i in range(context_length, len(features_arr)):
        X_list.append(features_arr[i - context_length: i])
        y_list.append(digits_arr[i])
        
    if not X_list:
        return np.empty((0, context_length, features_arr.shape[-1] if len(features_arr) > 0 else 17), dtype=np.float32), np.empty((0,), dtype=np.int64)
        
    return np.asarray(X_list, dtype=np.float32), np.asarray(y_list, dtype=np.int64)
