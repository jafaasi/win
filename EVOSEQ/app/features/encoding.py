import numpy as np

def one_hot(value: int, size: int) -> np.ndarray:
    """Generates a 1-hot categorical vector of given dimension."""
    result = np.zeros(size, dtype=np.float32)
    if 0 <= value < size:
        result[value] = 1.0
    return result

def encode_outcome(
    digit: int,
    size: int,
    color: int,
    parity: int,
) -> np.ndarray:
    """
    Concatenates categorical representations:
    [digit (10), size (2), color (3), parity (2)] -> 17-dimensional base vector.
    """
    return np.concatenate([
        one_hot(digit, 10),
        one_hot(size, 2),
        one_hot(color, 3),
        one_hot(parity, 2),
    ]).astype(np.float32)
