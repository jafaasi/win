from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class SequenceFeatures:
    """Canonical feature representation contract for all downstream sequence models."""
    digit: int
    size: int
    color: int
    parity: int
    entropy_digit: float
    entropy_size: float
    entropy_color: float
    entropy_parity: float
    conditional_entropy_1: float
    conditional_entropy_2: float
    conditional_entropy_3: float
    information_gain_1: float
    information_gain_2: float
    information_gain_3: float
    run_digit: int
    run_size: int
    run_color: int
    run_parity: int
    autocorrelation_1: float
    autocorrelation_2: float
    autocorrelation_3: float
    transition_entropy: float
    lz_complexity: float
    vector: np.ndarray
