import numpy as np
from typing import List, Dict

def to_big_small(val: int) -> int:
    return 1 if val >= 5 else 0

class DataCollector:
    """
    Normalizes incoming outcome data into formats useful for statistical tests and models.
    """
    def __init__(self, raw_sequence: List[int]):
        self.raw_sequence = raw_sequence
        
    def get_integers(self) -> np.ndarray:
        return np.array(self.raw_sequence, dtype=np.int32)
        
    def get_bitwise(self) -> np.ndarray:
        return np.array([to_big_small(x) for x in self.raw_sequence], dtype=np.int8)
        
    def describe(self) -> Dict:
        return {
            "length": len(self.raw_sequence),
            "unique": len(set(self.raw_sequence)),
            "min": min(self.raw_sequence) if self.raw_sequence else 0,
            "max": max(self.raw_sequence) if self.raw_sequence else 0
        }
