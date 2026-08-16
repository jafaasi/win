from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Fold:
    """Represents a strictly sequential (Train -> Validation -> Test) temporal fold."""
    train_end: int
    validation_end: int
    test_end: int

def create_folds(
    n: int,
    initial_train: int = 1000,
    validation_size: int = 200,
    test_size: int = 100,
    step: int = 100
) -> List[Fold]:
    """
    Constructs walk-forward expanding window folds across n total observations:
    Fold 1: Train (0..T1) -> Val (T1..T1+V) -> Test (T1+V..T1+V+Te)
    Fold 2: Train (0..T1+step) -> Val (T1+step..T1+step+V) -> Test (...)
    """
    folds = []
    train_end = initial_train
    while (train_end + validation_size + test_size) <= n:
        val_end = train_end + validation_size
        test_end = val_end + test_size
        folds.append(Fold(train_end=train_end, validation_end=val_end, test_end=test_end))
        train_end += step
    return folds
