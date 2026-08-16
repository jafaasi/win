from typing import Sequence
from .entropy import categorical_entropy
from .conditional_entropy import conditional_entropy

def information_gain(
    sequence: Sequence[int],
    order: int = 1,
    cardinality: int = 10,
) -> float:
    """
    Computes Information Gain (Mutual Information with past order-K context):
    I(X_t; X_{t-1:t-order}) = H(X_t) - H(X_t | X_{t-1:t-order}).
    A value significantly greater than 0 indicates exploitable memory in the sequence.
    """
    h_current = categorical_entropy(sequence, cardinality=cardinality)
    h_conditional = conditional_entropy(sequence, order=order, cardinality=cardinality)
    return max(0.0, float(h_current - h_conditional))
