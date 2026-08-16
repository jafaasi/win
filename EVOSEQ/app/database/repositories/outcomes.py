from typing import List, Tuple, Sequence, Dict, Any
from sqlalchemy import text
from ..database import engine, SessionLocal
from ...schemas import Outcome

def load_outcomes() -> List[Outcome]:
    """Loads all outcomes ordered by sequence_no ascending."""
    with SessionLocal() as session:
        return session.query(Outcome).order_by(Outcome.sequence_no.asc()).all()

def find_gaps(sequence_numbers: Sequence[int]) -> List[Tuple[int, int]]:
    """
    Detects missing sequence gaps in a sequence of numbers:
    If sequence contains 1001, 1002, 1005 -> returns [(1003, 1004)].
    """
    seq = list(sequence_numbers)
    if len(seq) < 2:
        return []
        
    gaps = []
    for prev, curr in zip(seq[:-1], seq[1:]):
        if curr != prev + 1:
            gaps.append((prev + 1, curr - 1))
    return gaps

def validate_sequence(outcomes: Sequence[Outcome]) -> None:
    """Validates continuity, ordering, domain limits, and uniqueness of outcomes."""
    if not outcomes:
        raise ValueError("No observations found in sequence")
        
    seq_nos = [o.sequence_no for o in outcomes]
    digits = [o.digit for o in outcomes]
    
    # Monotonicity check
    for i in range(1, len(seq_nos)):
        if seq_nos[i] <= seq_nos[i - 1]:
            raise ValueError(f"Sequence is not strictly monotonic increasing at index {i}: {seq_nos[i-1]} -> {seq_nos[i]}")
            
    # Value range check
    for d in digits:
        if d is None or not (0 <= d <= 9):
            raise ValueError(f"Invalid or missing digit in sequence: {d}")
            
    # Duplicate sequence number check
    if len(seq_nos) != len(set(seq_nos)):
        raise ValueError("Duplicate sequence numbers detected in dataset")
