from typing import List, Sequence, Any
import numpy as np

def audit_dataframe(df: Any) -> List[str]:
    """
    Comprehensive data quality audit for historical outcome sequence:
    - Checks non-empty dataset
    - Checks duplicate sequence numbers
    - Checks strict monotonic sequence ordering
    - Checks valid digit bounds [0, 9]
    - Checks sequence gaps (b != a + 1)
    """
    errors = []
    if df is None or len(df) == 0:
        errors.append("Dataset is empty")
        return errors
        
    # Extract sequence numbers & digits
    if hasattr(df, "sequence_no"):
        seq_nos = df.sequence_no.to_numpy()
        digits = df.digit.to_numpy()
    else:
        # Dictionary / list fallback
        seq_nos = np.array([row["sequence_no"] for row in df])
        digits = np.array([row["digit"] for row in df])
        
    # 1. Duplicates
    if len(seq_nos) != len(np.unique(seq_nos)):
        errors.append("Duplicate sequence numbers detected")
        
    # 2. Strict Monotonicity
    if np.any(np.diff(seq_nos) <= 0):
        errors.append("Sequence is not strictly monotonic increasing")
        
    # 3. Digit Bounds
    if np.any(digits < 0) or np.any(digits > 9):
        errors.append("Invalid digit detected (must be between 0 and 9)")
        
    # 4. Gap Detection
    diffs = np.diff(seq_nos)
    gap_count = int(np.sum(diffs > 1))
    if gap_count > 0:
        errors.append(f"Sequence gaps detected: {gap_count}")
        
    return errors
