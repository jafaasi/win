from typing import Dict, Any, Union
import numpy as np

def recurrence_matrix(X: np.ndarray, epsilon: float = 0.5) -> np.ndarray:
    """
    Computes unthresholded / binary recurrence plot matrix:
    R_{ij} = 1[ ||x_i - x_j|| <= epsilon ]
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]
        
    distances = np.linalg.norm(X_arr[:, None, :] - X_arr[None, :, :], axis=-1)
    return (distances <= epsilon).astype(np.uint8)

def recurrence_quantification_analysis(R: np.ndarray, min_diag_len: int = 2) -> Dict[str, float]:
    """
    Computes standard Recurrence Quantification Analysis (RQA) diagnostics:
    - Recurrence Rate (RR): density of recurrence points
    - Determinism (DET): percentage of recurrence points forming diagonal lines
    - Average Diagonal Length (L)
    - Laminarity (LAM): percentage of recurrence points forming vertical structures
    """
    N = R.shape[0]
    if N <= 1:
        return {"recurrence_rate": 0.0, "determinism": 0.0, "avg_diagonal_len": 0.0, "laminarity": 0.0}
        
    # 1. Recurrence Rate (excluding main diagonal)
    total_points = N * (N - 1)
    rec_points = int(np.sum(R) - N) # subtract identity line
    rr = float(rec_points / max(1, total_points))
    
    # 2. Diagonal lines length extraction
    diag_lengths = []
    for k in range(1, N):
        diag = np.diagonal(R, offset=k)
        # count contiguous 1s
        length = 0
        for val in diag:
            if val == 1:
                length += 1
            else:
                if length >= min_diag_len:
                    diag_lengths.append(length)
                length = 0
        if length >= min_diag_len:
            diag_lengths.append(length)
            
    diag_points_in_lines = sum(diag_lengths) * 2 # symmetric
    det = float(diag_points_in_lines / max(1, rec_points)) if rec_points > 0 else 0.0
    avg_l = float(np.mean(diag_lengths)) if diag_lengths else 0.0
    
    # 3. Vertical lines length extraction (Laminarity)
    vert_lengths = []
    for col in range(N):
        length = 0
        for row in range(N):
            if row != col and R[row, col] == 1:
                length += 1
            else:
                if length >= min_diag_len:
                    vert_lengths.append(length)
                length = 0
        if length >= min_diag_len:
            vert_lengths.append(length)
            
    lam_points = sum(vert_lengths)
    lam = float(lam_points / max(1, rec_points)) if rec_points > 0 else 0.0
    
    return {
        "recurrence_rate": round(rr, 4),
        "determinism": round(min(1.0, det), 4),
        "avg_diagonal_len": round(avg_l, 2),
        "laminarity": round(min(1.0, lam), 4)
    }
