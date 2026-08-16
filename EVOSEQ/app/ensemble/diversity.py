from typing import Sequence, List, Union
import numpy as np

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Computes Jensen-Shannon divergence JS(p, q) in bits."""
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    
    # Simplex normalization
    p_norm = p_arr / max(p_arr.sum(), 1e-12)
    q_norm = q_arr / max(q_arr.sum(), 1e-12)
    m = 0.5 * (p_norm + q_norm)
    
    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
        
    return 0.5 * _kl(p_norm, m) + 0.5 * _kl(q_norm, m)

def pairwise_diversity_matrix(predictions: Sequence[np.ndarray]) -> np.ndarray:
    """
    Constructs M x M pairwise Jensen-Shannon diversity matrix:
    D_{i, j} = JS(P_i, P_j)
    """
    M = len(predictions)
    matrix = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        for j in range(i + 1, M):
            d = jensen_shannon_divergence(predictions[i], predictions[j])
            matrix[i, j] = d
            matrix[j, i] = d
    return matrix

def model_diversity_score(model_idx: int, predictions: Sequence[np.ndarray]) -> float:
    """
    Calculates complementary diversity score for model m vs rest of population:
    D_m = 1 / (M - 1) * sum_{j != m} JS(p_m, p_j)
    """
    M = len(predictions)
    if M <= 1:
        return 0.0
    p_m = predictions[model_idx]
    div_sum = sum(jensen_shannon_divergence(p_m, predictions[j]) for j in range(M) if j != model_idx)
    return float(div_sum / (M - 1))

def diversity_adjusted_weights(
    losses: Sequence[float],
    predictions: Sequence[np.ndarray],
    alpha: float = 1.0,
    gamma: float = 0.1,
    beta: float = 2.0
) -> np.ndarray:
    """
    Computes diversity-adjusted ensemble weights:
    Score_m = -alpha * L_m + gamma * D_m
    w_m = softmax(beta * Score_m)
    """
    M = len(losses)
    if M == 0:
        return np.empty(0)
    if M == 1:
        return np.array([1.0], dtype=np.float64)
        
    scores = []
    for m in range(M):
        d_m = model_diversity_score(m, predictions)
        score = -alpha * float(losses[m]) + gamma * d_m
        scores.append(score)
        
    scores_arr = np.asarray(scores, dtype=np.float64)
    logits = beta * scores_arr
    logits -= np.max(logits)
    w = np.exp(logits)
    return w / np.sum(w)
