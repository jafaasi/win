from typing import Dict, Sequence, Union, Any, Tuple, Optional
import numpy as np

def entropy_bits(probs: np.ndarray) -> float:
    """Computes Shannon entropy in bits."""
    p = np.asarray(probs, dtype=np.float64)
    mask = p > 0
    if not np.any(mask):
        return 0.0
    return float(-np.sum(p[mask] * np.log2(p[mask])))

def decompose_uncertainty(
    model_predictions: Sequence[np.ndarray],
    weights: Optional[Sequence[float]] = None
) -> Dict[str, float]:
    """
    Decomposes ensemble uncertainty into:
    1. Aleatoric / Individual entropy: mean_H = sum_m w_m H(p_m)
    2. Combined predictive entropy: total_H = H(p_ensemble)
    3. Population Disagreement (JS divergence): D = total_H - mean_H
    """
    preds = [np.asarray(p, dtype=np.float64) for p in model_predictions]
    M = len(preds)
    if M == 0:
        return {"aleatoric_entropy": 0.0, "total_entropy": 0.0, "disagreement": 0.0}
        
    if weights is None:
        w = np.full(M, 1.0 / M)
    else:
        w_arr = np.asarray(weights, dtype=np.float64)
        w = w_arr / max(w_arr.sum(), 1e-12)
        
    individual_entropies = [entropy_bits(p) for p in preds]
    mean_aleatoric = float(np.sum([w[i] * individual_entropies[i] for i in range(M)]))
    
    # Combined ensemble distribution
    combined = np.sum([w[i] * preds[i] for i in range(M)], axis=0)
    total_entropy = entropy_bits(combined)
    
    disagreement = max(0.0, total_entropy - mean_aleatoric)
    
    return {
        "aleatoric_entropy": round(mean_aleatoric, 4),
        "total_entropy": round(total_entropy, 4),
        "disagreement": round(disagreement, 4)
    }

def evolution_pressure(
    loss_degradation: float,
    drift: float,
    disagreement: float,
    calibration_error: float
) -> float:
    """
    Calculates evolutionary pressure:
    Pressure = 0.40 * degradation + 0.25 * drift + 0.20 * disagreement + 0.15 * calibration_error
    """
    p = (
        0.40 * max(0.0, float(loss_degradation)) +
        0.25 * max(0.0, float(drift)) +
        0.20 * max(0.0, float(disagreement)) +
        0.15 * max(0.0, float(calibration_error))
    )
    return float(p)

def evolution_state(
    pressure: float,
    threshold1: float = 0.25,
    threshold2: float = 0.55
) -> str:
    """Classifies controller state: STABLE, INVESTIGATE, or EVOLVE."""
    if pressure < threshold1:
        return "STABLE"
    elif pressure < threshold2:
        return "INVESTIGATE"
    else:
        return "EVOLVE"
