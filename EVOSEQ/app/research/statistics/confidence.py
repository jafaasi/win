import numpy as np
from typing import Sequence, Dict, Any, Callable
from ...models.base import SequenceModel
from ...evaluation.walk_forward import evaluate_model_walk_forward

def empirical_p_value(observed: float, null_scores: Sequence[float]) -> float:
    """
    Computes empirical Monte Carlo p-value:
    p = (1 + sum(S_null >= S_obs)) / (N + 1).
    """
    null_arr = np.asarray(null_scores, dtype=np.float64)
    if len(null_arr) == 0:
        return 1.0
    return float((1.0 + np.sum(null_arr >= observed)) / (len(null_arr) + 1.0))

def compare_model_to_null(
    model: SequenceModel,
    observed_sequence: Sequence[int],
    null_generator_fn: Callable[[int, Optional[int]], np.ndarray],
    repetitions: int = 30,
    initial_train_size: int = 40,
) -> Dict[str, Any]:
    """
    Executes the model-vs-null laboratory experiment:
    Evaluates model on observed sequence vs N synthetic null sequences under identical walk-forward protocols.
    """
    obs_eval = evaluate_model_walk_forward(model, observed_sequence, initial_train_size=initial_train_size)
    obs_score = obs_eval["null_advantage"] - (obs_eval["mean_brier_score"] * 5.0)
    
    null_scores = []
    L = len(observed_sequence)
    for seed in range(repetitions):
        synthetic_seq = null_generator_fn(L, seed)
        null_eval = evaluate_model_walk_forward(model, synthetic_seq, initial_train_size=initial_train_size)
        n_score = null_eval["null_advantage"] - (null_eval["mean_brier_score"] * 5.0)
        null_scores.append(n_score)
        
    null_arr = np.asarray(null_scores)
    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr) + 1e-9)
    p_val = empirical_p_value(obs_score, null_scores)
    
    return {
        "observed_score": round(obs_score, 4),
        "null_mean": round(null_mean, 4),
        "null_std": round(null_std, 4),
        "delta_vs_null": round(obs_score - null_mean, 4),
        "z_score": round((obs_score - null_mean) / null_std, 2),
        "p_value": round(p_val, 4),
        "is_significant": (p_val < 0.05)
    }
