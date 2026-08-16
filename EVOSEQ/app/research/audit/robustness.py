from typing import Sequence, Dict, Any, List
import numpy as np
from ...models.base import SequenceModel
from ...evaluation.walk_forward import evaluate_model_walk_forward
from ..null_models.surrogate import SurrogateHierarchy, NullModelType

def evaluate_model_robustness(
    model: SequenceModel,
    observed_sequence: Sequence[int],
    initial_train_size: int = 30
) -> Dict[str, Any]:
    """
    Evaluates candidate model across a battery of adversarial and surrogate environments:
    - Observed sequence
    - IID null surrogate
    - 1st-Order Markov surrogate
    - Block-Shuffle surrogate
    - High-Entropy vs Low-Entropy partitions
    Computes composite Model Robustness Score R_m.
    """
    seq = list(observed_sequence)
    L = len(seq)
    if L < 40:
        return {"robustness_score": 0.0, "status": "INSUFFICIENT_DATA", "environment_scores": {}}
        
    env_scores = {}
    
    # 1. Observed environment
    eval_obs = evaluate_model_walk_forward(model, seq, initial_train_size=initial_train_size)
    env_scores["observed"] = eval_obs["null_advantage"] - (eval_obs["mean_brier_score"] * 5.0)
    
    # 2. IID Null environment
    iid_seq = SurrogateHierarchy.generate_surrogate(seq, NullModelType.IID, seed=42)
    eval_iid = evaluate_model_walk_forward(model, iid_seq, initial_train_size=initial_train_size)
    env_scores["iid_null"] = eval_iid["null_advantage"] - (eval_iid["mean_brier_score"] * 5.0)
    
    # 3. Markov Null environment
    mkv_seq = SurrogateHierarchy.generate_surrogate(seq, NullModelType.MARKOV_1, seed=42)
    eval_mkv = evaluate_model_walk_forward(model, mkv_seq, initial_train_size=initial_train_size)
    env_scores["markov_null"] = eval_mkv["null_advantage"] - (eval_mkv["mean_brier_score"] * 5.0)
    
    # 4. Block-Shuffle environment
    blk_seq = SurrogateHierarchy.generate_surrogate(seq, NullModelType.BLOCK_SHUFFLE, block_size=16, seed=42)
    eval_blk = evaluate_model_walk_forward(model, blk_seq, initial_train_size=initial_train_size)
    env_scores["block_shuffle"] = eval_blk["null_advantage"] - (eval_blk["mean_brier_score"] * 5.0)
    
    # 5. Segment partitions (Recent vs Historical)
    half = L // 2
    eval_recent = evaluate_model_walk_forward(model, seq[half:], initial_train_size=max(15, half // 3))
    env_scores["recent_half"] = eval_recent["null_advantage"] - (eval_recent["mean_brier_score"] * 5.0)
    
    scores_list = list(env_scores.values())
    mean_score = float(np.mean(scores_list))
    std_score = float(np.std(scores_list))
    
    # R_m = Mean - Std (rewards high consistent performance, penalizes variance / fragility)
    robustness_score = round(mean_score - (0.5 * std_score), 4)
    status = "HIGH" if robustness_score > -0.25 else "MODERATE" if robustness_score > -0.45 else "FRAGILE"
    
    return {
        "robustness_score": robustness_score,
        "mean_score": round(mean_score, 4),
        "score_std": round(std_score, 4),
        "status": status,
        "environment_scores": {k: round(v, 4) for k, v in env_scores.items()}
    }
