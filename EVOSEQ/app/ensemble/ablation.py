from typing import Dict, List, Sequence, Any
import numpy as np
from ..research.metrics import log_loss
from .tracker import combine_predictions

def compute_model_contributions(
    predictions_by_model: Dict[str, np.ndarray],
    targets: Sequence[int]
) -> Dict[str, Dict[str, float]]:
    """
    Ablation Contribution Test for Model Ecosystem:
    Contribution_m = Loss(Ensemble without m) - Loss(Full Ensemble)
    
    Positive Contribution: Model provides complementary out-of-sample edge.
    Zero / Negative Contribution: Model is redundant or degrades ensemble.
    """
    model_names = list(predictions_by_model.keys())
    M = len(model_names)
    if M <= 1:
        return {k: {"contribution": 0.0, "status": "SOLE_MODEL"} for k in model_names}
        
    y = np.asarray(targets, dtype=np.int64)
    all_preds = [predictions_by_model[k] for k in model_names]
    
    # 1. Full Ensemble Loss
    w_full = np.full(M, 1.0 / M)
    full_combined = combine_predictions(all_preds, w_full)
    full_loss = log_loss(full_combined, y)
    
    contributions = {}
    for i, name in enumerate(model_names):
        # 2. Ablated Ensemble (without model i)
        ablated_preds = [predictions_by_model[k] for j, k in enumerate(model_names) if j != i]
        w_ablated = np.full(M - 1, 1.0 / (M - 1))
        ablated_combined = combine_predictions(ablated_preds, w_ablated)
        ablated_loss = log_loss(ablated_combined, y)
        
        delta = float(ablated_loss - full_loss)
        
        status = "VALUABLE" if delta > 1e-4 else ("REDUNDANT" if abs(delta) <= 1e-4 else "HARMFUL")
        contributions[name] = {
            "contribution": round(delta, 5),
            "ablated_loss": round(float(ablated_loss), 4),
            "full_loss": round(float(full_loss), 4),
            "status": status
        }
        
    return contributions
