from typing import Dict, List, Sequence, Any, Tuple
import numpy as np
from ..research.metrics import log_loss
from ..core.mapping import map_digit
from ..features.vector import encode_observation

FEATURE_VERSIONS = {
    "v1": "digits (10-dim one-hot)",
    "v2": "digits + parity (12-dim)",
    "v3": "digits + parity + size (14-dim)",
    "v4": "full causal categorical: digits + size + color + parity (17-dim)",
    "v5": "v4 + online entropy + information gain (19-dim)",
    "v6": "v5 + autocorrelation + recurrence statistics (21-dim)",
    "v7": "v6 + lempel-ziv complexity indicator (22-dim)"
}

class FeatureAblationTester:
    """
    Evaluates marginal information contribution of feature subsets:
    Delta L_f = Loss(without feature f) - Loss(with full feature set)
    
    Positive Delta L: Feature provides genuine predictive reduction in log-loss.
    Zero / Negative Delta L: Feature is redundant and risks overfitting.
    """

    def __init__(self):
        self.feature_groups = ["digit_one_hot", "size_one_hot", "color_one_hot", "parity_one_hot", "temporal_entropy", "recurrence_rate"]

    def ablate_features(
        self,
        full_feature_matrix: np.ndarray,
        targets: Sequence[int],
        eval_fn: Any
    ) -> Dict[str, Dict[str, float]]:
        """
        Runs ablation test across feature dimensions.
        """
        y = np.asarray(targets, dtype=np.int64)
        full_preds = eval_fn(full_feature_matrix)
        full_loss = float(log_loss(full_preds, y))
        
        results = {}
        # Group slicing indices
        group_slices = {
            "size": slice(10, 12),
            "color": slice(12, 15),
            "parity": slice(15, 17)
        }
        
        for name, slc in group_slices.items():
            ablated_matrix = full_feature_matrix.copy()
            ablated_matrix[..., slc] = 0.0 # Zero out feature group
            ablated_preds = eval_fn(ablated_matrix)
            ablated_loss = float(log_loss(ablated_preds, y))
            delta = float(ablated_loss - full_loss)
            
            status = "INFORMATIVE" if delta > 1e-4 else ("REDUNDANT" if abs(delta) <= 1e-4 else "NOISY")
            results[name] = {
                "delta_loss": round(delta, 5),
                "ablated_loss": round(ablated_loss, 4),
                "full_loss": round(full_loss, 4),
                "status": status
            }
            
        return results
