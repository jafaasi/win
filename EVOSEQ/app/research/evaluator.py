from dataclasses import dataclass
from typing import Sequence, Dict, Any, List, Union
import numpy as np
from .metrics import log_loss, accuracy, brier_score

@dataclass(frozen=True)
class ModelEvaluation:
    """Consolidated out-of-sample evaluation summary across multiple temporal folds."""
    model_name: str
    mean_log_loss: float
    std_log_loss: float
    mean_accuracy: float
    mean_brier: float
    calibration_error: float
    robustness: float
    stability_score: float

def evaluate(
    probabilities: Union[Sequence[np.ndarray], np.ndarray],
    targets: Union[Sequence[int], np.ndarray]
) -> Dict[str, float]:
    """Computes immediate categorical evaluation metrics."""
    return {
        "accuracy": accuracy(probabilities, targets),
        "log_loss": log_loss(probabilities, targets),
        "brier_score": brier_score(probabilities, targets),
    }

def rank_models(results: List[ModelEvaluation]) -> List[ModelEvaluation]:
    """
    Ranks competing models based on out-of-sample log loss and generalization robustness:
    Primary: ascending mean_log_loss
    Secondary: descending robustness
    """
    return sorted(results, key=lambda x: (x.mean_log_loss, -x.robustness))
