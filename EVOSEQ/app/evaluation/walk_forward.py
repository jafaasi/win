import numpy as np
from typing import Sequence, List, Dict, Any, Tuple
from ..models.base import SequenceModel
from .metrics import log_loss, brier_score, entropy, calibration_error, calculate_null_advantage

def walk_forward(
    model: SequenceModel,
    sequence: Sequence[int],
    initial_train_size: int,
) -> List[Dict[str, Any]]:
    """
    Executes a leakage-free walk-forward validation on a sequence stream.
    Trains on sequence[:initial_train_size], then tests 1-step ahead incrementally.
    """
    sequence = list(sequence)
    predictions = []
    if len(sequence) <= initial_train_size:
        return predictions

    model.fit(sequence[:initial_train_size])
    
    for i in range(initial_train_size, len(sequence)):
        context = sequence[:i]
        probabilities = model.predict_proba(context)
        actual = sequence[i]
        
        loss_val = log_loss(probabilities, actual)
        brier_val = brier_score(probabilities, actual)
        ent_val = entropy(probabilities)
        pred_digit = int(np.argmax(probabilities))
        
        predictions.append({
            "index": i,
            "probabilities": probabilities.tolist(),
            "predicted_digit": pred_digit,
            "actual_digit": actual,
            "is_correct": (pred_digit == actual),
            "log_loss": loss_val,
            "brier_score": brier_val,
            "entropy": ent_val,
        })
        
        # Incremental online adaptation
        model.update(sequence[i-1:i+1])
        
    return predictions

def evaluate_model_walk_forward(
    model: SequenceModel,
    sequence: Sequence[int],
    initial_train_size: int = 500
) -> Dict[str, float]:
    """Computes aggregated evaluation metrics over the walk-forward evaluation run."""
    preds = walk_forward(model, sequence, initial_train_size)
    if not preds:
        return {
            "accuracy": 0.10,
            "mean_log_loss": 2.302,
            "mean_brier_score": 0.09,
            "calibration_error": 0.0,
            "null_advantage": 0.0,
            "observations": 0
        }
        
    actuals = [p["actual_digit"] for p in preds]
    probs = [np.array(p["probabilities"]) for p in preds]
    corrects = [p["is_correct"] for p in preds]
    
    acc = float(np.mean(corrects))
    mean_ll = float(np.mean([p["log_loss"] for p in preds]))
    mean_brier = float(np.mean([p["brier_score"] for p in preds]))
    ece = calibration_error(probs, actuals)
    null_adv = calculate_null_advantage(acc, null_accuracy=0.10)
    
    return {
        "accuracy": round(acc, 4),
        "mean_log_loss": round(mean_ll, 4),
        "mean_brier_score": round(mean_brier, 4),
        "calibration_error": round(ece, 4),
        "null_advantage": round(null_adv, 4),
        "observations": len(preds)
    }
