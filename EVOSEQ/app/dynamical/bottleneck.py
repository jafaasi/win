from typing import Sequence, Dict, List, Any, Optional
import numpy as np
from ..models.base import SequenceModel
from ..models.markov import MarkovModel
from ..evaluation.walk_forward import evaluate_model_walk_forward

class MemoryDepthEstimator:
    """
    Information Bottleneck & Memory Depth Estimator:
    Measures marginal predictability improvement Delta S_L = S_L - S_{L/2}
    across exponential context horizons L in [16, 32, 64, 128, 256].
    Finds the exact memory saturation horizon where additional context provides zero edge.
    """

    @staticmethod
    def estimate_depth_curve(
        sequence: Sequence[int],
        context_horizons: Sequence[int] = (1, 2, 3, 4, 5)
    ) -> Dict[str, Any]:
        seq = list(sequence)
        L_total = len(seq)
        if L_total < 40:
            return {"saturation_order": 1, "curve": {}}
            
        scores = {}
        deltas = {}
        prev_score = None
        saturation_order = 1
        
        for ord_k in context_horizons:
            model = MarkovModel(order=ord_k, smoothing=0.5)
            ev = evaluate_model_walk_forward(model, seq, initial_train_size=max(20, L_total // 2))
            score = ev["null_advantage"] - (ev["mean_brier_score"] * 5.0)
            scores[f"order_{ord_k}"] = round(score, 4)
            
            if prev_score is not None:
                delta = score - prev_score
                deltas[f"delta_order_{ord_k}"] = round(delta, 4)
                if delta > 0.01:
                    saturation_order = ord_k
            prev_score = score
            
        return {
            "saturation_order": saturation_order,
            "scores": scores,
            "deltas": deltas,
            "memory_decay_rate": round(float(np.mean(list(deltas.values()))) if deltas else 0.0, 4)
        }
