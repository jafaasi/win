from typing import List, Dict, Any
import numpy as np

class FeatureHealthMonitor:
    """
    Monitors distributional stability, Z-scores, and drift of all continuous features.
    """

    def __init__(self, history_window: int = 1000):
        self.history_window = history_window
        self.feature_history: Dict[str, List[float]] = {}

    def update(self, feature_dict: Dict[str, float]) -> None:
        for k, v in feature_dict.items():
            if isinstance(v, (int, float)):
                if k not in self.feature_history:
                    self.feature_history[k] = []
                self.feature_history[k].append(float(v))
                if len(self.feature_history[k]) > self.history_window:
                    self.feature_history[k].pop(0)

    def compute_health_report(self, current_features: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        report = {}
        for k, current_val in current_features.items():
            if not isinstance(current_val, (int, float)):
                continue
            hist = self.feature_history.get(k, [])
            if len(hist) < 10:
                report[k] = {
                    "current": float(current_val),
                    "mean": float(current_val),
                    "std": 0.0,
                    "z_score": 0.0,
                    "drift_status": "CALIBRATING"
                }
                continue
                
            mean = float(np.mean(hist))
            std = float(np.std(hist))
            z_score = float((current_val - mean) / (std + 1e-9))
            
            status = "CRITICAL" if abs(z_score) > 3.5 else "MODERATE" if abs(z_score) > 2.0 else "NORMAL"
            
            report[k] = {
                "current": round(float(current_val), 4),
                "mean": round(mean, 4),
                "std": round(std, 4),
                "z_score": round(z_score, 2),
                "drift_status": status
            }
        return report
