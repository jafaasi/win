import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import math

class DriftDetector:
    """
    Advanced drift detection for sequence predictability systems.
    Detects concept drift, distribution changes, and regime shifts.
    """
    
    def __init__(self, window_size: int = 100, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.reference_window = deque(maxlen=window_size)
        self.current_window = deque(maxlen=window_size)
        self.drift_history = deque(maxlen=50)
        self.regime_history = deque(maxlen=20)
        self.current_regime = "STABLE"
        self.drift_score = 0.0
        self.confidence_adjustment = 1.0
        
    def add_sample(self, value: int) -> None:
        """Add a new sample to the current window."""
        self.current_window.append(value)
        
    def set_reference(self, reference_sequence: List[int]) -> None:
        """Set the reference window from a sequence."""
        self.reference_window.clear()
        for val in reference_sequence[-self.window_size:]:
            self.reference_window.append(val)
    
    def calculate_distribution_stats(self, window: deque) -> Dict:
        """Calculate statistical properties of a window."""
        if len(window) < 2:
            return {
                "mean": 0.0, "std": 0.0, "skewness": 0.0, 
                "big_ratio": 0.5, "entropy": 0.0
            }
        
        arr = np.array(list(window))
        big_ratio = np.mean(arr >= 5)
        
        # Calculate entropy
        counts = np.bincount(arr, minlength=10)
        probs = counts / len(arr)
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)
        
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "skewness": float(self._calculate_skewness(arr)),
            "big_ratio": float(big_ratio),
            "entropy": float(entropy)
        }
    
    def _calculate_skewness(self, arr: np.ndarray) -> float:
        """Calculate skewness of the distribution."""
        if len(arr) < 3:
            return 0.0
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0.0
        n = len(arr)
        skewness = (n / ((n-1) * (n-2))) * np.sum(((arr - mean) / std) ** 3)
        return float(skewness)
    
    def population_stability_index(self, ref_stats: Dict, curr_stats: Dict) -> float:
        """Calculate PSI between reference and current distributions."""
        psi = 0.0
        
        # Compare key statistics
        for key in ["big_ratio", "entropy"]:
            ref_val = ref_stats[key]
            curr_val = curr_stats[key]
            
            # Avoid division by zero
            if ref_val == 0:
                ref_val = 0.001
            if curr_val == 0:
                curr_val = 0.001
            
            # PSI calculation
            psi += (curr_val - ref_val) * math.log(curr_val / ref_val)
        
        return abs(psi)
    
    def detect_drift(self) -> Tuple[bool, float, str]:
        """
        Detect if drift has occurred.
        Returns: (is_drift, drift_score, drift_type)
        """
        if len(self.reference_window) < self.window_size // 2 or len(self.current_window) < self.window_size // 2:
            return False, 0.0, "INSUFFICIENT_DATA"
        
        ref_stats = self.calculate_distribution_stats(self.reference_window)
        curr_stats = self.calculate_distribution_stats(self.current_window)
        
        # Calculate drift score using multiple metrics
        psi = self.population_stability_index(ref_stats, curr_stats)
        
        # Mean shift detection
        mean_shift = abs(ref_stats["mean"] - curr_stats["mean"]) / (ref_stats["std"] + 0.1)
        
        # Big ratio shift
        big_ratio_shift = abs(ref_stats["big_ratio"] - curr_stats["big_ratio"])
        
        # Entropy shift
        entropy_shift = abs(ref_stats["entropy"] - curr_stats["entropy"])
        
        # Combined drift score
        self.drift_score = 0.4 * psi + 0.3 * mean_shift + 0.2 * big_ratio_shift + 0.1 * entropy_shift
        
        # Determine drift type
        drift_type = "NONE"
        if self.drift_score > self.threshold:
            if big_ratio_shift > 0.15:
                drift_type = "DISTRIBUTION_SHIFT"
            elif mean_shift > 1.0:
                drift_type = "MEAN_SHIFT"
            elif entropy_shift > 0.3:
                drift_type = "ENTROPY_CHANGE"
            else:
                drift_type = "GENERAL_DRIFT"
        
        is_drift = self.drift_score > self.threshold
        
        # Update drift history
        self.drift_history.append({
            "score": self.drift_score,
            "type": drift_type,
            "is_drift": is_drift
        })
        
        return is_drift, self.drift_score, drift_type
    
    def detect_regime(self, sequence: List[int]) -> str:
        """
        Detect the current regime based on sequence characteristics.
        """
        if len(sequence) < 20:
            return "INSUFFICIENT_DATA"
        
        recent = sequence[-min(50, len(sequence)):]
        arr = np.array(recent)
        
        # Calculate regime indicators
        big_ratio = np.mean(arr >= 5)
        volatility = np.std(arr)
        entropy = self.calculate_distribution_stats(deque(recent))["entropy"]
        
        # Momentum detection
        if len(recent) >= 10:
            recent_big = np.mean(arr[-10:] >= 5)
            earlier_big = np.mean(arr[-20:-10] >= 5) if len(arr) >= 20 else recent_big
            momentum = recent_big - earlier_big
        else:
            momentum = 0.0
        
        # Regime classification
        if big_ratio > 0.70:
            regime = "STRONG_BIG_MOMENTUM"
        elif big_ratio < 0.30:
            regime = "STRONG_SMALL_MOMENTUM"
        elif big_ratio > 0.55:
            regime = "MODERATE_BIG_BIAS"
        elif big_ratio < 0.45:
            regime = "MODERATE_SMALL_BIAS"
        elif volatility > 3.0:
            regime = "HIGH_VOLATILITY"
        elif volatility < 1.5:
            regime = "LOW_VOLATILITY"
        elif momentum > 0.15:
            regime = "BIG_ACCELERATION"
        elif momentum < -0.15:
            regime = "SMALL_ACCELERATION"
        elif entropy < 2.8:
            regime = "LOW_ENTROPY_PATTERN"
        elif entropy > 3.25:
            regime = "HIGH_ENTROPY_RANDOM"
        else:
            regime = "EQUILIBRIUM"
        
        # Track regime changes
        if regime != self.current_regime:
            self.regime_history.append({
                "from": self.current_regime,
                "to": regime,
                "drift_score": self.drift_score
            })
            self.current_regime = regime
        
        return regime
    
    def get_confidence_adjustment(self) -> float:
        """
        Get confidence adjustment factor based on recent drift.
        Lower confidence during high drift periods.
        """
        if len(self.drift_history) < 5:
            return 1.0
        
        recent_drifts = [d["score"] for d in list(self.drift_history)[-10:]]
        avg_drift = np.mean(recent_drifts)
        
        # Reduce confidence during high drift
        if avg_drift > self.threshold * 2:
            self.confidence_adjustment = 0.85
        elif avg_drift > self.threshold:
            self.confidence_adjustment = 0.92
        else:
            self.confidence_adjustment = 1.0
        
        return self.confidence_adjustment
    
    def update_reference_if_needed(self) -> bool:
        """
        Update reference window if significant drift has occurred and stabilized.
        Returns True if reference was updated.
        """
        if len(self.drift_history) < 10:
            return False
        
        recent_drifts = [d["is_drift"] for d in list(self.drift_history)[-5:]]
        
        # If drift has stopped and we have stable recent data
        if not any(recent_drifts) and len(self.current_window) >= self.window_size:
            self.reference_window = deque(list(self.current_window))
            return True
        
        return False
    
    def get_state(self) -> Dict:
        """Get current state of the drift detector."""
        return {
            "current_regime": self.current_regime,
            "drift_score": round(self.drift_score, 4),
            "confidence_adjustment": round(self.confidence_adjustment, 3),
            "regime_history": list(self.regime_history)[-5:],
            "recent_drifts": list(self.drift_history)[-5:],
            "reference_window_size": len(self.reference_window),
            "current_window_size": len(self.current_window)
        }
    
    def reset(self) -> None:
        """Reset the drift detector."""
        self.reference_window.clear()
        self.current_window.clear()
        self.drift_history.clear()
        self.regime_history.clear()
        self.current_regime = "STABLE"
        self.drift_score = 0.0
        self.confidence_adjustment = 1.0