import numpy as np
import math
import logging
from typing import List, Dict, Tuple
from collections import deque

# Assuming the models are wrapped in the BaseModel interface from app.models.baseline
from app.models.baseline import UniformModel, FrequencyModel, MarkovModel, BitwiseModel

logger = logging.getLogger(__name__)

class EnhancedAdaptiveRNGPredictor:
    def __init__(self, output_space_size: int = 10, threshold: float = 0.03):
        self.N = output_space_size
        self.threshold = threshold
        
        # Initialize the multi-layered ensemble
        self.models = [
            UniformModel(self.N),
            FrequencyModel(self.N),
            MarkovModel(order=1, num_classes=self.N),
            MarkovModel(order=2, num_classes=self.N),
            MarkovModel(order=3, num_classes=self.N),
            BitwiseModel(),
            # Placeholder for Deep Models which run asynchronously
            # TransformerModel(),
            # SSMModel()
        ]
        
        # Start with equal weights
        self.weights = np.ones(len(self.models)) / len(self.models)
        self.alert = "Awaiting calibration"
        
        # Enhanced tracking for adaptive weighting
        self.performance_history = deque(maxlen=50)  # Track recent performance
        self.model_accuracies = np.zeros(len(self.models))
        self.model_confidences = np.ones(len(self.models)) * 0.5
        self.disagreement_score = 0.0
        self.regime_state = "UNKNOWN"
        
        # Adaptive parameters
        self.learning_rate = 0.1
        self.decay_factor = 0.95
        self.volatility_window = 20

    def softmax_negative(self, scores: List[float]) -> np.ndarray:
        # Lower score (loss) is better.
        # We negate the scores to apply softmax, weighting better models higher
        s = -np.array(scores)
        # Numerical stability shift
        s -= np.max(s)
        exp_s = np.exp(s)
        return exp_s / np.sum(exp_s)

    def softmax_temperature(self, scores: List[float], temperature: float = 1.0) -> np.ndarray:
        """Apply temperature scaling for sharper/softer probability distributions"""
        s = -np.array(scores) / temperature
        s -= np.max(s)
        exp_s = np.exp(s)
        return exp_s / np.sum(exp_s)

    def uniform_baseline_loss(self, sequence: np.ndarray) -> float:
        return math.log(self.N)

    def calculate_disagreement(self, recent_values: np.ndarray) -> float:
        """Calculate how much models disagree with each other"""
        if len(recent_values) < 2:
            return 0.0
            
        predictions = []
        for model in self.models:
            probs = model.predict_proba(recent_values)
            predictions.append(probs)
        
        # Calculate pairwise disagreement using Jensen-Shannon divergence
        disagreements = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                js_div = self.jensen_shannon_divergence(predictions[i], predictions[j])
                disagreements.append(js_div)
        
        return np.mean(disagreements) if disagreements else 0.0

    def jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence between two probability distributions"""
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        m = 0.5 * (p + q)
        return 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))

    def detect_regime(self, sequence: np.ndarray) -> str:
        """Detect current regime based on sequence characteristics"""
        if len(sequence) < 10:
            return "UNKNOWN"
            
        recent = sequence[-min(50, len(sequence)):]
        
        # Calculate basic statistics
        big_ratio = np.mean(recent >= 5)
        volatility = np.std(recent)
        
        # Detect momentum regime
        if big_ratio > 0.65:
            return "BIG_MOMENTUM"
        elif big_ratio < 0.35:
            return "SMALL_MOMENTUM"
        elif volatility < 2.0:
            return "LOW_VOLATILITY"
        elif volatility > 3.5:
            return "HIGH_VOLATILITY"
        else:
            return "EQUILIBRIUM"

    def adaptive_weighting(self, losses: List[float], recent_values: np.ndarray) -> np.ndarray:
        """Enhanced adaptive weighting with regime awareness"""
        base_weights = self.softmax_temperature(losses, temperature=0.5)
        
        # Apply regime-specific adjustments
        regime = self.detect_regime(recent_values)
        self.regime_state = regime
        
        # Boost weights for models that perform well in current regime
        if regime == "BIG_MOMENTUM":
            # Favor frequency and higher-order Markov models
            base_weights[1] *= 1.3  # FrequencyModel
            base_weights[4] *= 1.2  # Markov order 3
        elif regime == "SMALL_MOMENTUM":
            base_weights[1] *= 1.3
            base_weights[4] *= 1.2
        elif regime == "HIGH_VOLATILITY":
            # Favor simpler models that adapt quickly
            base_weights[2] *= 1.2  # Markov order 1
            base_weights[5] *= 1.1  # BitwiseModel
        elif regime == "LOW_VOLATILITY":
            # Favor complex models that capture subtle patterns
            base_weights[3] *= 1.3  # Markov order 2
            base_weights[4] *= 1.2
        
        # Normalize weights
        base_weights = base_weights / np.sum(base_weights)
        
        # Apply disagreement-based uncertainty adjustment
        disagreement = self.calculate_disagreement(recent_values)
        self.disagreement_score = disagreement
        
        if disagreement > 0.15:  # High disagreement - be more conservative
            # Boost uniform model weight when models disagree strongly
            base_weights[0] *= 1.5
            base_weights = base_weights / np.sum(base_weights)
        
        return base_weights

    def update_daily(self, new_sequence: np.ndarray, stats_engine=None):
        """
        Enhanced daily evolution logic with adaptive weighting and regime detection.
        """
        if len(new_sequence) < 3:
            return

        # 1. Update simple online models
        for model in self.models:
            model.partial_fit(new_sequence)

        # 2. Evaluate on latest held-out slice with multiple windows
        scores = []
        recent_values = new_sequence[-min(100, len(new_sequence)):]
        
        for model in self.models:
            # Evaluate on multiple time windows for robustness
            window_losses = []
            for window_size in [20, 50, min(100, len(new_sequence))]:
                if len(new_sequence) >= window_size:
                    loss = model.evaluate(new_sequence[-window_size:], val_window=min(10, window_size // 3))
                    window_losses.append(loss)
            
            # Use weighted average of window losses
            if window_losses:
                avg_loss = np.mean(window_losses)
                scores.append(avg_loss)
            else:
                scores.append(model.evaluate(new_sequence))

        # 3. Apply enhanced adaptive weighting
        self.weights = self.adaptive_weighting(scores, recent_values)

        # 4. Track model performance
        baseline_loss = self.uniform_baseline_loss(new_sequence)
        best_loss = min(scores)
        best_model_idx = np.argmin(scores)
        
        # Update accuracy tracking
        self.model_accuracies = self.decay_factor * self.model_accuracies
        self.model_accuracies[best_model_idx] += (1 - self.decay_factor)

        # 5. Enhanced exploitability detection
        advantage = baseline_loss - best_loss
        if advantage > self.threshold:
            self.alert = f"STRUCTURAL PREDICTABILITY DETECTED (Advantage: {advantage:.4f})"
            logger.warning(f"RNG ANOMALY: Model {best_model_idx} loss {best_loss:.4f} beat baseline {baseline_loss:.4f}")
        else:
            self.alert = "UNIFORM RANDOMNESS (No exploitable pattern)"
            # Partial collapse to uniform, but keep some diversity
            uniform_weight = 0.7
            diversity_weight = 0.3 / (len(self.models) - 1)
            self.weights = np.full(len(self.models), diversity_weight)
            self.weights[0] = uniform_weight

        # 6. Update performance history
        self.performance_history.append({
            'best_loss': best_loss,
            'baseline_loss': baseline_loss,
            'advantage': advantage,
            'regime': self.regime_state,
            'disagreement': self.disagreement_score
        })

    def predict_next(self, recent_values: np.ndarray) -> np.ndarray:
        """
        Returns weighted probability distribution with confidence calibration.
        """
        probs = np.zeros(self.N)
        
        for idx, model in enumerate(self.models):
            w = self.weights[idx]
            if w > 0.001:
                model_probs = model.predict_proba(recent_values)
                probs += w * model_probs
        
        # Apply confidence calibration based on disagreement
        if self.disagreement_score > 0.1:
            # Soften predictions when uncertain
            probs = probs * 0.7 + np.ones(self.N) / self.N * 0.3
                
        # Normalize
        probs /= np.sum(probs)
        return probs

    def get_state(self) -> Dict:
        return {
            "alert": self.alert,
            "weights": self.weights.tolist(),
            "models": [m.__class__.__name__ for m in self.models],
            "regime": self.regime_state,
            "disagreement": round(self.disagreement_score, 4),
            "model_accuracies": self.model_accuracies.tolist(),
            "performance_samples": len(self.performance_history)
        }

# Maintain backward compatibility
AdaptiveRNGPredictor = EnhancedAdaptiveRNGPredictor
