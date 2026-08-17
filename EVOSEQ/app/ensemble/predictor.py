import numpy as np
import math
import logging
from typing import List, Dict

# Assuming the models are wrapped in the BaseModel interface from app.models.baseline
from app.models.baseline import UniformModel, FrequencyModel, MarkovModel, BitwiseModel

logger = logging.getLogger(__name__)

class AdaptiveRNGPredictor:
    def __init__(self, output_space_size: int = 10, threshold: float = 0.05):
        self.N = output_space_size
        self.threshold = threshold
        
        # Initialize the multi-layered ensemble
        self.models = [
            UniformModel(self.N),
            FrequencyModel(self.N),
            MarkovModel(order=1, num_classes=self.N),
            MarkovModel(order=2, num_classes=self.N),
            BitwiseModel(),
            # Placeholder for Deep Models which run asynchronously
            # TransformerModel(),
            # SSMModel()
        ]
        
        # Start with equal weights
        self.weights = np.ones(len(self.models)) / len(self.models)
        self.alert = "Awaiting calibration"

    def softmax_negative(self, scores: List[float]) -> np.ndarray:
        # Lower score (loss) is better.
        # We negate the scores to apply softmax, weighting better models higher
        s = -np.array(scores)
        # Numerical stability shift
        s -= np.max(s)
        exp_s = np.exp(s)
        return exp_s / np.sum(exp_s)

    def uniform_baseline_loss(self, sequence: np.ndarray) -> float:
        return math.log(self.N)

    def update_daily(self, new_sequence: np.ndarray, stats_engine=None):
        """
        Runs the daily evolution logic based on new data.
        """
        if len(new_sequence) < 3:
            return

        # 1. Update simple online models
        for model in self.models:
            model.partial_fit(new_sequence)

        # 2. Evaluate on latest held-out slice (e.g. the sequence itself in streaming mode)
        # Note: In strict mode, we hold out a validation block
        scores = []
        for model in self.models:
            loss = model.evaluate(new_sequence)
            scores.append(loss)

        # 3. Convert losses to weights
        self.weights = self.softmax_negative(scores)

        # 4. Detect exploitability vs Baseline
        baseline_loss = self.uniform_baseline_loss(new_sequence)
        best_loss = min(scores)

        if best_loss < (baseline_loss - self.threshold):
            self.alert = "Possible structural predictability detected"
            logger.warning(f"RNG ANOMALY: Model loss {best_loss:.4f} beat baseline {baseline_loss:.4f}")
        else:
            self.alert = "No significant predictability (Uniform Randomness)"
            # Collapse weights to uniform baseline if no signal
            self.weights = np.zeros(len(self.models))
            self.weights[0] = 1.0 # Index 0 is UniformModel

    def predict_next(self, recent_values: np.ndarray) -> np.ndarray:
        """
        Returns a single weighted probability distribution over the N classes.
        """
        probs = np.zeros(self.N)
        
        for idx, model in enumerate(self.models):
            w = self.weights[idx]
            if w > 0.001:
                model_probs = model.predict_proba(recent_values)
                probs += w * model_probs
                
        # Normalize just in case
        probs /= np.sum(probs)
        return probs

    def get_state(self) -> Dict:
        return {
            "alert": self.alert,
            "weights": self.weights.tolist(),
            "models": [m.__class__.__name__ for m in self.models]
        }
