from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np

@dataclass
class EnvironmentState:
    """Canonical statistical environment state."""
    entropy: float
    conditional_entropy_1: float
    conditional_entropy_2: float
    information_gain_1: float
    information_gain_2: float
    autocorrelation_1: float
    autocorrelation_2: float
    autocorrelation_3: float
    lz_zscore: float
    drift_score: float
    model_disagreement: float
    regime_entropy: float

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.entropy,
            self.conditional_entropy_1,
            self.conditional_entropy_2,
            self.information_gain_1,
            self.information_gain_2,
            self.autocorrelation_1,
            self.autocorrelation_2,
            self.autocorrelation_3,
            self.lz_zscore,
            self.drift_score,
            self.model_disagreement,
            self.regime_entropy
        ], dtype=np.float64)

    def to_dict(self) -> Dict[str, float]:
        return {
            "entropy": self.entropy,
            "conditional_entropy_1": self.conditional_entropy_1,
            "conditional_entropy_2": self.conditional_entropy_2,
            "information_gain_1": self.information_gain_1,
            "information_gain_2": self.information_gain_2,
            "autocorrelation_1": self.autocorrelation_1,
            "autocorrelation_2": self.autocorrelation_2,
            "autocorrelation_3": self.autocorrelation_3,
            "lz_zscore": self.lz_zscore,
            "drift_score": self.drift_score,
            "model_disagreement": self.model_disagreement,
            "regime_entropy": self.regime_entropy
        }

@dataclass
class ModelDescriptor:
    """Descriptor vector for architecture candidates."""
    family: str
    context_length: int
    parameter_count: int
    hidden_size: int = 0
    layers: int = 0
    heads: int = 0
    reservoir_size: int = 0
    order: int = 0

    def to_vector(self) -> np.ndarray:
        fam_idx = {"Uniform": 0, "Markov": 1, "DiscreteHMM": 2, "EchoStateNetwork": 3, "Transformer": 4, "S4": 5, "Mamba": 6}.get(self.family, 0)
        return np.array([
            fam_idx,
            self.context_length,
            np.log1p(self.parameter_count),
            self.hidden_size,
            self.layers,
            self.heads,
            self.reservoir_size,
            self.order
        ], dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "context_length": self.context_length,
            "parameter_count": self.parameter_count,
            "hidden_size": self.hidden_size,
            "layers": self.layers,
            "heads": self.heads,
            "reservoir_size": self.reservoir_size,
            "order": self.order
        }

@dataclass
class ParetoPoint:
    """Candidate representation for Pareto multi-objective optimization."""
    candidate_id: str
    descriptor: ModelDescriptor
    log_loss: float
    brier_score: float
    calibration_error: float
    complexity: float
    latency: float
    robustness: float
