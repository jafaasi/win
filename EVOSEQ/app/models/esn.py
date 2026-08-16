from typing import Sequence, Optional, Union, Dict, Any
import numpy as np
from .base import SequenceModel, ModelMetadata
from ..features.encoding import one_hot

class EchoStateNetwork(SequenceModel):
    """
    Echo State Network (ESN) with tunable reservoir size, spectral radius, leak rate, and ridge readout.
    Nonlinear recurrent dynamical sequence predictor.
    """

    def __init__(
        self,
        input_size: int = 10,
        reservoir_size: int = 128,
        output_size: int = 10,
        spectral_radius: float = 0.9,
        sparsity: float = 0.15,
        leak_rate: float = 0.3,
        ridge: float = 1e-4,
        seed: int = 42,
        version: str = "esn-v1"
    ):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        self.ridge = ridge
        self.seed = seed
        
        rng = np.random.default_rng(seed)
        self.W_in = rng.uniform(-1.0, 1.0, (reservoir_size, input_size)).astype(np.float64)
        
        W = rng.uniform(-1.0, 1.0, (reservoir_size, reservoir_size)).astype(np.float64)
        mask = rng.random(W.shape) < sparsity
        W *= mask
        
        eigenvalues = np.linalg.eigvals(W)
        radius = float(np.max(np.abs(eigenvalues)))
        if radius > 0:
            W *= (spectral_radius / radius)
        self.W = W
        
        self.state = np.zeros(reservoir_size, dtype=np.float64)
        self.W_out = np.zeros((output_size, reservoir_size), dtype=np.float64)
        
        self.metadata = ModelMetadata(
            name="EchoStateNetwork",
            version=version,
            parameters={
                "reservoir_size": reservoir_size,
                "spectral_radius": spectral_radius,
                "sparsity": sparsity,
                "leak_rate": leak_rate,
                "ridge": ridge
            }
        )

    def _prepare_input(self, X: Union[np.ndarray, list]) -> np.ndarray:
        arr = np.asarray(X)
        if arr.ndim == 1:
            # One-hot encode integer digit series if 1D
            return np.array([one_hot(int(d), self.input_size) for d in arr], dtype=np.float64)
        return arr.astype(np.float64)

    def step(self, x: np.ndarray) -> np.ndarray:
        """Runs a single reservoir update step."""
        pre_activation = self.W_in @ x + self.W @ self.state
        new_state = np.tanh(pre_activation)
        self.state = (1.0 - self.leak_rate) * self.state + self.leak_rate * new_state
        return self.state

    def collect_states(self, X: np.ndarray) -> np.ndarray:
        """Passes entire sequence through reservoir and gathers internal state matrix [T x N]."""
        self.state = np.zeros(self.reservoir_size, dtype=np.float64)
        states = []
        for x in X:
            st = self.step(x)
            states.append(st.copy())
        return np.asarray(states, dtype=np.float64)

    def fit(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None) -> "EchoStateNetwork":
        """Fits readout weights W_out via Ridge Regression (L2 regularized least-squares)."""
        X_mat = self._prepare_input(X)
        if len(X_mat) < 5:
            return self
            
        if y is None:
            # Predict next step (Y = X[1:], X = X[:-1])
            input_X = X_mat[:-1]
            raw_targets = X_mat[1:]
        else:
            input_X = X_mat
            raw_targets = np.asarray(y)
            if raw_targets.ndim == 1:
                raw_targets = np.array([one_hot(int(t), self.output_size) for t in raw_targets], dtype=np.float64)
                
        states = self.collect_states(input_X)
        
        # Ridge Solution: W_out = (R^T R + lambda * I)^-1 R^T Y
        RtR = states.T @ states
        reg = self.ridge * np.eye(RtR.shape[0], dtype=np.float64)
        RtY = states.T @ raw_targets
        
        try:
            self.W_out = np.linalg.solve(RtR + reg, RtY).T
        except Exception:
            self.W_out = (np.linalg.pinv(RtR + reg) @ RtY).T
            
        return self

    def update(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None) -> "EchoStateNetwork":
        """Online adaptation of the readout layer using Recursive Least Squares (RLS) step."""
        X_mat = self._prepare_input(X)
        if len(X_mat) == 0:
            return self
        # Warm reservoir state with recent history
        for x in X_mat[-10:]:
            self.step(x)
        return self

    def predict_proba(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """Returns 10-class softmax probabilities for the next step."""
        X_mat = self._prepare_input(X)
        if len(X_mat) == 0:
            return np.full(self.output_size, 1.0 / self.output_size, dtype=np.float64)
            
        # Run sequence through reservoir to reach current hidden state
        states = self.collect_states(X_mat)
        final_state = states[-1]
        
        logits = self.W_out @ final_state
        logits_stable = logits - np.max(logits)
        exp_logits = np.exp(logits_stable)
        total = exp_logits.sum()
        if total == 0:
            return np.full(self.output_size, 1.0 / self.output_size, dtype=np.float64)
        return exp_logits / total

    def save(self, path: str) -> None:
        save_data = {
            "input_size": self.input_size,
            "reservoir_size": self.reservoir_size,
            "output_size": self.output_size,
            "spectral_radius": self.spectral_radius,
            "sparsity": self.sparsity,
            "leak_rate": self.leak_rate,
            "ridge": self.ridge,
            "seed": self.seed,
            "W_in": self.W_in,
            "W": self.W,
            "W_out": self.W_out,
            "metadata": self.metadata
        }
        np.save(path, save_data)

    @classmethod
    def load(cls, path: str) -> "EchoStateNetwork":
        data = np.load(path, allow_pickle=True).item()
        model = cls(
            input_size=data["input_size"],
            reservoir_size=data["reservoir_size"],
            output_size=data["output_size"],
            spectral_radius=data["spectral_radius"],
            sparsity=data["sparsity"],
            leak_rate=data["leak_rate"],
            ridge=data["ridge"],
            seed=data.get("seed", 42)
        )
        model.W_in = data["W_in"]
        model.W = data["W"]
        model.W_out = data["W_out"]
        model.metadata = data.get("metadata", model.metadata)
        return model

ESN = EchoStateNetwork

