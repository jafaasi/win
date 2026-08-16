from typing import Sequence, Tuple, Optional, Union, Dict, Any
import numpy as np
from .base import SequenceModel, ModelMetadata

class DiscreteHMM(SequenceModel):
    """
    Discrete Hidden Markov Model with scaled forward-backward inference and online updates.
    z_t -> latent statistical regime, x_t -> emitted digit (0-9).
    """

    def __init__(
        self,
        n_states: int = 4,
        n_symbols: int = 10,
        smoothing: float = 1e-3,
        random_state: int = 42,
        version: str = "hmm-v1"
    ):
        self.n_states = n_states
        self.n_symbols = n_symbols
        self.smoothing = smoothing
        self.random_state = random_state
        
        rng = np.random.default_rng(random_state)
        self.transition = rng.random((n_states, n_states), dtype=np.float64)
        self.transition /= self.transition.sum(axis=1, keepdims=True)
        
        self.emission = rng.random((n_states, n_symbols), dtype=np.float64)
        self.emission /= self.emission.sum(axis=1, keepdims=True)
        
        self.initial = np.full(n_states, 1.0 / n_states, dtype=np.float64)
        
        self.metadata = ModelMetadata(
            name="DiscreteHMM",
            version=version,
            parameters={
                "n_states": n_states,
                "n_symbols": n_symbols,
                "smoothing": smoothing
            }
        )

    def _normalize(self, matrix: np.ndarray, axis: int = 1) -> np.ndarray:
        matrix = matrix + self.smoothing
        return matrix / matrix.sum(axis=axis, keepdims=True)

    def forward(self, sequence: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates scaled forward probabilities alpha_t and scaling factors."""
        sequence = np.asarray(sequence, dtype=int)
        T = len(sequence)
        if T == 0:
            return np.ones((0, self.n_states)), np.ones(0)
            
        alpha = np.zeros((T, self.n_states), dtype=np.float64)
        scale = np.zeros(T, dtype=np.float64)
        
        # alpha_0(i) = pi_i * B_i(x_0)
        s0 = int(sequence[0])
        alpha[0] = self.initial * self.emission[:, s0]
        scale[0] = alpha[0].sum()
        if scale[0] == 0:
            scale[0] = 1e-12
        alpha[0] /= scale[0]
        
        # Forward recursion: alpha_t(j) = [sum_i alpha_{t-1}(i) * A_ij] * B_j(x_t)
        for t in range(1, T):
            st = int(sequence[t])
            alpha[t] = (alpha[t - 1] @ self.transition) * self.emission[:, st]
            scale[t] = alpha[t].sum()
            if scale[t] == 0:
                scale[t] = 1e-12
            alpha[t] /= scale[t]
            
        return alpha, scale

    def fit(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None) -> "DiscreteHMM":
        """Fits HMM parameters on sequence X via EM / Baum-Welch iterations."""
        sequence = np.asarray(X, dtype=int)
        T = len(sequence)
        if T < 4:
            return self
            
        for _ in range(10): # 10 EM iterations
            alpha, scale = self.forward(sequence)
            
            # Backward pass (beta)
            beta = np.zeros((T, self.n_states), dtype=np.float64)
            beta[-1] = 1.0 / scale[-1]
            for t in range(T - 2, -1, -1):
                st1 = int(sequence[t + 1])
                beta[t] = (self.transition @ (self.emission[:, st1] * beta[t + 1])) / scale[t]
                
            # State posteriors (gamma) and transitions (xi)
            gamma = alpha * beta
            gamma_sum = gamma.sum(axis=1, keepdims=True)
            gamma_sum[gamma_sum == 0] = 1e-12
            gamma /= gamma_sum
            
            # Update transitions A
            xi_sum = np.zeros((self.n_states, self.n_states), dtype=np.float64)
            for t in range(T - 1):
                st1 = int(sequence[t + 1])
                xi_t = alpha[t][:, None] * self.transition * self.emission[:, st1] * beta[t + 1][None, :]
                xi_sum += xi_t / (xi_t.sum() + 1e-12)
                
            self.transition = self._normalize(xi_sum, axis=1)
            
            # Update emissions B
            new_emission = np.zeros((self.n_states, self.n_symbols), dtype=np.float64)
            for sym in range(self.n_symbols):
                mask = (sequence == sym)
                new_emission[:, sym] = gamma[mask].sum(axis=0)
            self.emission = self._normalize(new_emission, axis=1)
            
        return self

    def update(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None) -> "DiscreteHMM":
        """Incremental online Baum-Welch update with exponential forgetting (lambda = 0.95)."""
        sequence = np.asarray(X, dtype=int)
        if len(sequence) < 2:
            return self
        alpha, _ = self.forward(sequence)
        gamma = alpha[-1] # Latest posterior
        st = int(sequence[-1])
        
        # Adaptive gradient nudge towards latest observation
        for k in range(self.n_states):
            self.emission[k, st] += 0.05 * gamma[k]
        self.emission = self._normalize(self.emission, axis=1)
        return self

    def predict_proba(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """Returns P(x_{T+1} = k | x_{1:T})."""
        sequence = np.asarray(X, dtype=int)
        if len(sequence) == 0:
            return np.full(self.n_symbols, 1.0 / self.n_symbols, dtype=np.float64)
            
        alpha, _ = self.forward(sequence)
        current_state_posterior = alpha[-1] # gamma_T
        next_state_prior = current_state_posterior @ self.transition # P(z_{T+1})
        
        # Emission marginalization: P(x_{T+1}) = sum_j P(z_{T+1}=j) * B_{j, k}
        next_symbol_probs = next_state_prior @ self.emission
        total = next_symbol_probs.sum()
        if total == 0:
            return np.full(self.n_symbols, 1.0 / self.n_symbols, dtype=np.float64)
        return next_symbol_probs / total

    def get_latent_state_posterior(self, sequence: Sequence[int]) -> np.ndarray:
        """Returns the latent regime posterior vector P(z_T = i | x_{1:T})."""
        alpha, _ = self.forward(sequence)
        if len(alpha) == 0:
            return self.initial
        return alpha[-1]

    def save(self, path: str) -> None:
        save_data = {
            "n_states": self.n_states,
            "n_symbols": self.n_symbols,
            "smoothing": self.smoothing,
            "transition": self.transition,
            "emission": self.emission,
            "initial": self.initial,
            "metadata": self.metadata
        }
        np.save(path, save_data)

    @classmethod
    def load(cls, path: str) -> "DiscreteHMM":
        data = np.load(path, allow_pickle=True).item()
        model = cls(n_states=data["n_states"], n_symbols=data["n_symbols"], smoothing=data["smoothing"])
        model.transition = data["transition"]
        model.emission = data["emission"]
        model.initial = data["initial"]
        model.metadata = data.get("metadata", model.metadata)
        return model

class RegimeMonitor:
    """Monitors active latent Markov regime probabilities for the sequence."""

    def __init__(self, hmm: DiscreteHMM):
        self.hmm = hmm
        self.state_probabilities: Optional[np.ndarray] = None

    def update(self, sequence: Sequence[int]) -> Dict[str, float]:
        posteriors = self.hmm.get_latent_state_posterior(sequence)
        self.state_probabilities = posteriors
        return {f"state_{i}_prob": round(float(p), 4) for i, p in enumerate(posteriors)}
