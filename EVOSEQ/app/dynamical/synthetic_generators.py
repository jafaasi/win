from typing import Sequence, Optional, Tuple, Dict, Any, List
import numpy as np

class ControlledGenerators:
    """
    Synthetic Benchmark Suite with known ground-truth parameters & latent states:
    Used for scientifically measuring state and parameter reconstruction fidelity.
    """

    @staticmethod
    def generator_iid(length: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generator A: Memoryless IID discrete uniform."""
        rng = np.random.default_rng(seed)
        obs = rng.integers(0, 10, size=length)
        states = np.zeros(length, dtype=np.int64)
        return obs, states

    @staticmethod
    def generator_markov(length: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generator B: 1st-order Markov chain with known transition matrix."""
        rng = np.random.default_rng(seed)
        trans = np.zeros((10, 10))
        for i in range(10):
            trans[i, (i + 1) % 10] = 0.7
            trans[i, (i + 2) % 10] = 0.3
            
        obs = np.empty(length, dtype=np.int64)
        obs[0] = rng.integers(0, 10)
        for t in range(1, length):
            obs[t] = rng.choice(10, p=trans[obs[t-1]])
        return obs, obs.copy()

    @staticmethod
    def generator_hmm(length: int, n_states: int = 3, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generator C: Hidden Markov Model with latent regime states z_t -> x_t."""
        rng = np.random.default_rng(seed)
        # Latent transitions
        A = np.array([
            [0.85, 0.10, 0.05],
            [0.05, 0.85, 0.10],
            [0.10, 0.05, 0.85]
        ])
        # Emission distributions
        B = np.array([
            [0.6, 0.3, 0.1, 0, 0, 0, 0, 0, 0, 0], # bias low
            [0, 0, 0, 0.3, 0.4, 0.3, 0, 0, 0, 0], # bias mid
            [0, 0, 0, 0, 0, 0, 0.1, 0.3, 0.4, 0.2] # bias high
        ])
        B = B / B.sum(axis=1, keepdims=True)
        
        states = np.empty(length, dtype=np.int64)
        obs = np.empty(length, dtype=np.int64)
        
        states[0] = rng.integers(0, n_states)
        obs[0] = rng.choice(10, p=B[states[0]])
        
        for t in range(1, length):
            states[t] = rng.choice(n_states, p=A[states[t-1]])
            obs[t] = rng.choice(10, p=B[states[t]])
            
        return obs, states

    @staticmethod
    def generator_fsm(length: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generator D: Deterministic Finite State Automaton."""
        rng = np.random.default_rng(seed)
        states = np.empty(length, dtype=np.int64)
        obs = np.empty(length, dtype=np.int64)
        
        z = rng.integers(0, 16)
        for t in range(length):
            states[t] = z
            obs[t] = (z * 7 + 3) % 10
            z = (z * 5 + 1) % 16
            
        return obs, states

    @staticmethod
    def generator_lcg(length: int, a: int = 17, c: int = 43, m: int = 10007, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Generator E: Linear Congruential PRNG z_{t+1} = (a * z_t + c) % m, x_t = z_t % 10."""
        states = np.empty(length, dtype=np.int64)
        obs = np.empty(length, dtype=np.int64)
        
        z = seed % m
        for t in range(length):
            states[t] = z
            obs[t] = z % 10
            z = (a * z + c) % m
            
        return obs, states
