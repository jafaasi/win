from typing import Tuple, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
from ..features.encoding import one_hot

class LatentStateEncoder(nn.Module):
    """
    Continuous Non-Linear Latent State Space Encoder:
    Encodes discrete observation sequence (x_1, ..., x_t) into compact latent state z_t
    and predicts next categorical distribution x_{t+1} = g(z_t).
    """

    def __init__(self, input_size: int = 10, hidden_size: int = 32, output_size: int = 10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = nn.GRU(input_size, hidden_size, batch_first=True)
        self.output_head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L, input_size]
        states, _ = self.encoder(x)
        final_state = states[:, -1, :] # z_t: [B, hidden_size]
        logits = self.output_head(final_state)
        return logits, final_state

class LatentStabilityTester:
    """
    Measures empirical stability of latent representations under controlled input perturbations:
    S = E[ ||z_t(x) - z_t(x')||_2 ].
    High stability implies the latent space captures invariant dynamical manifolds rather than noise.
    """

    @staticmethod
    def measure_stability(encoder: LatentStateEncoder, sequence: Sequence[int], perturbation_rate: float = 0.05) -> float:
        encoder.eval()
        seq = list(sequence)
        if len(seq) < 10:
            return 1.0
            
        with torch.no_grad():
            x_raw = np.array([one_hot(int(d), encoder.input_size) for d in seq], dtype=np.float32)
            t_orig = torch.tensor(x_raw).unsqueeze(0)
            _, z_orig = encoder(t_orig)
            
            # Perturb random token
            seq_pert = list(seq)
            pert_idx = np.random.randint(0, len(seq_pert))
            seq_pert[pert_idx] = (seq_pert[pert_idx] + 1) % encoder.input_size
            
            x_pert = np.array([one_hot(int(d), encoder.input_size) for d in seq_pert], dtype=np.float32)
            t_pert = torch.tensor(x_pert).unsqueeze(0)
            _, z_pert = encoder(t_pert)
            
            diff = torch.norm(z_orig - z_pert, p=2).item()
            return round(float(diff), 4)
