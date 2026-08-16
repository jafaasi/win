import torch
import torch.nn as nn
from typing import Optional

class MetaGate(nn.Module):
    """
    Environment-Conditioned Neural Meta-Gating Network.
    Maps continuous statistical environment fingerprints E_t -> dynamic model weights w_t.
    """

    def __init__(self, feature_size: int = 12, models: int = 5, hidden: int = 64):
        super().__init__()
        self.feature_size = feature_size
        self.models = models
        self.network = nn.Sequential(
            nn.Linear(feature_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, models),
        )

    def forward(self, environment: torch.Tensor) -> torch.Tensor:
        """
        environment: [Batch, feature_size] or [feature_size]
        Returns: [Batch, models] or [models] probability simplex weights.
        """
        logits = self.network(environment)
        return torch.softmax(logits, dim=-1)
