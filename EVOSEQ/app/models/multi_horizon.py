from typing import List, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHorizonHead(nn.Module):
    """
    Multi-Horizon Categorical Prediction Head:
    Computes distinct output representations and distributions for:
    - Horizon 1: P(X_{t+1} | X_{<=t})
    - Horizon 2: P(X_{t+2} | X_{<=t})
    - Horizon 3: P(X_{t+3} | X_{<=t})
    """

    def __init__(self, hidden_size: int, horizons: int = 3, classes: int = 10):
        super().__init__()
        self.horizons = horizons
        self.classes = classes
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, classes)
            )
            for _ in range(horizons)
        ])

    def forward(self, representation: torch.Tensor) -> List[torch.Tensor]:
        # representation: [B, hidden_size] or [B, L, hidden_size]
        if representation.ndim == 3:
            representation = representation[:, -1, :]
        return [head(representation) for head in self.heads]

    def compute_loss(
        self,
        logits_list: List[torch.Tensor],
        targets: torch.Tensor,
        weights: Sequence[float] = (1.0, 0.5, 0.25)
    ) -> torch.Tensor:
        """
        Computes composite multi-horizon cross-entropy loss:
        L = sum(lambda_h * CrossEntropy(logits_h, target_h))
        """
        total_loss = torch.tensor(0.0, device=targets.device)
        for h, logits in enumerate(logits_list):
            if h < targets.shape[1]:
                target_h = targets[:, h]
                loss_h = F.cross_entropy(logits, target_h)
                weight_h = weights[h] if h < len(weights) else 0.1
                total_loss = total_loss + (weight_h * loss_h)
        return total_loss
