from typing import List, Optional
import torch
import torch.nn as nn
from .positional import SinusoidalPositionEncoding

class NeuralTransformer(nn.Module):
    """
    Causal Multi-Horizon Neural Transformer for discrete sequence probability estimation.
    Applies strict upper-triangular masking to ensure zero future lookahead contamination.
    """

    def __init__(
        self,
        input_size: int = 17,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        feedforward: int = 512,
        dropout: float = 0.1,
        context_length: int = 128,
        horizons: int = 3,
        classes: int = 10,
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.horizons = horizons
        self.classes = classes
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.position = SinusoidalPositionEncoding(d_model, max_length=max(context_length, 2048))
        
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        
        self.heads = nn.ModuleList([
            nn.Linear(d_model, classes)
            for _ in range(horizons)
        ])

    def causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        """Constructs boolean upper-triangular causal attention mask."""
        return torch.triu(
            torch.ones(length, length, device=device, dtype=torch.bool),
            diagonal=1
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass:
        x: [Batch, Length, input_size]
        Returns: list of logits [[Batch, classes], ...] for each horizon H_1, ..., H_horizons
        """
        x = self.input_projection(x)
        x = self.position(x)
        mask = self.causal_mask(x.size(1), x.device)
        hidden = self.encoder(x, mask=mask)
        hidden = self.norm(hidden)
        state = hidden[:, -1]
        return [head(state) for head in self.heads]

    def predict_proba(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Inference mode returning calibrated Softmax probability simplex tensors."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return [torch.softmax(lg, dim=-1) for lg in logits]
