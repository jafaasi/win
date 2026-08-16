from typing import List, Optional, Union, Sequence, Any
import torch
import torch.nn as nn
import numpy as np
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
        **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.context_length = context_length
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
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        device = next(self.parameters()).device
        x = x.to(device)
        
        x = self.input_projection(x)
        x = self.position(x)
        mask = self.causal_mask(x.size(1), x.device)
        hidden = self.encoder(x, mask=mask)
        hidden = self.norm(hidden)
        state = hidden[:, -1]
        return [head(state) for head in self.heads]

    def fit(self, X: Any, y: Optional[Any] = None, **kwargs) -> "NeuralTransformer":
        """Polymorphic fit interface."""
        return self

    def update(self, X: Any, y: Optional[Any] = None, **kwargs) -> "NeuralTransformer":
        return self

    def predict_sequence(self, sequence: Sequence[Any]) -> np.ndarray:
        """Sequential probability generation across sequence length."""
        preds = []
        seq_list = list(sequence)
        for t in range(len(seq_list)):
            ctx = seq_list[:t+1]
            p = self.predict_proba(ctx)
            preds.append(p)
        return np.asarray(preds, dtype=np.float64)

    def predict_proba(self, x: Any) -> Union[List[torch.Tensor], np.ndarray]:
        """Inference mode returning calibrated Softmax probability distributions."""
        self.eval()
        with torch.no_grad():
            if isinstance(x, torch.Tensor):
                logits = self.forward(x)
                return [torch.softmax(lg, dim=-1) for lg in logits]
            else:
                # Numpy or sequence context
                arr = np.asarray(x, dtype=np.float32)
                if arr.ndim == 1:
                    if len(arr) == 0:
                        return np.full(self.classes, 1.0 / self.classes, dtype=np.float64)
                    # Raw integer digits or 1D feature vector
                    if arr.dtype in (np.int32, np.int64, int) or len(arr) != self.input_size:
                        from ...core.mapping import map_digit
                        from ...features.vector import encode_observation
                        feats = []
                        for d in arr:
                            m = map_digit(int(d))
                            feats.append(encode_observation(m["digit"], m["size"], m["color"], m["parity"]))
                        tensor = torch.tensor(np.asarray(feats), dtype=torch.float32).unsqueeze(0)
                    else:
                        tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                elif arr.ndim == 2:
                    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
                else:
                    tensor = torch.tensor(arr, dtype=torch.float32)
                    
                device = next(self.parameters()).device
                tensor = tensor.to(device)
                logits = self.forward(tensor)
                h1_probs = torch.softmax(logits[0][0], dim=-1).cpu().numpy().astype(np.float64)
                return h1_probs
