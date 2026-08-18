import math
from typing import Sequence, Optional, Union, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .base import SequenceModel, ModelMetadata
from .dataset import SequenceDataset
from ..features.encoding import one_hot

def temperature_scale(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Applies temperature scaling to prevent overconfident probability calibration."""
    temp = max(1e-4, float(temperature))
    return logits / temp

class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding for temporal order awareness."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        return x + self.pe[:, :x.size(1), :]

class CausalTransformer(nn.Module):
    """Causal Multi-Head Self-Attention Transformer with strict upper-triangular masking."""

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        output_size: int = 10,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output = nn.Linear(hidden_size, output_size)

    def causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F]
        proj = self.input_projection(x)
        pos_proj = self.pos_encoder(proj)
        length = pos_proj.size(1)
        mask = self.causal_mask(length, pos_proj.device)
        representation = self.encoder(pos_proj, mask=mask)
        final_state = representation[:, -1, :]
        return self.output(final_state)

class TransformerSequenceModel(SequenceModel):
    """
    EVOSEQ SequenceModel wrapper around Causal Transformer Architecture.
    Includes causal training batches, gradient clipping, and temperature calibration.
    """

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        context_length: int = 64,
        lr: float = 1e-3,
        temperature: float = 1.15,
        version: str = "transformer-v1"
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.heads = heads
        self.layers = layers
        self.dropout = dropout
        self.context_length = context_length
        self.lr = lr
        self.temperature = temperature
        
        # Force CPU for cost efficiency on cloud deployments
        # GPU acceleration provides speed but not accuracy improvements
        self.device = torch.device("cpu")
        print(f"[Transformer] Using CPU (cost-efficient, same intelligence)")
        
        self.net = CausalTransformer(
            input_size=input_size,
            hidden_size=hidden_size,
            layers=layers,
            heads=heads,
            dropout=dropout,
            output_size=10
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        
        self.metadata = ModelMetadata(
            name="Transformer",
            version=version,
            parameters={
                "hidden_size": hidden_size,
                "heads": heads,
                "layers": layers,
                "context_length": context_length,
                "lr": lr,
                "temperature": temperature
            }
        )

    def _prepare_tensor(self, X: Union[np.ndarray, list]) -> torch.Tensor:
        arr = np.asarray(X)
        if arr.ndim == 1:
            oh = np.array([one_hot(int(d), self.input_size) for d in arr], dtype=np.float32)
            return torch.from_numpy(oh).unsqueeze(0).to(self.device)
        elif arr.ndim == 2:
            return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).to(self.device)
        else:
            return torch.from_numpy(arr.astype(np.float32)).to(self.device)

    def fit(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None, epochs: int = 8) -> "TransformerSequenceModel":
        self.net.train()
        X_seq = list(X)
        if len(X_seq) <= self.context_length + 2:
            return self
            
        dataset = SequenceDataset(X_seq[:-1], X_seq[1:], context_length=min(self.context_length, len(X_seq)//2), input_size=self.input_size)
        loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)
        
        for _ in range(epochs):
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.net(batch_X)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optimizer.step()
                
        return self

    def update(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None) -> "TransformerSequenceModel":
        self.net.train()
        X_seq = list(X)
        if len(X_seq) < 4:
            return self
            
        ctx = X_seq[-4:-1]
        target = int(X_seq[-1])
        t_in = torch.tensor(np.array([[one_hot(int(d), self.input_size) for d in ctx]]), dtype=torch.float32).to(self.device)
        t_out = torch.tensor([target], dtype=torch.long).to(self.device)
        
        self.optimizer.zero_grad()
        logits = self.net(t_in)
        loss = self.criterion(logits, t_out)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        return self

    def predict_proba(self, X: Union[np.ndarray, list]) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            t_in = self._prepare_tensor(X)
            logits = self.net(t_in)
            scaled_logits = temperature_scale(logits, self.temperature)
            probs = torch.softmax(scaled_logits, dim=-1).squeeze(0).cpu().numpy().astype(np.float64)
            probs_sum = probs.sum()
            if probs_sum == 0:
                return np.full(10, 0.1, dtype=np.float64)
            return probs / probs_sum

    def save(self, path: str) -> None:
        torch.save({
            "model_state": self.net.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "heads": self.heads,
            "layers": self.layers,
            "dropout": self.dropout,
            "context_length": self.context_length,
            "lr": self.lr,
            "temperature": self.temperature,
            "metadata": self.metadata
        }, path)

    @classmethod
    def load(cls, path: str) -> "TransformerSequenceModel":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            heads=checkpoint["heads"],
            layers=checkpoint["layers"],
            dropout=checkpoint.get("dropout", 0.1),
            context_length=checkpoint.get("context_length", 64),
            lr=checkpoint["lr"],
            temperature=checkpoint["temperature"]
        )
        model.net.load_state_dict(checkpoint["model_state"])
        model.metadata = checkpoint.get("metadata", model.metadata)
        return model
