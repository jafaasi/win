from typing import Sequence, Optional, Union, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from .base import SequenceModel, ModelMetadata
from .torch_base import SequenceNetwork
from ..features.encoding import one_hot

def temperature_scale(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Applies temperature scaling to prevent overconfident probability calibration."""
    temp = max(1e-4, float(temperature))
    return logits / temp

class TransformerSequenceNet(SequenceNetwork):
    """PyTorch Multi-Head Self-Attention Encoder."""

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        heads: int = 4,
        layers: int = 2,
        output_size: int = 10
    ):
        super().__init__(input_size, hidden_size, output_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 2,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class TransformerSequenceModel(SequenceModel):
    """
    EVOSEQ SequenceModel wrapper around PyTorch Transformer Architecture.
    Includes SGD/Adam training and Temperature Probability Calibration.
    """

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        heads: int = 4,
        layers: int = 2,
        lr: float = 1e-3,
        temperature: float = 1.2,
        version: str = "transformer-v1"
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.heads = heads
        self.layers = layers
        self.lr = lr
        self.temperature = temperature
        
        self.net = TransformerSequenceNet(
            input_size=input_size,
            hidden_size=hidden_size,
            heads=heads,
            layers=layers,
            output_size=10
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        
        self.metadata = ModelMetadata(
            name="Transformer",
            version=version,
            parameters={
                "hidden_size": hidden_size,
                "heads": heads,
                "layers": layers,
                "lr": lr,
                "temperature": temperature
            }
        )

    def _prepare_tensor(self, X: Union[np.ndarray, list]) -> torch.Tensor:
        arr = np.asarray(X)
        if arr.ndim == 1:
            # Convert 1D integer sequence to one-hot: [1, L, 10]
            oh = np.array([one_hot(int(d), self.input_size) for d in arr], dtype=np.float32)
            return torch.from_numpy(oh).unsqueeze(0)
        elif arr.ndim == 2:
            return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
        else:
            return torch.from_numpy(arr.astype(np.float32))

    def fit(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None, epochs: int = 10) -> "TransformerSequenceModel":
        """Trains Transformer model on sequence X."""
        self.net.train()
        X_seq = list(X)
        if len(X_seq) < 8:
            return self
            
        context_len = min(32, len(X_seq) - 1)
        
        # Construct rolling training batches
        inputs, targets = [], []
        for i in range(context_len, len(X_seq)):
            ctx = X_seq[i - context_len:i]
            target = X_seq[i]
            inputs.append(np.array([one_hot(int(d), self.input_size) for d in ctx], dtype=np.float32))
            targets.append(int(target))
            
        if not inputs:
            return self
            
        t_in = torch.tensor(np.array(inputs), dtype=torch.float32)
        t_out = torch.tensor(targets, dtype=torch.long)
        
        for _ in range(epochs):
            self.optimizer.zero_grad()
            logits = self.net(t_in)
            loss = self.criterion(logits, t_out)
            loss.backward()
            self.optimizer.step()
            
        return self

    def update(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None) -> "TransformerSequenceModel":
        """Performs a single online gradient descent step on the latest observation."""
        self.net.train()
        X_seq = list(X)
        if len(X_seq) < 4:
            return self
            
        ctx = X_seq[-4:-1]
        target = int(X_seq[-1])
        
        t_in = torch.tensor(np.array([[one_hot(int(d), self.input_size) for d in ctx]]), dtype=torch.float32)
        t_out = torch.tensor([target], dtype=torch.long)
        
        self.optimizer.zero_grad()
        logits = self.net(t_in)
        loss = self.criterion(logits, t_out)
        loss.backward()
        self.optimizer.step()
        return self

    def predict_proba(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """Returns 10-class temperature-calibrated probabilities."""
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
            lr=checkpoint["lr"],
            temperature=checkpoint["temperature"]
        )
        model.net.load_state_dict(checkpoint["model_state"])
        model.metadata = checkpoint.get("metadata", model.metadata)
        return model
