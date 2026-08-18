from typing import Sequence, Optional, Union, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .base import SequenceModel, ModelMetadata
from .dataset import SequenceDataset
from .transformer import temperature_scale
from ..features.encoding import one_hot

class StateSpaceLayer(nn.Module):
    """
    Continuous State Space reference layer:
    h_t = tanh(W_in * x_t + W_state * h_{t-1})
    y_t = W_out * h_t
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.state_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L, input_size]
        B, L, _ = x.size()
        if state is None:
            state = torch.zeros(B, self.state_projection.in_features, device=x.device)
            
        outputs = []
        for t in range(L):
            current = x[:, t, :]
            state = torch.tanh(self.input_projection(current) + self.state_projection(state))
            outputs.append(self.output_projection(state))
            
        return torch.stack(outputs, dim=1), state

class S4Net(nn.Module):
    """Structured State Space (S4) Architecture with recurrent compressed hidden states."""

    def __init__(self, input_size: int = 10, hidden_size: int = 48, layers: int = 2, output_size: int = 10):
        super().__init__()
        self.layers = nn.ModuleList([
            StateSpaceLayer(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(layers)
        ])
        self.output_head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out, _ = layer(out)
        final_state = out[:, -1, :]
        return self.output_head(final_state)

class S4SequenceModel(SequenceModel):
    """EVOSEQ SequenceModel wrapper around S4 State Space Architecture."""

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 48,
        layers: int = 2,
        context_length: int = 64,
        lr: float = 1e-3,
        temperature: float = 1.1,
        version: str = "s4-v1"
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.context_length = context_length
        self.lr = lr
        self.temperature = temperature
        
        # Force CPU for cost efficiency on cloud deployments
        # GPU acceleration provides speed but not accuracy improvements
        self.device = torch.device("cpu")
        print(f"[S4] Using CPU (cost-efficient, same intelligence)")
        
        self.net = S4Net(input_size, hidden_size, layers, output_size=10).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        
        self.metadata = ModelMetadata(
            name="S4",
            version=version,
            parameters={
                "hidden_size": hidden_size,
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

    def fit(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None, epochs: int = 8) -> "S4SequenceModel":
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

    def update(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None) -> "S4SequenceModel":
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
            scaled = temperature_scale(logits, self.temperature)
            probs = torch.softmax(scaled, dim=-1).squeeze(0).cpu().numpy().astype(np.float64)
            probs_sum = probs.sum()
            if probs_sum == 0:
                return np.full(10, 0.1, dtype=np.float64)
            return probs / probs_sum

    def save(self, path: str) -> None:
        torch.save({
            "model_state": self.net.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "layers": self.layers,
            "context_length": self.context_length,
            "lr": self.lr,
            "temperature": self.temperature,
            "metadata": self.metadata
        }, path)

    @classmethod
    def load(cls, path: str) -> "S4SequenceModel":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            layers=checkpoint["layers"],
            context_length=checkpoint.get("context_length", 64),
            lr=checkpoint["lr"],
            temperature=checkpoint["temperature"]
        )
        model.net.load_state_dict(checkpoint["model_state"])
        model.metadata = checkpoint.get("metadata", model.metadata)
        return model

class SelectiveSSMLayer(nn.Module):
    """Selective State Space (Mamba) layer with input-dependent parameters Delta_t, B_t, C_t."""

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1)
        self.dt_proj = nn.Linear(1, d_model)
        self.A = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        B, L, D = x.size()
        h = torch.zeros(B, D, self.d_state, device=x.device)
        outputs = []
        
        for t in range(L):
            xt = x[:, t, :] # [B, D]
            proj = self.x_proj(xt) # [B, 2*d_state + 1]
            B_t = proj[:, :self.d_state] # [B, d_state]
            C_t = proj[:, self.d_state:2*self.d_state] # [B, d_state]
            dt_raw = proj[:, -1:] # [B, 1]
            dt = torch.sigmoid(self.dt_proj(dt_raw)) # [B, D]
            
            # Discretization
            dA = torch.exp(self.A.unsqueeze(0) * dt.unsqueeze(-1)) # [B, D, d_state]
            dB = dt.unsqueeze(-1) * B_t.unsqueeze(1) # [B, D, d_state]
            
            # Selective state step
            h = h * dA + dB * xt.unsqueeze(-1) # [B, D, d_state]
            yt = (h * C_t.unsqueeze(1)).sum(dim=-1) # [B, D]
            outputs.append(self.out_proj(yt))
            
        return torch.stack(outputs, dim=1)

class MambaNet(nn.Module):
    """Mamba Selective State Space Neural Architecture."""

    def __init__(self, input_size: int = 10, hidden_size: int = 48, layers: int = 2, output_size: int = 10):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList([SelectiveSSMLayer(hidden_size) for _ in range(layers)])
        self.output_head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_projection(x)
        for layer in self.layers:
            out = out + layer(out) # residual
        final_state = out[:, -1, :]
        return self.output_head(final_state)

class MambaSequenceModel(SequenceModel):
    """EVOSEQ SequenceModel wrapper around Mamba Selective State Space Architecture."""

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 48,
        layers: int = 2,
        context_length: int = 64,
        lr: float = 1e-3,
        temperature: float = 1.1,
        version: str = "mamba-v1"
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.context_length = context_length
        self.lr = lr
        self.temperature = temperature
        
        # Force CPU for cost efficiency on cloud deployments
        # GPU acceleration provides speed but not accuracy improvements
        self.device = torch.device("cpu")
        print(f"[Mamba] Using CPU (cost-efficient, same intelligence)")
        
        self.net = MambaNet(input_size, hidden_size, layers, output_size=10).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        
        self.metadata = ModelMetadata(
            name="Mamba",
            version=version,
            parameters={
                "hidden_size": hidden_size,
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

    def fit(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None, epochs: int = 8) -> "MambaSequenceModel":
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

    def update(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None) -> "MambaSequenceModel":
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
            scaled = temperature_scale(logits, self.temperature)
            probs = torch.softmax(scaled, dim=-1).squeeze(0).cpu().numpy().astype(np.float64)
            probs_sum = probs.sum()
            if probs_sum == 0:
                return np.full(10, 0.1, dtype=np.float64)
            return probs / probs_sum

    def save(self, path: str) -> None:
        torch.save({
            "model_state": self.net.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "layers": self.layers,
            "context_length": self.context_length,
            "lr": self.lr,
            "temperature": self.temperature,
            "metadata": self.metadata
        }, path)

    @classmethod
    def load(cls, path: str) -> "MambaSequenceModel":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            layers=checkpoint["layers"],
            context_length=checkpoint.get("context_length", 64),
            lr=checkpoint["lr"],
            temperature=checkpoint["temperature"]
        )
        model.net.load_state_dict(checkpoint["model_state"])
        model.metadata = checkpoint.get("metadata", model.metadata)
        return model
