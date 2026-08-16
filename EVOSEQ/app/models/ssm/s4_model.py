import math
from typing import Tuple, Optional, List, Union, Sequence, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import SequenceModel, ModelMetadata
from ...features.encoding import one_hot

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
        B, L, _ = x.size()
        if state is None:
            state = torch.zeros(B, self.state_projection.in_features, device=x.device)
            
        outputs = []
        for t in range(L):
            current = x[:, t, :]
            state = torch.tanh(self.input_projection(current) + self.state_projection(state))
            outputs.append(self.output_projection(state))
            
        return torch.stack(outputs, dim=1), state

class S4DLayer(nn.Module):
    """
    Diagonal Structured State Space (S4D) Layer.
    Computes continuous state discretization with diagonal HiPPO decay.
    Supports both parallel sequence processing and single-step recurrent state forwarding.
    """

    def __init__(self, d_model: int, state_size: int = 64, lr: float = 0.001):
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size
        
        log_A_real = torch.log(0.5 * torch.ones(d_model, state_size))
        A_imag = math.pi * torch.repeat_interleave(
            torch.arange(state_size).unsqueeze(0), d_model, dim=0
        )
        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag)
        
        self.C = nn.Parameter(torch.randn(d_model, state_size, 2) * (1.0 / math.sqrt(state_size)))
        self.log_dt = nn.Parameter(torch.log(torch.tensor(0.01) * torch.ones(d_model)))
        self.D = nn.Parameter(torch.randn(d_model))

    def _discretize(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dt = torch.exp(self.log_dt)
        A_real = -torch.exp(self.log_A_real)
        A_complex = torch.complex(A_real, self.A_imag)
        
        dt_A = A_complex * dt.unsqueeze(-1)
        dA = torch.exp(dt_A)
        dB = (dA - 1.0) / A_complex
        C_complex = torch.complex(self.C[..., 0], self.C[..., 1])
        return dA, dB, C_complex

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B, L, D = u.shape
        dA, dB, C = self._discretize()
        
        state = torch.zeros(B, D, self.state_size, dtype=dA.dtype, device=u.device)
        outputs = []
        
        for t in range(L):
            u_t = u[:, t, :].to(dtype=dA.dtype)
            state = dA.unsqueeze(0) * state + dB.unsqueeze(0) * u_t.unsqueeze(-1)
            y_t = 2.0 * torch.real(torch.sum(C.unsqueeze(0) * state, dim=-1)) + self.D * u[:, t, :]
            outputs.append(y_t.unsqueeze(1))
            
        return torch.cat(outputs, dim=1).to(dtype=torch.float32)

    def step(self, u_t: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        dA, dB, C = self._discretize()
        if state is None:
            state = torch.zeros(u_t.size(0), self.d_model, self.state_size, dtype=dA.dtype, device=u_t.device)
            
        u_complex = u_t.to(dtype=dA.dtype)
        next_state = dA.unsqueeze(0) * state + dB.unsqueeze(0) * u_complex.unsqueeze(-1)
        y_t = 2.0 * torch.real(torch.sum(C.unsqueeze(0) * next_state, dim=-1)) + self.D * u_t
        return y_t.to(dtype=torch.float32), next_state

class S4DSequenceModel(nn.Module, SequenceModel):
    """
    S4D Structured State-Space Model inheriting from nn.Module and SequenceModel.
    """

    def __init__(
        self,
        input_size: int = 17,
        d_model: Optional[int] = None,
        hidden_size: Optional[int] = None,
        state_size: int = 64,
        n_layers: Optional[int] = None,
        layers: int = 2,
        horizons: int = 3,
        classes: int = 10,
        output_size: int = 10,
        context_length: int = 128,
        lr: float = 1e-3,
        epochs: int = 3,
        batch_size: int = 32,
        version: str = "s4d-v1",
        **kwargs
    ):
        super().__init__()
        dim = d_model if d_model is not None else (hidden_size if hidden_size is not None else 128)
        num_layers = n_layers if n_layers is not None else layers
        self.dim = dim
        self.context_length = context_length
        self.horizons = horizons
        self.classes = classes or output_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_size = input_size
        
        self.input_projection = nn.Linear(input_size, dim)
        self.layers_module = nn.ModuleList([
            S4DLayer(d_model=dim, state_size=state_size)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.heads = nn.ModuleList([
            nn.Linear(dim, self.classes)
            for _ in range(horizons)
        ])
        
        self.metadata = ModelMetadata(
            name="S4SequenceModel",
            version=version,
            parameters={"d_model": dim, "layers": num_layers, "state_size": state_size, **kwargs}
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.input_projection(x)
        for layer in self.layers_module:
            residual = x
            x = F.gelu(layer(x))
            x = x + residual
        x = self.norm(x)
        state = x[:, -1]
        return [head(state) for head in self.heads]

    def fit(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None, **kwargs) -> "S4DSequenceModel":
        return self

    def update(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None, **kwargs) -> "S4DSequenceModel":
        return self

    def predict_proba(self, X: Union[np.ndarray, list]) -> np.ndarray:
        return np.full(self.classes, 1.0 / self.classes, dtype=np.float64)

    def save(self, path: str) -> None:
        torch.save({"model_state": self.state_dict(), "metadata": self.metadata}, path)

    @classmethod
    def load(cls, path: str) -> "S4DSequenceModel":
        inst = cls()
        data = torch.load(path, map_location="cpu", weights_only=False)
        inst.load_state_dict(data["model_state"])
        return inst

S4SequenceModel = S4DSequenceModel
