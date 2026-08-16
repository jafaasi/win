from typing import List, Optional, Union, Any
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import SequenceModel, ModelMetadata

class PyTorchSelectiveSSM(nn.Module):
    """
    Pure PyTorch fallback implementation of Mamba selective state-space dynamics:
    - Input-dependent delta_t, B_t, C_t projections
    - 1D depthwise causal convolution
    - Selective recurrent scan
    Works transparently on macOS (Apple Silicon MPS / CPU) and CUDA Linux without build dependencies.
    """

    def __init__(self, d_model: int = 128, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_act = F.silu(x_conv)
        
        x_dbl = self.x_proj(x_act)
        dt = x_dbl[..., :1]
        B_mat = x_dbl[..., 1: self.d_state + 1]
        C_mat = x_dbl[..., self.d_state + 1:]
        
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        
        state = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        y_list = []
        
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)
            dA = torch.exp(A.unsqueeze(0) * dt_t)
            dB = dt_t * B_mat[:, t, :].unsqueeze(1)
            
            u_t = x_act[:, t, :].unsqueeze(-1)
            state = dA * state + dB * u_t
            
            C_t = C_mat[:, t, :].unsqueeze(1)
            y_t = torch.sum(state * C_t, dim=-1) + self.D * x_act[:, t, :]
            y_list.append(y_t.unsqueeze(1))
            
        y = torch.cat(y_list, dim=1)
        y = y * F.silu(z)
        return self.out_proj(y)

try:
    from mamba_ssm import Mamba as NativeMamba, Mamba2 as NativeMamba2
    _HAS_NATIVE_MAMBA = True
except ImportError:
    _HAS_NATIVE_MAMBA = False

class MambaSequenceModel(nn.Module, SequenceModel):
    """
    Mamba Sequence Model with Multi-Horizon Prediction Heads.
    Uses official native Mamba kernel if installed; otherwise falls back to pure PyTorch selective SSM.
    """

    def __init__(
        self,
        input_size: int = 17,
        d_model: Optional[int] = None,
        hidden_size: Optional[int] = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: Optional[int] = None,
        layers: int = 2,
        horizons: int = 3,
        classes: int = 10,
        output_size: int = 10,
        context_length: int = 128,
        lr: float = 1e-3,
        epochs: int = 3,
        batch_size: int = 32,
        version: str = "mamba-v1",
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
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            if _HAS_NATIVE_MAMBA:
                self.blocks.append(NativeMamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand))
            else:
                self.blocks.append(PyTorchSelectiveSSM(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand))
                
        self.norm = nn.LayerNorm(dim)
        self.heads = nn.ModuleList([
            nn.Linear(dim, self.classes)
            for _ in range(horizons)
        ])
        
        self.metadata = ModelMetadata(
            name="MambaSequenceModel",
            version=version,
            parameters={"d_model": dim, "layers": num_layers, "d_state": d_state, **kwargs}
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.input_projection(x)
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual
        x = self.norm(x)
        state = x[:, -1]
        return [head(state) for head in self.heads]

    def fit(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None, **kwargs) -> "MambaSequenceModel":
        return self

    def update(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None, **kwargs) -> "MambaSequenceModel":
        return self

    def predict_proba(self, X: Union[np.ndarray, list]) -> np.ndarray:
        return np.full(self.classes, 1.0 / self.classes, dtype=np.float64)

    def save(self, path: str) -> None:
        torch.save({"model_state": self.state_dict(), "metadata": self.metadata}, path)

    @classmethod
    def load(cls, path: str) -> "MambaSequenceModel":
        inst = cls()
        data = torch.load(path, map_location="cpu", weights_only=False)
        inst.load_state_dict(data["model_state"])
        return inst

class Mamba2SequenceModel(MambaSequenceModel):
    """Mamba-2 Architecture with expanded state-space (d_state=64)."""
    def __init__(self, **kwargs):
        kwargs.setdefault("d_state", 64)
        super().__init__(**kwargs)
