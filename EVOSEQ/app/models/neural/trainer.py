from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def get_device() -> torch.device:
    """Selects highest-performance hardware accelerator available (CUDA, Apple MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def calculate_loss(
    outputs: List[torch.Tensor],
    targets: torch.Tensor,
    weights: Optional[List[float]] = None
) -> Tuple[torch.Tensor, List[float]]:
    """
    Computes weighted multi-horizon cross entropy loss:
    L = sum_h w_h * CE(outputs_h, targets[:, h])
    Default weights: [0.5, 0.3, 0.2]
    """
    horizons = len(outputs)
    if weights is None:
        if horizons == 3:
            weights = [0.5, 0.3, 0.2]
        else:
            weights = [1.0 / horizons] * horizons
            
    total = torch.tensor(0.0, device=outputs[0].device)
    individual = []
    
    for h in range(horizons):
        loss = F.cross_entropy(outputs[h], targets[:, h])
        individual.append(float(loss.detach().item()))
        total = total + (weights[h] * loss)
        
    return total, individual

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weights: Optional[List[float]] = None
) -> float:
    """Executes a single training epoch with gradient clipping."""
    model.train()
    total_loss = 0.0
    
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        outputs = model(X)
        loss, _ = calculate_loss(outputs, y, weights)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())
        
    return total_loss / max(len(loader), 1)

def evaluate_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    weights: Optional[List[float]] = None
) -> float:
    """Evaluates validation loss without gradient computation."""
    model.eval()
    losses = []
    
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            loss, _ = calculate_loss(outputs, y, weights)
            losses.append(float(loss.item()))
            
    return float(sum(losses) / max(len(losses), 1))

class EarlyStopping:
    """Prevents overfitting by halting optimization when validation loss stagnates."""

    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_epochs = 0

    def update(self, value: float) -> bool:
        """Returns True if training should be stopped early."""
        if value < (self.best - self.min_delta):
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
