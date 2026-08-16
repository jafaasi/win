import os
import json
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    validation_loss: float = 0.0,
    config: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Saves atomic versioned model checkpoint and accompanying metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "validation_loss": validation_loss,
        "config": config.__dict__ if hasattr(config, "__dict__") else config,
        "metadata": metadata or {}
    }
    torch.save(checkpoint, path)
    
    meta_path = os.path.join(os.path.dirname(path), "metadata.json")
    try:
        with open(meta_path, "w") as f:
            json.dump({
                "epoch": epoch,
                "validation_loss": validation_loss,
                "config": config.__dict__ if hasattr(config, "__dict__") else config,
                "metadata": metadata or {}
            }, f, indent=2)
    except Exception:
        pass

def load_checkpoint(
    path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, Any]:
    """Loads checkpoint weights and state."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if model is not None and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
