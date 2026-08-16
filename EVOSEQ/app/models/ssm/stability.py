from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np

class StateNormMonitor:
    """
    Monitors recurrent and state-space numerical stability:
    - Tracks Euclidean state norm: ||h_t||_2
    - Tracks gradient norms
    - Detects numerical instability (NaN / Inf)
    """

    def __init__(self, max_norm_threshold: float = 100.0):
        self.max_norm_threshold = max_norm_threshold
        self.state_norms = []
        self.gradient_norms = []
        self.nan_inf_count = 0

    def check_tensor(self, tensor: torch.Tensor, name: str = "state") -> Dict[str, Any]:
        """Inspects tensor for finite values and computes mean L2 norm."""
        is_finite = bool(torch.isfinite(tensor).all().item())
        if not is_finite:
            self.nan_inf_count += 1
            return {
                "name": name,
                "is_finite": False,
                "mean_norm": float("nan"),
                "max_norm": float("nan"),
                "status": "UNSTABLE_NAN_INF"
            }
            
        l2_norms = tensor.norm(dim=-1)
        mean_norm = float(l2_norms.mean().item())
        max_norm = float(l2_norms.max().item())
        
        self.state_norms.append(mean_norm)
        
        status = "HEALTHY"
        if max_norm > self.max_norm_threshold:
            status = "WARNING_HIGH_NORM"
            
        return {
            "name": name,
            "is_finite": True,
            "mean_norm": round(mean_norm, 4),
            "max_norm": round(max_norm, 4),
            "status": status
        }
