from enum import Enum
from dataclasses import dataclass
from typing import Sequence, Dict, Any, List, Optional
import numpy as np
from ..features.basic import digit_distribution
from ..features.transitions import transition_matrix

class DriftState(Enum):
    STABLE = "stable"
    WATCH = "watch"
    WARNING = "warning"
    EVOLVE = "evolve"

@dataclass
class MultiDimensionalDriftResult:
    is_significant: bool
    composite_drift: float
    state: DriftState
    dimension_drifts: Dict[str, float]

    @property
    def level(self) -> str:
        if self.state == DriftState.EVOLVE:
            return "CRITICAL"
        elif self.state in [DriftState.WARNING, DriftState.WATCH]:
            return "MODERATE"
        return "LOW"

    @property
    def js_divergence(self) -> float:
        return self.composite_drift



def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Computes Kullback-Leibler divergence D_KL(P || Q)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / np.maximum(q[mask], 1e-12))))

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Computes symmetric Jensen-Shannon divergence D_JS(P, Q) in range [0, 1]."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum == 0 or q_sum == 0:
        return 0.0
    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)
    js = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return float(np.clip(js, 0.0, 1.0))

def calculate_multidimensional_drift(
    digits: Sequence[int],
    sizes: Optional[Sequence[int]] = None,
    colors: Optional[Sequence[int]] = None,
    parities: Optional[Sequence[int]] = None,
    recent_window: int = 200,
    historical_window: int = 1000,
    weights: Optional[Dict[str, float]] = None
) -> MultiDimensionalDriftResult:
    """
    Computes composite multi-dimensional drift across:
    - Digits (0-9)
    - Sizes (Big/Small)
    - Colors (Red/Green/Violet)
    - Parities (Odd/Even)
    - 1-step Transition Matrix
    """
    digits = list(digits)
    n = len(digits)
    if n < recent_window * 2:
        return MultiDimensionalDriftResult(
            is_significant=False,
            composite_drift=0.0,
            state=DriftState.STABLE,
            dimension_drifts={"digit": 0.0, "size": 0.0, "color": 0.0, "parity": 0.0, "transition": 0.0}
        )

    if sizes is None:
        sizes = [1 if d >= 5 else 0 for d in digits]
    if colors is None:
        colors = [1 if d in [1, 3, 7, 9] else 2 if d in [0, 5] else 0 for d in digits]
    if parities is None:
        parities = [1 if d % 2 != 0 else 0 for d in digits]

    w = weights or {"digit": 0.40, "size": 0.20, "color": 0.15, "parity": 0.15, "transition": 0.10}

    # Slices
    d_rec, d_hist = digits[-recent_window:], digits[-min(n, historical_window + recent_window):-recent_window]
    s_rec, s_hist = sizes[-recent_window:], sizes[-min(n, historical_window + recent_window):-recent_window]
    c_rec, c_hist = colors[-recent_window:], colors[-min(n, historical_window + recent_window):-recent_window]
    p_rec, p_hist = parities[-recent_window:], parities[-min(n, historical_window + recent_window):-recent_window]

    # JS Divergence per dimension
    js_digit = js_divergence(digit_distribution(d_rec), digit_distribution(d_hist))
    js_size = js_divergence(np.bincount(s_rec, minlength=2), np.bincount(s_hist, minlength=2))
    js_color = js_divergence(np.bincount(c_rec, minlength=3), np.bincount(c_hist, minlength=3))
    js_parity = js_divergence(np.bincount(p_rec, minlength=2), np.bincount(p_hist, minlength=2))
    
    # Transition matrix drift (flattened)
    t_rec = transition_matrix(d_rec, 10).flatten()
    t_hist = transition_matrix(d_hist, 10).flatten()
    js_trans = js_divergence(t_rec, t_hist)

    dim_drifts = {
        "digit": round(js_digit, 4),
        "size": round(js_size, 4),
        "color": round(js_color, 4),
        "parity": round(js_parity, 4),
        "transition": round(js_trans, 4)
    }

    composite = (
        w["digit"] * js_digit +
        w["size"] * js_size +
        w["color"] * js_color +
        w["parity"] * js_parity +
        w["transition"] * js_trans
    )
    composite = round(composite, 4)

    controller = DriftController()
    state = controller.update(composite)

    return MultiDimensionalDriftResult(
        is_significant=(state in [DriftState.WARNING, DriftState.EVOLVE]),
        composite_drift=composite,
        state=state,
        dimension_drifts=dim_drifts
    )

class DriftController:
    """Hysteresis state machine for drift management."""

    def __init__(
        self,
        watch_threshold: float = 0.02,
        warning_threshold: float = 0.05,
        evolve_threshold: float = 0.10,
    ):
        self.watch_threshold = watch_threshold
        self.warning_threshold = warning_threshold
        self.evolve_threshold = evolve_threshold
        self.state = DriftState.STABLE

    def update(self, score: float) -> DriftState:
        if score >= self.evolve_threshold:
            self.state = DriftState.EVOLVE
        elif score >= self.warning_threshold:
            self.state = DriftState.WARNING
        elif score >= self.watch_threshold:
            self.state = DriftState.WATCH
        else:
            self.state = DriftState.STABLE
        return self.state

# Backward compatible helper
def calculate_drift(history: Sequence[int], recent_window: int = 500, historical_window: int = 2500, threshold: float = 0.08):
    res = calculate_multidimensional_drift(history, recent_window=recent_window, historical_window=historical_window)
    return res
