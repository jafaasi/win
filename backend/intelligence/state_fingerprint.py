from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class StateFingerprint:
    sequence_no: int = 0
    sequence_hash: str = ""

    short_window_size: int = 10
    medium_window_size: int = 50
    long_window_size: int = 200

    recent_sequence: str = ""
    short_big_rate: float = 0.5
    medium_big_rate: float = 0.5
    long_big_rate: float = 0.5

    current_streak: int = 0
    streak_value: int = 0

    entropy: float = 0.0
    transition_entropy: float = 0.0
    autocorr_lag1: float = 0.0
    autocorr_lag2: float = 0.0
    autocorr_lag3: float = 0.0

    conditional_entropy_1: float = 0.0
    conditional_entropy_2: float = 0.0

    information_gain_1: float = 0.0
    information_gain_2: float = 0.0

    lz_complexity: float = 0.0
    drift_score: float = 0.0
    regime_id: str = "UNKNOWN"

    feature_vector: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_numpy(self) -> np.ndarray:
        return np.array(self.feature_vector, dtype=np.float32)

    def fingerprint_id(self) -> str:
        """Short numeric ID for quick matching: uses key features."""
        key = (
            self.recent_sequence[-8:] if len(self.recent_sequence) >= 8 else self.recent_sequence,
            round(self.entropy, 3),
            round(self.short_big_rate, 2),
            round(self.medium_big_rate, 2),
            self.current_streak,
        )
        return hashlib.md5(repr(key).encode()).hexdigest()


def _size_from_digit(d: int) -> int:
    return 1 if int(d) >= 5 else 0


def _categorical_entropy(values: Sequence[int], cardinality: int) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    counts = np.zeros(cardinality, dtype=np.float64)
    for v in values:
        if 0 <= int(v) < cardinality:
            counts[int(v)] += 1.0
    probs = counts / counts.sum()
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log2(p)
    return h


def _windowed_big_rate(sizes: List[int], window: int) -> float:
    if len(sizes) == 0:
        return 0.5
    w = sizes[-window:] if len(sizes) >= window else sizes
    return float(np.mean(w)) if len(w) > 0 else 0.5


def _current_streak(sizes: List[int]) -> Tuple[int, int]:
    """Return (streak_length, streak_value)."""
    if len(sizes) == 0:
        return 0, 0
    last = sizes[-1]
    length = 0
    for i in range(len(sizes) - 1, -1, -1):
        if sizes[i] == last:
            length += 1
        else:
            break
    return length, int(last)


def _autocorr(x: Sequence[float], lag: int) -> float:
    arr = np.asarray(x, dtype=np.float64)
    if len(arr) <= lag + 1:
        return 0.0
    arr = arr - arr.mean()
    denom = (arr * arr).sum()
    if denom <= 0:
        return 0.0
    num = (arr[:-lag] * arr[lag:]).sum()
    return float(num / denom)


def _transition_entropy(sizes: List[int]) -> float:
    if len(sizes) < 2:
        return 1.0
    t_counts = np.zeros((2, 2), dtype=np.float64)
    for i in range(1, len(sizes)):
        a = sizes[i - 1]
        b = sizes[i]
        if 0 <= a < 2 and 0 <= b < 2:
            t_counts[a, b] += 1.0
    row_sums_arr = t_counts.sum(axis=1)
    total = len(sizes) - 1
    h = 0.0
    for a in range(2):
        rs = float(row_sums_arr[a])
        if rs <= 0:
            continue
        for b in range(2):
            p = float(t_counts[a, b]) / rs
            if p > 0:
                h -= (rs / total) * p * math.log2(p)
    return float(h)


def _conditional_entropy(values: Sequence[int], order: int, cardinality: int = 10) -> float:
    vals = list(values)
    if len(vals) <= order:
        return 0.0
    joint_counts = {}
    cond_counts = {}
    for i in range(order, len(vals)):
        context = tuple(vals[i - order:i])
        next_val = vals[i]
        joint_counts[(context, next_val)] = joint_counts.get((context, next_val), 0) + 1
        cond_counts[context] = cond_counts.get(context, 0) + 1
    total = sum(joint_counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for (ctx, nxt), c in joint_counts.items():
        p_joint = c / total
        p_cond = c / cond_counts[ctx]
        if p_cond > 0:
            h -= p_joint * math.log2(p_cond)
    return h


def _information_gain(values: Sequence[int], order: int, cardinality: int = 10) -> float:
    vals = list(values)
    if len(vals) <= order + 1:
        return 0.0
    marginal_h = _categorical_entropy(vals, cardinality)
    cond_h = _conditional_entropy(vals, order, cardinality)
    return max(0.0, float(marginal_h - cond_h))


def _lz_complexity(values: Sequence[int]) -> float:
    """Lempel-Ziv complexity estimate, normalized."""
    v = list(values)
    n = len(v)
    if n < 2:
        return 0.0
    s = tuple(v)
    words = set()
    i = 0
    while i < n:
        j = i + 1
        while j <= n and s[i:j] in words:
            j += 1
        if j <= n:
            words.add(s[i:j])
        i = j
    # Normalize by n/log2(n) for 0..1-ish range
    denom = max(1.0, n / max(1.0, math.log2(max(2, n))))
    return min(1.0, float(len(words)) / denom)


def _detect_regime(sizes: List[int], entropy_val: float, drift: float) -> str:
    if len(sizes) < 30:
        return "UNKNOWN"
    recent = sizes[-50:] if len(sizes) >= 50 else sizes
    big_ratio = float(np.mean(recent))
    vol = float(np.std(recent)) if len(recent) > 1 else 0.0

    if drift > 0.25:
        return "REGIME_CHANGE"
    if big_ratio > 0.65:
        return "BIG_MOMENTUM"
    if big_ratio < 0.35:
        return "SMALL_MOMENTUM"
    if entropy_val > 0.98 and vol < 0.45:
        return "HIGH_ENTROPY_EQUILIBRIUM"
    if vol < 0.35:
        return "LOW_VOLATILITY"
    if vol > 0.49:
        return "HIGH_VOLATILITY"
    return "EQUILIBRIUM"


def compute_state_fingerprint(
    digits: Sequence[int],
    sequence_no: int = 0,
    drift_score_override: Optional[float] = None,
) -> StateFingerprint:
    """
    Converts a digit sequence into a complete StateFingerprint.
    Strictly causal: uses only digits in the provided order.
    """
    fp = StateFingerprint()
    fp.sequence_no = int(sequence_no)

    digits_list = [int(d) for d in digits]
    n = len(digits_list)
    if n == 0:
        return fp

    sizes = [_size_from_digit(d) for d in digits_list]

    fp.short_big_rate = _windowed_big_rate(sizes, 10)
    fp.medium_big_rate = _windowed_big_rate(sizes, 50)
    fp.long_big_rate = _windowed_big_rate(sizes, 200)

    recent_n = min(16, n)
    fp.recent_sequence = "".join(["B" if s == 1 else "S" for s in sizes[-recent_n:]])
    fp.sequence_hash = hashlib.sha256(fp.recent_sequence.encode()).hexdigest()

    streak_len, streak_val = _current_streak(sizes)
    fp.current_streak = streak_len
    fp.streak_value = streak_val

    # Entropy on sizes (binary)
    fp.entropy = _categorical_entropy(sizes[-min(200, n):], 2)
    fp.transition_entropy = _transition_entropy(sizes[-min(100, n):])

    if n >= 4:
        fp.autocorr_lag1 = _autocorr(sizes, 1)
    if n >= 5:
        fp.autocorr_lag2 = _autocorr(sizes, 2)
    if n >= 6:
        fp.autocorr_lag3 = _autocorr(sizes, 3)

    if n >= 10:
        fp.conditional_entropy_1 = _conditional_entropy(digits_list[-min(500, n):], order=1, cardinality=10)
    if n >= 20:
        fp.conditional_entropy_2 = _conditional_entropy(digits_list[-min(500, n):], order=2, cardinality=10)

    if n >= 10:
        fp.information_gain_1 = _information_gain(digits_list[-min(500, n):], order=1, cardinality=10)
    if n >= 20:
        fp.information_gain_2 = _information_gain(digits_list[-min(500, n):], order=2, cardinality=10)

    fp.lz_complexity = _lz_complexity(sizes[-min(200, n):]) if n >= 4 else 0.0

    # Simple drift estimate: distribution difference between short and long windows
    if drift_score_override is not None:
        fp.drift_score = float(drift_score_override)
    else:
        short = sizes[-min(30, n):]
        long_win = sizes[-min(200, n):]
        if len(short) >= 10 and len(long_win) >= 30:
            diff = abs(float(np.mean(short)) - float(np.mean(long_win)))
            fp.drift_score = min(1.0, diff * 5.0)
        else:
            fp.drift_score = 0.0

    fp.regime_id = _detect_regime(sizes, fp.entropy, fp.drift_score)

    fp.feature_vector = [
        fp.short_big_rate,
        fp.medium_big_rate,
        fp.long_big_rate,
        float(fp.current_streak) / 20.0,
        float(fp.streak_value),
        fp.entropy,
        fp.transition_entropy,
        (fp.autocorr_lag1 + 1.0) / 2.0,
        (fp.autocorr_lag2 + 1.0) / 2.0,
        (fp.autocorr_lag3 + 1.0) / 2.0,
        fp.conditional_entropy_1 / math.log2(10),
        fp.conditional_entropy_2 / math.log2(10),
        fp.information_gain_1 / math.log2(10),
        fp.information_gain_2 / math.log2(10),
        fp.lz_complexity,
        fp.drift_score,
    ]

    return fp
