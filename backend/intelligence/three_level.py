from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Sequence, Tuple
import numpy as np


@dataclass
class ThreeLevelProbabilities:
    # Level 1: P(next round success)
    p_success_l1: float
    p_big_l1: float
    p_small_l1: float

    # Level 2: P(at least 1 success within next 2 rounds)
    p_success_l2: float
    p_big_l2: float
    p_small_l2: float

    # Level 3: P(at least 1 success within next 3 rounds)
    p_success_l3: float
    p_big_l3: float
    p_small_l3: float

    # Empirical evidence counts
    l1_total_samples: int = 0
    l1_correct_samples: int = 0
    l2_total_samples: int = 0
    l2_correct_samples: int = 0
    l3_total_samples: int = 0
    l3_correct_samples: int = 0

    empirical: bool = False  # True if computed from historical data, not modeled

    def to_dict(self) -> dict:
        return asdict(self)


class ThreeLevelAnalysis:
    """
    TRUE THREE-LEVEL ANALYSIS — not manufactured from L1.

    L1: P(outcome at next round | current state)
    L2: P(success within next 2 rounds | current state)
    L3: P(success within next 3 rounds | current state)

    Both paths are provided:
      a) Model-based: derived from the proposed single-round probability
         with mild correlation penalty when historical evidence is absent.
      b) Empirical: walk-forward conditioned on historical similar states,
         if enough data exists.

    This module also stores rolling estimates computed from the audit log
    so L2/L3 are not merely synthetic derivations of L1.
    """

    def __init__(self, correlation_rho: float = 0.06):
        # Mild round-to-round correlation penalty; prevents L2/L3 claims from being
        # overly optimistic when consecutive rounds are not truly independent.
        self.correlation_rho = correlation_rho

        # Rolling empirical counters for calibration
        self.l1_attempts = 0
        self.l1_correct = 0
        self.l2_attempts = 0
        self.l2_correct = 0
        self.l3_attempts = 0
        self.l3_correct = 0

        # Per-confidence-bucket rollups for empirical calibration
        self.bucket_l1: Dict[str, Tuple[int, int]] = {}
        self.bucket_l2: Dict[str, Tuple[int, int]] = {}
        self.bucket_l3: Dict[str, Tuple[int, int]] = {}

    # ------------------------------------------------------------------
    # Online calibration updates from resolved predictions
    # ------------------------------------------------------------------

    def _bucket(self, p: float) -> str:
        lo = int(math.floor(max(0.0, min(0.99, p)) * 10)) / 10.0
        return f"{lo:.1f}-{lo + 0.1:.1f}"

    def update_empirical(
        self,
        p_l1: float,
        p_l2: float,
        p_l3: float,
        l1_win: bool,
        l2_win: bool,
        l3_win: bool,
    ) -> None:
        self.l1_attempts += 1
        if l1_win:
            self.l1_correct += 1
        self.l2_attempts += 1
        if l2_win:
            self.l2_correct += 1
        self.l3_attempts += 1
        if l3_win:
            self.l3_correct += 1

        def add(d: Dict[str, Tuple[int, int]], key: str, win: bool):
            a, w = d.get(key, (0, 0))
            d[key] = (a + 1, w + (1 if win else 0))

        add(self.bucket_l1, self._bucket(p_l1), l1_win)
        add(self.bucket_l2, self._bucket(p_l2), l2_win)
        add(self.bucket_l3, self._bucket(p_l3), l3_win)

    def empirical_win_rates(self) -> Tuple[float, float, float]:
        def safe(a, b):
            return (a / b) if b > 0 else 0.0

        return (
            safe(self.l1_correct, self.l1_attempts),
            safe(self.l2_correct, self.l2_attempts),
            safe(self.l3_correct, self.l3_attempts),
        )

    # ------------------------------------------------------------------
    # Empirical historical walk-forward evaluation on provided history
    # ------------------------------------------------------------------

    def _history_side(self, d: int) -> int:
        return 1 if int(d) >= 5 else 0

    def evaluate_empirically(
        self,
        history_digits: Sequence[int],
        proposed_side: str,
        p_big_l1: float,
        window: int = 200,
        min_samples: int = 50,
    ) -> ThreeLevelProbabilities:
        """
        Walk forward through `history_digits` using the same short-window
        frequency heuristic the predictor uses, and count how often the
        predicted side wins within 1 / 2 / 3 rounds.

        This is a ground-truth empirical sanity check — NOT a synthetic formula.
        """
        digits = [int(d) for d in history_digits]
        n = len(digits)
        if n < window + 4:
            # Not enough history → return model-based
            return self.model_based(p_big_l1)

        predict_big = proposed_side == "Big"

        total_l1 = 0
        win_l1 = 0
        total_l2 = 0
        win_l2 = 0
        total_l3 = 0
        win_l3 = 0

        # Walk forward using a simple rolling short-window frequency heuristic
        for i in range(window, n - 3):
            ctx = digits[i - window:i]
            # Simple heuristic: recent frequency → predicted side
            sizes = [1 if d >= 5 else 0 for d in ctx]
            big_rate = float(np.mean(sizes[-20:])) if len(sizes) >= 20 else float(np.mean(sizes))
            heuristic_side = big_rate >= 0.5
            # Only consider cases where the heuristic aligns with the proposed_side
            # (i.e., measure when the engine would have predicted the same direction)
            if heuristic_side != predict_big:
                continue

            target = 1 if predict_big else 0
            s1 = self._history_side(digits[i])
            s2 = self._history_side(digits[i + 1])
            s3 = self._history_side(digits[i + 2])

            total_l1 += 1
            total_l2 += 1
            total_l3 += 1

            if s1 == target:
                win_l1 += 1
                win_l2 += 1
                win_l3 += 1
            elif s2 == target:
                win_l2 += 1
                win_l3 += 1
            elif s3 == target:
                win_l3 += 1

        # Use empirical if sufficient evidence
        if total_l1 >= min_samples:
            p1 = (win_l1 + 1) / (total_l1 + 2)
            p2 = (win_l2 + 1) / (total_l2 + 2)
            p3 = (win_l3 + 1) / (total_l3 + 2)
        else:
            mb = self.model_based(p_big_l1)
            p1, p2, p3 = mb.p_success_l1, mb.p_success_l2, mb.p_success_l3

        if predict_big:
            return ThreeLevelProbabilities(
                p_success_l1=p1,
                p_big_l1=p1,
                p_small_l1=1 - p1,
                p_success_l2=p2,
                p_big_l2=p2,
                p_small_l2=1 - p2,
                p_success_l3=p3,
                p_big_l3=p3,
                p_small_l3=1 - p3,
                l1_total_samples=total_l1,
                l1_correct_samples=win_l1,
                l2_total_samples=total_l2,
                l2_correct_samples=win_l2,
                l3_total_samples=total_l3,
                l3_correct_samples=win_l3,
                empirical=total_l1 >= min_samples,
            )
        else:
            return ThreeLevelProbabilities(
                p_success_l1=p1,
                p_big_l1=1 - p1,
                p_small_l1=p1,
                p_success_l2=p2,
                p_big_l2=1 - p2,
                p_small_l2=p2,
                p_success_l3=p3,
                p_big_l3=1 - p3,
                p_small_l3=p3,
                l1_total_samples=total_l1,
                l1_correct_samples=win_l1,
                l2_total_samples=total_l2,
                l2_correct_samples=win_l2,
                l3_total_samples=total_l3,
                l3_correct_samples=win_l3,
                empirical=total_l1 >= min_samples,
            )

    # ------------------------------------------------------------------
    # Model-based derivation (honest; not claiming independence)
    # ------------------------------------------------------------------

    def model_based(self, p_big_l1: float) -> ThreeLevelProbabilities:
        """
        Derive L2/L3 from L1 with honest round-to-round correlation penalty.

        P_single = p
        P(L2) = 1 - (1-p) * (1 - p2) where p2 is *slightly* degraded from p
        P(L3) = 1 - (1-p) * (1 - p2) * (1 - p3) with more degradation

        This matches the calibration logic historically used in the Telegram
        display, but we add an explicit correlation penalty so that when
        `correlation_rho > 0` the joint probability is strictly LESS than
        an independence-based formula would claim.
        """
        p1 = min(0.99, max(0.01, float(p_big_l1)))
        # Successive rounds have slightly degraded single-round probability
        p2 = 0.5 + 0.94 * (p1 - 0.5)
        p3 = 0.5 + 0.88 * (p1 - 0.5)

        # Apply correlation penalty: reduce each successive "independence gain"
        rho = self.correlation_rho
        p_joint2_raw = 1.0 - (1.0 - p1) * (1.0 - p2)
        p_joint3_raw = p_joint2_raw + (1.0 - p_joint2_raw) * p3
        # Shrink toward L1 to compensate for correlation
        p_joint2 = p1 + (p_joint2_raw - p1) * (1.0 - rho)
        p_joint3 = p_joint2 + (p_joint3_raw - p_joint2) * (1.0 - rho * 1.2)

        # Clip strictly below 1.0 and above p1 (monotonicity)
        p_joint2 = min(0.995, max(p1, p_joint2))
        p_joint3 = min(0.999, max(p_joint2, p_joint3))

        big = p1
        return ThreeLevelProbabilities(
            p_success_l1=p1,
            p_big_l1=big,
            p_small_l1=1 - p1,
            p_success_l2=p_joint2,
            p_big_l2=big if big >= 0.5 else (1 - p_joint2),
            p_small_l2=1 - big if big < 0.5 else (1 - p_joint2),
            p_success_l3=p_joint3,
            p_big_l3=big if big >= 0.5 else (1 - p_joint3),
            p_small_l3=1 - big if big < 0.5 else (1 - p_joint3),
            empirical=False,
        )
