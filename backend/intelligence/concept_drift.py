from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Sequence, Tuple
import numpy as np


@dataclass
class DriftResult:
    ks_score: float
    js_divergence: float
    wasserstein_distance: float
    composite: float

    drift_detected: bool
    severity: str
    regime_before: str
    regime_after: str
    recommendation: str

    def to_dict(self) -> dict:
        return asdict(self)


class ConceptDriftDetector:
    """
    Detects when historical behavior changes (concept drift):

      - Compares long-term distribution vs recent distribution
      - Measures distribution shift via KS / JS / Wasserstein
      - Detects model degradation (recent accuracy vs long-term accuracy)
      - Detects state-frequency changes and transition-statistic changes
      - Detects calibration degradation

    When drift is detected:
      1. Reduce trust in stale models (signal to meta-learner via recommendation)
      2. Increase recency weighting (signal to ensemble)
      3. Recommend additional validation / trigger a new generation
      4. Never blindly assume the new regime is predictable
    """

    def __init__(
        self,
        reference_window: int = 1000,
        recent_window: int = 150,
        composite_threshold: float = 0.25,
    ):
        self.reference_window = reference_window
        self.recent_window = recent_window
        self.composite_threshold = composite_threshold

        # Rolling performance history for drift detection
        self._per_round_correct_long: List[bool] = []
        self._per_round_correct_recent: List[bool] = []
        self._historical_composite: List[float] = []

    # ------------------------------------------------------------------
    # Distribution comparison
    # ------------------------------------------------------------------

    def _to_sizes(self, digits: Sequence[int]) -> np.ndarray:
        return np.array([1 if int(d) >= 5 else 0 for d in digits], dtype=np.float64)

    def _ks_statistic(self, a: np.ndarray, b: np.ndarray) -> float:
        """Kolmogorov-Smirnov statistic on binary samples."""
        if len(a) == 0 or len(b) == 0:
            return 0.0
        # ECDF over the 2 possible values {0, 1}
        a0 = float(np.mean(a == 0))
        b0 = float(np.mean(b == 0))
        a1 = 1.0 - a0
        b1 = 1.0 - b0
        ks = max(abs(a0 - b0), abs(1.0 - (1 - a0) - (1 - b0)))
        # KS-like max difference including the 1.0 endpoint
        ks = max(abs(a0 - b0), abs((a0 + a1) - (b0 + b1)))
        return ks

    def _js_divergence_binary(self, pa: float, pb: float) -> float:
        pa = min(0.999, max(0.001, pa))
        pb = min(0.999, max(0.001, pb))
        p = np.array([pa, 1.0 - pa])
        q = np.array([pb, 1.0 - pb])
        m = 0.5 * (p + q)
        js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
        return float(max(0.0, js))

    def _wasserstein_binary(self, pa: float, pb: float) -> float:
        return float(abs(pa - pb))

    def _transition_shift(self, sizes: Sequence[int], ref_n: int, rec_n: int) -> float:
        """Compare transition matrices in reference vs recent windows."""
        if ref_n < 10 or rec_n < 10:
            return 0.0
        ref = sizes[-ref_n - 1:-rec_n] if rec_n < ref_n else sizes[:-rec_n]
        rec = sizes[-rec_n:]
        if len(ref) < 3 or len(rec) < 3:
            return 0.0

        def t_matrix(seq):
            m = np.zeros((2, 2)) + 0.1
            for i in range(1, len(seq)):
                a = int(seq[i - 1])
                b = int(seq[i])
                if 0 <= a < 2 and 0 <= b < 2:
                    m[a, b] += 1
            return m / m.sum(axis=1, keepdims=True)

        m1 = t_matrix(list(ref))
        m2 = t_matrix(list(rec))
        return float(np.mean(np.abs(m1 - m2)))

    # ------------------------------------------------------------------
    # Detect drift
    # ------------------------------------------------------------------

    def detect(
        self,
        digits: Sequence[int],
        long_accuracy: Optional[float] = None,
        recent_accuracy: Optional[float] = None,
        calibration_error_before: float = 0.05,
    ) -> DriftResult:
        n = len(digits)
        if n < 2 * self.recent_window:
            # Not enough data → no drift claim
            return DriftResult(
                ks_score=0.0,
                js_divergence=0.0,
                wasserstein_distance=0.0,
                composite=0.0,
                drift_detected=False,
                severity="NONE",
                regime_before="UNKNOWN",
                regime_after="UNKNOWN",
                recommendation="INSUFFICIENT_HISTORY",
            )

        sizes = self._to_sizes(digits)
        ref_n = min(self.reference_window, n - self.recent_window)
        rec_n = self.recent_window
        ref_sizes = sizes[-(ref_n + rec_n):-rec_n]
        rec_sizes = sizes[-rec_n:]

        p_ref = float(np.mean(ref_sizes))
        p_rec = float(np.mean(rec_sizes))

        ks = self._ks_statistic(ref_sizes, rec_sizes)
        js = self._js_divergence_binary(p_ref, p_rec)
        wass = self._wasserstein_binary(p_ref, p_rec)
        tshift = self._transition_shift(list(sizes), ref_n, rec_n)

        # Accuracy drift component: if recent_accuracy is much worse than long
        accuracy_component = 0.0
        if long_accuracy is not None and recent_accuracy is not None and long_accuracy > 0:
            drop = max(0.0, long_accuracy - recent_accuracy)
            accuracy_component = min(1.0, drop / max(0.02, (long_accuracy - 0.5) + 0.01))

        # Calibration drift
        cal_component = min(1.0, calibration_error_before / 0.20)

        # Composite (weighted)
        composite = min(
            1.0,
            0.30 * ks * 3.0
            + 0.25 * js * 4.0
            + 0.20 * wass * 3.0
            + 0.10 * tshift * 2.5
            + 0.10 * accuracy_component
            + 0.05 * cal_component,
        )

        self._historical_composite.append(composite)
        if len(self._historical_composite) > 500:
            self._historical_composite.pop(0)

        detected = composite >= self.composite_threshold

        # Severity
        if composite >= 0.6:
            severity = "SEVERE"
        elif composite >= 0.4:
            severity = "MODERATE"
        elif detected:
            severity = "MILD"
        else:
            severity = "NONE"

        # Regime heuristic
        def classify(p):
            if p > 0.65:
                return "BIG_MOMENTUM"
            if p < 0.35:
                return "SMALL_MOMENTUM"
            return "EQUILIBRIUM"

        regime_before = classify(p_ref)
        regime_after = classify(p_rec)

        # Recommendation
        if not detected:
            recommendation = "STABLE"
        elif severity == "SEVERE":
            recommendation = "TRIGGER_NEW_GENERATION"
        elif severity == "MODERATE":
            recommendation = "INCREASE_RECENCY_WEIGHT"
        else:
            recommendation = "REDUCE_CONFIDENCE"

        return DriftResult(
            ks_score=float(ks),
            js_divergence=float(js),
            wasserstein_distance=float(wass),
            composite=float(composite),
            drift_detected=bool(detected),
            severity=severity,
            regime_before=regime_before,
            regime_after=regime_after,
            recommendation=recommendation,
        )

    def mean_drift(self, window: int = 50) -> float:
        data = self._historical_composite[-window:] if len(self._historical_composite) >= window else self._historical_composite
        return float(np.mean(data)) if data else 0.0
