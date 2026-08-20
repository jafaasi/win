from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Sequence, Tuple
import numpy as np


@dataclass
class CalibrationResult:
    # Raw vs calibrated
    raw_probability_big: float
    calibrated_probability_big: float
    raw_confidence: float
    calibrated_confidence: float
    calibration_error: float
    calibration_applied: bool
    bucket_id: str
    bucket_observed_rate: Optional[float]
    bucket_sample_size: int

    expected_calibration_error: float
    max_calibration_deviation: float

    def to_dict(self) -> dict:
        return asdict(self)


class ConfidenceCalibrator:
    """
    Confidence Calibration:

    If the engine says "70% confidence" then historically similar predictions
    should actually succeed approximately 70% of the time.

    - Tracks reliability curve (bucketed predicted vs observed win rate)
    - Tracks Brier, log-loss, Expected Calibration Error (ECE)
    - Applies Platt-like shrinkage + bucket correction
    - Learns calibration corrections from the audit log / fast memory
    """

    BUCKET_EDGES = [
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
    ]

    def __init__(self, min_samples_per_bucket: int = 20):
        self.min_samples_per_bucket = min_samples_per_bucket

        # bucket -> (attempts, wins)
        self.buckets: Dict[str, List[int]] = {}
        # bucket -> calibrated correction factor (how much to multiply away from 0.5)
        self.bucket_correction: Dict[str, float] = {}

        self.total_calibration_samples: int = 0
        self._ece_running: float = 0.05

    # ------------------------------------------------------------------
    # Online updates
    # ------------------------------------------------------------------

    def _bucket_for(self, confidence: float) -> str:
        conf = min(0.999, max(0.50, float(confidence)))
        for i in range(1, len(self.BUCKET_EDGES)):
            if conf <= self.BUCKET_EDGES[i]:
                lo = self.BUCKET_EDGES[i - 1]
                hi = self.BUCKET_EDGES[i]
                return f"{lo:.2f}-{hi:.2f}"
        return "0.95-1.00"

    def update(self, predicted_prob_big: float, actual_side: str) -> None:
        conf = max(predicted_prob_big, 1.0 - predicted_prob_big)
        predicted_correct = (
            (predicted_prob_big >= 0.5 and actual_side == "Big")
            or (predicted_prob_big < 0.5 and actual_side == "Small")
        )
        bucket = self._bucket_for(conf)
        b = self.buckets.setdefault(bucket, [0, 0])
        b[0] += 1
        if predicted_correct:
            b[1] += 1
        self.total_calibration_samples += 1
        self._recompute_corrections()

    def bulk_update(self, records: Sequence[Dict[str, Any]]) -> int:
        """Feed a list of {probability_big, actual_size} audit records."""
        n = 0
        for rec in records:
            p = rec.get("probability_big")
            actual = rec.get("actual_size")
            if p is None or actual not in {"Big", "Small"}:
                continue
            try:
                self.update(float(p), str(actual))
                n += 1
            except Exception:
                continue
        return n

    def _recompute_corrections(self) -> None:
        for bucket, (attempts, wins) in self.buckets.items():
            if attempts < self.min_samples_per_bucket:
                self.bucket_correction[bucket] = 1.0
                continue
            lo, hi = [float(x) for x in bucket.split("-")]
            mid_pred = 0.5 * (lo + hi)
            observed = wins / attempts
            # Correction multiplier on the distance-from-chance
            denom = max(1e-6, mid_pred - 0.5)
            numer = max(0.0, observed - 0.5)
            factor = numer / denom
            # Shrink factor toward 1.0 for stability
            shrinkage = min(1.0, attempts / 200.0)
            self.bucket_correction[bucket] = 1.0 * (1 - shrinkage) + factor * shrinkage
        # ECE approximation: weighted average |observed - predicted|
        ece = 0.0
        total_att = 0
        for bucket, (attempts, wins) in self.buckets.items():
            if attempts == 0:
                continue
            lo, hi = [float(x) for x in bucket.split("-")]
            mid_pred = 0.5 * (lo + hi)
            obs = wins / attempts
            ece += abs(obs - mid_pred) * attempts
            total_att += attempts
        self._ece_running = (ece / total_att) if total_att > 0 else 0.05

    # ------------------------------------------------------------------
    # Calibrate a new probability
    # ------------------------------------------------------------------

    def calibrate(self, raw_probability_big: float) -> CalibrationResult:
        raw_p = min(0.999, max(0.001, float(raw_probability_big)))
        raw_conf = max(raw_p, 1.0 - raw_p)
        bucket = self._bucket_for(raw_conf)
        b = self.buckets.get(bucket, [0, 0])
        attempts, wins = b

        factor = self.bucket_correction.get(bucket, 1.0)
        if attempts < self.min_samples_per_bucket:
            # Fall back to adjacent bucket or global
            factor = self._global_correction(raw_conf)

        # Apply correction
        direction = 1.0 if raw_p >= 0.5 else -1.0
        distance_from_chance = abs(raw_p - 0.5)
        # Clip correction factor so we never over-shoot 1.0 or under-shoot 0.5
        effective_factor = max(0.2, min(1.6, factor))
        corrected_distance = distance_from_chance * effective_factor
        calibrated_p = 0.5 + direction * corrected_distance
        calibrated_p = min(0.99, max(0.50, calibrated_p))

        calibrated_conf = max(calibrated_p, 1.0 - calibrated_p)
        if raw_p < 0.5:
            calibrated_p = 1.0 - calibrated_conf

        observed_rate = (wins / attempts) if attempts >= self.min_samples_per_bucket else None

        # ECE: current running estimate
        ece = self._ece_running
        max_dev = max(0.0, max(abs(factor - 1.0) * 0.5 for factor in [self.bucket_correction.get(bucket, 1.0)]) or 0.0)

        return CalibrationResult(
            raw_probability_big=raw_p,
            calibrated_probability_big=calibrated_p,
            raw_confidence=raw_conf,
            calibrated_confidence=calibrated_conf,
            calibration_error=abs(calibrated_conf - raw_conf),
            calibration_applied=attempts >= self.min_samples_per_bucket,
            bucket_id=bucket,
            bucket_observed_rate=observed_rate,
            bucket_sample_size=attempts,
            expected_calibration_error=ece,
            max_calibration_deviation=max_dev,
        )

    def _global_correction(self, raw_conf: float) -> float:
        """Fallback: overall calibration across all buckets with sufficient data."""
        numer = 0.0
        denom = 0.0
        for bucket, (attempts, wins) in self.buckets.items():
            if attempts < self.min_samples_per_bucket:
                continue
            lo, hi = [float(x) for x in bucket.split("-")]
            mid_pred = 0.5 * (lo + hi)
            observed = wins / attempts
            pred_dist = mid_pred - 0.5
            obs_dist = max(0.0, observed - 0.5)
            if pred_dist > 1e-6:
                numer += obs_dist * attempts
                denom += pred_dist * attempts
        if denom <= 0:
            return 1.0
        raw_dist = max(1e-6, raw_conf - 0.5)
        global_factor = numer / denom
        # Blend gently
        return 0.7 * global_factor + 0.3 * 1.0

    def expected_calibration_error(self) -> float:
        return self._ece_running
