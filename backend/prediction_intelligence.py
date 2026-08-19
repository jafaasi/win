"""Continuous calibration for online binary predictions.

Every round receives a forecast.  Resolved, timestamped forecasts from the
database continuously calibrate the next forecast's confidence, so the system
learns day by day without suppressing predictions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.database import PredictionAudit


@dataclass(frozen=True)
class EvidencePolicy:
    min_resolved: int = 200
    min_local_samples: int = 40
    local_bandwidth: float = 0.06
    min_brier_improvement: float = 0.002


def _wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    """Conservative 95% lower confidence bound for directional accuracy."""
    if total <= 0:
        return 0.0
    p = wins / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return max(0.0, (centre - spread) / denominator)


def evaluate_records(
    records: Iterable[Mapping[str, object]], raw_probability_big: float, policy: EvidencePolicy = EvidencePolicy()
) -> dict:
    """Calibrate a proposed probability using only resolved historical audits.

    ``records`` must contain a probability_big and an actual_size.  It is a
    pure function so its acceptance criteria can be regression-tested without a
    database or model runtime.
    """
    raw_probability_big = min(1.0 - 1e-6, max(1e-6, float(raw_probability_big)))
    clean = []
    for record in records:
        actual = record.get("actual_size")
        probability = record.get("probability_big")
        if actual not in {"Big", "Small"} or probability is None:
            continue
        try:
            probability = min(1.0 - 1e-6, max(1e-6, float(probability)))
        except (TypeError, ValueError):
            continue
        clean.append((probability, 1.0 if actual == "Big" else 0.0))

    proposed_side = "Big" if raw_probability_big >= 0.5 else "Small"
    raw_confidence = max(raw_probability_big, 1.0 - raw_probability_big)
    if not clean:
        return {
            "action": "FORECAST",
            "reason": "LEARNING_FROM_FIRST_OUTCOME",
            "confidence": 0.5,
            "raw_confidence": raw_confidence,
            "resolved_predictions": 0,
            "local_samples": 0,
            "brier_improvement": 0.0,
            "accuracy_lower_bound": 0.0,
        }

    base_rate = sum(target for _, target in clean) / len(clean)
    model_brier = sum((probability - target) ** 2 for probability, target in clean) / len(clean)
    null_brier = sum((base_rate - target) ** 2 for _, target in clean) / len(clean)
    brier_improvement = null_brier - model_brier

    # Calibrate against previous predictions with similar stated confidence.
    comparable = [
        (probability, target)
        for probability, target in clean
        if abs(max(probability, 1.0 - probability) - raw_confidence) <= policy.local_bandwidth
    ]
    if not comparable:
        comparable = clean
    wins = sum(
        1
        for probability, target in comparable
        if (probability >= 0.5 and target == 1.0) or (probability < 0.5 and target == 0.0)
    )
    local_samples = len(comparable)
    # A weak Beta(1, 1) prior prevents a tiny perfect sample from looking certain.
    calibrated = (wins + 1) / (local_samples + 2)
    lower_bound = _wilson_lower_bound(wins, local_samples)
    validated_edge = (
        len(clean) >= policy.min_resolved
        and local_samples >= policy.min_local_samples
        and brier_improvement >= policy.min_brier_improvement
        and lower_bound > 0.5
    )
    return {
        # Forecast every issue.  The model's historical results change only
        # the calibrated confidence and its evidence label, never whether a
        # user receives a prediction.
        "action": "FORECAST",
        "reason": "VALIDATED_EDGE" if validated_edge else "LEARNING_CALIBRATION",
        "confidence": calibrated,
        "raw_confidence": raw_confidence,
        "resolved_predictions": len(clean),
        "local_samples": local_samples,
        "brier_improvement": brier_improvement,
        "accuracy_lower_bound": lower_bound,
        "proposed_side": proposed_side,
        "validated_edge": validated_edge,
    }


class EvidenceGate:
    """Loads resolved predictions and applies the immutable evidence policy."""

    def __init__(self, policy: EvidencePolicy = EvidencePolicy(), history_limit: int = 5000):
        self.policy = policy
        self.history_limit = history_limit

    def assess(self, db, raw_probability_big: float) -> dict:
        # Keep the statistical core importable in lightweight environments (and
        # in tests) that do not have the database driver installed.
        from backend.database import PredictionAudit

        rows = (
            db.query(PredictionAudit)
            .filter(PredictionAudit.actual_size.isnot(None))
            .order_by(PredictionAudit.id.desc())
            .limit(self.history_limit)
            .all()
        )
        records = [
            {"probability_big": row.probability_big, "actual_size": row.actual_size}
            for row in rows
        ]
        return evaluate_records(records, raw_probability_big, self.policy)
