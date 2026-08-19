"""Continuous calibration for online binary predictions.

Every round receives a forecast.  Resolved, timestamped forecasts from the
database continuously calibrate the next forecast's confidence, so the system
learns day by day without suppressing predictions.

This module is explicitly tuned for a **win-within-3-levels (Martingale)**
strategy.  The calibrated probability and the validated-edge flag therefore
measure not only per-round accuracy but also the joint probability of at
least one win across three sequential forecasts, which is the quantity that
determines Martingale survival.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.database import PredictionAudit


# ============================================================================
# Policy: stricter once the engine has seen real history, so 3-level win
# targets (P >= 0.90+) are only claimed when the audit history supports it.
# ============================================================================

@dataclass(frozen=True)
class EvidencePolicy:
    min_resolved: int = 50           # Scaled dynamically based on sample volume
    min_local_samples: int = 20      # comparable (raw-confidence bucket) size
    local_bandwidth: float = 0.07    # ± 7% band on raw confidence
    min_brier_improvement: float = 0.0010
    # 3-level specific: we only claim validated edge if the rolling 3-forecast
    # joint win probability has a Wilson lower bound above 0.88.
    min_3level_lower_bound: float = 0.88
    # A minimum per-round calibrated accuracy needed for 3-level to make sense.
    min_per_round_lower_bound: float = 0.53
    recency_window: int = 200        # focus on recent window for dynamic shifts


def _wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    """Conservative 95% lower confidence bound for directional accuracy."""
    if total <= 0:
        return 0.0
    p = wins / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return max(0.0, (centre - spread) / denominator)


def _rolling_three_level_win_rate(ordered_results: List[Tuple[float, float]]) -> Tuple[int, int]:
    """Return (wins_in_3, total_3_tuples) from ordered (prob_big, actual_0_1).

    For each consecutive 3-forecast window we count a "window win" iff at
    least 1 of the 3 rounds was correct.
    """
    if len(ordered_results) < 3:
        return 0, 0
    wins = 0
    total = 0
    # Stride by 1 so every position is evaluated (conservative evaluation).
    for i in range(len(ordered_results) - 2):
        win = False
        for j in range(3):
            p_big, actual = ordered_results[i + j]
            predicted_big = p_big >= 0.5
            correct = (predicted_big and actual >= 0.5) or ((not predicted_big) and actual < 0.5)
            if correct:
                win = True
                break
        if win:
            wins += 1
        total += 1
    return wins, total


def _per_round_correctness(ordered_results: List[Tuple[float, float]]) -> Tuple[int, int]:
    wins = 0
    total = 0
    for p_big, actual in ordered_results:
        predicted_big = p_big >= 0.5
        correct = (predicted_big and actual >= 0.5) or ((not predicted_big) and actual < 0.5)
        if correct:
            wins += 1
        total += 1
    return wins, total


def _platt_shrink(raw_confidence: float, observed_win_rate: float, n: int) -> float:
    """Bayesian shrinkage of raw stated confidence toward the observed rate.

    The greater ``n`` the more we trust the historical rate over the raw
    statement. Returns a calibrated probability in (0, 1).
    """
    effective_prior_weight = 5.0
    a = effective_prior_weight * observed_win_rate + n * observed_win_rate
    b = effective_prior_weight * (1.0 - observed_win_rate) + n * (1.0 - observed_win_rate)
    posterior_mean = a / (a + b) if (a + b) > 0 else observed_win_rate
    # Blend toward observed: when lots of evidence exists, follow it closely.
    trust = min(1.0, n / 250.0)
    calibrated = trust * posterior_mean + (1.0 - trust) * raw_confidence
    # Honest floor that respects signal weakness
    floor = 0.50
    return max(floor, min(0.985, calibrated))


def _joint3_from_single(p_single: float, rho: float = 0.06) -> float:
    """Joint P(at least 1 win in 3) from a single-round calibrated p.

    ``rho`` is a mild correlation penalty so that overlapping rounds are not
    treated as independent.
    """
    h2 = 0.5 + 0.94 * (p_single - 0.5)
    h3 = 0.5 + 0.88 * (p_single - 0.5)
    p_indep = 1.0 - (1.0 - p_single) * (1.0 - h2) * (1.0 - h3)
    return 0.5 + (p_indep - 0.5) * (1.0 - rho)


def evaluate_records(
    records: Iterable[Mapping[str, object]], raw_probability_big: float, policy: EvidencePolicy = EvidencePolicy()
) -> dict:
    """Calibrate a proposed probability using only resolved historical audits.

    ``records`` must contain a probability_big and an actual_size. It is a
    pure function so its acceptance criteria can be regression-tested without a
    database or model runtime.
    """
    raw_probability_big = min(1.0 - 1e-6, max(1e-6, float(raw_probability_big)))

    # Preserve insertion order: records typically come in DESC id order, so
    # reverse to chronological order for 3-forecast rolling windows.
    clean_desc: List[Tuple[float, float]] = []
    for record in records:
        actual = record.get("actual_size")
        probability = record.get("probability_big")
        if actual not in {"Big", "Small"} or probability is None:
            continue
        try:
            probability = min(1.0 - 1e-6, max(1e-6, float(probability)))
        except (TypeError, ValueError):
            continue
        actual_val = 1.0 if actual == "Big" else 0.0
        clean_desc.append((probability, actual_val))
    clean_chron = list(reversed(clean_desc))

    proposed_side = "Big" if raw_probability_big >= 0.5 else "Small"
    raw_confidence = max(raw_probability_big, 1.0 - raw_probability_big)
    n_total = len(clean_chron)

    if n_total == 0:
        return {
            "action": "FORECAST",
            "reason": "LEARNING_FROM_FIRST_OUTCOME",
            "confidence": 0.5,
            "raw_confidence": raw_confidence,
            "resolved_predictions": 0,
            "local_samples": 0,
            "brier_improvement": 0.0,
            "accuracy_lower_bound": 0.0,
            "three_level_win_rate": 0.0,
            "three_level_lower_bound": 0.0,
            "per_round_win_rate": 0.0,
            "recent_win_rate": 0.0,
            "joint3_probability": _joint3_from_single(0.5),
            "validated_edge": False,
        }

    # Dynamic scaling of required resolved sample threshold based on sample availability
    effective_min_resolved = min(policy.min_resolved, max(30, n_total // 3))

    base_rate = sum(target for _, target in clean_chron) / n_total
    model_brier = sum((p - t) ** 2 for p, t in clean_chron) / n_total
    null_brier = sum((base_rate - t) ** 2 for _, t in clean_chron) / n_total
    brier_improvement = null_brier - model_brier

    # Recency-weighted evaluation: separate recent window
    recent_slice = clean_chron[-policy.recency_window:] if len(clean_chron) > policy.recency_window else clean_chron
    recent_wins, recent_n = _per_round_correctness(recent_slice)
    recent_win_rate = recent_wins / recent_n if recent_n > 0 else 0.5

    # Calibrate against previous predictions with similar stated confidence.
    comparable = [
        (p, t) for p, t in clean_chron if abs(max(p, 1.0 - p) - raw_confidence) <= policy.local_bandwidth
    ]
    if len(comparable) < policy.min_local_samples:
        comparable = clean_chron  # fall back to all history

    wins_per_round, n_local = _per_round_correctness(comparable)
    observed_win_rate = wins_per_round / n_local if n_local > 0 else 0.5

    # Blend observed rate with recent win rate (65% comparable, 35% recent window)
    blended_observed = 0.65 * observed_win_rate + 0.35 * recent_win_rate

    # Platt-beta shrinkage of raw confidence to observed rate
    calibrated = _platt_shrink(raw_confidence, blended_observed, n_local)

    lower_bound = _wilson_lower_bound(wins_per_round, n_local)

    # 3-level win rate over chronological history (rolling windows of 3)
    wins_3, n_3 = _rolling_three_level_win_rate(clean_chron)
    if n_3 > 0:
        three_level_win_rate = wins_3 / n_3
        three_level_lower = _wilson_lower_bound(wins_3, n_3)
    else:
        three_level_win_rate = _joint3_from_single(blended_observed)
        three_level_lower = 0.0
    joint3_probability = _joint3_from_single(calibrated)

    # Validated edge: check both per-round and 3-level joint performance
    per_round_ok = n_total >= effective_min_resolved and lower_bound > policy.min_per_round_lower_bound
    three_level_ok = (
        n_3 >= max(30, policy.min_local_samples)
        and three_level_lower > policy.min_3level_lower_bound
    )
    brier_ok = brier_improvement >= policy.min_brier_improvement
    validated_edge = bool(per_round_ok and three_level_ok and brier_ok)

    if validated_edge:
        reason = "VALIDATED_3LEVEL_EDGE"
    elif n_total < effective_min_resolved:
        reason = "LEARNING_CALIBRATION"
    elif per_round_ok and not three_level_ok:
        reason = "PER_ROUND_OK_3LEVEL_PENDING"
    elif lower_bound > 0.50:
        reason = "MARGINAL_EDGE"
    else:
        reason = "HONEST_CALIBRATION"

    return {
        "action": "FORECAST",
        "reason": reason,
        "confidence": round(float(calibrated), 4),
        "raw_confidence": round(float(raw_confidence), 4),
        "resolved_predictions": int(n_total),
        "local_samples": int(n_local),
        "brier_improvement": round(float(brier_improvement), 6),
        "accuracy_lower_bound": round(float(lower_bound), 4),
        "per_round_win_rate": round(float(observed_win_rate), 4),
        "recent_win_rate": round(float(recent_win_rate), 4),
        "three_level_win_rate": round(float(three_level_win_rate), 4),
        "three_level_lower_bound": round(float(three_level_lower), 4),
        "joint3_probability": round(float(joint3_probability), 4),
        "proposed_side": proposed_side,
        "validated_edge": validated_edge,
    }


class EvidenceGate:
    """Loads resolved predictions and applies the immutable evidence policy."""

    def __init__(self, policy: EvidencePolicy = EvidencePolicy(), history_limit: int = 8000):
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
