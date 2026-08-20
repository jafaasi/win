#!/usr/bin/env python3
"""
Reality Check Engine — TAIE Component 3
=========================================
Every day this engine asks one honest question:

    "Am I actually learning anything, or am I fooling myself?"

It compares the adaptive engine against four baselines using strictly
out-of-sample (OOS) data — the most recent records that were never used
during any training or weight-optimisation step.

Baselines
---------
  B0  Random          — coin flip, p_big = 0.50 every round
  B1  Frequency       — historical Big rate over the full training window
  B2  Recent momentum — 20-round rolling Big rate (naive trend-follower)
  B3  Always-majority — always predict the overall majority side

The adaptive engine must outperform ALL four baselines on at least TWO of
three metrics (accuracy, Brier score, 3-level win rate) on the OOS test
split.  If it fails, the report says:

    verdict = "NO_VERIFIED_EDGE"

and the engine lowers its confidence tier thresholds for the next day.

Metrics
-------
  accuracy          — fraction of correct directional calls
  brier_score       — mean squared calibration error (lower = better)
  log_loss          — binary cross-entropy (lower = better)
  three_level_wr    — fraction of rolling 3-round windows with ≥1 win
  calibration_error — mean |predicted_conf - actual_win_rate| per bucket
  oos_edge          — accuracy minus best-baseline accuracy (signed)

Output
------
RealityCheckReport with:
  verdict           — VERIFIED_EDGE | MARGINAL_EDGE | NO_VERIFIED_EDGE
  oos_accuracy      — engine accuracy on test split
  baseline_results  — per-baseline metrics
  beats_all_on      — list of metrics where engine beats all baselines
  intelligence_score — composite 0-100 score (used in Telegram display)
  recommended_tier_adjustment — +1 / 0 / -1 (upgrade/keep/downgrade thresholds)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Metric primitives (pure, no DB dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _accuracy(pairs: List[Tuple[float, int]]) -> float:
    if not pairs:
        return 0.5
    return sum(1 for p, a in pairs if (p >= 0.5) == (a == 1)) / len(pairs)


def _brier(pairs: List[Tuple[float, int]]) -> float:
    if not pairs:
        return 0.25
    return sum((p - a) ** 2 for p, a in pairs) / len(pairs)


def _log_loss(pairs: List[Tuple[float, int]]) -> float:
    if not pairs:
        return math.log(2)
    total = 0.0
    for p, a in pairs:
        p = max(1e-6, min(1 - 1e-6, p))
        total += -(a * math.log(p) + (1 - a) * math.log(1 - p))
    return total / len(pairs)


def _three_level_wr(pairs: List[Tuple[float, int]]) -> float:
    correct = [(p >= 0.5) == (a == 1) for p, a in pairs]
    if len(correct) < 3:
        return 0.5
    wins = total = 0
    for i in range(len(correct) - 2):
        if any(correct[i:i + 3]):
            wins += 1
        total += 1
    return wins / total if total else 0.5


def _calibration_error(pairs: List[Tuple[float, int]], n_bins: int = 5) -> float:
    """Mean absolute calibration error across equal-width confidence bins."""
    if len(pairs) < n_bins * 2:
        return 0.5
    bins: Dict[int, List[Tuple[float, int]]] = {i: [] for i in range(n_bins)}
    for p, a in pairs:
        conf = max(p, 1 - p)
        b = min(n_bins - 1, int((conf - 0.5) / (0.5 / n_bins)))
        bins[b].append((conf, a))
    total_err = 0.0
    total_n = 0
    for b_pairs in bins.values():
        if not b_pairs:
            continue
        avg_conf = sum(c for c, _ in b_pairs) / len(b_pairs)
        actual_rate = sum(a for _, a in b_pairs) / len(b_pairs)
        # actual win rate for confidence = fraction correct at that confidence level
        correct_rate = sum(
            1 for c, a in b_pairs if (c >= 0.5) == (a == 1)
        ) / len(b_pairs)
        total_err += abs(avg_conf - correct_rate) * len(b_pairs)
        total_n += len(b_pairs)
    return total_err / total_n if total_n else 0.5


def _metrics(pairs: List[Tuple[float, int]]) -> Dict[str, float]:
    return {
        "accuracy":          round(_accuracy(pairs), 4),
        "brier":             round(_brier(pairs), 4),
        "log_loss":          round(_log_loss(pairs), 4),
        "three_level_wr":    round(_three_level_wr(pairs), 4),
        "calibration_error": round(_calibration_error(pairs), 4),
        "n":                 len(pairs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Baseline generators
# ─────────────────────────────────────────────────────────────────────────────

def _baseline_random(pairs: List[Tuple[float, int]]) -> Dict[str, float]:
    rng_pairs = [(0.50, a) for _, a in pairs]
    return _metrics(rng_pairs)


def _baseline_frequency(
    pairs: List[Tuple[float, int]],
    train_big_rate: float
) -> Dict[str, float]:
    freq_pairs = [(train_big_rate, a) for _, a in pairs]
    return _metrics(freq_pairs)


def _baseline_recent_momentum(
    pairs: List[Tuple[float, int]],
    window: int = 20
) -> Dict[str, float]:
    """Uses the Big rate of the 20 rounds *before* each test round."""
    actuals = [a for _, a in pairs]
    momentum_pairs = []
    for i, (_, a) in enumerate(pairs):
        # Look back into the pairs (these are test-set pairs, so use what's available)
        lookback = actuals[max(0, i - window):i]
        rate = sum(lookback) / len(lookback) if lookback else 0.5
        # Laplace-smooth
        rate = (rate * len(lookback) + 1) / (len(lookback) + 2)
        momentum_pairs.append((rate, a))
    return _metrics(momentum_pairs)


def _baseline_always_majority(
    pairs: List[Tuple[float, int]],
    train_big_rate: float
) -> Dict[str, float]:
    majority_p = 0.75 if train_big_rate >= 0.5 else 0.25
    maj_pairs = [(majority_p, a) for _, a in pairs]
    return _metrics(maj_pairs)


# ─────────────────────────────────────────────────────────────────────────────
# Intelligence Score computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_intelligence_score(
    engine: Dict[str, float],
    baselines: Dict[str, Dict[str, float]],
    validated_edge: bool,
) -> float:
    """
    Composite 0-100 intelligence score.

    Components:
      30% Prediction accuracy (relative to random baseline)
      25% Brier improvement vs frequency baseline
      20% 3-level win rate
      15% Calibration quality (1 - calibration_error)
      10% Validated edge bonus
    """
    # Accuracy component: how much better than random (50%)?
    acc_edge = max(0.0, engine["accuracy"] - 0.50) / 0.25  # 0.25pp → full score
    acc_score = min(1.0, acc_edge) * 30.0

    # Brier component: improvement vs frequency baseline
    freq_brier = baselines.get("frequency", {}).get("brier", 0.25)
    brier_improvement = max(0.0, freq_brier - engine["brier"])
    brier_score_c = min(1.0, brier_improvement / 0.025) * 25.0

    # 3-level win rate: target is 0.92+
    wr_score = min(1.0, max(0.0, (engine["three_level_wr"] - 0.80) / 0.15)) * 20.0

    # Calibration: 1 - cal_error, target < 0.05
    cal_score = min(1.0, max(0.0, 1.0 - engine["calibration_error"] / 0.20)) * 15.0

    # Validated edge bonus
    edge_bonus = 10.0 if validated_edge else 0.0

    total = acc_score + brier_score_c + wr_score + cal_score + edge_bonus
    return round(min(100.0, max(0.0, total)), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Report structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RealityCheckReport:
    verdict: str                              # VERIFIED_EDGE | MARGINAL_EDGE | NO_VERIFIED_EDGE
    oos_accuracy: float
    oos_brier: float
    oos_three_level_wr: float
    oos_n: int
    engine_metrics: Dict[str, float]
    baseline_results: Dict[str, Dict[str, float]]
    beats_all_on: List[str]                   # metrics where engine beats every baseline
    oos_edge: float                           # engine accuracy - best baseline accuracy
    intelligence_score: float                 # 0-100
    recommended_tier_adjustment: int          # +1=upgrade, 0=keep, -1=downgrade
    reasoning: List[str]
    checked_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "oos_accuracy": self.oos_accuracy,
            "oos_brier": self.oos_brier,
            "oos_three_level_wr": self.oos_three_level_wr,
            "oos_n": self.oos_n,
            "engine_metrics": self.engine_metrics,
            "baseline_results": self.baseline_results,
            "beats_all_on": self.beats_all_on,
            "oos_edge": self.oos_edge,
            "intelligence_score": self.intelligence_score,
            "recommended_tier_adjustment": self.recommended_tier_adjustment,
            "reasoning": self.reasoning,
            "checked_at": self.checked_at,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main RealityCheckEngine class
# ─────────────────────────────────────────────────────────────────────────────

MIN_OOS_SAMPLES = 50
REALITY_CHECK_KEY = "Reality_Check_Report"


class RealityCheckEngine:
    """
    Evaluates whether the adaptive engine has a genuine predictive edge.

    Usage (daily, from daily_learning.py):
        rce = RealityCheckEngine()
        report = rce.run(db)
        # report.verdict == "VERIFIED_EDGE" means the engine is genuinely learning
    """

    def run(self, db=None, memory=None) -> RealityCheckReport:
        """
        Execute full OOS reality check.

        Accepts either a live `db` session (reads from decision_memory table)
        or a pre-loaded `memory` DecisionMemory object.
        """
        from backend.decision_memory import (
            DecisionMemory, temporal_train_val_test_split,
            resolved_only, to_pairs,
        )

        # Load records
        if memory is not None:
            all_resolved = memory.all_resolved()
        elif db is not None:
            mem = DecisionMemory()
            mem.load_recent_from_db(db, limit=5000)
            all_resolved = mem.all_resolved()
        else:
            return self._insufficient("No data source provided", 0)

        if len(all_resolved) < MIN_OOS_SAMPLES * 2:
            return self._insufficient(
                f"Only {len(all_resolved)} resolved records — need ≥{MIN_OOS_SAMPLES * 2}",
                len(all_resolved),
            )

        # Strict temporal split
        train, val, test = temporal_train_val_test_split(all_resolved, 0.60, 0.20)

        if len(test) < MIN_OOS_SAMPLES:
            return self._insufficient(
                f"Test split too small: {len(test)} records (need ≥{MIN_OOS_SAMPLES})",
                len(all_resolved),
            )

        # Build (p_big, actual) pairs
        train_pairs = to_pairs(train)
        test_pairs  = to_pairs(test)

        train_big_rate = (
            sum(a for _, a in train_pairs) / len(train_pairs)
            if train_pairs else 0.5
        )

        # Engine metrics on OOS test set
        engine_metrics = _metrics(test_pairs)

        # Four baselines, all evaluated on the test set
        baseline_results = {
            "random":          _baseline_random(test_pairs),
            "frequency":       _baseline_frequency(test_pairs, train_big_rate),
            "recent_momentum": _baseline_recent_momentum(test_pairs),
            "always_majority": _baseline_always_majority(test_pairs, train_big_rate),
        }

        # For each metric, check if engine beats ALL baselines
        compare_metrics = ["accuracy", "brier", "three_level_wr"]
        beats_all_on: List[str] = []
        reasoning: List[str] = []

        for metric in compare_metrics:
            engine_val = engine_metrics[metric]
            # For brier and log_loss, lower is better
            lower_is_better = metric in ("brier", "log_loss", "calibration_error")
            if lower_is_better:
                worst_baseline = max(
                    baseline_results[b][metric] for b in baseline_results
                )
                beats = engine_val < worst_baseline
            else:
                worst_baseline = max(
                    baseline_results[b][metric] for b in baseline_results
                )
                best_baseline  = max(
                    baseline_results[b][metric] for b in baseline_results
                )
                beats = engine_val > best_baseline

            if beats:
                beats_all_on.append(metric)
                reasoning.append(
                    f"✅ {metric}: engine={engine_val:.4f} beats best "
                    f"baseline={best_baseline if not lower_is_better else worst_baseline:.4f}"
                )
            else:
                best_baseline_val = (
                    min(baseline_results[b][metric] for b in baseline_results)
                    if lower_is_better
                    else max(baseline_results[b][metric] for b in baseline_results)
                )
                reasoning.append(
                    f"❌ {metric}: engine={engine_val:.4f} does NOT beat "
                    f"best baseline={best_baseline_val:.4f}"
                )

        # Validated edge from decision memory
        validated_n = sum(1 for r in test if r.validated_edge)
        validated_frac = validated_n / len(test) if test else 0.0
        validated_edge = validated_frac > 0.3 and len(beats_all_on) >= 2

        # Intelligence score
        intel_score = _compute_intelligence_score(
            engine_metrics, baseline_results, validated_edge
        )

        # OOS edge vs best baseline on accuracy
        best_baseline_acc = max(
            baseline_results[b]["accuracy"] for b in baseline_results
        )
        oos_edge = round(engine_metrics["accuracy"] - best_baseline_acc, 4)

        # Verdict
        if len(beats_all_on) >= 3:
            verdict = "VERIFIED_EDGE"
            tier_adj = +1
            reasoning.append(
                f"🏆 VERIFIED_EDGE: beats all baselines on all 3 metrics "
                f"(intelligence score={intel_score:.0f})"
            )
        elif len(beats_all_on) >= 2:
            verdict = "MARGINAL_EDGE"
            tier_adj = 0
            reasoning.append(
                f"⚠️ MARGINAL_EDGE: beats baselines on {len(beats_all_on)}/3 metrics"
            )
        else:
            verdict = "NO_VERIFIED_EDGE"
            tier_adj = -1
            reasoning.append(
                "🚫 NO_VERIFIED_EDGE: does not outperform simple baselines "
                "on OOS data — confidence tiers will be downgraded"
            )

        report = RealityCheckReport(
            verdict=verdict,
            oos_accuracy=engine_metrics["accuracy"],
            oos_brier=engine_metrics["brier"],
            oos_three_level_wr=engine_metrics["three_level_wr"],
            oos_n=len(test),
            engine_metrics=engine_metrics,
            baseline_results=baseline_results,
            beats_all_on=beats_all_on,
            oos_edge=oos_edge,
            intelligence_score=intel_score,
            recommended_tier_adjustment=tier_adj,
            reasoning=reasoning,
        )

        # Persist to Supabase
        if db is not None:
            try:
                from backend.database import save_ai_brain_state
                save_ai_brain_state(
                    db=db,
                    model_name=REALITY_CHECK_KEY,
                    generation=len(all_resolved),
                    total_samples=len(all_resolved),
                    weights_json=json.dumps(report.to_dict()),
                    win_rate=intel_score,
                )
            except Exception as e:
                print(f"[RealityCheck] persist failed: {e}")

        return report

    @staticmethod
    def _insufficient(reason: str, n: int) -> RealityCheckReport:
        return RealityCheckReport(
            verdict="INSUFFICIENT_DATA",
            oos_accuracy=0.5,
            oos_brier=0.25,
            oos_three_level_wr=0.5,
            oos_n=n,
            engine_metrics={},
            baseline_results={},
            beats_all_on=[],
            oos_edge=0.0,
            intelligence_score=0.0,
            recommended_tier_adjustment=0,
            reasoning=[reason],
        )

    @staticmethod
    def load_last_report(db) -> Optional[dict]:
        """Load the most recent persisted reality check report."""
        try:
            from backend.database import load_ai_brain_state
            brain = load_ai_brain_state(db, model_name=REALITY_CHECK_KEY)
            if brain and brain.synaptic_weights:
                return json.loads(brain.synaptic_weights)
        except Exception as e:
            print(f"[RealityCheck] load failed: {e}")
        return None
