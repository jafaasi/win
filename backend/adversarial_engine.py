#!/usr/bin/env python3
"""
Adversarial Devil's Advocate Engine — TAIE Component 1
=======================================================
Before any prediction is accepted, this engine asks:

    "What evidence suggests the proposed prediction is WRONG?"

It independently generates contradictory signals and computes an
adversarial_score that measures how hard it is to contradict the
main ensemble.  High adversarial_score = strong contradicting evidence
= downgrade or abstain.  Low adversarial_score = little contradiction
= the main prediction holds up to scrutiny.

Architecture
------------
The engine runs 7 independent adversarial probes, each trying to find
evidence FOR the OPPOSITE side:

  1. Anti-momentum probe    — does recent momentum oppose the prediction?
  2. Regression-to-mean     — is the predicted side recently over-represented?
  3. Streak exhaustion      — is the current streak unusually long?
  4. Anti-ACF probe         — do significant lags predict the opposite?
  5. Entropy probe          — is the sequence too random to predict?
  6. Variance probe         — is volatility so high that any signal is noise?
  7. Cross-model dissent    — how many of the 12 sub-models disagree?

Each probe returns a score in [0, 1] representing how strongly it
contradicts the proposed prediction.  The final adversarial_score is
a weighted blend.

Output
------
AdversarialReport dataclass with:
  adversarial_score    — [0, 1], 0 = no contradiction, 1 = strong opposition
  probe_scores         — dict of individual probe results
  contradicting_signals — list of human-readable reasons
  supporting_signals   — list of reasons the prediction might be right
  verdict              — "HOLD" | "CAUTION" | "OVERRIDE" | "ABSTAIN"
  recommendation       — modified confidence multiplier [0.5, 1.0]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Result structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AdversarialReport:
    adversarial_score: float          # [0, 1] — how strong is the contradiction
    probe_scores: Dict[str, float]    # individual probe results
    contradicting_signals: List[str]  # human-readable opposition evidence
    supporting_signals: List[str]     # human-readable supporting evidence
    verdict: str                      # HOLD | CAUTION | OVERRIDE | ABSTAIN
    recommendation: float             # confidence multiplier [0.5, 1.0]
    net_score: float                  # supporting - contradicting (signed)

    def to_dict(self) -> dict:
        return {
            "adversarial_score": round(self.adversarial_score, 4),
            "probe_scores": {k: round(v, 4) for k, v in self.probe_scores.items()},
            "contradicting_signals": self.contradicting_signals,
            "supporting_signals": self.supporting_signals,
            "verdict": self.verdict,
            "recommendation": round(self.recommendation, 4),
            "net_score": round(self.net_score, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Individual adversarial probes
# ─────────────────────────────────────────────────────────────────────────────

def _probe_anti_momentum(
    sides: List[int], proposed_side: int, window: int = 20
) -> Tuple[float, str]:
    """
    Anti-momentum: if recent short-term momentum strongly favors the proposed
    side, a reversal is more likely than a continuation — this is opposing evidence.

    Returns (adversarial_strength [0,1], reason_string).
    """
    if len(sides) < window:
        return 0.0, ""
    recent = sides[-window:]
    rate = sum(recent) / len(recent)   # fraction that were Big
    # Laplace-smoothed rate for proposed side
    if proposed_side == 1:  # Big proposed
        overextension = max(0.0, rate - 0.62)  # above 62% Big = overextended
    else:
        overextension = max(0.0, (1 - rate) - 0.62)

    score = min(1.0, overextension / 0.20)  # saturates at 82% one-sided
    if score > 0.3:
        direction = "Big" if proposed_side == 1 else "Small"
        opposite = "Small" if proposed_side == 1 else "Big"
        return score, (
            f"Anti-momentum: {direction} appeared {rate*100:.0f}% of last {window} "
            f"rounds — regression toward {opposite} is likely"
        )
    return score, ""


def _probe_regression_to_mean(
    sides: List[int], proposed_side: int,
    short_win: int = 10, long_win: int = 200
) -> Tuple[float, str]:
    """
    If the short-term rate deviates strongly from the long-term mean on the
    proposed side, mean-reversion is the contrarian bet.
    """
    n = len(sides)
    if n < long_win:
        return 0.0, ""

    short_rate = sum(sides[-short_win:]) / short_win
    long_rate  = sum(sides[-long_win:]) / long_win

    deviation = short_rate - long_rate   # positive = recent skew toward Big

    if proposed_side == 1:
        contradiction = max(0.0, deviation)    # Big proposed but over-Big recently
    else:
        contradiction = max(0.0, -deviation)   # Small proposed but over-Small recently

    score = min(1.0, abs(contradiction) / 0.15)  # 15pp deviation → full score
    if score > 0.25:
        direction = "Big" if proposed_side == 1 else "Small"
        return score, (
            f"Regression-to-mean: short-term {direction} rate "
            f"({short_rate*100:.0f}%) is {abs(deviation)*100:.0f}pp "
            f"above long-term mean ({long_rate*100:.0f}%) — reversion likely"
        )
    return score, ""


def _probe_streak_exhaustion(
    sides: List[int], proposed_side: int
) -> Tuple[float, str]:
    """
    Measures current run length.  Very long streaks have historically lower
    continuation probability — opposing evidence for a continuation prediction.
    """
    if not sides:
        return 0.0, ""
    current = sides[-1]
    run = 1
    for i in range(len(sides) - 2, -1, -1):
        if sides[i] == current:
            run += 1
        else:
            break

    # Only adversarial if proposing CONTINUATION of the current streak
    if proposed_side != current:
        return 0.0, ""  # proposing reversal — streak exhaustion actually supports it

    # Sigmoid-style: run length 1 → 0, run 5 → 0.5, run 10 → 0.85
    score = 1.0 - math.exp(-0.15 * max(0, run - 3))
    score = min(0.90, max(0.0, score))
    if score > 0.3:
        direction = "Big" if current == 1 else "Small"
        return score, (
            f"Streak exhaustion: {direction} streak is {run} rounds long — "
            f"reversal probability elevated"
        )
    return score, ""


def _probe_anti_acf(
    sides: List[int], proposed_side: int, max_lag: int = 12
) -> Tuple[float, str]:
    """
    If significant ACF lags predict the OPPOSITE of the proposed side,
    this is direct contradictory evidence.
    """
    n = len(sides)
    if n < max_lag + 10:
        return 0.0, ""

    arr = np.array(sides[-min(500, n):], dtype=np.float64)
    mu = arr.mean()
    arr_c = arr - mu
    var = float((arr_c ** 2).mean()) + 1e-12
    threshold = 1.5 / math.sqrt(len(arr))

    opposite_vote_weight = 0.0
    total_weight = 0.0
    contradicting_lags = []

    for lag in range(1, min(max_lag + 1, len(arr))):
        acf = float(np.mean(arr_c[lag:] * arr_c[:-lag])) / var
        if abs(acf) <= threshold:
            continue
        w = abs(acf) / (lag ** 0.5)
        total_weight += w
        # If acf > 0 → lag predicts continuation; if acf < 0 → predicts reversal
        past_side = int(sides[-lag]) if lag <= len(sides) else 1
        if acf > 0:
            lag_vote = past_side   # continuation
        else:
            lag_vote = 1 - past_side  # reversal

        if lag_vote != proposed_side:
            opposite_vote_weight += w
            contradicting_lags.append(lag)

    if total_weight < 1e-9:
        return 0.0, ""

    contradiction_fraction = opposite_vote_weight / total_weight
    score = min(1.0, contradiction_fraction * 1.5)
    if score > 0.3 and contradicting_lags:
        return score, (
            f"Anti-ACF: lags {contradicting_lags[:4]} predict opposite side "
            f"(contradiction fraction {contradiction_fraction:.2f})"
        )
    return score, ""


def _probe_entropy(
    digits: List[int], window: int = 100
) -> Tuple[float, str]:
    """
    High entropy = near-random sequence = any prediction is unreliable.
    This is an adversarial probe against the ENTIRE ensemble, not one side.
    Score → proportion of randomness.
    """
    if len(digits) < window:
        return 0.0, ""
    counts = np.bincount(np.array(digits[-window:]) % 10, minlength=10).astype(np.float64)
    counts = np.clip(counts, 1e-9, None)
    probs = counts / counts.sum()
    h = float(-np.sum(probs * np.log(probs)))
    h_max = math.log(10)
    h_norm = h / h_max
    # Near-maximum entropy (≥ 0.985) → strong contradiction against any prediction
    score = max(0.0, (h_norm - 0.965) / 0.035)
    score = min(1.0, score)
    if score > 0.3:
        return score, (
            f"Entropy probe: normalised Shannon entropy = {h_norm:.3f} "
            f"(≈ {h_norm*100:.0f}% of maximum) — sequence near-random, "
            f"predictions are unreliable"
        )
    return score, ""


def _probe_volatility(
    sides: List[int], window: int = 30
) -> Tuple[float, str]:
    """
    Very high short-term volatility (frequent side switches) means the
    sequence is unstable — any directional bet is high-risk.
    """
    if len(sides) < window:
        return 0.0, ""
    recent = sides[-window:]
    switches = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i - 1])
    switch_rate = switches / (len(recent) - 1)
    # Pure alternating (rate ≈ 1) or very high noise (>0.65) → adversarial
    score = max(0.0, (switch_rate - 0.55) / 0.30)
    score = min(1.0, score)
    if score > 0.3:
        return score, (
            f"Volatility probe: {switch_rate*100:.0f}% side-switch rate "
            f"in last {window} rounds — high noise environment"
        )
    return score, ""


def _probe_cross_model_dissent(
    model_p_big: np.ndarray, proposed_side: int
) -> Tuple[float, str]:
    """
    Measures what fraction of the 12 sub-models disagree with the proposed
    side.  High dissent = the ensemble majority is weak = adversarial.
    """
    if model_p_big is None or len(model_p_big) == 0:
        return 0.0, ""
    n = len(model_p_big)
    dissent = sum(1 for p in model_p_big if (p < 0.5) == (proposed_side == 1))
    dissent_fraction = dissent / n
    score = max(0.0, (dissent_fraction - 0.20) / 0.50)  # 20% dissent = 0, 70% = 1
    score = min(1.0, score)
    if score > 0.25:
        return score, (
            f"Cross-model dissent: {dissent}/{n} sub-models oppose proposed "
            f"prediction ({dissent_fraction*100:.0f}% dissent rate)"
        )
    return score, ""


# ─────────────────────────────────────────────────────────────────────────────
# Supporting evidence probes (these REDUCE adversarial credibility)
# ─────────────────────────────────────────────────────────────────────────────

def _supporting_exploit_confirmed(
    exploit_score: float, reject_iid: bool
) -> Tuple[float, str]:
    if reject_iid and exploit_score > 0.4:
        strength = min(1.0, exploit_score / 0.8)
        return strength, (
            f"Exploit confirmed: IID null rejected, exploit score = {exploit_score:.2f} "
            f"— non-random structure detected"
        )
    return 0.0, ""


def _supporting_model_consensus(
    model_consensus: float
) -> Tuple[float, str]:
    if model_consensus > 0.70:
        return model_consensus, (
            f"Model consensus: {model_consensus*100:.0f}% of sub-models agree "
            f"— strong directional signal"
        )
    return 0.0, ""


def _supporting_validated_edge(
    validated_edge: bool, three_level_lower: float
) -> Tuple[float, str]:
    if validated_edge and three_level_lower > 0.88:
        return 0.85, (
            f"Validated edge: historical 3-level win rate lower bound = "
            f"{three_level_lower*100:.1f}% — statistically significant"
        )
    return 0.0, ""


# ─────────────────────────────────────────────────────────────────────────────
# Verdict engine
# ─────────────────────────────────────────────────────────────────────────────

def _compute_verdict(
    adversarial_score: float,
    supporting_strength: float,
    net_score: float,
) -> Tuple[str, float]:
    """
    Maps adversarial + supporting balance to a verdict and confidence multiplier.

    Verdicts:
      HOLD    — contradiction is weak, proceed normally
      CAUTION — moderate contradiction, reduce confidence
      OVERRIDE — strong contradiction overrides weak support
      ABSTAIN — adversarial evidence so strong, signal should be withheld
    """
    if adversarial_score < 0.25 or net_score > 0.3:
        return "HOLD", 1.0
    elif adversarial_score < 0.45 and net_score > -0.1:
        return "CAUTION", 0.85
    elif adversarial_score < 0.65:
        return "CAUTION", 0.70
    elif adversarial_score < 0.80 and supporting_strength < 0.4:
        return "OVERRIDE", 0.55
    else:
        return "ABSTAIN", 0.50


# ─────────────────────────────────────────────────────────────────────────────
# Main AdversarialEngine class
# ─────────────────────────────────────────────────────────────────────────────

# Probe weights — how much each probe contributes to the final adversarial score
PROBE_WEIGHTS = {
    "anti_momentum":       0.15,
    "regression_to_mean":  0.15,
    "streak_exhaustion":   0.12,
    "anti_acf":            0.20,
    "entropy":             0.18,
    "volatility":          0.10,
    "cross_model_dissent": 0.10,
}


class AdversarialEngine:
    """
    Devil's advocate.  Call analyse() before every prediction is finalised.

    Usage:
        engine = AdversarialEngine()
        report = engine.analyse(
            proposed_side="Big",
            digits=history,
            sides=side_series,
            model_p_big=model_p_big_array,
            exploit_score=0.45,
            reject_iid=True,
            model_consensus=0.75,
            validated_edge=True,
            three_level_lower=0.89,
        )
        if report.verdict == "ABSTAIN":
            action = "SKIP"
        elif report.verdict == "OVERRIDE":
            # flip prediction or skip
            ...
    """

    def __init__(self):
        self._probe_history: list = []   # for diagnostics

    def analyse(
        self,
        proposed_side: str,             # "Big" or "Small"
        digits: List[int],
        sides: List[int],
        model_p_big: Optional[np.ndarray] = None,
        exploit_score: float = 0.0,
        reject_iid: bool = False,
        model_consensus: float = 0.5,
        validated_edge: bool = False,
        three_level_lower: float = 0.0,
    ) -> AdversarialReport:

        side_int = 1 if proposed_side == "Big" else 0

        # ── Run all 7 adversarial probes ─────────────────────────────────────
        p_anti_mom,  r_anti_mom  = _probe_anti_momentum(sides, side_int)
        p_reg,       r_reg       = _probe_regression_to_mean(sides, side_int)
        p_streak,    r_streak    = _probe_streak_exhaustion(sides, side_int)
        p_acf,       r_acf       = _probe_anti_acf(sides, side_int)
        p_entropy,   r_entropy   = _probe_entropy(digits)
        p_vol,       r_vol       = _probe_volatility(sides)
        p_dissent,   r_dissent   = _probe_cross_model_dissent(
            model_p_big if model_p_big is not None else np.array([]),
            side_int
        )

        probe_scores = {
            "anti_momentum":       p_anti_mom,
            "regression_to_mean":  p_reg,
            "streak_exhaustion":   p_streak,
            "anti_acf":            p_acf,
            "entropy":             p_entropy,
            "volatility":          p_vol,
            "cross_model_dissent": p_dissent,
        }

        # Weighted adversarial score
        adversarial_score = sum(
            PROBE_WEIGHTS[k] * v for k, v in probe_scores.items()
        )
        adversarial_score = round(min(1.0, adversarial_score), 4)

        # Collect non-empty contradiction reasons
        contradicting = [r for r in [
            r_anti_mom, r_reg, r_streak, r_acf, r_entropy, r_vol, r_dissent
        ] if r]

        # ── Run supporting probes ─────────────────────────────────────────────
        s_exploit,    r_s_exploit    = _supporting_exploit_confirmed(exploit_score, reject_iid)
        s_consensus,  r_s_consensus  = _supporting_model_consensus(model_consensus)
        s_validated,  r_s_validated  = _supporting_validated_edge(validated_edge, three_level_lower)

        supporting_strength = (s_exploit * 0.35 + s_consensus * 0.35 + s_validated * 0.30)
        supporting = [r for r in [r_s_exploit, r_s_consensus, r_s_validated] if r]

        # ── Net score (positive = prediction well-supported, negative = contested)
        net_score = round(supporting_strength - adversarial_score, 4)

        # ── Verdict ──────────────────────────────────────────────────────────
        verdict, recommendation = _compute_verdict(
            adversarial_score, supporting_strength, net_score
        )

        report = AdversarialReport(
            adversarial_score=adversarial_score,
            probe_scores=probe_scores,
            contradicting_signals=contradicting,
            supporting_signals=supporting,
            verdict=verdict,
            recommendation=recommendation,
            net_score=net_score,
        )
        self._probe_history.append({
            "adversarial_score": adversarial_score,
            "verdict": verdict,
            "net_score": net_score,
        })
        if len(self._probe_history) > 500:
            self._probe_history = self._probe_history[-500:]

        return report

    def recent_override_rate(self, window: int = 50) -> float:
        """Fraction of recent rounds where verdict was OVERRIDE or ABSTAIN."""
        recent = self._probe_history[-window:]
        if not recent:
            return 0.0
        bad = sum(1 for r in recent if r["verdict"] in ("OVERRIDE", "ABSTAIN"))
        return bad / len(recent)

    def average_adversarial_score(self, window: int = 50) -> float:
        recent = self._probe_history[-window:]
        if not recent:
            return 0.0
        return sum(r["adversarial_score"] for r in recent) / len(recent)
