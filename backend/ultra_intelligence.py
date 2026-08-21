#!/usr/bin/env python3
"""
Ultra Intelligence Engine for WinGo 30s  — v2.0
=================================================
Unified top-level intelligence orchestrating 12 sub-models:

  1.  HIP CTW            — Variable-order Markov via Context Tree Weighting
  2.  HIP N-gram         — N-gram blend orders 1-5
  3.  HIP Streak         — Bayesian Beta-Binomial streak belief
  4.  HIP Frequency      — Rolling side frequency (75-step EWA)
  5.  EVOSEQ Ensemble    — Transformer + Mamba + baseline models
  6.  Decay Markov       — Exponentially decay-weighted transition matrix
  7.  Session Bias       — Session-position temporal features (6 bins)
  8.  Exploit Detector   — 11 statistical tests + 4 predictive engines
  9.  Pattern Intelligence — ACF lag discovery, streak reversal profile,
                             conditional freq table, digit gap detector
  10. 3-Level Martingale — Calibrated Martingale-level-aware prediction
  11. Volatility Regime  — Rolling volatility band model (Big/Small rate)
  12. Cross-Round Corr   — Multi-lag cross-correlation ensemble

New in v2.0:
  • 4 new sub-models (9-12 above) added to 12-model Hedge ensemble
  • Smarter SKIP logic: multi-factor consensus gate, not just cal_single
  • Hedge eta loaded from daily_learning.py recommendations
  • PatternIntelligence and 3-Level state persisted/restored from Supabase
  • DailyLearningScheduler integrated — fires at midnight UTC automatically

Usage:
    from backend.ultra_intelligence import UltraIntelligenceEngine
    engine = UltraIntelligenceEngine()
    result = engine.predict(history, registry_state, db)
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.high_intelligence_predictor import (
    HighIntelligencePredictor,
    three_level_win_probability,
    recommend_strike_level,
)
from backend.exhaustive_exploit import (
    ExhaustiveExploitDetector,
    markov_side_matrix,
    ngram_next_probs,
)
from backend.prediction_intelligence import EvidenceGate
from backend.pattern_intelligence import PatternIntelligence
from backend.three_level_winning import ThreeLevelWinningAlgorithm
from backend.adversarial_engine import AdversarialEngine
from backend.state_memory import StateMemory
from backend.meta_learner import MetaLearner
from backend.evolution_controller import EvolutionController
from backend.decision_memory import DecisionMemory, DecisionRecord
from backend.intelligence.state_fingerprint import compute_state_fingerprint


# ============================================================================
# 1. Decay-weighted Markov transition matrix
# ============================================================================

class DecayMarkov:
    """Exponentially decay-weighted transition matrix for digits 0-9.

    Recent transitions have more influence than old ones (lambda per step).
    This adapts faster to regime changes than a flat-count matrix.
    """

    def __init__(self, decay: float = 0.995, alphabet: int = 10):
        self.decay = decay
        self.alphabet = alphabet
        # T[i, j] = weighted count of transitions i -> j
        self.T = np.full((alphabet, alphabet), 0.1, dtype=np.float64)
        self._total_steps = 0

    def update(self, prev_digit: int, curr_digit: int) -> None:
        # Decay existing counts
        self.T *= self.decay
        # Add new observation
        self.T[prev_digit % self.alphabet, curr_digit % self.alphabet] += 1.0
        self._total_steps += 1

    def update_sequence(self, sequence: Sequence[int]) -> None:
        for i in range(len(sequence) - 1):
            self.update(int(sequence[i]), int(sequence[i + 1]))

    def predict_proba(self, last_digit: int) -> np.ndarray:
        """P(next_digit | last_digit) from the decay-weighted matrix."""
        row = self.T[last_digit % self.alphabet].copy()
        row = np.clip(row, 1e-9, None)
        return row / row.sum()

    def predict_side_proba(self, last_digit: int) -> Tuple[float, float]:
        """Return (P_big, P_small) from the decay-weighted matrix."""
        probs = self.predict_proba(last_digit)
        p_big = float(probs[5:].sum())
        return p_big, 1.0 - p_big


# ============================================================================
# 2. Session-position temporal feature extractor
# ============================================================================

class SessionPositionFeatures:
    """Track and exploit temporal patterns within game sessions.

    Many PRNGs show different behavior at the start vs. middle vs. end
    of a session. We track the position within the current session and
    compute bias features at each position.
    """

    def __init__(self, session_length: int = 120, n_bins: int = 6):
        self.session_length = session_length  # ~120 rounds per hour
        self.n_bins = n_bins
        # bin_stats[bin_idx] = (big_count, total_count)
        self.bin_stats: Dict[int, List[int]] = {
            i: [0, 0] for i in range(n_bins)
        }
        self._position = 0

    def observe(self, digit: int) -> None:
        bin_idx = min(self.n_bins - 1, self._position * self.n_bins // self.session_length)
        side = 1 if digit >= 5 else 0
        self.bin_stats[bin_idx][0] += side
        self.bin_stats[bin_idx][1] += 1
        self._position = (self._position + 1) % self.session_length

    def observe_sequence(self, sequence: Sequence[int]) -> None:
        for d in sequence:
            self.observe(int(d))

    def current_bin_bias(self) -> Tuple[float, float, int]:
        """Return (p_big, confidence_weight, bin_index) for current position."""
        bin_idx = min(self.n_bins - 1, self._position * self.n_bins // self.session_length)
        big, total = self.bin_stats[bin_idx]
        if total < 10:
            return 0.5, 0.0, bin_idx  # Not enough data, no bias
        p_big = (big + 1.0) / (total + 2.0)  # Laplace smoothed
        # Confidence weight: higher when we have more samples and deviation from 0.5
        deviation = abs(p_big - 0.5)
        weight = min(0.15, deviation * min(1.0, total / 200.0))
        return p_big, weight, bin_idx


# ============================================================================
# 3. Adaptive Hedge ensemble (online regret-minimizing)
# ============================================================================

class HedgeEnsemble:
    """Aggregating Algorithm / Hedge-style online ensemble.

    Unlike fixed blending ratios, this learns which sub-model is performing
    best and automatically shifts weight toward it. Uses multiplicative
    weight updates with adaptive learning rate.
    """

    def __init__(self, n_models: int, eta: float = 0.12):
        self.n = n_models
        self.eta = eta
        self.weights = np.ones(n_models, dtype=np.float64) / n_models
        self._cumulative_loss = np.zeros(n_models, dtype=np.float64)
        self._rounds = 0

    def predict(self, model_p_big: np.ndarray) -> float:
        """Combine model P(Big) predictions via weighted average."""
        w = self.weights / self.weights.sum()
        return float(np.clip(np.dot(w, model_p_big), 0.005, 0.995))

    def predict_digits(self, model_digit_dists: List[np.ndarray]) -> np.ndarray:
        """Combine model digit distributions via weighted average."""
        w = self.weights / self.weights.sum()
        blended = np.zeros(10, dtype=np.float64)
        for i, dist in enumerate(model_digit_dists):
            blended += w[i] * dist
        blended = np.clip(blended, 1e-8, None)
        return blended / blended.sum()

    def update(self, model_p_big: np.ndarray, actual_side_int: int) -> None:
        """Update weights based on log-loss of each model."""
        p = np.clip(model_p_big, 1e-4, 1 - 1e-4)
        if actual_side_int == 1:
            losses = -np.log(p)
        else:
            losses = -np.log(1 - p)
        self._cumulative_loss += losses
        self._rounds += 1
        # Multiplicative weight update
        self.weights *= np.exp(-self.eta * losses)
        # L1 regularization to prevent degeneracy
        self.weights += 1e-5
        self.weights /= self.weights.sum()

    def get_weights_dict(self, model_names: List[str]) -> Dict[str, float]:
        w = self.weights / self.weights.sum()
        return {name: round(float(w[i]), 4) for i, name in enumerate(model_names)}


# ============================================================================
# 3.5  Volatility Regime Model
# ============================================================================

class VolatilityRegimeModel:
    """Models the rolling Big/Small rate in fast vs slow volatility windows.

    Computes three windows (10, 50, 200 rounds) of the Big-rate and detects
    momentum / mean-reversion regimes.  Returns a P(Big) adjusted for the
    current volatility environment.

    Fast window > slow window  → momentum regime  (follow trend)
    Fast window < slow window  → mean-reversion  (fade trend)
    """

    def __init__(self, windows: Tuple[int, int, int] = (10, 50, 200)):
        self.w_fast, self.w_mid, self.w_slow = windows

    def predict_p_big(self, sides: List[int]) -> float:
        n = len(sides)
        if n < self.w_fast:
            return 0.5

        def laplace_rate(window: int) -> float:
            sl = sides[-min(window, n):]
            return (sum(sl) + 1.0) / (len(sl) + 2.0)

        r_fast = laplace_rate(self.w_fast)
        r_mid  = laplace_rate(self.w_mid)
        r_slow = laplace_rate(self.w_slow)

        # Momentum signal: fast vs slow deviation
        momentum = r_fast - r_slow
        # Mean-reversion signal: slow rate pulls back
        mean_rev = r_slow - r_fast

        # Volatility (std of fast window)
        fast_sl = np.array(sides[-self.w_fast:], dtype=np.float64)
        vol = float(fast_sl.std()) if len(fast_sl) >= 3 else 0.5

        # In high-volatility environment follow momentum; low-vol follow slow rate
        if vol > 0.48:
            # High volatility: momentum tends to persist briefly
            p = 0.5 + 0.5 * momentum
        else:
            # Low volatility: mean-reversion more likely
            p = r_slow + 0.3 * mean_rev

        # Shrink toward 0.5 — this is a complementary signal, not a primary one
        p = 0.5 + 0.55 * (p - 0.5)
        return float(max(0.02, min(0.98, p)))


# ============================================================================
# 3.6  Cross-Round Correlation Ensemble
# ============================================================================

class CrossRoundCorrelation:
    """Multi-lag weighted cross-correlation predictor.

    For lags 1..12 computes the Pearson correlation between the side at
    position t-lag and the side at position t.  Uses this to build a
    combined forecast:  P(Big_t) = sigmoid( sum_lag  w_lag * side_{t-lag} )
    where w_lag = signed ACF at lag (positive = continuation, negative = reversal).

    Weights are re-estimated on a rolling window so they adapt to regime changes.
    """

    MAX_LAG = 12
    WINDOW = 500   # rolling window for ACF estimation

    def __init__(self):
        self._weights: np.ndarray = np.zeros(self.MAX_LAG, dtype=np.float64)
        self._n_updates: int = 0

    def _refit(self, sides: List[int]) -> None:
        """Re-estimate lag weights from the last WINDOW observations."""
        arr = np.array(sides[-self.WINDOW:], dtype=np.float64)
        n = len(arr)
        if n < 30:
            return
        mu = arr.mean()
        arr_c = arr - mu
        var = float((arr_c ** 2).mean()) + 1e-12
        for lag in range(1, self.MAX_LAG + 1):
            if lag >= n:
                self._weights[lag - 1] = 0.0
                continue
            acf = float(np.mean(arr_c[lag:] * arr_c[:-lag])) / var
            # Shrink small autocorrelations to zero (noise guard)
            threshold = 1.2 / math.sqrt(n)
            self._weights[lag - 1] = acf if abs(acf) > threshold else 0.0
        self._n_updates += 1

    def predict_p_big(self, sides: List[int]) -> float:
        n = len(sides)
        if n < self.MAX_LAG + 5:
            return 0.5

        # Re-fit every 50 new observations
        if self._n_updates == 0 or n % 50 == 0:
            self._refit(sides)

        # Linear score: sum over lags
        score = 0.0
        for lag in range(1, self.MAX_LAG + 1):
            w = self._weights[lag - 1]
            if abs(w) < 1e-9 or lag > n:
                continue
            # sides[-lag] is 0 or 1; center around 0.5
            score += w * (sides[-lag] - 0.5)

        # sigmoid → P(Big)
        p = 1.0 / (1.0 + math.exp(-6.0 * score))
        # Shrink toward 0.5
        p = 0.5 + 0.65 * (p - 0.5)
        return float(max(0.02, min(0.98, p)))


# ============================================================================
# 4. Streak tracker with persistence
# ============================================================================

@dataclass
class StreakState:
    """Tracks win/loss streaks with persistence support."""
    win_streak: int = 0
    loss_streak: int = 0
    total_wins: int = 0
    total_losses: int = 0
    recent_results: Deque[bool] = field(default_factory=lambda: deque(maxlen=20))
    session_results: Deque[bool] = field(default_factory=lambda: deque(maxlen=100))

    def record(self, won: bool) -> None:
        self.recent_results.append(won)
        self.session_results.append(won)
        if won:
            self.win_streak += 1
            self.loss_streak = 0
            self.total_wins += 1
        else:
            self.loss_streak += 1
            self.win_streak = 0
            self.total_losses += 1

    @property
    def recent_win_rate(self) -> float:
        if not self.recent_results:
            return 0.5
        return sum(self.recent_results) / len(self.recent_results)

    @property
    def session_win_rate(self) -> float:
        if not self.session_results:
            return 0.5
        return sum(self.session_results) / len(self.session_results)

    @property
    def total_games(self) -> int:
        return self.total_wins + self.total_losses

    def to_dict(self) -> dict:
        return {
            "win_streak": self.win_streak,
            "loss_streak": self.loss_streak,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "recent_results": list(self.recent_results),
            "session_results": list(self.session_results),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StreakState":
        s = cls()
        s.win_streak = d.get("win_streak", 0)
        s.loss_streak = d.get("loss_streak", 0)
        s.total_wins = d.get("total_wins", 0)
        s.total_losses = d.get("total_losses", 0)
        for r in d.get("recent_results", []):
            s.recent_results.append(bool(r))
        for r in d.get("session_results", []):
            s.session_results.append(bool(r))
        return s

    def skip_threshold(self) -> float:
        """Dynamic skip threshold based on streak state.

        After consecutive losses, widen the skip zone to avoid tilt-betting.
        """
        base = 0.54  # Minimum single-round P(correct) to not skip
        if self.loss_streak >= 5:
            return min(0.62, base + 0.02 * (self.loss_streak - 4))
        elif self.loss_streak >= 3:
            return base + 0.01 * (self.loss_streak - 2)
        return base


# ============================================================================
# 5.5  TAIE Decision Tree — STRONG / MODERATE / WEAK / ABSTAIN
# ============================================================================

def compute_taie_tier(
    cal_single: float,
    p_win_in_3: float,
    model_consensus: float,
    adversarial_score: float,
    adversarial_verdict: str,
    exploit_score: float,
    reject_iid: bool,
    state_memory_verdict: str,
    validated_edge: bool,
    evolution_edge_status: str,
) -> str:
    """
    Classify the current prediction into one of four TAIE tiers.

    STRONG   — multiple independent confirmations, low adversarial opposition
    MODERATE — decent signal, some caveats
    WEAK     — signal present but not well-confirmed
    ABSTAIN  — contradictions outweigh support; engine should not bet

    This is NOT the same as strikeQuality (which is a Telegram display tier).
    TAIE tier is a scientific assessment used by EvolutionController and
    DecisionMemory to segment historical performance.
    """
    # ABSTAIN: adversarial engine says to stay out, OR edge status is NO_EDGE
    if adversarial_verdict == "ABSTAIN":
        return "ABSTAIN"
    if evolution_edge_status == "NO_EDGE" and cal_single < 0.56:
        return "ABSTAIN"

    # Positive evidence score (0-6)
    pos = 0
    if cal_single >= 0.58:          pos += 1
    if p_win_in_3 >= 0.90:          pos += 1
    if model_consensus >= 0.70:     pos += 1
    if exploit_score >= 0.35:       pos += 1
    if validated_edge:              pos += 1
    if state_memory_verdict == "RELIABLE": pos += 1

    # Negative evidence score (adversarial pressure)
    neg = 0
    if adversarial_score >= 0.45:   neg += 1
    if adversarial_verdict in ("CAUTION", "OVERRIDE"): neg += 1
    if not reject_iid:              neg += 1
    if evolution_edge_status == "NO_EDGE": neg += 1

    net = pos - neg

    if net >= 4:
        return "STRONG"
    elif net >= 2:
        return "MODERATE"
    elif net >= 0:
        return "WEAK"
    else:
        return "ABSTAIN"


# ============================================================================
# 5. Confidence band classifier
# ============================================================================

def classify_confidence(
    cal_single: float,
    p_win_in_3: float,
    exploit_score: float,
    reject_iid: bool,
    engines_agree: int,
    streak: StreakState,
    pattern_p_big: float = 0.5,
    model_consensus: float = 0.5,
) -> Tuple[str, float, str]:
    """Classify confidence into honest bands with multi-factor SKIP logic.

    Returns (strike_quality, display_confidence_pct, action).

    Action:
      - "SKIP"     : no exploitable edge — do not bet
      - "CAUTION"  : marginal edge — minimum stake
      - "FORECAST" : normal edge
      - "STRIKE"   : strong validated edge

    v2: SKIP now requires *multiple* weak signals to trigger, not just a
    single low cal_single.  A strong pattern signal or model consensus can
    override a borderline skip.
    """
    skip_thresh = streak.skip_threshold()

    # ── Multi-factor SKIP gate ─────────────────────────────────────────────
    # Count how many individual factors suggest "no edge"
    skip_factors = 0
    if cal_single < skip_thresh:
        skip_factors += 1
    if not reject_iid:
        skip_factors += 1
    if exploit_score < 0.25:
        skip_factors += 1
    if engines_agree <= 1:
        skip_factors += 1
    # Pattern signal — if strongly directional it can rescue a borderline skip
    pattern_strength = abs(pattern_p_big - 0.5)
    if pattern_strength < 0.03:
        skip_factors += 1
    # Model consensus (fraction of 12 models on same side)
    if model_consensus < 0.55:
        skip_factors += 1

    # Only SKIP when ≥4 factors simultaneously suggest no edge
    if skip_factors >= 4:
        conf_pct = round(cal_single * 100, 1)
        return "SKIP", conf_pct, "SKIP"

    # ── CAUTION: marginal edge ─────────────────────────────────────────────
    if cal_single < 0.57 or (p_win_in_3 < 0.87 and engines_agree < 3):
        conf_pct = round(cal_single * 100, 1)
        return "LOW_CONFIDENCE", conf_pct, "CAUTION"

    # ── Confidence bands ───────────────────────────────────────────────────
    if p_win_in_3 >= 0.985 and cal_single >= 0.65 and engines_agree >= 4:
        strike = "ULTIMATE_CONVICTION"
        action = "STRIKE"
    elif p_win_in_3 >= 0.965 and cal_single >= 0.62 and engines_agree >= 3:
        strike = "BEAST_CONVICTION"
        action = "STRIKE"
    elif p_win_in_3 >= 0.94 and cal_single >= 0.59:
        strike = "HIGH_CONVICTION"
        action = "FORECAST"
    elif p_win_in_3 >= 0.90 and cal_single >= 0.56:
        strike = "MODERATE_CONVICTION"
        action = "FORECAST"
    else:
        strike = "CONSERVATIVE"
        action = "FORECAST"

    # Boost: if pattern and exploit both confirm, upgrade one level
    if pattern_strength >= 0.07 and exploit_score >= 0.50 and reject_iid:
        upgrades = {
            "CONSERVATIVE": "MODERATE_CONVICTION",
            "MODERATE_CONVICTION": "HIGH_CONVICTION",
            "HIGH_CONVICTION": "BEAST_CONVICTION",
        }
        strike = upgrades.get(strike, strike)
        if action == "FORECAST" and strike in ("BEAST_CONVICTION",):
            action = "STRIKE"

    # Loss-streak dampener: downgrade after 3+ losses
    if streak.loss_streak >= 3 and strike in ("ULTIMATE_CONVICTION", "BEAST_CONVICTION"):
        strike = "HIGH_CONVICTION"
        action = "FORECAST"
    elif streak.loss_streak >= 5 and strike == "HIGH_CONVICTION":
        strike = "MODERATE_CONVICTION"

    conf_pct = round(cal_single * 100, 1)
    return strike, conf_pct, action


# ============================================================================
# 6. Main Ultra Intelligence Engine
# ============================================================================

class UltraIntelligenceEngine:
    """Unified intelligence engine for WinGo 30s predictions — v2.0.

    12-model Hedge ensemble with evolving daily learning, 3-level Martingale
    awareness, pattern memory, volatility regime and cross-round correlation.
    """

    # Model names for the 12-model Hedge ensemble
    MODEL_NAMES = [
        "hip_ctw",            # 0: HIP CTW variable-order Markov
        "hip_ngram",          # 1: HIP n-gram (orders 1-5)
        "hip_streak",         # 2: HIP Bayesian streak
        "hip_frequency",      # 3: HIP side frequency
        "evoseq_ensemble",    # 4: Full EVOSEQ (Transformer + Mamba + baseline)
        "decay_markov",       # 5: Decay-weighted transition matrix
        "session_bias",       # 6: Session-position temporal bias
        "exploit_detector",   # 7: ExhaustiveExploitDetector blended prediction
        "pattern_intelligence", # 8: ACF + streak-reversal + cond-freq + gap
        "three_level_ml",     # 9: Martingale-level-calibrated ML prediction
        "volatility_regime",  # 10: Rolling volatility band momentum/mean-rev
        "cross_round_corr",   # 11: Multi-lag cross-round correlation
    ]

    def __init__(self, max_history: int = 50000):
        # Core sub-engines
        self.hip = HighIntelligencePredictor(max_history=max_history)
        self.exploit = ExhaustiveExploitDetector(history_limit=5000, permutation_rounds=200)
        self.evidence_gate = EvidenceGate()
        self.decay_markov = DecayMarkov(decay=0.995)
        self.session_features = SessionPositionFeatures(session_length=120, n_bins=6)

        # New v2 sub-engines
        self.pattern_intel = PatternIntelligence()
        self.three_level = ThreeLevelWinningAlgorithm()
        self.volatility_regime = VolatilityRegimeModel()
        self.cross_round = CrossRoundCorrelation()

        # Hedge ensemble: 12 models
        self.hedge = HedgeEnsemble(n_models=12, eta=0.10)

        # Streak tracking
        self.streak = StreakState()

        # State tracking
        self._last_issue: Optional[str] = None
        self._last_prediction_side: Optional[str] = None
        self._last_model_probs: Optional[np.ndarray] = None
        self._initialized_history = False
        self._last_sides: List[int] = []

        # Daily learning scheduler (started in load_streak or on first predict)
        self._daily_scheduler_started = False

    # ---- History initialization (idempotent) --------------------------------

    def _ensure_history(self, history: List[int]) -> None:
        """Feed history into sub-engines that need it."""
        int_history = [int(x) for x in history]
        sides = [1 if d >= 5 else 0 for d in int_history]

        # HIP: feed only new observations
        if len(self.hip.history) < len(int_history):
            for d in int_history[len(self.hip.history):]:
                self.hip.add_observation(d)

        # Decay Markov: rebuild from full history if not initialized
        if not self._initialized_history and len(int_history) > 1:
            self.decay_markov.update_sequence(int_history)
            self.session_features.observe_sequence(int_history[-500:])
            # PatternIntelligence: full update on init
            self.pattern_intel.update(int_history)
            # CrossRoundCorrelation: refit on init
            self.cross_round._refit(sides)
            self._initialized_history = True

        self._last_sides = sides

    # ---- Reward / feedback from resolved outcomes ----------------------------

    def reward_resolved(self, actual_digit: int, issue_closed: Optional[str] = None) -> None:
        """Feed the resolved outcome back into all sub-engines."""
        if actual_digit is None:
            return
        actual_digit = int(actual_digit) % 10
        actual_side = "Big" if actual_digit >= 5 else "Small"
        side_int = 1 if actual_digit >= 5 else 0

        # Update HIP
        self.hip.reward(actual_digit)

        # Update decay Markov (needs previous digit)
        if self.hip.history and len(self.hip.history) >= 2:
            prev = int(self.hip.history[-2])
            self.decay_markov.update(prev, actual_digit)

        # Update session features
        self.session_features.observe(actual_digit)

        # Update pattern intelligence win tracker
        self.pattern_intel.reward(actual_digit)

        # Update 3-level Martingale outcome
        if issue_closed and self.three_level.state.last_predicted_side:
            self.three_level.record_outcome(issue_closed, actual_side, db=None)

        # Update exploit detector calibration
        if self._last_model_probs is not None:
            p_big = float(self._last_model_probs[4])  # EVOSEQ ensemble p_big
            self.exploit.reward(p_big, actual_digit)

        # Update Hedge ensemble weights
        if self._last_model_probs is not None:
            self.hedge.update(self._last_model_probs, side_int)
            self._last_model_probs = None

        # Update streak
        if self._last_prediction_side is not None:
            won = (self._last_prediction_side == actual_side)
            self.streak.record(won)
            self._last_prediction_side = None

    # ---- Persistence --------------------------------------------------------

    def save_streak(self, db) -> None:
        """Persist streak and sub-engine states to database."""
        from backend.database import save_ai_brain_state
        save_ai_brain_state(
            db=db,
            model_name="Ultra_Streak_State",
            generation=self.streak.total_games,
            total_samples=self.streak.total_games,
            weights_json=json.dumps(self.streak.to_dict()),
            win_rate=self.streak.session_win_rate * 100,
        )
        # Periodically persist PatternIntelligence (every 100 games)
        if self.streak.total_games % 100 == 0:
            try:
                self.pattern_intel.save_state(db)
            except Exception:
                pass
        # Persist 3-Level state
        try:
            self.three_level.save_state(db)
        except Exception:
            pass

    def load_streak(self, db) -> None:
        """Load persisted streak and all sub-engine states from database."""
        from backend.database import load_ai_brain_state

        # Streak
        brain = load_ai_brain_state(db, model_name="Ultra_Streak_State")
        if brain and brain.synaptic_weights:
            try:
                d = json.loads(brain.synaptic_weights)
                self.streak = StreakState.from_dict(d)
                print(f"[ULTRA] Loaded streak: W{self.streak.win_streak}/L{self.streak.loss_streak} "
                      f"| Session {self.streak.session_win_rate*100:.1f}%")
            except Exception as e:
                print(f"[ULTRA] Could not load streak: {e}")

        # PatternIntelligence
        try:
            self.pattern_intel.load_state(db)
        except Exception as e:
            print(f"[ULTRA] PatternIntelligence load failed: {e}")

        # 3-Level Martingale
        try:
            self.three_level.load_state(db)
        except Exception as e:
            print(f"[ULTRA] 3-Level load failed: {e}")

        # Hedge eta from daily learning recommendation
        try:
            eta_brain = load_ai_brain_state(db, model_name="Ensemble_Hyperparams")
            if eta_brain and eta_brain.synaptic_weights:
                hp = json.loads(eta_brain.synaptic_weights)
                eta = float(hp.get("hedge_eta", 0.10))
                self.hedge.eta = eta
                print(f"[ULTRA] Loaded hedge eta={eta:.4f} from daily learning")
        except Exception as e:
            print(f"[ULTRA] Could not load hedge eta: {e}")

        # Start daily learning scheduler
        if not self._daily_scheduler_started:
            try:
                from backend.daily_learning import DailyLearningScheduler
                sched = DailyLearningScheduler()
                sched.start()
                self._daily_scheduler_started = True
                print("[ULTRA] DailyLearningScheduler started")
            except Exception as e:
                print(f"[ULTRA] DailyLearningScheduler start failed: {e}")

    # ---- Main prediction ----------------------------------------------------

    def predict(
        self,
        history: List[int],
        registry_state: dict,
        db,
    ) -> Optional[dict]:
        """Produce a unified prediction with exploit gating and honest confidence.

        Returns a dict compatible with the existing Telegram bot format, or None
        if insufficient data.
        """
        if not registry_state:
            return None

        li = registry_state.get("live_inference") or registry_state
        if not isinstance(li, dict):
            return None
        if "probability_big" not in li and "prediction" not in li:
            return None

        int_history = [int(x) for x in history]
        if len(int_history) < 10:
            return None

        # 0. Ensure all sub-engines have seen the history
        self._ensure_history(int_history)

        # ====================================================================
        # STAGE 1: Collect predictions from all 12 sub-models
        # ====================================================================

        # --- HIP sub-models (0-3) ---
        hip_result = self.hip.predict()
        hip_ctw_p_big = float(hip_result.digit_distribution[5:].sum())
        hip_ngram_p_big = max(0.01, min(0.99, 0.5 + (hip_result.probability_big - 0.5) *
                                        float(hip_result.markov_weight) / max(0.01, float(hip_result.ctw_weight))))
        streak_p_big_val, _ = self.hip._streak_side_probs()
        freq_p_big_val, _ = self.hip._side_frequency_probs(window=75)

        # --- EVOSEQ ensemble (4) ---
        evoseq_p_big = float(li["probability_big"])

        # --- Decay Markov (5) ---
        last_digit = int_history[-1] if int_history else 0
        decay_p_big, _ = self.decay_markov.predict_side_proba(last_digit)

        # --- Session-position bias (6) ---
        session_p_big, session_weight, session_bin = self.session_features.current_bin_bias()

        # --- Exploit detector (7) ---
        exploit_report = self.exploit.analyse(int_history[-5000:])
        exploit_p_big = float(exploit_report.tests.get("blended_p_big", 0.5))

        # --- Pattern Intelligence (8) ---
        sides = [1 if d >= 5 else 0 for d in int_history]
        try:
            self.pattern_intel.update(int_history)
            pattern_pred = self.pattern_intel.predict(int_history)
            pattern_p_big = float(pattern_pred.p_big)
        except Exception as _pe:
            pattern_p_big = 0.5
            pattern_pred = None

        # --- 3-Level Martingale ML (9) ---
        try:
            three_level_result = self.three_level.predict(int_history, db=db)
            if three_level_result:
                tl_side = three_level_result["prediction"]
                tl_conf = three_level_result["confidence"] / 100.0
                three_level_p_big = tl_conf if tl_side == "Big" else (1.0 - tl_conf)
            else:
                three_level_p_big = 0.5
                three_level_result = {}
        except Exception as _tl:
            three_level_p_big = 0.5
            three_level_result = {}

        # --- Volatility Regime (10) ---
        try:
            volatility_p_big = self.volatility_regime.predict_p_big(sides)
        except Exception:
            volatility_p_big = 0.5

        # --- Cross-Round Correlation (11) ---
        try:
            cross_round_p_big = self.cross_round.predict_p_big(sides)
        except Exception:
            cross_round_p_big = 0.5

        # Collect all 12 model P(Big) values
        model_p_big = np.array([
            hip_ctw_p_big,        # 0
            hip_ngram_p_big,      # 1
            streak_p_big_val,     # 2
            freq_p_big_val,       # 3
            evoseq_p_big,         # 4
            decay_p_big,          # 5
            session_p_big,        # 6
            exploit_p_big,        # 7
            pattern_p_big,        # 8  NEW
            three_level_p_big,    # 9  NEW
            volatility_p_big,     # 10 NEW
            cross_round_p_big,    # 11 NEW
        ], dtype=np.float64)
        model_p_big = np.clip(model_p_big, 0.01, 0.99)

        # Compute consensus fraction (how many models agree with majority)
        big_votes = float(np.sum(model_p_big >= 0.5))
        model_consensus = max(big_votes, 12 - big_votes) / 12.0

        # Store for later reward update
        self._last_model_probs = model_p_big.copy()

        # ====================================================================
        # STAGE 2: Hedge ensemble blending
        # ====================================================================

        blended_p_big = self.hedge.predict(model_p_big)
        blended_p_small = 1.0 - blended_p_big

        # Resolve side
        if blended_p_big >= 0.5:
            prediction = "Big"
            raw_side_p = blended_p_big
        else:
            prediction = "Small"
            raw_side_p = blended_p_small

        # ====================================================================
        # STAGE 3: Calibration
        # ====================================================================

        # HIP calibration (rolling Platt + isotonic)
        hip_cal_single = float(hip_result.calibrated_p_single)

        # Blend HIP calibration with raw side probability
        cal_single = 0.70 * hip_cal_single + 0.30 * raw_side_p
        cal_single = max(0.50, min(0.99, cal_single))

        # Multi-horizon attenuation
        cal_h2 = 0.5 + 0.94 * (cal_single - 0.5)
        cal_h3 = 0.5 + 0.87 * (cal_single - 0.5)
        p_win_in_3 = three_level_win_probability(cal_single, cal_h2, cal_h3, rho=0.06)

        # ====================================================================
        # STAGE 4: Digit-level target/hedge selection
        # ====================================================================

        # Collect digit distributions from key models
        hip_digits = hip_result.digit_distribution
        decay_digits = self.decay_markov.predict_proba(last_digit)
        exploit_digits = exploit_report.digit_distribution

        # Build EVOSEQ digit distribution
        evoseq_target = int(li["targetNum"])
        evoseq_hedge = int(li["hedgeNum"])
        evoseq_digits = np.full(10, 0.03, dtype=np.float64)
        evoseq_digits[evoseq_target] = 0.40
        evoseq_digits[evoseq_hedge] = 0.20
        side_slice = slice(5, 10) if prediction == "Big" else slice(0, 5)
        evoseq_digits[side_slice] += 0.37 / 5.0
        evoseq_digits /= evoseq_digits.sum()

        # Hedge-weighted digit blending
        digit_dists = [hip_digits, decay_digits, evoseq_digits, exploit_digits]
        # Use simplified weights: 35% HIP, 20% decay, 30% EVOSEQ, 15% exploit
        digit_weights = np.array([0.35, 0.20, 0.30, 0.15])
        final_digits = np.zeros(10, dtype=np.float64)
        for i, dist in enumerate(digit_dists):
            final_digits += digit_weights[i] * dist
        final_digits = np.clip(final_digits, 1e-6, None)
        final_digits /= final_digits.sum()

        sorted_idx = np.argsort(final_digits)[::-1]
        targetNum = int(sorted_idx[0])
        hedgeNum = int(sorted_idx[1])

        # ====================================================================
        # STAGE 5: Exploit gating + honest confidence classification
        # ====================================================================

        strike_quality, display_conf, action = classify_confidence(
            cal_single=cal_single,
            p_win_in_3=p_win_in_3,
            exploit_score=exploit_report.exploit_score,
            reject_iid=exploit_report.reject_iid,
            engines_agree=exploit_report.engines_agree,
            streak=self.streak,
            pattern_p_big=pattern_p_big,
            model_consensus=model_consensus,
        )

        # ====================================================================
        # STAGE 6: Evidence gate (outcome-calibrated, final step)
        # ====================================================================

        evidence = self.evidence_gate.assess(db, blended_p_big)

        # Blend evidence-calibrated confidence with our classification
        evidence_conf = float(evidence.get("confidence", 0.5))
        if evidence.get("validated_edge"):
            # Strong evidence: trust it more
            final_conf = 0.60 * (evidence_conf * 100.0) + 0.40 * display_conf
            if strike_quality not in ("SKIP", "LOW_CONFIDENCE"):
                strike_quality = "VALIDATED"
        else:
            final_conf = 0.25 * (evidence_conf * 100.0) + 0.75 * display_conf

        # Apply honest floor (no artificial inflation)
        final_conf = round(max(50.0, min(99.0, final_conf)), 1)

        # If we're in SKIP mode, cap confidence display
        if action == "SKIP":
            final_conf = min(final_conf, 55.0)

        # ====================================================================
        # STAGE 7: Multi-horizon forecasts
        # ====================================================================

        drift_level = registry_state.get("drift_level", "STABLE")
        h1 = np.array(hip_result.h1, dtype=np.float64)
        h2 = np.array(hip_result.h2, dtype=np.float64)
        h3 = np.array(hip_result.h3, dtype=np.float64)
        if drift_level in ("STRONG_BIG_MOMENTUM", "MODERATE_BIG_BIAS"):
            for h in (h1, h2, h3):
                h[5:] *= 1.10
                h /= h.sum()
        elif drift_level in ("STRONG_SMALL_MOMENTUM", "MODERATE_SMALL_BIAS"):
            for h in (h1, h2, h3):
                h[:5] *= 1.10
                h /= h.sum()

        # ====================================================================
        # STAGE 8: Build result dict
        # ====================================================================

        dominant_prob = max(blended_p_big, blended_p_small)
        advantage = max(0.001, dominant_prob - 0.50)
        entropy = float(-np.sum(final_digits * np.log(final_digits + 1e-12)))

        # Scorecard for Telegram display
        scorecard = {
            "recent_20": "".join("W" if w else "L" for w in self.streak.recent_results),
            "win_streak": self.streak.win_streak,
            "loss_streak": self.streak.loss_streak,
            "session_win_rate": round(self.streak.session_win_rate * 100, 1),
            "total_wins": self.streak.total_wins,
            "total_losses": self.streak.total_losses,
        }

        # Build insight string
        insight_parts = [
            f"Regime: {drift_level}",
            f"P(win3): {p_win_in_3:.3f}",
            f"P(single): {cal_single:.3f}",
            f"Exploit: {exploit_report.exploit_score:.2f}",
            f"IID-reject: {'Y' if exploit_report.reject_iid else 'N'}",
            f"Agree: {exploit_report.engines_agree}/4",
            f"Consensus: {model_consensus:.0%}",
            f"Streak: W{self.streak.win_streak}/L{self.streak.loss_streak}",
            f"Session: {self.streak.session_win_rate*100:.0f}%",
            f"3Lvl: L{three_level_result.get('level', 1)}",
        ]
        loophole_insight = "Ultra v2.0 | " + " | ".join(insight_parts)

        # Append evidence trail
        loophole_insight += (
            f" | Evidence: {evidence.get('reason', 'COLLECTING')} "
            f"(n={evidence.get('resolved_predictions', 0)}, "
            f"Brier Δ={float(evidence.get('brier_improvement', 0)):.4f})"
        )

        result = {
            "prediction": prediction,
            "confidence": final_conf,
            "targetNum": targetNum,
            "hedgeNum": hedgeNum,
            "probability_big": round(blended_p_big, 4),
            "probability_small": round(blended_p_small, 4),
            "patternName": f"🧠 Ultra v2.0 {drift_level}",
            "loopholeInsight": loophole_insight,
            "strikeQuality": strike_quality,
            "action": action,
            "h1": [round(float(x), 4) for x in h1.tolist()],
            "h2": [round(float(x), 4) for x in h2.tolist()],
            "h3": [round(float(x), 4) for x in h3.tolist()],
            # Calibration
            "calibratedPSingle": round(cal_single, 4),
            "calibratedPWinIn3": round(p_win_in_3, 4),
            # Exploit detection
            "exploitScore": round(exploit_report.exploit_score, 4),
            "rejectIID": exploit_report.reject_iid,
            "enginesAgree": exploit_report.engines_agree,
            "unanimousBig": exploit_report.unanimous_big,
            "unanimousSmall": exploit_report.unanimous_small,
            # Model consensus
            "modelConsensus": round(model_consensus, 3),
            # Scoring & diagnostics
            "predictiveScore": round(dominant_prob, 3),
            "calibrationQuality": round(float(registry_state.get("calibration_quality", 0.92)), 3),
            "stabilityScore": round(float(registry_state.get("stability_score", 0.88)), 3),
            "brierScore": round(float(registry_state.get("brier_score", 0.15)), 3),
            "logLoss": round(float(registry_state.get("log_loss", 0.55)), 3),
            "nullAdvantage": round(advantage, 3),
            "entropy": round(entropy, 3),
            "driftLevel": drift_level,
            "driftScore": round(float(registry_state.get("drift_score", 0.05)), 3),
            "changeProbability": round(float(hip_result.change_probability), 4),
            "regimeStrength": round(float(hip_result.regime_strength), 3),
            "streakRunLength": int(hip_result.streak_run_length),
            "modelsTested": int(registry_state.get("models_tested", 1)),
            "activeChallengers": int(registry_state.get("active_challengers", 1)),
            "retiredModels": int(registry_state.get("retired_models", 0)),
            # Ensemble weights (12-model)
            "ensembleWeights": self.hedge.get_weights_dict(self.MODEL_NAMES),
            "familyWeights": {
                "hip_statistical": round(float(self.hedge.weights[:4].sum()) / self.hedge.weights.sum(), 3),
                "evoseq_deep": round(float(self.hedge.weights[4]) / self.hedge.weights.sum(), 3),
                "decay_markov": round(float(self.hedge.weights[5]) / self.hedge.weights.sum(), 3),
                "session_temporal": round(float(self.hedge.weights[6]) / self.hedge.weights.sum(), 3),
                "exploit_statistical": round(float(self.hedge.weights[7]) / self.hedge.weights.sum(), 3),
                "pattern_intelligence": round(float(self.hedge.weights[8]) / self.hedge.weights.sum(), 3),
                "three_level_ml": round(float(self.hedge.weights[9]) / self.hedge.weights.sum(), 3),
                "volatility_regime": round(float(self.hedge.weights[10]) / self.hedge.weights.sum(), 3),
                "cross_round_corr": round(float(self.hedge.weights[11]) / self.hedge.weights.sum(), 3),
            },
            # Streak / scorecard
            "scorecard": scorecard,
            # 3-Level Martingale status
            "martingaleLevel": three_level_result.get("level", 1),
            "martingaleLevelLabel": three_level_result.get("level_label", "🟢 CONSERVATIVE"),
            "martingaleStake": three_level_result.get("stake_multiplier", 1.0),
            "martingaleLossStreak": self.three_level.state.loss_streak,
            "martingaleLevelWinRates": self.three_level.state.level_win_rate,
            # Martingale hint (backward compat)
            "martingale3Hint": {
                "pWinIn3": round(p_win_in_3, 4),
                "pCorrectSingle": round(cal_single, 4),
                "strike": strike_quality,
                "level": three_level_result.get("level", 1),
                "stake_multiplier": three_level_result.get("stake_multiplier", 1.0),
            },
            # Pattern intelligence diagnostics
            "patternSignificantLags": (
                [{"lag": l, "strength": round(s, 4)} for l, s in pattern_pred.significant_lags]
                if pattern_pred else []
            ),
            "patternCurrentStreak": pattern_pred.current_streak if pattern_pred else 0,
            "patternReversalProb": round(pattern_pred.p_reversal_at_streak, 4) if pattern_pred else 0.5,
            # Per-model P(Big) vector
            "modelPBigVector": {
                name: round(float(model_p_big[i]), 4)
                for i, name in enumerate(self.MODEL_NAMES)
            },
            # Evidence
            "evidence": evidence,
            "rawConfidence": display_conf,
            "environmentVector": [
                float(hip_result.entropy),
                float(registry_state.get("drift_score", 0.05)),
                sum(1 for x in int_history[-100:] if x >= 5) / max(1, len(int_history[-100:])),
                sum(1 for x in int_history[-500:] if x >= 5) / max(1, len(int_history[-500:])),
                advantage,
                float(registry_state.get("calibration_quality", 0.92)),
                float(registry_state.get("stability_score", 0.88)),
                float(registry_state.get("disagreement_score", 0.0)),
                float(registry_state.get("momentum_score", 0.0)),
                float(registry_state.get("cyclical_strength", 0.0)),
            ],
            "adaptive_tuning": registry_state.get("adaptive_tuning", {}),
        }

        # Remember for next-round reward
        self._last_prediction_side = prediction

        return result
