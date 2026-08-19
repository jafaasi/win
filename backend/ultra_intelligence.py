#!/usr/bin/env python3
"""
Ultra Intelligence Engine for WinGo 30s
========================================
Unified top-level intelligence that replaces the fragmented blending in
BeastPredictor by orchestrating:

  1. ExhaustiveExploitDetector — 11 statistical tests + unanimous engine filter
  2. HighIntelligencePredictor — CTW, Bayesian streak, BOCPD, rolling calibration
  3. EVOSEQ ensemble — Transformer + Mamba + statistical models
  4. Adaptive Hedge/EWA online ensemble weighting
  5. Decay-weighted Markov transition matrix
  6. Session-position temporal features
  7. Honest confidence bands with SKIP signal
  8. Persisted win/loss streak state

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
# 5. Confidence band classifier
# ============================================================================

def classify_confidence(
    cal_single: float,
    p_win_in_3: float,
    exploit_score: float,
    reject_iid: bool,
    engines_agree: int,
    streak: StreakState,
) -> Tuple[str, float, str]:
    """Classify confidence into honest bands.

    Returns (strike_quality, display_confidence_pct, action).

    Action is one of:
      - "SKIP"     : no edge, don't bet
      - "CAUTION"  : weak edge, bet minimum
      - "FORECAST" : normal edge, bet normally
      - "STRIKE"   : strong validated edge
    """
    skip_thresh = streak.skip_threshold()

    # SKIP: no exploitable edge
    if cal_single < skip_thresh and not reject_iid:
        conf_pct = round(cal_single * 100, 1)
        return "SKIP", conf_pct, "SKIP"

    if cal_single < skip_thresh and exploit_score < 0.3:
        conf_pct = round(cal_single * 100, 1)
        return "SKIP", conf_pct, "SKIP"

    # CAUTION: marginal edge
    if cal_single < 0.58 or (p_win_in_3 < 0.88 and engines_agree < 3):
        conf_pct = round(cal_single * 100, 1)
        return "LOW_CONFIDENCE", conf_pct, "CAUTION"

    # Map to confidence bands
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

    # Loss-streak dampener: downgrade conviction after 3+ losses
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
    """Unified intelligence engine for WinGo 30s predictions.

    Combines all sub-engines through a clean, regret-minimizing ensemble
    with exploit gating and honest confidence bands.
    """

    # Model names for the Hedge ensemble
    MODEL_NAMES = [
        "hip_ctw",          # 0: HIP CTW variable-order Markov
        "hip_ngram",        # 1: HIP n-gram (orders 1-5)
        "hip_streak",       # 2: HIP Bayesian streak
        "hip_frequency",    # 3: HIP side frequency
        "evoseq_ensemble",  # 4: Full EVOSEQ (Transformer + Mamba + baseline)
        "decay_markov",     # 5: Decay-weighted transition matrix
        "session_bias",     # 6: Session-position temporal bias
        "exploit_detector", # 7: ExhaustiveExploitDetector blended prediction
    ]

    def __init__(self, max_history: int = 50000):
        # Sub-engines
        self.hip = HighIntelligencePredictor(max_history=max_history)
        self.exploit = ExhaustiveExploitDetector(history_limit=5000, permutation_rounds=200)
        self.evidence_gate = EvidenceGate()
        self.decay_markov = DecayMarkov(decay=0.995)
        self.session_features = SessionPositionFeatures(session_length=120, n_bins=6)

        # Hedge ensemble: 8 models
        self.hedge = HedgeEnsemble(n_models=8, eta=0.10)

        # Streak tracking
        self.streak = StreakState()

        # State tracking
        self._last_issue: Optional[str] = None
        self._last_prediction_side: Optional[str] = None
        self._last_model_probs: Optional[np.ndarray] = None
        self._initialized_history = False

    # ---- History initialization (idempotent) --------------------------------

    def _ensure_history(self, history: List[int]) -> None:
        """Feed history into sub-engines that need it."""
        int_history = [int(x) for x in history]

        # HIP: feed only new observations
        if len(self.hip.history) < len(int_history):
            for d in int_history[len(self.hip.history):]:
                self.hip.add_observation(d)

        # Decay Markov: rebuild from full history if not initialized
        if not self._initialized_history and len(int_history) > 1:
            self.decay_markov.update_sequence(int_history)
            self.session_features.observe_sequence(int_history[-500:])
            self._initialized_history = True

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
        """Persist streak state to database."""
        from backend.database import save_ai_brain_state
        save_ai_brain_state(
            db=db,
            model_name="Ultra_Streak_State",
            generation=self.streak.total_games,
            total_samples=self.streak.total_games,
            weights_json=json.dumps(self.streak.to_dict()),
            win_rate=self.streak.session_win_rate * 100,
        )

    def load_streak(self, db) -> None:
        """Load persisted streak state from database."""
        from backend.database import load_ai_brain_state
        brain = load_ai_brain_state(db, model_name="Ultra_Streak_State")
        if brain and brain.synaptic_weights:
            try:
                d = json.loads(brain.synaptic_weights)
                self.streak = StreakState.from_dict(d)
                print(f"[ULTRA] Loaded streak: W{self.streak.win_streak}/L{self.streak.loss_streak} "
                      f"| Session {self.streak.session_win_rate*100:.1f}%")
            except Exception as e:
                print(f"[ULTRA] Could not load streak: {e}")

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
        if not registry_state or not registry_state.get("live_inference"):
            return None

        li = registry_state["live_inference"]
        int_history = [int(x) for x in history]
        if len(int_history) < 10:
            return None

        # 0. Ensure all sub-engines have seen the history
        self._ensure_history(int_history)

        # ====================================================================
        # STAGE 1: Collect predictions from all 8 sub-models
        # ====================================================================

        # --- HIP sub-models ---
        hip_result = self.hip.predict()
        hip_ctw_p_big = float(hip_result.digit_distribution[5:].sum())
        # For ngram/streak/freq, we extract from HIP's internal weights
        hip_ngram_p_big = max(0.01, min(0.99, 0.5 + (hip_result.probability_big - 0.5) *
                                        float(hip_result.markov_weight) / max(0.01, float(hip_result.ctw_weight))))
        streak_p_big_val, _ = self.hip._streak_side_probs()
        freq_p_big_val, _ = self.hip._side_frequency_probs(window=75)

        # --- EVOSEQ ensemble ---
        evoseq_p_big = float(li["probability_big"])

        # --- Decay Markov ---
        last_digit = int_history[-1] if int_history else 0
        decay_p_big, _ = self.decay_markov.predict_side_proba(last_digit)

        # --- Session-position bias ---
        session_p_big, session_weight, session_bin = self.session_features.current_bin_bias()

        # --- Exploit detector ---
        exploit_report = self.exploit.analyse(int_history[-5000:])
        exploit_p_big = float(exploit_report.tests.get("blended_p_big", 0.5))

        # Collect all model P(Big) into array
        model_p_big = np.array([
            hip_ctw_p_big,      # 0: hip_ctw
            hip_ngram_p_big,    # 1: hip_ngram
            streak_p_big_val,   # 2: hip_streak
            freq_p_big_val,     # 3: hip_frequency
            evoseq_p_big,       # 4: evoseq_ensemble
            decay_p_big,        # 5: decay_markov
            session_p_big,      # 6: session_bias
            exploit_p_big,      # 7: exploit_detector
        ], dtype=np.float64)
        model_p_big = np.clip(model_p_big, 0.01, 0.99)

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
            f"Streak: W{self.streak.win_streak}/L{self.streak.loss_streak}",
            f"Session: {self.streak.session_win_rate*100:.0f}%",
        ]
        loophole_insight = "Ultra v1.0 | " + " | ".join(insight_parts)

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
            "patternName": f"🧠 Ultra v1.0 {drift_level}",
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
            # Ensemble weights
            "ensembleWeights": self.hedge.get_weights_dict(self.MODEL_NAMES),
            "familyWeights": {
                "hip_statistical": round(sum(self.hedge.weights[:4]) / self.hedge.weights.sum(), 3),
                "evoseq_deep": round(float(self.hedge.weights[4]) / self.hedge.weights.sum(), 3),
                "decay_markov": round(float(self.hedge.weights[5]) / self.hedge.weights.sum(), 3),
                "session_temporal": round(float(self.hedge.weights[6]) / self.hedge.weights.sum(), 3),
                "exploit_statistical": round(float(self.hedge.weights[7]) / self.hedge.weights.sum(), 3),
            },
            # Streak / scorecard
            "scorecard": scorecard,
            # Martingale hint
            "martingale3Hint": {
                "pWinIn3": round(p_win_in_3, 4),
                "pCorrectSingle": round(cal_single, 4),
                "strike": strike_quality,
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
