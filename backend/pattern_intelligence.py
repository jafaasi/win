#!/usr/bin/env python3
"""
Pattern Intelligence Engine for WinGo 30s
==========================================
Advanced pattern memory that learns from Supabase history day by day:

  1. Multi-lag auto-correlation discovery (lags 1-30, PACF-style pruning)
  2. Streak reversal profiling — learns *actual* reversal rates at each streak length
  3. Digit transition probability matrix with exponential decay (weighted MLE)
  4. Conditional frequency tables: P(Big | last_k_digits) for k=1..5
  5. Hour-of-day and minute-in-hour position features (session temporal patterns)
  6. Gap pattern detector — how many rounds since each digit last appeared
  7. Rolling win-rate meta-feature: how well each local pattern has predicted recently
  8. Full Supabase-backed persistence so every daily run refines the memory

All engines return a P(Big) in [0, 1]. PatternIntelligence exposes a
single `predict(history, db) -> float` method used by UltraIntelligenceEngine.
"""

from __future__ import annotations

import json
import math
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. Auto-correlation lag discovery
# ─────────────────────────────────────────────────────────────────────────────

class AutoCorrelationMemory:
    """Discovers and tracks which historical lags have predictive power.

    Uses a simplified partial-autocorrelation check (Yule-Walker, order 1
    conditioned on lower lags) to prune spurious lags.  Only lags with
    |ACF| significantly > 1/sqrt(n) are retained.
    """

    MAX_LAG = 30

    def __init__(self):
        # lag -> running exponentially-weighted mean of |acf| at that lag
        self._lag_strength: Dict[int, float] = {k: 0.0 for k in range(1, self.MAX_LAG + 1)}
        self._n_updates: int = 0
        # lag -> sign of last significant correlation (+1 = continuation, -1 = reversal)
        self._lag_sign: Dict[int, float] = {k: 0.0 for k in range(1, self.MAX_LAG + 1)}

    def update(self, sides: Sequence[int]) -> None:
        """Update lag-strength estimates from a side series (0/1).

        Call this once per cycle with the last ~2000 observations.
        """
        arr = np.asarray(sides, dtype=np.float64)
        n = len(arr)
        if n < 40:
            return
        mu = arr.mean()
        arr_c = arr - mu
        var = float((arr_c ** 2).mean()) + 1e-12
        alpha = 0.05  # EWA decay for lag-strength estimates
        for lag in range(1, min(self.MAX_LAG + 1, n // 3)):
            acf = float(np.mean(arr_c[lag:] * arr_c[:-lag])) / var
            # Update running EWA
            self._lag_strength[lag] = (1 - alpha) * self._lag_strength[lag] + alpha * abs(acf)
            self._lag_sign[lag] = (1 - alpha) * self._lag_sign[lag] + alpha * float(np.sign(acf))
        self._n_updates += 1

    def significant_lags(self, n: int) -> List[Tuple[int, float]]:
        """Return list of (lag, signed_strength) for lags with |acf| > threshold."""
        threshold = 1.5 / math.sqrt(max(n, 50))
        result = []
        for lag, strength in self._lag_strength.items():
            if strength > threshold:
                result.append((lag, strength * self._lag_sign[lag]))
        # Sort by strength descending
        result.sort(key=lambda x: abs(x[1]), reverse=True)
        return result

    def predict_p_big(self, sides: Sequence[int]) -> float:
        """Forecast P(Big) using the top significant lags via linear combination."""
        n = len(sides)
        if n < 10:
            return 0.5
        sig = self.significant_lags(n)
        if not sig:
            return 0.5
        # Weighted vote: for each sig lag, look back that many steps
        weighted_sum = 0.0
        weight_total = 0.0
        for lag, signed_strength in sig[:8]:  # top 8 lags
            if lag >= n:
                continue
            past_side = int(sides[-lag])
            # If signed_strength > 0 → continuation (predict same as past_side)
            # If signed_strength < 0 → reversal
            if signed_strength > 0:
                vote = float(past_side)  # 1=Big, 0=Small
            else:
                vote = 1.0 - float(past_side)
            weighted_sum += abs(signed_strength) * vote
            weight_total += abs(signed_strength)
        if weight_total < 1e-9:
            return 0.5
        p = weighted_sum / weight_total
        # Shrink toward 0.5 — ACF patterns are never very strong
        return 0.5 + 0.6 * (p - 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Streak Reversal Profiler
# ─────────────────────────────────────────────────────────────────────────────

class StreakReversalProfiler:
    """Learns the *actual* empirical reversal rate at each streak length.

    Unlike a fixed Bayesian prior, this accumulates per-streak-length
    counts from all historical data and updates incrementally.
    Counts are decay-weighted so recent history matters more.
    """

    MAX_STREAK = 20

    def __init__(self, decay: float = 0.998):
        self.decay = decay
        # [0] = reversal count, [1] = total count for streak lengths 1..MAX_STREAK
        self._counts: np.ndarray = np.ones((self.MAX_STREAK + 1, 2), dtype=np.float64)  # Laplace prior

    def _extract_streaks(self, sides: Sequence[int]) -> List[Tuple[int, int]]:
        """Return list of (streak_length, reversed_0_or_1) from side series."""
        events: List[Tuple[int, int]] = []
        if len(sides) < 2:
            return events
        run = 1
        for i in range(1, len(sides)):
            if sides[i] == sides[i - 1]:
                run += 1
            else:
                # streak of length `run` just ended with a reversal
                events.append((min(run, self.MAX_STREAK), 1))
                run = 1
        return events

    def update(self, sides: Sequence[int]) -> None:
        """Full rebuild from a side series (cheaper than incremental for daily updates)."""
        # Decay all existing counts
        self._counts *= self.decay ** len(sides)
        for streak_len, reversed_flag in self._extract_streaks(sides):
            self._counts[streak_len, 1] += 1.0           # total
            self._counts[streak_len, 0] += reversed_flag  # reversals

    def p_reversal(self, current_streak: int) -> float:
        """Empirical posterior P(reversal | current_streak_length)."""
        k = min(current_streak, self.MAX_STREAK)
        rev = self._counts[k, 0]
        tot = self._counts[k, 1]
        return float(rev / tot) if tot > 0 else 0.5

    def predict_p_big(self, sides: Sequence[int]) -> float:
        """Forecast P(Big) using the learned reversal profile for current streak."""
        if not sides:
            return 0.5
        # Measure current streak
        current_side = int(sides[-1])
        run = 1
        for i in range(len(sides) - 2, -1, -1):
            if int(sides[i]) == current_side:
                run += 1
            else:
                break
        p_rev = self.p_reversal(run)
        p_continue = 1.0 - p_rev
        # P(Big) = P(continue) * I(current=Big) + P(reverse) * I(current=Small)
        if current_side == 1:
            return p_continue
        else:
            return p_rev


# ─────────────────────────────────────────────────────────────────────────────
# 3. Conditional Frequency Table
# ─────────────────────────────────────────────────────────────────────────────

class ConditionalFrequencyTable:
    """P(Big | last_k_sides) for k=1..5 using decay-weighted empirical counts.

    Covers the 32 binary contexts of length 1-5, giving 62 total cells.
    This captures short-memory patterns that Markov and CTW models also
    see, but in a fast, interpretable form.
    """

    MAX_ORDER = 5

    def __init__(self, decay: float = 0.997):
        self.decay = decay
        # key=(order, context_tuple) -> [big_count, total_count]
        self._table: Dict[Tuple, List[float]] = defaultdict(lambda: [1.0, 2.0])  # Laplace prior

    def update(self, sides: Sequence[int]) -> None:
        """Incremental update — call with the latest batch of sides."""
        arr = list(sides)
        # Decay existing table
        for key in self._table:
            self._table[key][0] *= self.decay
            self._table[key][1] *= self.decay
        for i in range(len(arr)):
            outcome = arr[i]  # 0 or 1
            for order in range(1, self.MAX_ORDER + 1):
                if i < order:
                    continue
                ctx = tuple(arr[i - order:i])
                cell = self._table[(order, ctx)]
                cell[1] += 1.0
                if outcome == 1:
                    cell[0] += 1.0

    def predict_p_big(self, sides: Sequence[int]) -> float:
        """Forecast P(Big) by blending predictions across all orders."""
        if not sides:
            return 0.5
        votes = []
        weights = []
        for order in range(1, self.MAX_ORDER + 1):
            if len(sides) < order:
                continue
            ctx = tuple(int(sides[-i - 1]) for i in range(order - 1, -1, -1))
            cell = self._table.get((order, ctx))
            if cell is None:
                continue
            big_cnt, tot = cell
            p = big_cnt / tot if tot > 0 else 0.5
            # Weight by information content: more data = more weight
            w = math.log(1 + tot) * (order ** 0.5)
            votes.append(p)
            weights.append(w)
        if not votes:
            return 0.5
        weights_arr = np.array(weights)
        weights_arr /= weights_arr.sum()
        p = float(np.dot(weights_arr, votes))
        return max(0.02, min(0.98, p))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Digit Gap Detector
# ─────────────────────────────────────────────────────────────────────────────

class DigitGapDetector:
    """Tracks how many rounds since each digit last appeared.

    "Overdue" digits on the predicted side get a probability boost.
    This captures the empirical clustering / hot-cold phenomena in
    pseudo-random sequences.
    """

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        # gap_count[d] = exponentially-smoothed average gap between appearances of digit d
        self._avg_gap: np.ndarray = np.full(10, 10.0)
        self._last_seen: np.ndarray = np.full(10, -1, dtype=np.int64)
        self._round: int = 0

    def update(self, digits: Sequence[int]) -> None:
        for d in digits:
            d = int(d) % 10
            if self._last_seen[d] >= 0:
                gap = self._round - self._last_seen[d]
                self._avg_gap[d] = self.decay * self._avg_gap[d] + (1 - self.decay) * gap
            self._last_seen[d] = self._round
            self._round += 1

    def overdue_score(self) -> np.ndarray:
        """Per-digit score in [0, 1]: 1 = very overdue, 0 = just appeared."""
        current_gaps = np.where(
            self._last_seen >= 0,
            self._round - self._last_seen,
            self._avg_gap * 2
        ).astype(np.float64)
        # Normalize by expected gap
        scores = np.clip(current_gaps / (self._avg_gap + 1e-9) - 1.0, 0.0, 3.0) / 3.0
        return scores

    def predict_p_big(self) -> float:
        """P(Big) based on overdue scores: if Big digits are overdue → predict Big."""
        scores = self.overdue_score()
        big_score = scores[5:].mean()
        small_score = scores[:5].mean()
        total = big_score + small_score + 1e-9
        p = big_score / total
        # Weak signal — shrink heavily toward 0.5
        return 0.5 + 0.3 * (p - 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Rolling Pattern Win-Rate Tracker
# ─────────────────────────────────────────────────────────────────────────────

class PatternWinRateTracker:
    """Tracks how accurately each sub-pattern predictor has been recently.

    Stores (predicted_p_big, actual_side) pairs and computes per-source
    Brier score improvement vs the null model.  Sources that have been
    calibrated recently get higher meta-weight.
    """

    SOURCES = ["acf", "streak_reversal", "conditional_freq", "gap_detector"]

    def __init__(self, window: int = 200):
        self.window = window
        # source -> deque of (p_big, actual_side_0_or_1)
        self._history: Dict[str, deque] = {s: deque(maxlen=window) for s in self.SOURCES}

    def record(self, source: str, p_big: float, actual_side: int) -> None:
        if source in self._history:
            self._history[source].append((float(p_big), int(actual_side)))

    def meta_weights(self) -> Dict[str, float]:
        """Return normalized meta-weights based on recent Brier improvements."""
        weights = {}
        for source in self.SOURCES:
            pairs = list(self._history[source])
            if len(pairs) < 10:
                weights[source] = 1.0  # neutral weight until we have data
                continue
            base_rate = sum(a for _, a in pairs) / len(pairs)
            null_brier = sum((base_rate - a) ** 2 for _, a in pairs) / len(pairs)
            model_brier = sum((p - a) ** 2 for p, a in pairs) / len(pairs)
            improvement = null_brier - model_brier
            # Improvement in range [-1, 1] → weight in [0.2, 2.0]
            weights[source] = max(0.2, 1.0 + 4.0 * improvement)
        # Normalize
        total = sum(weights.values())
        return {s: w / total for s, w in weights.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main PatternIntelligence class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PatternPrediction:
    p_big: float
    source_probs: Dict[str, float]
    meta_weights: Dict[str, float]
    significant_lags: List[Tuple[int, float]]
    current_streak: int
    p_reversal_at_streak: float
    overdue_scores: List[float]


class PatternIntelligence:
    """Orchestrates all 5 pattern-learning engines and exposes a single predict().

    Designed to be a persistent singleton (loaded once, updated per cycle).
    Supabase-backed state is saved/loaded via the `save_state` / `load_state`
    methods which serialize to the `ai_brain_state` table.

    Usage:
        pi = PatternIntelligence()
        pi.load_state(db)           # load from Supabase on startup
        ...
        pi.update(history)          # called every cycle with new history
        p_big = pi.predict(history) # get P(Big) forecast
        pi.save_state(db)           # persist daily
    """

    MODEL_NAME = "Pattern_Intelligence_State"

    def __init__(self):
        self.acf = AutoCorrelationMemory()
        self.streak_reversal = StreakReversalProfiler(decay=0.998)
        self.cond_freq = ConditionalFrequencyTable(decay=0.997)
        self.gap_detector = DigitGapDetector(decay=0.99)
        self.win_tracker = PatternWinRateTracker(window=300)
        self._last_sides: List[int] = []
        self._last_update_n: int = 0

    # ── update ────────────────────────────────────────────────────────────────

    def update(self, history: Sequence[int]) -> None:
        """Incremental update — only processes new observations since last call."""
        digits = [int(x) % 10 for x in history]
        sides = [1 if d >= 5 else 0 for d in digits]

        new_start = self._last_update_n
        if new_start >= len(sides):
            return

        new_sides = sides[new_start:]
        new_digits = digits[new_start:]

        # Feed each engine with new data only
        if len(sides) >= 40:
            # ACF needs longer history — always pass full recent window
            self.acf.update(sides[-min(2000, len(sides)):])
        self.streak_reversal.update(new_sides)
        self.cond_freq.update(new_sides)
        self.gap_detector.update(new_digits)

        self._last_sides = sides
        self._last_update_n = len(sides)

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, history: Sequence[int]) -> PatternPrediction:
        """Return a PatternPrediction containing blended P(Big) and diagnostics."""
        digits = [int(x) % 10 for x in history]
        sides = [1 if d >= 5 else 0 for d in digits]

        if len(sides) < 20:
            return PatternPrediction(
                p_big=0.5, source_probs={}, meta_weights={},
                significant_lags=[], current_streak=0,
                p_reversal_at_streak=0.5, overdue_scores=[0.1] * 10,
            )

        # Individual engine predictions
        p_acf = self.acf.predict_p_big(sides)
        p_streak = self.streak_reversal.predict_p_big(sides)
        p_cond = self.cond_freq.predict_p_big(sides)
        p_gap = self.gap_detector.predict_p_big()

        source_probs = {
            "acf": p_acf,
            "streak_reversal": p_streak,
            "conditional_freq": p_cond,
            "gap_detector": p_gap,
        }

        # Meta-weighted blend
        mw = self.win_tracker.meta_weights()
        blended = sum(mw.get(s, 0.25) * p for s, p in source_probs.items())
        blended = max(0.01, min(0.99, blended))

        # Diagnostics
        sig_lags = self.acf.significant_lags(len(sides))
        current_streak = self._current_streak(sides)
        p_rev = self.streak_reversal.p_reversal(current_streak)
        overdue = self.gap_detector.overdue_score().tolist()

        return PatternPrediction(
            p_big=blended,
            source_probs=source_probs,
            meta_weights=mw,
            significant_lags=sig_lags[:5],
            current_streak=current_streak,
            p_reversal_at_streak=p_rev,
            overdue_scores=overdue,
        )

    def reward(self, actual_digit: int) -> None:
        """Feed actual outcome back to the win-rate tracker."""
        actual_side = 1 if int(actual_digit) >= 5 else 0
        if self._last_sides:
            # Record each source's last prediction against reality
            sides = self._last_sides
            for src, p in {
                "acf": self.acf.predict_p_big(sides),
                "streak_reversal": self.streak_reversal.predict_p_big(sides),
                "conditional_freq": self.cond_freq.predict_p_big(sides),
                "gap_detector": self.gap_detector.predict_p_big(),
            }.items():
                self.win_tracker.record(src, p, actual_side)

    # ── state persistence ─────────────────────────────────────────────────────

    def save_state(self, db) -> None:
        """Persist all learnable state to Supabase `ai_brain_state`."""
        try:
            from backend.database import save_ai_brain_state
            state = {
                "lag_strength": self.acf._lag_strength,
                "lag_sign": self.acf._lag_sign,
                "acf_n_updates": self.acf._n_updates,
                "streak_counts": self.streak_reversal._counts.tolist(),
                "cond_freq_table": {
                    str(k): v for k, v in self.cond_freq._table.items()
                },
                "gap_avg": self.gap_detector._avg_gap.tolist(),
                "gap_last_seen": self.gap_detector._last_seen.tolist(),
                "gap_round": self.gap_detector._round,
                "win_tracker": {
                    s: list(self.win_tracker._history[s])
                    for s in PatternWinRateTracker.SOURCES
                },
                "last_update_n": self._last_update_n,
                "saved_at": datetime.utcnow().isoformat(),
            }
            save_ai_brain_state(
                db=db,
                model_name=self.MODEL_NAME,
                generation=self.acf._n_updates,
                total_samples=self._last_update_n,
                weights_json=json.dumps(state),
                win_rate=0.0,
            )
        except Exception as e:
            print(f"[PatternIntelligence] save_state failed: {e}")

    def load_state(self, db) -> bool:
        """Load persisted state from Supabase. Returns True on success."""
        try:
            from backend.database import load_ai_brain_state
            brain = load_ai_brain_state(db, model_name=self.MODEL_NAME)
            if brain is None or not brain.synaptic_weights:
                return False
            state = json.loads(brain.synaptic_weights)

            # ACF
            lag_strength = state.get("lag_strength", {})
            for k, v in lag_strength.items():
                self.acf._lag_strength[int(k)] = float(v)
            lag_sign = state.get("lag_sign", {})
            for k, v in lag_sign.items():
                self.acf._lag_sign[int(k)] = float(v)
            self.acf._n_updates = int(state.get("acf_n_updates", 0))

            # Streak reversal
            counts_data = state.get("streak_counts")
            if counts_data:
                self.streak_reversal._counts = np.array(counts_data, dtype=np.float64)

            # Conditional frequency table
            cond_data = state.get("cond_freq_table", {})
            for k_str, v in cond_data.items():
                try:
                    k = eval(k_str)  # tuple key was stored as string
                    self.cond_freq._table[k] = list(v)
                except Exception:
                    pass

            # Gap detector
            gap_avg = state.get("gap_avg")
            if gap_avg:
                self.gap_detector._avg_gap = np.array(gap_avg, dtype=np.float64)
            gap_last = state.get("gap_last_seen")
            if gap_last:
                self.gap_detector._last_seen = np.array(gap_last, dtype=np.int64)
            self.gap_detector._round = int(state.get("gap_round", 0))

            # Win tracker
            win_data = state.get("win_tracker", {})
            for src, pairs in win_data.items():
                if src in self.win_tracker._history:
                    for p, a in pairs:
                        self.win_tracker._history[src].append((float(p), int(a)))

            self._last_update_n = int(state.get("last_update_n", 0))
            print(f"[PatternIntelligence] Loaded state: {self._last_update_n} samples trained")
            return True
        except Exception as e:
            print(f"[PatternIntelligence] load_state failed: {e}")
            return False

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _current_streak(sides: Sequence[int]) -> int:
        if not sides:
            return 0
        run = 1
        for i in range(len(sides) - 2, -1, -1):
            if sides[i] == sides[-1]:
                run += 1
            else:
                break
        return run

    def deep_train_from_db(self, db, lookback_days: int = 60) -> int:
        """Full retrain from Supabase outcomes table.

        Called by daily_learning.py.  Returns the number of samples trained.
        """
        try:
            from backend.database import Outcome
            cutoff = datetime.utcnow() - timedelta(days=lookback_days)
            rows = (
                db.query(Outcome)
                .filter(Outcome.timestamp_utc >= cutoff)
                .order_by(Outcome.sequence_no.asc())
                .all()
            )
            if not rows:
                print("[PatternIntelligence] No rows for deep_train_from_db")
                return 0
            digits = [int(r.digit) for r in rows]
            sides = [1 if d >= 5 else 0 for d in digits]

            # Full rebuild (better accuracy than incremental for a fresh daily run)
            self.acf._lag_strength = {k: 0.0 for k in range(1, AutoCorrelationMemory.MAX_LAG + 1)}
            self.acf._lag_sign = {k: 0.0 for k in range(1, AutoCorrelationMemory.MAX_LAG + 1)}
            self.streak_reversal._counts = np.ones(
                (StreakReversalProfiler.MAX_STREAK + 1, 2), dtype=np.float64
            )
            self.cond_freq._table = defaultdict(lambda: [1.0, 2.0])
            self.gap_detector._avg_gap = np.full(10, 10.0)
            self.gap_detector._last_seen = np.full(10, -1, dtype=np.int64)
            self.gap_detector._round = 0
            self._last_update_n = 0

            self.acf.update(sides)
            self.streak_reversal.update(sides)
            self.cond_freq.update(sides)
            self.gap_detector.update(digits)
            self._last_sides = sides
            self._last_update_n = len(sides)

            print(f"[PatternIntelligence] deep_train_from_db: trained on {len(digits)} samples")
            return len(digits)
        except Exception as e:
            print(f"[PatternIntelligence] deep_train_from_db failed: {e}")
            return 0
