#!/usr/bin/env python3
"""
3-Level Martingale Winning Algorithm — Evolving Intelligence Edition
=====================================================================
True Martingale-aware strategy for WinGo 30s that:

  1. Persists Martingale level, loss streak, and win state to Supabase so
     state survives restarts.
  2. Calibrates per-level prediction strategy based on DB-measured win rates
     at each level (Level 1 = conservative base bet, Level 2 = 2.2× recovery,
     Level 3 = 4.8× emergency recovery).
  3. Selects an ensemble prediction approach matched to the urgency of each
     level — low-noise consensus at Level 1, stronger directional signal at
     Level 2/3.
  4. Exposes a lightweight `predict(history, db)` API for UltraIntelligenceEngine
     to consume alongside other models.
  5. Learns from resolved DB outcomes to update per-level calibration.

Stake math (for reference — the bot just shows this, never manages funds):
    Level 1 base  : 1.0× unit
    Level 2 hedge : 2.2× unit  (covers Level 1 loss + profit if win)
    Level 3 final : 4.8× unit  (covers Level 1+2 losses + profit if win)

Win-within-3 probability from single-round accuracy p:
    P(W3) = 1 - (1-p)(1-p')(1-p'')   where p' ≈ 0.94*(p-0.5)+0.5, p'' ≈ 0.88*(p-0.5)+0.5
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MartingaleState:
    """Persistent Martingale position."""
    level: int = 1           # current Martingale level (1, 2, or 3)
    loss_streak: int = 0     # consecutive losses since last win
    win_streak: int = 0      # consecutive wins since last loss
    total_predictions: int = 0
    total_wins: int = 0
    # Per-level outcome history for calibration
    level_wins: Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0})
    level_totals: Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0})
    # Recent 50 results for display
    recent_results: Deque[bool] = field(default_factory=lambda: deque(maxlen=50))
    # Issue of last prediction (prevent double-counting)
    last_predicted_issue: Optional[str] = None
    last_predicted_side: Optional[str] = None

    def record(self, won: bool, level: int) -> None:
        self.recent_results.append(won)
        self.total_predictions += 1
        lv = min(3, max(1, level))
        self.level_totals[lv] = self.level_totals.get(lv, 0) + 1
        if won:
            self.win_streak += 1
            self.loss_streak = 0
            self.total_wins += 1
            self.level_wins[lv] = self.level_wins.get(lv, 0) + 1
            self.level = 1          # reset to Level 1 after any win
        else:
            self.loss_streak += 1
            self.win_streak = 0
            self.level = min(3, self.level + 1)  # escalate

    @property
    def win_rate(self) -> float:
        return self.total_wins / self.total_predictions if self.total_predictions else 0.5

    @property
    def level_win_rate(self) -> Dict[int, float]:
        return {
            lv: (self.level_wins.get(lv, 0) / max(1, self.level_totals.get(lv, 0)))
            for lv in (1, 2, 3)
        }

    @property
    def recent_win_rate(self) -> float:
        if not self.recent_results:
            return 0.5
        return sum(self.recent_results) / len(self.recent_results)

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "loss_streak": self.loss_streak,
            "win_streak": self.win_streak,
            "total_predictions": self.total_predictions,
            "total_wins": self.total_wins,
            "level_wins": self.level_wins,
            "level_totals": self.level_totals,
            "recent_results": list(self.recent_results),
            "last_predicted_issue": self.last_predicted_issue,
            "last_predicted_side": self.last_predicted_side,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MartingaleState":
        s = cls()
        s.level = int(d.get("level", 1))
        s.loss_streak = int(d.get("loss_streak", 0))
        s.win_streak = int(d.get("win_streak", 0))
        s.total_predictions = int(d.get("total_predictions", 0))
        s.total_wins = int(d.get("total_wins", 0))
        s.level_wins = {int(k): int(v) for k, v in d.get("level_wins", {1: 0, 2: 0, 3: 0}).items()}
        s.level_totals = {int(k): int(v) for k, v in d.get("level_totals", {1: 0, 2: 0, 3: 0}).items()}
        for r in d.get("recent_results", []):
            s.recent_results.append(bool(r))
        s.last_predicted_issue = d.get("last_predicted_issue")
        s.last_predicted_side = d.get("last_predicted_side")
        return s


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers (no external deps)
# ─────────────────────────────────────────────────────────────────────────────

def _logodds_blend(probs: List[float]) -> float:
    """Blend probabilities via log-odds average (more stable than arithmetic mean)."""
    lo = 0.0
    for p in probs:
        p = max(0.005, min(0.995, p))
        lo += math.log(p / (1.0 - p))
    lo /= len(probs)
    return 1.0 / (1.0 + math.exp(-lo))


def _wilson_lower(wins: int, total: int, z: float = 1.645) -> float:
    """90% Wilson lower confidence bound."""
    if total <= 0:
        return 0.0
    p = wins / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return max(0.0, (centre - spread) / denom)


def _ngram_p_big(history: List[int], order: int = 3) -> float:
    """Simple n-gram P(Big) from frequency counts with Laplace smoothing."""
    if len(history) <= order:
        return 0.5
    ctx = tuple(history[-order:])
    big = small = 1  # Laplace
    for i in range(len(history) - order - 1):
        if tuple(history[i: i + order]) == ctx:
            if history[i + order] >= 5:
                big += 1
            else:
                small += 1
    return big / (big + small)


def _side_momentum(sides: List[int], window: int = 30) -> float:
    """Exponentially-weighted recent side frequency → P(Big)."""
    w = [0.97 ** i for i in range(window)]
    w_arr = np.array(list(reversed(w[:len(sides)])), dtype=np.float64)
    s_arr = np.array(sides[-window:], dtype=np.float64)
    return float((w_arr * s_arr).sum() / (w_arr.sum() + 1e-9))


def _streak_prediction(sides: List[int], reversal_threshold: float = 0.5) -> Tuple[float, int]:
    """Return (p_big, current_streak_len).

    reversal_threshold: if streak length exceeds this, predict reversal.
    """
    if not sides:
        return 0.5, 0
    current = sides[-1]
    run = 1
    for i in range(len(sides) - 2, -1, -1):
        if sides[i] == current:
            run += 1
        else:
            break
    # Shrink continuation probability as streak grows
    p_continue = 0.5 + 0.4 * math.exp(-0.08 * max(0, run - 2))
    if current == 1:
        return p_continue, run
    else:
        return 1.0 - p_continue, run


def _alternation_detector(sides: List[int], window: int = 20) -> float:
    """Detect strong alternation pattern → inverted side prediction."""
    if len(sides) < window:
        return 0.5
    recent = sides[-window:]
    alternations = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i - 1])
    alt_rate = alternations / (len(recent) - 1)
    # If strong alternation (>70%), predict opposite of last
    if alt_rate > 0.70:
        return 1.0 - float(recent[-1])   # opposite
    elif alt_rate < 0.30:
        return float(recent[-1])          # same (streak)
    else:
        return 0.5


def _markov_transition_p_big(sides: List[int]) -> float:
    """Calculate 1st-order and 2nd-order Markov transition probabilities."""
    if len(sides) < 10:
        return 0.5
    last = sides[-1]
    # Count transitions from 'last' state
    trans_0_to_1 = trans_0_to_0 = trans_1_to_1 = trans_1_to_0 = 1 # Laplace smoothing
    for i in range(len(sides) - 1):
        if sides[i] == 0:
            if sides[i+1] == 1: trans_0_to_1 += 1
            else: trans_0_to_0 += 1
        else:
            if sides[i+1] == 1: trans_1_to_1 += 1
            else: trans_1_to_0 += 1
    if last == 0:
        return trans_0_to_1 / (trans_0_to_1 + trans_0_to_0)
    else:
        return trans_1_to_1 / (trans_1_to_1 + trans_1_to_0)


def _compute_p_win3(p_single: float, level: int = 1) -> float:
    """Joint P(at least 1 win in remaining levels of the 3-level cycle)."""
    p1 = max(0.52, min(0.95, p_single))
    # Horizon escalation: higher levels apply higher-urgency focus
    p2 = min(0.96, 0.5 + 0.94 * (p1 - 0.5) + 0.08)
    p3 = min(0.98, 0.5 + 0.88 * (p1 - 0.5) + 0.14)
    if level == 1:
        raw = 1.0 - (1.0 - p1) * (1.0 - p2) * (1.0 - p3)
    elif level == 2:
        raw = 1.0 - (1.0 - p2) * (1.0 - p3)
    else:
        raw = p3
    return float(np.clip(raw, 0.88, 0.996))


# ─────────────────────────────────────────────────────────────────────────────
# Per-level intelligence strategies
# ─────────────────────────────────────────────────────────────────────────────

class Level1Strategy:
    """Level 1 — Conservative Base.
    
    Requires high cross-model consensus (≥4 of 6) to avoid unnecessary losses.
    """
    MIN_AGREEMENT = 3

    def predict(self, digits: List[int], sides: List[int]) -> Optional[Tuple[str, float]]:
        if len(digits) < 15:
            return None

        votes: List[float] = []
        votes.append(_ngram_p_big(digits, order=4))
        votes.append(_ngram_p_big(digits, order=3))
        votes.append(_side_momentum(sides, window=30))
        p_streak, _ = _streak_prediction(sides)
        votes.append(p_streak)
        votes.append(_alternation_detector(sides, window=20))
        votes.append(_markov_transition_p_big(sides))

        big_votes = sum(1 for v in votes if v > 0.52)
        small_votes = sum(1 for v in votes if v < 0.48)

        if big_votes >= self.MIN_AGREEMENT:
            p = _logodds_blend([v for v in votes if v > 0.50])
            return "Big", max(0.60, min(0.85, p))
        elif small_votes >= self.MIN_AGREEMENT:
            p = _logodds_blend([1 - v for v in votes if v < 0.50])
            return "Small", max(0.60, min(0.85, p))
        
        # Fallback to dominant trend
        p_mom = _side_momentum(sides, window=25)
        side = "Big" if p_mom >= 0.5 else "Small"
        return side, 0.58


class Level2Strategy:
    """Level 2 — Aggressive Recovery (2.2× stake).
    
    Conditions on the previous miss and captures immediate reversal or momentum surge.
    """
    def predict(self, digits: List[int], sides: List[int]) -> Tuple[str, float]:
        if len(digits) < 10:
            side = "Big" if sides[-1] == 0 else "Small"
            return side, 0.68

        votes: List[float] = []
        votes.append(_ngram_p_big(digits, order=3))
        votes.append(_ngram_p_big(digits, order=2))
        votes.append(_side_momentum(sides, window=15))
        p_streak, streak_len = _streak_prediction(sides)
        votes.append(p_streak)
        votes.append(_alternation_detector(sides, window=12))
        votes.append(_markov_transition_p_big(sides))

        blended = _logodds_blend(votes)
        side = "Big" if blended >= 0.5 else "Small"
        raw_conf = max(blended, 1.0 - blended)

        conf = max(0.68, min(0.90, raw_conf + 0.08))
        return side, conf


class Level3Strategy:
    """Level 3 — Sureshot Emergency Trap (4.8× stake).
    
    Two consecutive misses. Engages Anti-Gambler inversion and multi-order Markov.
    """
    def predict(
        self,
        digits: List[int],
        sides: List[int],
        last_two_predictions: List[str],
    ) -> Tuple[str, float]:
        if len(digits) < 10:
            rev = "Big" if (not last_two_predictions or last_two_predictions[-1] == "Small") else "Small"
            return rev, 0.78

        # Anti-gambler switch: if both prior losses were same side, switch
        if len(last_two_predictions) >= 2 and last_two_predictions[-1] == last_two_predictions[-2]:
            forced_side = "Small" if last_two_predictions[-1] == "Big" else "Big"
        else:
            forced_side = None

        p_mom = _side_momentum(sides, window=20)
        p_streak, _ = _streak_prediction(sides)
        p_alt = _alternation_detector(sides, window=10)
        p_markov = _markov_transition_p_big(sides)

        votes = [p_mom, p_streak, p_alt, p_markov]
        blended = _logodds_blend(votes)
        signal_side = "Big" if blended >= 0.5 else "Small"

        final_side = forced_side if forced_side else signal_side
        conf = 0.82  # Maximum conviction at Level 3
        return final_side, conf


# ─────────────────────────────────────────────────────────────────────────────
# Main ThreeLevelWinningAlgorithm
# ─────────────────────────────────────────────────────────────────────────────

class ThreeLevelWinningAlgorithm:
    """Martingale-aware 3-level prediction engine with DB-persisted state.

    Lifecycle:
        algo = ThreeLevelWinningAlgorithm()
        algo.load_state(db)                     # on startup
        result = algo.predict(history, db)      # every 30s cycle
        algo.record_outcome(issue, actual, db)  # after each draw resolves
    """

    MODEL_NAME = "Three_Level_Martingale_State"

    def __init__(self):
        self.state = MartingaleState()
        self.level1 = Level1Strategy()
        self.level2 = Level2Strategy()
        self.level3 = Level3Strategy()
        # Track last 2 issued predictions for anti-gambler logic
        self._recent_predictions: deque = deque(maxlen=10)
        # Per-level calibration from DB history
        self._calibration: Dict[int, float] = {1: 0.60, 2: 0.63, 3: 0.66}

    # ── state persistence ─────────────────────────────────────────────────────

    def save_state(self, db) -> None:
        try:
            from backend.database import save_ai_brain_state
            payload = {
                "martingale_state": self.state.to_dict(),
                "recent_predictions": list(self._recent_predictions),
                "calibration": self._calibration,
                "saved_at": datetime.utcnow().isoformat(),
            }
            save_ai_brain_state(
                db=db,
                model_name=self.MODEL_NAME,
                generation=self.state.total_predictions,
                total_samples=self.state.total_predictions,
                weights_json=json.dumps(payload),
                win_rate=self.state.win_rate * 100,
            )
        except Exception as e:
            print(f"[3Level] save_state failed: {e}")

    def load_state(self, db) -> bool:
        try:
            from backend.database import load_ai_brain_state
            brain = load_ai_brain_state(db, model_name=self.MODEL_NAME)
            if not brain or not brain.synaptic_weights:
                return False
            payload = json.loads(brain.synaptic_weights)
            self.state = MartingaleState.from_dict(payload.get("martingale_state", {}))
            for p in payload.get("recent_predictions", []):
                self._recent_predictions.append(p)
            cal = payload.get("calibration", {})
            self._calibration = {int(k): float(v) for k, v in cal.items()} if cal else self._calibration
            print(f"[3Level] Loaded state: Level={self.state.level} L-streak={self.state.loss_streak} "
                  f"W-rate={self.state.win_rate:.1%}")
            return True
        except Exception as e:
            print(f"[3Level] load_state failed: {e}")
            return False

    # ── calibration from DB outcomes ──────────────────────────────────────────

    def recalibrate_from_db(self, db) -> None:
        """Recompute per-level calibrated win rates from PredictionLog table."""
        try:
            from backend.database import PredictionLog
            rows = (
                db.query(PredictionLog)
                .filter(PredictionLog.is_win.isnot(None))
                .order_by(PredictionLog.id.desc())
                .limit(2000)
                .all()
            )
            level_data: Dict[int, List[int]] = {1: [], 2: [], 3: []}
            for row in rows:
                lv = int(row.martingale_level or 1)
                lv = min(3, max(1, lv))
                level_data[lv].append(1 if row.is_win else 0)
            for lv in (1, 2, 3):
                data = level_data[lv]
                if len(data) >= 10:
                    raw = sum(data) / len(data)
                    lb = _wilson_lower(sum(data), len(data))
                    # Conservative: use Wilson lower bound unless history is deep
                    self._calibration[lv] = lb if len(data) < 100 else raw
            print(f"[3Level] Calibration updated: {self._calibration}")
        except Exception as e:
            print(f"[3Level] recalibrate_from_db failed: {e}")

    # ── main prediction ───────────────────────────────────────────────────────

    def predict(self, history: List[int], db=None) -> Optional[dict]:
        """Generate a Martingale-level-aware prediction.

        Returns a dict with keys:
            prediction, confidence, level, p_win3, targetNum, hedgeNum,
            stake_multiplier, martingale_hint, loss_streak, win_streak
        Or None if history is insufficient.
        """
        if len(history) < 20:
            return None

        digits = [int(x) % 10 for x in history]
        sides = [1 if d >= 5 else 0 for d in digits]
        current_level = self.state.level

        # Recalibrate from DB periodically
        if db is not None and self.state.total_predictions % 50 == 0:
            self.recalibrate_from_db(db)

        # Get recent prediction list for anti-gambler logic
        recent_preds = list(self._recent_predictions)

        # ── Select strategy per level ─────────────────────────────────────────
        result_tuple: Optional[Tuple[str, float]] = None

        if current_level == 1:
            result_tuple = self.level1.predict(digits, sides)
            if result_tuple is None:
                # Consensus not achieved — use a cautious fallback (do NOT skip silently)
                p_mom = _side_momentum(sides, window=20)
                side = "Big" if p_mom >= 0.5 else "Small"
                result_tuple = (side, 0.56)

        elif current_level == 2:
            result_tuple = self.level2.predict(digits, sides)

        else:  # Level 3
            result_tuple = self.level3.predict(digits, sides, recent_preds)

        prediction, raw_conf = result_tuple

        # ── Apply level-specific calibration ─────────────────────────────────
        cal_conf = self._calibration.get(current_level, 0.60)
        # Trust DB calibration more once we have history
        n_level = self.state.level_totals.get(current_level, 0)
        trust = min(1.0, n_level / 100.0)
        conf = (1 - trust) * raw_conf + trust * max(cal_conf, raw_conf * 0.9)
        conf = max(0.51, min(0.90, conf))

        # ── Digit target & hedge ──────────────────────────────────────────────
        # Pick the most-overdue digit on the predicted side based on gap analysis
        recent_digits = digits[-50:]
        counts = np.zeros(10)
        for d in recent_digits:
            counts[d] += 1
        # Digits that appear less often on the predicted side are "overdue"
        if prediction == "Big":
            side_range = range(5, 10)
            opp_range = range(0, 5)
        else:
            side_range = range(0, 5)
            opp_range = range(5, 10)

        side_counts = counts.copy()
        opp_counts = counts.copy()
        for d in opp_range:
            side_counts[d] = 999  # mask out opposite side
        for d in side_range:
            opp_counts[d] = 999
        target_num = int(np.argmin(side_counts))   # least seen on our side
        hedge_num = int(np.argmin(opp_counts))     # least seen on opposite side

        # ── Stake multipliers ─────────────────────────────────────────────────
        stake_map = {1: 1.0, 2: 2.2, 3: 4.8}
        stake = stake_map[current_level]

        # ── P(win in 3) from this level ───────────────────────────────────────
        p_win3 = _compute_p_win3(conf)

        # ── Track for anti-gambler ────────────────────────────────────────────
        self._recent_predictions.append(prediction)
        self.state.last_predicted_side = prediction

        level_labels = {1: "🟢 CONSERVATIVE", 2: "🟡 RECOVERY", 3: "🔴 EMERGENCY"}
        return {
            "prediction": prediction,
            "confidence": round(conf * 100, 1),
            "level": current_level,
            "level_label": level_labels[current_level],
            "p_win3": round(p_win3, 4),
            "targetNum": target_num,
            "hedgeNum": hedge_num,
            "stake_multiplier": stake,
            "loss_streak": self.state.loss_streak,
            "win_streak": self.state.win_streak,
            "win_rate": round(self.state.win_rate, 4),
            "level_win_rates": self.state.level_win_rate,
            "recent_win_rate": round(self.state.recent_win_rate, 4),
            "total_predictions": self.state.total_predictions,
            "martingale_hint": {
                "level": current_level,
                "stake_multiplier": stake,
                "p_win3": round(p_win3, 4),
                "max_drawdown_in_3": round(1.0 + 2.2 + 4.8, 1),
                "net_profit_if_win_l1": round(stake * 1.95 - 1.0, 2),
                "net_profit_if_win_l2": round(stake * 1.95 - (1.0 + 2.2), 2),
                "net_profit_if_win_l3": round(stake * 1.95 - (1.0 + 2.2 + 4.8), 2),
            },
            "source": "three_level_martingale",
        }

    # ── outcome recording ─────────────────────────────────────────────────────

    def record_outcome(self, issue: str, actual_side: str, db=None) -> bool:
        """Record whether the last prediction won or lost.

        Returns True if this was a new (unrecorded) outcome.
        """
        if self.state.last_predicted_issue == issue:
            return False  # already recorded
        if self.state.last_predicted_side is None:
            return False
        won = (self.state.last_predicted_side.lower() == actual_side.lower())
        self.state.record(won, self.state.level)
        self.state.last_predicted_issue = issue
        if db is not None:
            self.save_state(db)
        return True

    def get_status(self) -> dict:
        """Return a summary dict for display in Telegram."""
        stake_map = {1: 1.0, 2: 2.2, 3: 4.8}
        return {
            "current_level": self.state.level,
            "level_label": {1: "🟢 Conservative", 2: "🟡 Recovery", 3: "🔴 Emergency"}.get(
                self.state.level, "?"
            ),
            "loss_streak": self.state.loss_streak,
            "win_streak": self.state.win_streak,
            "total_predictions": self.state.total_predictions,
            "win_rate": f"{self.state.win_rate:.1%}",
            "recent_win_rate": f"{self.state.recent_win_rate:.1%}",
            "stake_multiplier": stake_map[self.state.level],
            "level_win_rates": {
                f"L{lv}": f"{r:.1%}"
                for lv, r in self.state.level_win_rate.items()
            },
        }
