#!/usr/bin/env python3
"""
Telegram Bot for WinGo Predictions — Ultra Intelligence v2.0
=============================================================
Upgraded with:
  - 3-Level Martingale status panel (current level, stake, per-level win rates)
  - Daily Learning status (/learn command and dashboard tab)
  - Improved prediction card: 12-model consensus bar, win-probability curve
  - Auto-recovery hints when loss streak ≥ 2
  - Pattern intelligence diagnostics display
  - Model consensus percentage display
  - All new result dict fields surfaced cleanly
"""

import asyncio
from html import escape
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Set

import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

# ── Project path ──────────────────────────────────────────────────────────────
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(project_dir, ".env"))
    load_dotenv(os.path.join(project_dir, "backend", ".env"))
except ImportError:
    pass

from backend.database import SessionLocal, AIBrainState, PredictionAudit, Draw
from backend.telegram_card import (
    render_forecast_card,
    render_metrics_card,
    render_status_card,
    render_martingale_card,
)

# ── Config ────────────────────────────────────────────────────────────────────
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
API_URL = os.environ.get("PREDICTION_API_URL", "http://localhost:8000/api/state")
CHECK_INTERVAL = 3.0
NOTIFICATION_COOLDOWN = 15
MAX_NOTIFICATIONS_PER_HOUR = 20

# ── In-memory state ───────────────────────────────────────────────────────────
user_filters: Dict[int, str] = {}       # user_id -> "all" | "high" | "strike"
subscribed_users: Set[int] = set()
last_prediction_issue = None
last_prediction = None
last_notification_time: Dict[int, float] = {}
notification_count: Dict[int, int] = {}

logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("TELEGRAM_BOT")


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _text(value, fallback="—") -> str:
    if value is None or value == "":
        return fallback
    return escape(str(value))


def _pct(value, fallback="—") -> str:
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return fallback


def _bar(ratio: float, width: int = 12) -> str:
    """ASCII progress bar for probability display."""
    filled = round(max(0.0, min(1.0, ratio)) * width)
    return "█" * filled + "░" * (width - filled)


def _win_prob_curve(p1: float) -> str:
    """Compact textual win-probability curve for L1/L2/L3."""
    p2 = 0.5 + 0.94 * (p1 - 0.5)
    p3 = 0.5 + 0.88 * (p1 - 0.5)
    j2 = 1.0 - (1 - p1) * (1 - p2)
    j3 = 1.0 - (1 - p1) * (1 - p2) * (1 - p3)
    return (
        f"L1 {_bar(p1)} {p1*100:.0f}%\n"
        f"L2 {_bar(j2)} {j2*100:.0f}%\n"
        f"L3 {_bar(j3)} {j3*100:.0f}%"
    )


def _recovery_hint(loss_streak: int, prediction: str) -> str:
    """Generate an auto-recovery hint when on a losing streak."""
    if loss_streak <= 0:
        return ""
    if loss_streak == 1:
        return f"\n⚠️ <b>Recovery L2:</b> Next bet is 2.2× — stay disciplined."
    if loss_streak == 2:
        return (
            f"\n🚨 <b>Emergency L3 active:</b> 4.8× stake. "
            f"Anti-gambler engaged — going <b>{prediction}</b>."
        )
    return (
        f"\n🛑 <b>{loss_streak} consecutive losses.</b> "
        f"Consider pausing if budget limit reached."
    )


async def get_prediction() -> Optional[dict]:
    """Fetch the latest prediction blob from the FastAPI gateway."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
            r = await client.get(API_URL)
            if r.status_code != 200:
                return None
            data = r.json()
            return data if isinstance(data, dict) else None
    except Exception as e:
        logger.warning("Prediction API error: %s", e)
        return None


def get_metrics_from_db() -> Optional[dict]:
    """Pull resolved accuracy metrics directly from PredictionAudit table."""
    try:
        db = SessionLocal()
        rows = (
            db.query(PredictionAudit)
            .filter(PredictionAudit.actual_size.isnot(None))
            .order_by(PredictionAudit.id.desc())
            .limit(2000)
            .all()
        )
        db.close()
        if not rows:
            return None
        n = len(rows)
        correct = sum(1 for r in rows if r.is_correct)
        brier = sum(r.brier_score or 0.25 for r in rows) / n
        loss = sum(r.log_loss or 0.693 for r in rows) / n
        return {
            "resolved_predictions": n,
            "directional_accuracy": correct / n,
            "brier_score": brier,
            "log_loss": loss,
        }
    except Exception as e:
        logger.warning("DB metrics query failed: %s", e)
        return None


async def check_win_loss(prev: dict) -> Optional[dict]:
    """Check whether the previous prediction won by querying the Draw table."""
    try:
        issue = prev.get("nextIssue") or prev.get("currentIssue")
        if not issue:
            return None
        db = SessionLocal()
        draw = db.query(Draw).filter(Draw.issue_number == str(issue)).first()
        db.close()
        if not draw:
            return None
        actual = "Big" if int(draw.number) >= 5 else "Small"
        return {
            "won": actual == prev.get("prediction"),
            "issue": issue,
            "actual": actual,
            "number": draw.number,
        }
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Keyboards
# ═════════════════════════════════════════════════════════════════════════════

def main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✨ Live Forecast", callback_data="forecast"),
            InlineKeyboardButton("📊 Performance",   callback_data="metrics"),
        ],
        [
            InlineKeyboardButton("🎯 3-Level Status", callback_data="level_status"),
            InlineKeyboardButton("🏆 Scorecard",      callback_data="scorecard"),
        ],
        [
            InlineKeyboardButton("💎 Strategy",       callback_data="martingale"),
            InlineKeyboardButton("🧬 Models",         callback_data="models"),
        ],
        [
            InlineKeyboardButton("🎓 Daily Learning", callback_data="learn"),
            InlineKeyboardButton("⚙️ Settings",       callback_data="filter_menu"),
        ],
        [
            InlineKeyboardButton("🟢 Status", callback_data="status"),
            InlineKeyboardButton("🔔 Updates", callback_data="toggle_sub"),
        ],
    ])


def back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("← Dashboard", callback_data="forecast")]
    ])


def filter_keyboard(current_filter: str = "all") -> InlineKeyboardMarkup:
    marks = {k: "✅ " for k in ("all", "high", "strike")}
    marks = {k: ("✅ " if k == current_filter else "") for k in marks}
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"{marks['all']}All Rounds (Recommended)",         callback_data="set_filter_all")],
        [InlineKeyboardButton(f"{marks['high']}High Conviction Only (≥70%)",     callback_data="set_filter_high")],
        [InlineKeyboardButton(f"{marks['strike']}Strike Only (Beast/Ultimate)",  callback_data="set_filter_strike")],
        [InlineKeyboardButton("← Dashboard", callback_data="forecast")],
    ])


# ═════════════════════════════════════════════════════════════════════════════
# Message Formatters
# ═════════════════════════════════════════════════════════════════════════════

def format_prediction_message(data: Optional[dict], previous_result: Optional[dict] = None) -> str:
    """Main forecast card — text fallback when PIL card unavailable."""
    if not data:
        return "<b>✨ EVOSEQ Ultra v2</b>\n\n⏳ Intelligence service connecting..."

    prediction   = _text(data.get("prediction"))
    confidence   = _pct(data.get("confidence"))
    target_num   = _text(data.get("targetNum"))
    hedge_num    = _text(data.get("hedgeNum"))
    next_issue   = _text(data.get("nextIssue"))
    strike       = _text(data.get("strikeQuality", "CONSERVATIVE")).replace("_", " ")
    action       = str(data.get("action", "FORECAST"))
    p_win3       = data.get("calibratedPWinIn3")
    p_single     = float(data.get("calibratedPSingle", 0.55))
    consensus    = data.get("modelConsensus", 0.5)
    ml_level     = int(data.get("martingaleLevel", 1))
    ml_label     = _text(data.get("martingaleLevelLabel", "🟢 CONSERVATIVE"))
    ml_stake     = data.get("martingaleStake", 1.0)
    loss_streak  = int(data.get("martingaleLossStreak", 0))
    reject_iid   = data.get("rejectIID", False)
    exploit_sc   = data.get("exploitScore", 0.0)

    side_emoji = "🔵" if prediction.lower() == "big" else "🟡"

    action_line = {
        "SKIP":     "⏭️ <b>NO EDGE — SKIP THIS ROUND</b>",
        "CAUTION":  f"⚠️ <b>LOW CONFIDENCE</b> | {strike}",
        "STRIKE":   f"⚡ <b>STRIKE</b> | {strike}",
        "FORECAST": f"📊 <b>FORECAST</b> | {strike}",
    }.get(action, f"📊 {strike}")

    p3_str = f"{float(p_win3)*100:.1f}%" if p_win3 is not None else "—"

    # Consensus bar
    consensus_bar = _bar(consensus)
    consensus_str = f"{consensus*100:.0f}%"

    # Win-prob curve
    curve = _win_prob_curve(p_single)

    # Result line
    result_line = ""
    if previous_result:
        emoji = "✅" if previous_result["won"] else "❌"
        result_line = (
            f"\n{emoji} <b>Last Result:</b> "
            f"{'WON' if previous_result['won'] else 'LOST'} "
            f"(#{_text(previous_result.get('issue', ''))[-6:]} → {_text(previous_result.get('actual'))})"
        )

    # Recovery hint
    recovery = _recovery_hint(loss_streak, prediction)

    # Exploit signal
    iid_tag = "🔬 <b>PRNG EXPLOIT DETECTED</b>" if reject_iid else "🎲 Monitoring..."

    return f"""
<b>✨ EVOSEQ Ultra v2.0</b>
{action_line}
━━━━━━━━━━━━━━━━━━━━━━
{side_emoji} <b>{prediction.upper()}</b>  |  Confidence: <b>{confidence}</b>
🎯 Target: <b>{target_num}</b>  •  Hedge: <b>{hedge_num}</b>
📈 P(win in 3): <b>{p3_str}</b>
🔢 Round: <code>#{next_issue[-8:] if len(next_issue) >= 8 else next_issue}</code>

<b>Win Probability Curve:</b>
<code>{curve}</code>

<b>12-Model Consensus:</b> {consensus_bar} {consensus_str}
{iid_tag}

<b>🎰 Martingale:</b> {ml_label}  |  Stake: {ml_stake}×{recovery}{result_line}

<i>Ultra Intelligence v2.0 — 12-model adaptive ensemble</i>
""".strip()


def format_forecast_caption(data: dict, previous_result=None) -> str:
    """Short caption for photo cards."""
    action   = str(data.get("action", "FORECAST"))
    strike   = _text(data.get("strikeQuality", "CONSERVATIVE")).replace("_", " ")
    p3       = data.get("calibratedPWinIn3")
    p3_txt   = f"{float(p3)*100:.1f}%" if p3 else "—"
    ml_level = int(data.get("martingaleLevel", 1))
    ml_stake = data.get("martingaleStake", 1.0)

    if action == "SKIP":
        header = "⏭️ <b>NO EDGE — SKIP</b>"
    elif action == "STRIKE":
        header = f"⚡ <b>STRIKE • {strike.upper()}</b>"
    else:
        header = f"◈ <b>QUANT CALL • {strike.upper()}</b>"

    caption = (
        f"{header}\n"
        f"Target: <code>#{_text(data.get('nextIssue'))}</code>  •  "
        f"<b>{_text(data.get('prediction')).upper()}</b> ({_pct(data.get('confidence'))})\n"
        f"Nums: <b>{_text(data.get('targetNum'))}</b>/{_text(data.get('hedgeNum'))}  •  "
        f"P(win3): <b>{p3_txt}</b>  •  L{ml_level} @ {ml_stake}×"
    )
    if previous_result:
        out = "✅ WON" if previous_result.get("won") else "❌ LOST"
        caption += f"\nLast: <b>{out}</b> (#{_text(previous_result.get('issue', ''))[-6:]})"
    return caption


async def reply_with_forecast(message, data: Optional[dict], previous_result=None):
    """Deliver visual card first, fall back to text."""
    if data:
        try:
            card = render_forecast_card(data, previous_result)
            if card:
                return await message.reply_photo(
                    photo=card,
                    caption=format_forecast_caption(data, previous_result),
                    parse_mode="HTML",
                    reply_markup=main_keyboard(),
                )
        except Exception as e:
            logger.warning("Card render failed: %s", e)
    return await message.reply_text(
        format_prediction_message(data, previous_result),
        parse_mode="HTML",
        reply_markup=main_keyboard(),
    )


def format_level_status_message(data: Optional[dict]) -> str:
    """3-Level Martingale status panel."""
    if not data:
        return "<b>🎯 3-Level Status</b>\n\n⏳ Loading..."

    ml_level     = int(data.get("martingaleLevel", 1))
    ml_label     = _text(data.get("martingaleLevelLabel", "🟢 CONSERVATIVE"))
    ml_stake     = data.get("martingaleStake", 1.0)
    loss_streak  = int(data.get("martingaleLossStreak", 0))
    lv_rates     = data.get("martingaleLevelWinRates", {})
    p_win3       = data.get("calibratedPWinIn3", 0.875)
    p_single     = float(data.get("calibratedPSingle", 0.55))
    hint         = data.get("martingale3Hint", {})

    # Per-level win rate bars
    def lv_line(lv: int) -> str:
        rate = float(lv_rates.get(lv, 0.0))
        return f"L{lv}: {_bar(rate, 10)} {rate*100:.1f}%"

    lv_bars = "\n".join(lv_line(lv) for lv in (1, 2, 3))

    # Recovery hint
    recovery = _recovery_hint(loss_streak, _text(data.get("prediction", "Big")))

    # Stake table
    stakes = {1: 1.0, 2: 2.2, 3: 4.8}
    stake_table = "\n".join(
        f"{'→' if lv == ml_level else ' '} L{lv}: <b>{stakes[lv]}×</b>"
        + (" ← <i>ACTIVE</i>" if lv == ml_level else "")
        for lv in (1, 2, 3)
    )

    # P(win) curve
    curve = _win_prob_curve(p_single)

    return f"""
<b>🎯 3-LEVEL MARTINGALE STATUS</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<b>Current Level:</b> {ml_label}
<b>Loss Streak:</b> {loss_streak}  |  <b>P(win in 3):</b> {float(p_win3)*100:.1f}%

<b>Stake Progression:</b>
<code>{stake_table}</code>

<b>Win Probability Curve:</b>
<code>{curve}</code>

<b>Per-Level Historical Win Rates:</b>
<code>{lv_bars}</code>
{recovery}

<b>Rules:</b>
• <b>Win</b> → immediately reset to Level 1
• <b>Never exceed Level 3</b> (max 3-step discipline)
• Max drawdown per cycle: <b>1 + 2.2 + 4.8 = 8×</b>
""".strip()


def format_martingale_message(data: Optional[dict]) -> str:
    """Full Martingale strategy explanation with live calibration."""
    p_single = float(data.get("calibratedPSingle", 0.58)) if data else 0.58
    ml_level = int(data.get("martingaleLevel", 1)) if data else 1
    ml_label = _text(data.get("martingaleLevelLabel", "🟢 CONSERVATIVE")) if data else "🟢 CONSERVATIVE"
    loss_streak = int(data.get("martingaleLossStreak", 0)) if data else 0

    p1 = p_single
    p2 = 0.5 + 0.94 * (p1 - 0.5)
    p3 = 0.5 + 0.88 * (p1 - 0.5)
    j2 = 1.0 - (1 - p1) * (1 - p2)
    j3 = 1.0 - (1 - p1) * (1 - p2) * (1 - p3)

    recovery = _recovery_hint(loss_streak, _text(data.get("prediction", "Big")) if data else "Big")

    return f"""
<b>💎 3-LEVEL MARTINGALE LADDER</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<b>Current Position:</b> {ml_label}  (Loss streak: {loss_streak}){recovery}

<b>STEP 1 — Initial Entry</b>
• Allocation: <b>1.0× Base Unit</b>
• Win Probability: <b>{p1*100:.1f}%</b>
• Bar: <code>{_bar(p1)}</code>

<b>STEP 2 — Recovery Strike</b>
• Allocation: <b>2.2× Base Unit</b>
• Cumulative P(win): <b>{j2*100:.1f}%</b>
• Bar: <code>{_bar(j2)}</code>

<b>STEP 3 — Max Conviction Strike</b>
• Allocation: <b>4.8× Base Unit</b>
• Cumulative P(win): <b>{j3*100:.1f}%</b>
• Bar: <code>{_bar(j3)}</code>

<b>🛡️ RULES:</b>
1. <b>Reset to Step 1</b> immediately upon ANY win
2. <b>Never extend to Step 4</b> — 3-step maximum
3. Max drawdown = 8× base (1 + 2.2 + 4.8)
""".strip()


def format_scorecard_message(data: Optional[dict]) -> str:
    scorecard    = data.get("scorecard", {}) if data else {}
    win_streak   = scorecard.get("win_streak", 0)
    loss_streak  = scorecard.get("loss_streak", 0)
    session_rate = scorecard.get("session_win_rate", 50.0)
    total_wins   = scorecard.get("total_wins", 0)
    total_losses = scorecard.get("total_losses", 0)
    recent_20    = scorecard.get("recent_20", "")
    ml_level     = int(data.get("martingaleLevel", 1)) if data else 1
    consensus    = float(data.get("modelConsensus", 0.5)) if data else 0.5

    if win_streak > 0:
        streak_str = f"🔥 <b>{win_streak} CONSECUTIVE WINS</b>"
    elif loss_streak > 0:
        streak_str = f"❄️ <b>{loss_streak} LOSSES — Recovery L{ml_level} active</b>"
    else:
        streak_str = "<b>Neutral Streak</b>"

    recent_visual = " ".join("🟢" if c == "W" else "🔴" for c in recent_20)
    consensus_bar = _bar(consensus)

    return f"""
<b>🏆 LIVE SESSION SCORECARD</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<b>Current Run:</b> {streak_str}
<b>Session Win Rate:</b> <b>{session_rate:.1f}%</b>
<b>Record:</b> <b>{total_wins}W</b> / <b>{total_losses}L</b>  (Total: {total_wins+total_losses})

<b>Last 20 Rounds:</b>
{recent_visual if recent_visual else "Collecting…"}

<b>12-Model Consensus:</b>
<code>{consensus_bar}</code> {consensus*100:.0f}%

<i>Scores sync from live Supabase reconciliations.</i>
""".strip()


def format_models_message(data: Optional[dict]) -> str:
    weights  = data.get("ensembleWeights", {}) if data else {}
    family   = data.get("familyWeights", {}) if data else {}
    vec      = data.get("modelPBigVector", {}) if data else {}
    consensus = float(data.get("modelConsensus", 0.5)) if data else 0.5

    def w(key: str, default: float = 0.08) -> str:
        v = float(weights.get(key, default))
        return f"{v*100:.1f}%"

    def pv(key: str) -> str:
        v = vec.get(key)
        if v is None:
            return "—"
        return f"{float(v)*100:.0f}%"

    return f"""
<b>🧬 AI ENSEMBLE — 12-MODEL BREAKDOWN</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<i>Hedge weights shift every 30s via online regret minimization</i>

<b>Statistical Models:</b>
• CTW Context Tree:    <code>{w('hip_ctw')}</code>  P={pv('hip_ctw')}
• N-Gram (ord 1-5):   <code>{w('hip_ngram')}</code>  P={pv('hip_ngram')}
• Bayesian Streak:    <code>{w('hip_streak')}</code>  P={pv('hip_streak')}
• Side Frequency:     <code>{w('hip_frequency')}</code>  P={pv('hip_frequency')}

<b>Deep Sequence Models:</b>
• Transformer+Mamba:  <code>{w('evoseq_ensemble')}</code>  P={pv('evoseq_ensemble')}

<b>Structural Models:</b>
• Decay Markov:       <code>{w('decay_markov')}</code>  P={pv('decay_markov')}
• Session Position:   <code>{w('session_bias')}</code>  P={pv('session_bias')}
• Exploit Detector:   <code>{w('exploit_detector')}</code>  P={pv('exploit_detector')}

<b>New v2 Models:</b>
• Pattern Memory:     <code>{w('pattern_intelligence')}</code>  P={pv('pattern_intelligence')}
• 3-Level ML:         <code>{w('three_level_ml')}</code>  P={pv('three_level_ml')}
• Volatility Regime:  <code>{w('volatility_regime')}</code>  P={pv('volatility_regime')}
• Cross-Round Corr:   <code>{w('cross_round_corr')}</code>  P={pv('cross_round_corr')}

<b>Family Totals:</b>
• Statistical:  <b>{family.get('hip_statistical',0)*100:.1f}%</b>
• Deep EVOSEQ:  <b>{family.get('evoseq_deep',0)*100:.1f}%</b>
• Pattern/Corr: <b>{(family.get('pattern_intelligence',0)+family.get('cross_round_corr',0))*100:.1f}%</b>
• Regime/Level: <b>{(family.get('volatility_regime',0)+family.get('three_level_ml',0))*100:.1f}%</b>

<b>Overall Consensus:</b> <code>{_bar(consensus)}</code> {consensus*100:.0f}%
""".strip()


def format_metrics_message(metrics: Optional[dict]) -> str:
    if not metrics or metrics.get("resolved_predictions", 0) == 0:
        return (
            "<b>◈ LIVE METRICS</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>Collecting Evidence</b>\n\n"
            "Reconciled metrics appear after outcomes resolve."
        )
    acc = metrics.get("directional_accuracy")
    acc_str = f"{float(acc)*100:.1f}%" if isinstance(acc, (int, float)) else "—"
    brier = metrics.get("brier_score")
    loss  = metrics.get("log_loss")
    three_lvl = metrics.get("three_level_win_rate")
    three_str = f"{float(three_lvl)*100:.1f}%" if isinstance(three_lvl, (int, float)) else "—"

    return f"""
<b>◈ LIVE EVIDENCE METRICS</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<b>Resolved Forecasts:</b>  {_text(metrics.get('resolved_predictions'))}
<b>Directional Accuracy:</b> <b>{acc_str}</b>
<b>3-Level Win Rate:</b>    <b>{three_str}</b>
<b>Brier Score:</b>         {f"{float(brier):.4f}" if isinstance(brier,(int,float)) else "—"}
<b>Log Loss:</b>            {f"{float(loss):.4f}" if isinstance(loss,(int,float)) else "—"}

<i>Strict OOS evaluation — each prediction timestamped before outcome.</i>
""".strip()


def format_learn_message() -> str:
    """Daily learning status card — reads from Supabase."""
    try:
        from backend.daily_learning import get_last_learning_report
        report = get_last_learning_report()
    except Exception:
        report = None

    if not report:
        return (
            "<b>🎓 DAILY LEARNING ENGINE</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "⏳ No learning report found yet.\n\n"
            "The engine runs automatically every midnight UTC.\n"
            "Force a run: <code>python -m backend.daily_learning --force</code>"
        )

    status    = report.get("status", "?")
    started   = report.get("started_at", "?")[:16].replace("T", " ")
    elapsed   = report.get("elapsed_seconds", 0)
    samples   = report.get("samples_trained", 0)
    errors    = report.get("errors", [])
    audit     = report.get("metrics", {}).get("audit", {})
    three_lv  = report.get("metrics", {}).get("level_performance", {})
    eta       = report.get("metrics", {}).get("hedge_eta", "—")

    status_emoji = "✅" if "COMPLETED" in status else "⚠️" if "ERROR" in status else "⏳"

    acc_str = f"{float(audit.get('accuracy',0))*100:.1f}%" if audit.get("accuracy") else "—"
    brier_str = f"{float(audit.get('brier',0)):.4f}" if audit.get("brier") else "—"

    def lv_line(lv: int) -> str:
        d = three_lv.get(str(lv)) or three_lv.get(lv, {})
        wr = float(d.get("win_rate", 0))
        tot = int(d.get("total", 0))
        return f"L{lv}: {_bar(wr, 10)} {wr*100:.1f}% ({tot} rounds)"

    lv_lines = "\n".join(lv_line(lv) for lv in (1, 2, 3))
    err_section = ""
    if errors:
        err_section = f"\n⚠️ <b>Warnings:</b> {len(errors)}"
        for e in errors[:2]:
            err_section += f"\n  • {escape(str(e)[:80])}"

    return f"""
<b>🎓 DAILY LEARNING ENGINE</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{status_emoji} <b>Status:</b> {status}
<b>Last Run:</b> {started} UTC  ({elapsed:.0f}s)
<b>Samples Trained:</b> {samples:,}

<b>Current Accuracy:</b> {acc_str}  •  Brier: {brier_str}
<b>Hedge Eta:</b> {eta}

<b>Per-Level Win Rates:</b>
<code>{lv_lines}</code>
{err_section}

<i>Learns nightly from Supabase history — day by day.</i>
""".strip()


def format_pattern_message(data: Optional[dict]) -> str:
    """Pattern Intelligence diagnostics."""
    if not data:
        return "<b>🔬 Pattern Intelligence</b>\n\nNo data available."
    lags = data.get("patternSignificantLags", [])
    streak = data.get("patternCurrentStreak", 0)
    reversal_prob = data.get("patternReversalProb", 0.5)
    exploit_sc = float(data.get("exploitScore", 0))
    reject_iid = data.get("rejectIID", False)

    lags_text = ""
    if lags:
        for item in lags[:5]:
            lag = item.get("lag", "?")
            strength = float(item.get("strength", 0))
            direction = "→ continue" if strength > 0 else "↩ reverse"
            lags_text += f"  Lag {lag:>2}: {abs(strength):.3f} ({direction})\n"
    else:
        lags_text = "  No significant lags found\n"

    iid_status = "🔬 <b>PRNG non-random detected</b>" if reject_iid else "🎲 <i>No exploit pattern</i>"

    return f"""
<b>🔬 PATTERN INTELLIGENCE</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{iid_status}
<b>Exploit Score:</b> {exploit_sc:.3f}  <code>{_bar(exploit_sc)}</code>

<b>Significant Auto-Correlation Lags:</b>
<code>{lags_text.rstrip()}</code>

<b>Current Streak:</b> {streak} rounds
<b>Reversal Probability:</b> {reversal_prob*100:.1f}%
<code>{_bar(reversal_prob)}</code>

<i>Pattern memory learns from every Supabase draw.</i>
""".strip()


def format_status_message(data: Optional[dict]) -> str:
    online = data is not None and not data.get("error")
    status_str = "🟢 <b>All Systems Operational</b>" if online else "🔴 <b>Cluster Offline</b>"
    issue = _text(data.get("currentIssue")) if online else "—"
    engine = _text(data.get("patternName", "Ultra v2.0")) if online else "—"
    ml_label = _text(data.get("martingaleLevelLabel", "—")) if online else "—"
    drift = _text(data.get("driftLevel", "STABLE")) if online else "—"
    n_samples = _text(data.get("totalSamplesTrained")) if online else "—"

    return f"""
<b>◈ SYSTEM TELEMETRY</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{status_str}

<b>Issue:</b>          <code>#{issue}</code>
<b>Engine:</b>         {engine}
<b>Drift Regime:</b>   {drift}
<b>Martingale:</b>     {ml_label}
<b>Samples:</b>        {n_samples}
<b>Database:</b>       PostgreSQL / Supabase ✅
<b>Dispatch:</b>       {CHECK_INTERVAL}s polling
""".strip()


def premium_help_message() -> str:
    return """
<b>✨ EVOSEQ Ultra v2.0</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>Commands:</b>
/forecast  — Live prediction card
/level     — 3-Level Martingale status
/strategy  — Recovery ladder
/learn     — Daily learning report
/pattern   — Pattern intelligence diagnostics
/models    — 12-model ensemble weights
/results   — Session scorecard
/stats     — Accuracy metrics
/status    — System health
/settings  — Notification filter
/subscribe / /unsubscribe — Auto-stream

<b>Features:</b>
• 12-model adaptive ensemble
• 3-level Martingale auto-recovery
• Daily self-learning from Supabase
• PRNG exploit detection
• Pattern memory (lags + streak + gap)

<i>Learns every day. Adapts every round.</i>
""".strip()


# ═════════════════════════════════════════════════════════════════════════════
# Command Handlers
# ═════════════════════════════════════════════════════════════════════════════

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_chat.id
    subscribed_users.add(user_id)
    user_filters.setdefault(user_id, "all")
    welcome = """
<b>✨ EVOSEQ Ultra v2.0</b>
<i>12-model intelligence, daily self-learning</i>
━━━━━━━━━━━━━━━━━━

Welcome! Your premium dashboard is ready.

🔔 <b>Auto-updates:</b> Enabled
🎯 <b>Filter:</b> All rounds (adjustable)
🎓 <b>Daily Learning:</b> Active (midnight UTC)

Tap below to explore.
""".strip()
    await update.message.reply_text(welcome, parse_mode="HTML", reply_markup=main_keyboard())


async def predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prediction_data = await get_prediction()
    await reply_with_forecast(update.message, prediction_data)


async def level_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = await get_prediction()
    await update.message.reply_text(
        format_level_status_message(data), parse_mode="HTML", reply_markup=main_keyboard()
    )


async def martingale_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = await get_prediction()
    p_single = float(data.get("calibratedPSingle", 0.58)) if data else 0.58
    card = render_martingale_card(p_single)
    text = format_martingale_message(data)
    if card:
        await update.message.reply_photo(photo=card, caption=text[:1024], parse_mode="HTML", reply_markup=main_keyboard())
    else:
        await update.message.reply_text(text, parse_mode="HTML", reply_markup=main_keyboard())


async def learn_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        format_learn_message(), parse_mode="HTML", reply_markup=main_keyboard()
    )


async def pattern_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = await get_prediction()
    await update.message.reply_text(
        format_pattern_message(data), parse_mode="HTML", reply_markup=main_keyboard()
    )


async def scorecard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = await get_prediction()
    await update.message.reply_text(
        format_scorecard_message(data), parse_mode="HTML", reply_markup=main_keyboard()
    )


async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = await get_prediction()
    await update.message.reply_text(
        format_models_message(data), parse_mode="HTML", reply_markup=main_keyboard()
    )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    metrics = get_metrics_from_db()
    if not metrics:
        metrics = await get_prediction()
    card = render_metrics_card(metrics) if metrics else None
    text = format_metrics_message(metrics)
    if card:
        await update.message.reply_photo(photo=card, caption=text[:1024], parse_mode="HTML", reply_markup=main_keyboard())
    else:
        await update.message.reply_text(text, parse_mode="HTML", reply_markup=main_keyboard())


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = await get_prediction()
    status_card = render_status_card({"online": data is not None, "issue": data.get("currentIssue") if data else None})
    text = format_status_message(data)
    if status_card:
        await update.message.reply_photo(photo=status_card, caption=text[:1024], parse_mode="HTML", reply_markup=main_keyboard())
    else:
        await update.message.reply_text(text, parse_mode="HTML", reply_markup=main_keyboard())


async def filter_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_chat.id
    await update.message.reply_text(
        "<b>⚙️ NOTIFICATION FILTER</b>\nChoose which prediction alerts to receive:",
        parse_mode="HTML",
        reply_markup=filter_keyboard(user_filters.get(user_id, "all")),
    )


async def subscribe_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    subscribed_users.add(update.effective_chat.id)
    await update.message.reply_text("🔔 <b>Auto-stream enabled!</b>", parse_mode="HTML", reply_markup=main_keyboard())


async def unsubscribe_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    subscribed_users.discard(update.effective_chat.id)
    await update.message.reply_text("🔕 <b>Auto-stream paused.</b>", parse_mode="HTML", reply_markup=main_keyboard())


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(premium_help_message(), parse_mode="HTML", reply_markup=main_keyboard())


# ═════════════════════════════════════════════════════════════════════════════
# Dashboard Callbacks
# ═════════════════════════════════════════════════════════════════════════════

async def dashboard_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = query.message.chat.id

    async def edit(text: str, kb=None):
        try:
            await query.edit_message_text(
                text, parse_mode="HTML", reply_markup=kb or back_keyboard()
            )
        except Exception:
            pass

    d = query.data

    if d == "forecast":
        pred = await get_prediction()
        await edit(format_prediction_message(pred), main_keyboard())

    elif d == "level_status":
        pred = await get_prediction()
        await edit(format_level_status_message(pred))

    elif d == "martingale":
        pred = await get_prediction()
        await edit(format_martingale_message(pred))

    elif d == "scorecard":
        pred = await get_prediction()
        await edit(format_scorecard_message(pred))

    elif d == "models":
        pred = await get_prediction()
        await edit(format_models_message(pred))

    elif d == "metrics":
        metrics = get_metrics_from_db()
        await edit(format_metrics_message(metrics))

    elif d == "learn":
        await edit(format_learn_message())

    elif d == "pattern":
        pred = await get_prediction()
        await edit(format_pattern_message(pred))

    elif d == "status":
        pred = await get_prediction()
        await edit(format_status_message(pred))

    elif d == "filter_menu":
        await query.edit_message_text(
            "<b>⚙️ NOTIFICATION FILTER</b>",
            parse_mode="HTML",
            reply_markup=filter_keyboard(user_filters.get(user_id, "all")),
        )

    elif d.startswith("set_filter_"):
        f_type = d.replace("set_filter_", "")
        user_filters[user_id] = f_type
        desc = {"all": "All Rounds", "high": "High Conviction (≥70%)", "strike": "Strike Only"}.get(f_type, f_type)
        await edit(f"✅ <b>Filter set:</b> {desc}", back_keyboard())

    elif d == "toggle_sub":
        if user_id in subscribed_users:
            subscribed_users.discard(user_id)
            await edit("🔕 <b>Auto-stream paused.</b>", back_keyboard())
        else:
            subscribed_users.add(user_id)
            await edit("🔔 <b>Auto-stream enabled!</b>", back_keyboard())


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Update %s caused error: %s", update, context.error)


# ═════════════════════════════════════════════════════════════════════════════
# Cycle-Synchronized Background Dispatcher
# ═════════════════════════════════════════════════════════════════════════════

def should_send_to_user(user_id: int, data: dict) -> bool:
    f = user_filters.get(user_id, "all")
    if f == "all":
        return True
    strike = data.get("strikeQuality", "CONSERVATIVE")
    action = data.get("action", "FORECAST")
    conf   = float(data.get("confidence", 0))
    if f == "high":
        return action != "SKIP" and (conf >= 70.0 or strike in (
            "HIGH_CONVICTION", "BEAST_CONVICTION", "ULTIMATE_CONVICTION", "VALIDATED"
        ))
    if f == "strike":
        return action == "STRIKE" or strike in ("BEAST_CONVICTION", "ULTIMATE_CONVICTION")
    return True


async def check_and_send_predictions(context):
    global last_prediction_issue, last_prediction
    try:
        data = await get_prediction()
        if not data:
            return
        current_issue = data.get("currentIssue")
        if not current_issue or current_issue == last_prediction_issue:
            return

        logger.info("New round: %s", current_issue)
        last_prediction_issue = current_issue

        previous_result = None
        if last_prediction:
            previous_result = await check_win_loss(last_prediction)
            if previous_result:
                logger.info("Previous %s: issue #%s", "WON" if previous_result["won"] else "LOST", previous_result["issue"])

        card_bytes = None
        try:
            card_bytes = render_forecast_card(data, previous_result)
        except Exception as e:
            logger.warning("Card render failed: %s", e)

        msg_text    = format_prediction_message(data, previous_result)
        caption_txt = format_forecast_caption(data, previous_result)
        loop_time   = asyncio.get_event_loop().time()

        for user_id in list(subscribed_users):
            if not should_send_to_user(user_id, data):
                continue
            last_t = last_notification_time.get(user_id, 0)
            if loop_time - last_t < NOTIFICATION_COOLDOWN and previous_result is None:
                continue
            try:
                if card_bytes:
                    card_bytes.seek(0)
                    await context.bot.send_photo(
                        chat_id=user_id,
                        photo=card_bytes,
                        caption=caption_txt,
                        parse_mode="HTML",
                        reply_markup=main_keyboard(),
                    )
                else:
                    await context.bot.send_message(
                        chat_id=user_id,
                        text=msg_text,
                        parse_mode="HTML",
                        reply_markup=main_keyboard(),
                    )
                last_notification_time[user_id] = loop_time
            except Exception as e:
                logger.warning("Send to %s failed: %s", user_id, e)
                subscribed_users.discard(user_id)

        last_prediction = data

    except Exception as e:
        logger.error("Dispatcher error: %s", e)


async def prediction_updater(bot):
    logger.info("Background dispatcher running (interval=%.1fs)", CHECK_INTERVAL)
    while True:
        try:
            class _Ctx:
                def __init__(self, b):
                    self.bot = b
            await check_and_send_predictions(_Ctx(bot))
        except Exception as e:
            logger.error("Updater error: %s", e)
        await asyncio.sleep(CHECK_INTERVAL)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

async def main():
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing — set it in .env")

    logger.info("Starting WinGo Ultra Intelligence v2.0 Telegram Bot...")
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start",       start_command))
    app.add_handler(CommandHandler("predict",     predict_command))
    app.add_handler(CommandHandler("forecast",    predict_command))
    app.add_handler(CommandHandler("level",       level_command))
    app.add_handler(CommandHandler("martingale",  martingale_command))
    app.add_handler(CommandHandler("strategy",    martingale_command))
    app.add_handler(CommandHandler("learn",       learn_command))
    app.add_handler(CommandHandler("pattern",     pattern_command))
    app.add_handler(CommandHandler("scorecard",   scorecard_command))
    app.add_handler(CommandHandler("results",     scorecard_command))
    app.add_handler(CommandHandler("models",      models_command))
    app.add_handler(CommandHandler("stats",       stats_command))
    app.add_handler(CommandHandler("status",      status_command))
    app.add_handler(CommandHandler("filter",      filter_command))
    app.add_handler(CommandHandler("settings",    filter_command))
    app.add_handler(CommandHandler("help",        help_command))
    app.add_handler(CommandHandler("subscribe",   subscribe_command))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe_command))
    app.add_handler(CallbackQueryHandler(dashboard_callback))
    app.add_error_handler(error_handler)

    initial = await get_prediction()
    if initial:
        logger.info("Connected — Issue #%s", initial.get("currentIssue"))

    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    updater_task = asyncio.create_task(prediction_updater(app.bot))
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        updater_task.cancel()
        try:
            await updater_task
        except asyncio.CancelledError:
            pass
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
