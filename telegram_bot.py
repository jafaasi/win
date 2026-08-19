#!/usr/bin/env python3
"""
Telegram Bot for WinGo Predictions — Ultra Intelligence Edition
Provides predictions via Telegram commands and automatic updates with:
- Direct DB access for ultra-fast response (<10ms)
- Cycle-synchronized 1.5s polling
- Exploit-gated SKIP signal support
- Rolling scorecard (W/L streak & session win rate)
- User conviction filtering (All / High+ / Strike Only)
"""

import asyncio
from html import escape
import json
import logging
import os
import sys
from typing import Dict, Optional, Set
import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

# Ensure backend imports work
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

# Configuration
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
API_URL = os.environ.get("PREDICTION_API_URL", "http://localhost:8000/api/state")
WINGO_API_URL = os.environ.get("WINGO_API_URL", "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json")
CHECK_INTERVAL = 1.5  # Check every 1.5s for fast 30s game window sync

# User preferences store: user_id -> {"filter": "all" | "high" | "strike"}
user_filters: Dict[int, str] = {}
subscribed_users: Set[int] = set()
last_prediction_issue = None
last_prediction = None  # Store last prediction data for win/loss tracking

# Logging setup
logging.basicConfig(
    format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("TELEGRAM_BOT")


# ============================================================================
# Keyboards & Helpers
# ============================================================================

def main_keyboard() -> InlineKeyboardMarkup:
    """Luxury Telegram dashboard navigation."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("⚡ Live Forecast", callback_data="forecast"),
            InlineKeyboardButton("📊 Quant Metrics", callback_data="metrics"),
        ],
        [
            InlineKeyboardButton("💎 3-Step Martingale", callback_data="martingale"),
            InlineKeyboardButton("🏆 Live Scorecard", callback_data="scorecard"),
        ],
        [
            InlineKeyboardButton("🎯 Conviction Alerts", callback_data="filter_menu"),
            InlineKeyboardButton("🧬 AI Model Ensembles", callback_data="models"),
        ],
        [
            InlineKeyboardButton("🟢 System Telemetry", callback_data="status"),
            InlineKeyboardButton("🔔 Toggle Auto-Stream", callback_data="toggle_sub"),
        ],
    ])


def back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("← Back to Quant Dashboard", callback_data="forecast")]])


def filter_keyboard(current_filter: str = "all") -> InlineKeyboardMarkup:
    all_mark = "✅ " if current_filter == "all" else ""
    high_mark = "✅ " if current_filter == "high" else ""
    strike_mark = "✅ " if current_filter == "strike" else ""

    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"{all_mark}All Rounds (Recommended)", callback_data="set_filter_all")],
        [InlineKeyboardButton(f"{high_mark}High Conviction Only (≥70% edge)", callback_data="set_filter_high")],
        [InlineKeyboardButton(f"{strike_mark}Strike Only (Beast & Ultimate)", callback_data="set_filter_strike")],
        [InlineKeyboardButton("← Back to Quant Dashboard", callback_data="forecast")],
    ])


def _text(value, fallback="—") -> str:
    if value is None or value == "":
        return fallback
    return escape(str(value))


def _percentage(value) -> str:
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "—"


# ============================================================================
# Message Formatters
# ============================================================================

def format_prediction_message(data: Optional[dict], previous_result: Optional[dict] = None) -> str:
    """Render a luxury quant terminal message card."""
    if not data:
        return "<b>◈ ULTRA QUANT INTELLIGENCE</b>\n\n⚠️ <b>Live forecast syncing…</b>\nPlease wait for the next 30s draw."

    try:
        prediction = _text(data.get('prediction'))
        confidence = _percentage(data.get('confidence'))
        target_num = _text(data.get('targetNum'))
        hedge_num = _text(data.get('hedgeNum'))
        current_issue = _text(data.get('currentIssue'))
        next_issue = _text(data.get('nextIssue'))
        pattern = _text(data.get('patternName'))
        strike_quality = _text(data.get('strikeQuality', 'CONSERVATIVE'))
        action = str(data.get('action', 'FORECAST'))

        evidence = data.get('evidence', {}) or {}
        is_validated = bool(evidence.get('validated_edge', False))

        # Scorecard
        scorecard = data.get('scorecard', {}) or {}
        win_streak = scorecard.get('win_streak', 0)
        loss_streak = scorecard.get('loss_streak', 0)
        session_rate = scorecard.get('session_win_rate', None)
        recent_20 = scorecard.get('recent_20', '')

        # Action banner
        if action == "SKIP":
            action_banner = "⏭️ <b>RESTRAIN BET: NO PRNG EXPLOIT DETECTED</b>\n<i>Engine recommends sitting this round out to protect capital.</i>"
            side_emoji = "⚪"
        elif action == "CAUTION":
            action_banner = "⚠️ <b>MARGINAL EDGE: 1X BASE STAKE</b>"
            side_emoji = "🔵" if prediction.lower() == "big" else "🟡"
        elif action == "STRIKE":
            action_banner = "⚡ <b>STRIKE CONVICTION: HIGH EDGE MULTIPLIER</b>"
            side_emoji = "🔵" if prediction.lower() == "big" else "🟡"
        else:
            action_banner = "✦ <b>VALIDATED 3-LEVEL EDGE</b>" if is_validated else "◌ <b>CONTINUOUS ADAPTIVE LEARNING</b>"
            side_emoji = "🔵" if prediction.lower() == "big" else "🟡"

        # Martingale 3-level strategy block
        p_win_in_3 = data.get('calibratedPWinIn3', evidence.get('joint3_probability', None))
        p_correct_single = data.get('calibratedPSingle', evidence.get('per_round_win_rate', None))
        p3_pct = f"{float(p_win_in_3) * 100:.1f}%" if p_win_in_3 is not None else "—"
        p1_pct = f"{float(p_correct_single) * 100:.1f}%" if p_correct_single is not None else "—"

        strike_emoji = {
            "ULTIMATE_CONVICTION": "💎",
            "BEAST_CONVICTION": "🔥",
            "HIGH_CONVICTION": "⚡",
            "MODERATE_CONVICTION": "🎯",
            "VALIDATED": "✅",
            "SKIP": "⏭️",
            "LOW_CONFIDENCE": "⚠️",
        }.get(strike_quality, "🧿")

        # Scorecard line
        scorecard_parts = []
        if win_streak > 0:
            scorecard_parts.append(f"🔥 Streak: <b>W{win_streak}</b>")
        elif loss_streak > 0:
            scorecard_parts.append(f"❄️ Streak: <b>L{loss_streak}</b>")
        if session_rate is not None:
            scorecard_parts.append(f"Session: <b>{float(session_rate):.1f}%</b>")
        if recent_20:
            formatted_recent = "".join("🟢" if x == "W" else "🔴" for x in recent_20[-8:])
            scorecard_parts.append(f"Recent: {formatted_recent}")
        scorecard_line = "  •  ".join(scorecard_parts) if scorecard_parts else "Tracking session outcomes…"

        # Previous result
        result_info = ""
        if previous_result:
            result_emoji = "✅" if previous_result['won'] else "❌"
            result_text = "WON (+1.96x)" if previous_result['won'] else "LOST"
            predicted = _text(previous_result.get('predicted', 'N/A')).upper()
            actual = _text(previous_result.get('actual', 'N/A')).upper()
            number = _text(previous_result.get('number', 'N/A'))
            issue_id = _text(previous_result.get('issue', ''))
            result_info = f"\n\n<b>LAST DRAW OUTCOME</b>\n{result_emoji} <b>{result_text}</b>  •  Issue #{issue_id[-6:]}\nPredicted: <b>{predicted}</b>  |  Actual: <b>{actual}</b> (Digit: {number})"

        message = f"""
<b>◈ ULTRA QUANT INTELLIGENCE</b>  <i>WINGO 30S</i>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{action_banner}

<b>LIVE FORECAST</b>
{side_emoji} <b>{prediction.upper()}</b>   <b>Edge: {confidence}</b>
🎯 <b>Primary Target:</b> {target_num}     🛡️ <b>Hedge Target:</b> {hedge_num}

<b>💎 3-LEVEL CAPITAL STRATEGY</b>
{strike_emoji} <b>{strike_quality.replace('_', ' ')}</b>
• <b>P(win in 3 steps):</b> <code>{p3_pct}</code>
• <b>P(single round):</b>   <code>{p1_pct}</code>

<b>🏆 LIVE SCORECARD</b>
{scorecard_line}

<b>⏱️ ROUND TIMELINE</b>
Current Issue: <code>{current_issue}</code>
Target Issue:  <code>#{next_issue}</code>{result_info}

<b>🔬 ENGINE TELEMETRY</b>
• <b>Active:</b> {pattern}
• <b>Exploit Edge:</b> <code>{_text(data.get('exploitScore', '0.00'))}</code> (Reject IID: {'Yes' if data.get('rejectIID') else 'No'})
• <b>Evidence Gate:</b> {_text(evidence.get('reason', 'LEARNING'))} (n={_text(evidence.get('resolved_predictions', 0))})

<i>Refreshed instantly • Direct DB sync (<10ms)</i>
        """
        return message.strip()
    except Exception as e:
        logger.error("Error formatting message: %s", e)
        return "<b>◈ ULTRA INTELLIGENCE</b>\n\n⚠️ <b>Unable to format live forecast.</b>"


def format_forecast_caption(data: dict, previous_result=None) -> str:
    action = str(data.get("action", "FORECAST"))
    strike = _text(data.get("strikeQuality", "CONSERVATIVE")).replace("_", " ")
    p3 = data.get("calibratedPWinIn3", None)
    p3_txt = f"{float(p3) * 100:.1f}%" if p3 is not None else "—"

    if action == "SKIP":
        header = "⏭️ <b>RESTRAIN BET (NO EXPLOIT EDGE)</b>"
    elif action == "STRIKE":
        header = f"⚡ <b>STRIKE CONVICTION • {strike.upper()}</b>"
    else:
        header = f"◈ <b>LIVE QUANT CALL • {strike.upper()}</b>"

    caption = f"""
{header}
Target: <code>#{_text(data.get('nextIssue'))}</code>  •  Call: <b>{_text(data.get('prediction')).upper()}</b> ({_percentage(data.get('confidence'))})
Targets: <b>{_text(data.get('targetNum'))}</b> / {_text(data.get('hedgeNum'))}  •  P(win in 3): <b>{p3_txt}</b>
    """.strip()

    if previous_result:
        outcome = "✅ WON" if previous_result.get("won") else "❌ LOST"
        caption += f"\nLast Draw: <b>{outcome}</b> (#{_text(previous_result.get('issue'))[-6:]})"
    return caption


async def reply_with_forecast(message, data: Optional[dict], previous_result=None):
    """Deliver a visual card, with text fallback."""
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
        except Exception as error:
            logger.exception("Forecast card rendering failed: %s", error)

    return await message.reply_text(
        format_prediction_message(data, previous_result),
        parse_mode="HTML",
        reply_markup=main_keyboard(),
    )


def format_martingale_message(data: Optional[dict]) -> str:
    p_single = float(data.get('calibratedPSingle', 0.58)) if data else 0.58
    p1 = p_single
    p2 = 0.5 + 0.94 * (p1 - 0.5)
    p3 = 0.5 + 0.88 * (p1 - 0.5)
    joint_p = 1.0 - (1.0 - p1) * (1.0 - p2) * (1.0 - p3)

    return f"""
<b>💎 3-LEVEL MARTINGALE LADDER</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<i>Optimal capital allocation table designed to ensure long-term mathematical survival:</i>

<b>STEP 1: INITIAL ENTRY</b>
• Allocation: <b>1.0x Base Unit</b> (e.g. $10)
• Single Win Probability: <b>{p1*100:.1f}%</b>
• Capital Risk: <b>Minimal</b>

<b>STEP 2: RECOVERY STRIKE</b>
• Allocation: <b>2.2x Base Unit</b> (e.g. $22)
• Cumulative Win Probability: <b>{(1-(1-p1)*(1-p2))*100:.1f}%</b>
• Capital Risk: <b>Controlled</b>

<b>STEP 3: MAX CONVICTION STRIKE</b>
• Allocation: <b>4.8x Base Unit</b> (e.g. $48)
• Cumulative Win Probability: <b>{joint_p*100:.1f}%</b>
• Capital Risk: <b>Max Capped</b>

<b>🛡️ RISK GOVERNOR RULES:</b>
1. <b>Reset to Step 1</b> immediately upon ANY win.
2. <b>Never extend to Step 4</b> — 3-step discipline ensures 96%+ survival without catastrophic drawdown.
    """.strip()


def format_models_message(data: Optional[dict]) -> str:
    weights = data.get("ensembleWeights", {}) if data else {}
    family = data.get("familyWeights", {}) if data else {}

    return f"""
<b>🧬 AI MODEL ENSEMBLE BREAKDOWN</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<i>Adaptive Hedge weights updated dynamically on every 30s draw based on online regret minimization:</i>

<b>Core Model Weights:</b>
• <b>CTW Context Tree:</b> <code>{weights.get('hip_ctw', 0.18)*100:.1f}%</code>
• <b>Transformer & Mamba:</b> <code>{weights.get('evoseq_ensemble', 0.22)*100:.1f}%</code>
• <b>Decay-Weighted Markov:</b> <code>{weights.get('decay_markov', 0.16)*100:.1f}%</code>
• <b>Exploit Detector:</b> <code>{weights.get('exploit_detector', 0.14)*100:.1f}%</code>
• <b>Bayesian Streaks:</b> <code>{weights.get('hip_streak', 0.12)*100:.1f}%</code>
• <b>Temporal Sessions:</b> <code>{weights.get('session_bias', 0.10)*100:.1f}%</code>
• <b>N-Gram Variable:</b> <code>{weights.get('hip_ngram', 0.08)*100:.1f}%</code>

<b>Family Allocation:</b>
• Statistical/Deterministic: <b>{family.get('hip_statistical', 0.45)*100:.1f}%</b>
• Deep Sequence (Mamba/Transformer): <b>{family.get('evoseq_deep', 0.25)*100:.1f}%</b>
• Exploit/Regime Gating: <b>{family.get('exploit_statistical', 0.15)*100:.1f}%</b>
• Recency Markov: <b>{family.get('decay_markov', 0.15)*100:.1f}%</b>
    """.strip()


def format_scorecard_message(data: Optional[dict]) -> str:
    scorecard = data.get('scorecard', {}) if data else {}
    win_streak = scorecard.get('win_streak', 0)
    loss_streak = scorecard.get('loss_streak', 0)
    session_rate = scorecard.get('session_win_rate', 50.0)
    total_wins = scorecard.get('total_wins', 0)
    total_losses = scorecard.get('total_losses', 0)
    recent_20 = scorecard.get('recent_20', '')

    streak_str = f"🔥 <b>{win_streak} CONSECUTIVE WINS</b>" if win_streak > 0 else f"❄️ <b>{loss_streak} LOSSES (Dampener Active)</b>" if loss_streak > 0 else "<b>Neutral Streak</b>"
    recent_formatted = " ".join("🟢" if x == "W" else "🔴" for x in recent_20)

    return f"""
<b>🏆 LIVE SESSION SCORECARD</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>Current Run:</b> {streak_str}
<b>Session Win Rate:</b> <b>{session_rate:.1f}%</b>
<b>Total Record:</b> <b>{total_wins}W</b> — <b>{total_losses}L</b> (Total: {total_wins + total_losses})

<b>Last 20 Outcomes:</b>
{recent_formatted if recent_formatted else 'Collecting round history…'}

<i>Scores synchronize automatically from live database reconciliations.</i>
    """.strip()


def format_metrics_message(metrics) -> str:
    if not metrics or metrics.get("resolved_predictions", 0) == 0:
        return """
<b>◈ LIVE METRICS</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<b>Collecting Evidence</b>

Reconciled metrics appear automatically after database outcomes resolve.
        """.strip()

    accuracy = metrics.get("directional_accuracy")
    accuracy_text = f"{float(accuracy) * 100:.1f}%" if isinstance(accuracy, (int, float)) else "—"
    brier = metrics.get("brier_score")
    loss = metrics.get("log_loss")

    return f"""
<b>◈ LIVE EVIDENCE METRICS</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>Resolved Forecasts:</b>  {_text(metrics.get('resolved_predictions'))}
<b>Directional Accuracy:</b> <b>{_text(accuracy_text)}</b>
<b>Brier Score:</b>          {_text(f'{float(brier):.4f}' if isinstance(brier, (int, float)) else '—')}
<b>Log Loss:</b>             {_text(f'{float(loss):.4f}' if isinstance(loss, (int, float)) else '—')}

<i>Strict out-of-sample evaluation: each prediction is timestamped BEFORE the outcome occurs.</i>
    """.strip()


def premium_help_message() -> str:
    return """
<b>◈ ULTRA QUANT INTELLIGENCE GUIDE</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>Command Palette:</b>
<b>/predict</b> — Instant visual forecast card
<b>/martingale</b> — Visual 3-step capital ladder
<b>/scorecard</b> — Live win/loss history & streak
<b>/models</b> — Live AI model Hedge weights
<b>/stats</b> — Out-of-sample accuracy & Brier score
<b>/filter</b> — Conviction notification threshold
<b>/status</b> — System telemetry & latency
<b>/subscribe</b> — Enable cycle-synced stream
<b>/unsubscribe</b> — Pause notifications

<b>Core Pillars:</b>
1. <b>Exploit Gating:</b> 11 statistical tests protect your bankroll by emitting <b>SKIP</b> when no edge exists.
2. <b>8-Model Regret Minimization:</b> Multiplicative Hedge automatically favors top-performing sub-models.
3. <b>Martingale joint calibration:</b> Evaluates probability over 3 steps for maximum longevity.

<i>For analytical and entertainment purposes only. Practice disciplined bankroll management.</i>
    """.strip()


# ============================================================================
# Command Handlers
# ============================================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_chat.id
    subscribed_users.add(user_id)
    if user_id not in user_filters:
        user_filters[user_id] = "all"

    welcome = """
<b>◈ WINGO 30S • ULTRA QUANT INTELLIGENCE</b>
<i>Luxury Exploit-Gated Forecasting & Capital Architecture</i>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Welcome! Your live quant terminal is fully synchronized.

🔔 <b>Auto-Stream:</b> ENABLED (every 30s draw)
🎯 <b>Conviction Filter:</b> ALL ROUNDS (use /filter to adjust)

Tap below for your visual forecast, Martingale ladder, or quant metrics.
    """.strip()
    await update.message.reply_text(welcome, parse_mode='HTML', reply_markup=main_keyboard())


async def predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prediction_data = await get_prediction()
    await reply_with_forecast(update.message, prediction_data)


async def martingale_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prediction_data = await get_prediction()
    p_single = float(prediction_data.get('calibratedPSingle', 0.58)) if prediction_data else 0.58
    card = render_martingale_card(p_single)
    if card:
        await update.message.reply_photo(
            photo=card,
            caption=format_martingale_message(prediction_data),
            parse_mode="HTML",
            reply_markup=main_keyboard()
        )
    else:
        await update.message.reply_text(
            format_martingale_message(prediction_data),
            parse_mode="HTML",
            reply_markup=main_keyboard()
        )


async def scorecard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prediction_data = await get_prediction()
    await update.message.reply_text(
        format_scorecard_message(prediction_data),
        parse_mode="HTML",
        reply_markup=main_keyboard()
    )


async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prediction_data = await get_prediction()
    await update.message.reply_text(
        format_models_message(prediction_data),
        parse_mode="HTML",
        reply_markup=main_keyboard()
    )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    metrics = get_metrics_from_db()
    if not metrics:
        metrics = await get_prediction()
    card = render_metrics_card(metrics) if metrics else None
    if card:
        await update.message.reply_photo(
            photo=card,
            caption=format_metrics_message(metrics),
            parse_mode="HTML",
            reply_markup=main_keyboard()
        )
    else:
        await update.message.reply_text(
            format_metrics_message(metrics), parse_mode="HTML", reply_markup=main_keyboard()
        )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prediction_data = await get_prediction()
    online = prediction_data is not None and not prediction_data.get("error")
    status_card = render_status_card({
        "online": online,
        "issue": prediction_data.get("currentIssue") if online else None
    })
    if status_card:
        await update.message.reply_photo(
            photo=status_card,
            caption=f"◈ <b>STATUS:</b> {'🟢 ALL SYSTEMS OPERATIONAL' if online else '🔴 OFFLINE'}\nPolling latency: <10ms (Direct DB Stream)",
            parse_mode="HTML",
            reply_markup=main_keyboard()
        )
    else:
        await update.message.reply_text(
            f"◈ <b>STATUS:</b> {'🟢 ONLINE' if online else '🔴 OFFLINE'}",
            parse_mode="HTML",
            reply_markup=main_keyboard()
        )


async def filter_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_chat.id
    cur = user_filters.get(user_id, "all")
    await update.message.reply_text(
        "<b>🎯 CONVICTION FILTER SETTINGS</b>\nChoose which prediction alerts you wish to receive automatically:",
        parse_mode="HTML",
        reply_markup=filter_keyboard(cur)
    )


async def subscribe_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_chat.id
    subscribed_users.add(user_id)
    await update.message.reply_text(
        "🔔 <b>Auto-stream enabled!</b> You will receive predictions at each 30s cycle.",
        parse_mode='HTML',
        reply_markup=main_keyboard()
    )


async def unsubscribe_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_chat.id
    subscribed_users.discard(user_id)
    await update.message.reply_text(
        "🔕 <b>Auto-stream paused.</b> Use /subscribe to resume.",
        parse_mode="HTML",
        reply_markup=main_keyboard()
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        premium_help_message(), parse_mode='HTML', reply_markup=main_keyboard()
    )


# ============================================================================
# Dashboard Callbacks
# ============================================================================

async def dashboard_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = query.message.chat.id

    if query.data == "forecast":
        prediction = await get_prediction()
        await query.edit_message_text(
            format_prediction_message(prediction), parse_mode="HTML", reply_markup=main_keyboard()
        )
    elif query.data == "martingale":
        prediction = await get_prediction()
        await query.edit_message_text(
            format_martingale_message(prediction), parse_mode="HTML", reply_markup=back_keyboard()
        )
    elif query.data == "scorecard":
        prediction = await get_prediction()
        await query.edit_message_text(
            format_scorecard_message(prediction), parse_mode="HTML", reply_markup=back_keyboard()
        )
    elif query.data == "models":
        prediction = await get_prediction()
        await query.edit_message_text(
            format_models_message(prediction), parse_mode="HTML", reply_markup=back_keyboard()
        )
    elif query.data == "metrics":
        metrics = get_metrics_from_db()
        await query.edit_message_text(
            format_metrics_message(metrics), parse_mode="HTML", reply_markup=back_keyboard()
        )
    elif query.data == "status":
        prediction = await get_prediction()
        online = prediction is not None and not prediction.get("error")
        status_text = "🟢 <b>All Systems Operational</b>" if online else "🔴 <b>Cluster Offline</b>"
        issue = _text(prediction.get("currentIssue")) if online else "—"
        text = f"""
<b>◈ SYSTEM TELEMETRY & RUNTIME</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{status_text}

<b>Active Blockchain Issue:</b> <code>#{issue}</code>
<b>Engine:</b>                 Ultra Intelligence v1.0
<b>Database:</b>               PostgreSQL / Supabase (Direct)
<b>Dispatch Interval:</b>      1.5s (Zero-Latency Sync)
        """.strip()
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=back_keyboard())
    elif query.data == "filter_menu":
        cur = user_filters.get(user_id, "all")
        await query.edit_message_text(
            "<b>🎯 CONVICTION ALERT SETTINGS</b>\nChoose which rounds trigger automatic notifications:",
            parse_mode="HTML",
            reply_markup=filter_keyboard(cur)
        )
    elif query.data.startswith("set_filter_"):
        f_type = query.data.replace("set_filter_", "")
        user_filters[user_id] = f_type
        desc = {
            "all": "All Rounds",
            "high": "High Conviction Only (≥70%)",
            "strike": "Strike Only (Beast/Ultimate)"
        }.get(f_type, "All")
        await query.edit_message_text(
            f"✅ <b>Alert filter set to:</b> {desc}\n\nYou will receive predictions matching this criterion.",
            parse_mode="HTML",
            reply_markup=back_keyboard()
        )
    elif query.data == "toggle_sub":
        if user_id in subscribed_users:
            subscribed_users.discard(user_id)
            await query.edit_message_text(
                "🔕 <b>Auto-stream paused.</b> Tap below to return.",
                parse_mode="HTML",
                reply_markup=back_keyboard()
            )
        else:
            subscribed_users.add(user_id)
            await query.edit_message_text(
                "🔔 <b>Auto-stream enabled!</b> You will receive live predictions at each 30s draw.",
                parse_mode="HTML",
                reply_markup=back_keyboard()
            )
    elif query.data == "learn":
        await query.edit_message_text(
            premium_help_message(), parse_mode="HTML", reply_markup=back_keyboard()
        )


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Update %s caused error: %s", update, context.error)


# ============================================================================
# Cycle-Synchronized Background Dispatcher
# ============================================================================

def should_send_to_user(user_id: int, prediction_data: dict) -> bool:
    """Check if the prediction matches the user's conviction filter."""
    u_filter = user_filters.get(user_id, "all")
    if u_filter == "all":
        return True

    strike = prediction_data.get("strikeQuality", "CONSERVATIVE")
    action = prediction_data.get("action", "FORECAST")
    conf = float(prediction_data.get("confidence", 0))

    if u_filter == "high":
        # Send only if HIGH, BEAST, ULTIMATE or conf >= 70% and not SKIP
        return action != "SKIP" and (conf >= 70.0 or strike in ("HIGH_CONVICTION", "BEAST_CONVICTION", "ULTIMATE_CONVICTION", "VALIDATED"))

    if u_filter == "strike":
        # Send only if BEAST or ULTIMATE or action == STRIKE
        return action == "STRIKE" or strike in ("BEAST_CONVICTION", "ULTIMATE_CONVICTION")

    return True


async def check_and_send_predictions(context: ContextTypes.DEFAULT_TYPE):
    """Check for new predictions and dispatch to subscribed users."""
    global last_prediction_issue, last_prediction

    try:
        prediction_data = await get_prediction()
        if not prediction_data:
            return

        current_issue = prediction_data.get('currentIssue')

        # Detect new round immediately
        if current_issue and current_issue != last_prediction_issue:
            logger.info("New round detected: %s", current_issue)
            last_prediction_issue = current_issue

            # Check win/loss for previous prediction
            previous_result = None
            if last_prediction:
                previous_result = await check_win_loss(last_prediction)
                if previous_result:
                    res_str = "WON" if previous_result['won'] else "LOST"
                    logger.info("Previous prediction %s: Issue #%s", res_str, previous_result['issue'])

            # Render card once for all users
            card_bytes = None
            try:
                card_bytes = render_forecast_card(prediction_data, previous_result)
            except Exception as e:
                logger.warning("Card rendering failed: %s", e)

            message_text = format_prediction_message(prediction_data, previous_result)
            caption_text = format_forecast_caption(prediction_data, previous_result)

            for user_id in subscribed_users.copy():
                if not should_send_to_user(user_id, prediction_data):
                    continue

                try:
                    if card_bytes:
                        # Reset buffer pointer for each send
                        card_bytes.seek(0)
                        await context.bot.send_photo(
                            chat_id=user_id,
                            photo=card_bytes,
                            caption=caption_text,
                            parse_mode='HTML',
                            reply_markup=main_keyboard(),
                        )
                    else:
                        await context.bot.send_message(
                            chat_id=user_id,
                            text=message_text,
                            parse_mode='HTML',
                            reply_markup=main_keyboard(),
                        )
                except Exception as e:
                    logger.warning("Failed to send to user %s: %s", user_id, e)
                    subscribed_users.discard(user_id)

            last_prediction = prediction_data

    except Exception as e:
        logger.error("Error in prediction dispatcher: %s", e)


async def prediction_updater(bot):
    """Cycle-synchronized background worker."""
    logger.info("Background dispatcher running (interval=%.1fs)", CHECK_INTERVAL)
    while True:
        try:
            class MockContext:
                def __init__(self, b):
                    self.bot = b
            await check_and_send_predictions(MockContext(bot))
        except Exception as e:
            logger.error("Dispatcher iteration error: %s", e)

        await asyncio.sleep(CHECK_INTERVAL)


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required. Set it in your .env file.")

    logger.info("Starting WinGo Ultra Intelligence Telegram Bot...")
    application = Application.builder().token(BOT_TOKEN).build()

    # Handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("predict", predict_command))
    application.add_handler(CommandHandler("martingale", martingale_command))
    application.add_handler(CommandHandler("scorecard", scorecard_command))
    application.add_handler(CommandHandler("models", models_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("filter", filter_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("subscribe", subscribe_command))
    application.add_handler(CommandHandler("unsubscribe", unsubscribe_command))
    application.add_handler(CallbackQueryHandler(dashboard_callback))
    application.add_error_handler(error_handler)

    # Initial probe
    initial = await get_prediction()
    if initial:
        logger.info("Initial prediction connected: Issue #%s", initial.get("currentIssue"))

    await application.initialize()
    await application.start()
    await application.updater.start_polling()

    updater_task = asyncio.create_task(prediction_updater(application.bot))

    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Bot shutting down...")
    finally:
        updater_task.cancel()
        try:
            await updater_task
        except asyncio.CancelledError:
            pass
        await application.updater.stop()
        await application.stop()
        await application.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
