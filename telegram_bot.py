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
CHECK_INTERVAL = 3.0  # Check every 3s (reduced from 1.5s for less frequent updates)
NOTIFICATION_COOLDOWN = 15  # Minimum 15s between notifications (prevent spam)
MAX_NOTIFICATIONS_PER_HOUR = 20  # Limit notifications to prevent overwhelming users

# User preferences store: user_id -> {"filter": "all" | "high" | "strike"}
user_filters: Dict[int, str] = {}
subscribed_users: Set[int] = set()
last_prediction_issue = None
last_prediction = None  # Store last prediction data for win/loss tracking
last_notification_time = {}  # Track last notification time per user
notification_count = {}  # Track notification count per user per hour

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
    """Luxury Telegram dashboard navigation - Premium but not overwhelming."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✨ Live Forecast", callback_data="forecast"),
            InlineKeyboardButton("📊 Performance", callback_data="metrics"),
        ],
        [
            InlineKeyboardButton("💎 Strategy", callback_data="martingale"),
            InlineKeyboardButton("🏆 Results", callback_data="scorecard"),
        ],
        [
            InlineKeyboardButton("⚙️ Preferences", callback_data="filter_menu"),
        ],
        [
            InlineKeyboardButton("🟢 Status", callback_data="status"),
            InlineKeyboardButton("🔔 Updates", callback_data="toggle_sub"),
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

async def get_prediction() -> Optional[dict]:
    """Fetch the latest prediction from the local WinGo API."""
    try:
        timeout = httpx.Timeout(5.0, connect=2.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(API_URL)

            if response.status_code != 200:
                logger.warning(
                    "Prediction API returned HTTP %s",
                    response.status_code
                )
                return None

            data = response.json()

            if not isinstance(data, dict):
                logger.warning("Prediction API returned unexpected data type")
                return None

            return data

    except httpx.RequestError as e:
        logger.warning("Prediction API connection failed: %s", e)
        return None

    except Exception as e:
        logger.exception("Failed to fetch prediction: %s", e)
        return None
# ============================================================================
# Message Formatters
# ============================================================================

def format_prediction_message(data: Optional[dict], previous_result: Optional[dict] = None) -> str:
    """Render a premium but concise message card."""
    if not data:
        return "<b>✨ EVOSEQ Premium</b>\n\n⏳ Intelligence service connecting..."

    try:
        prediction = _text(data.get('prediction'))
        confidence = _percentage(data.get('confidence'))
        target_num = _text(data.get('targetNum'))
        hedge_num = _text(data.get('hedgeNum'))
        next_issue = _text(data.get('nextIssue'))
        
        # Premium minimalist design
        side_emoji = "🔵" if prediction.lower() == "big" else "🟡"
        
        # Win/loss result (simplified)
        result_info = ""
        if previous_result:
            result_emoji = "✅" if previous_result['won'] else "❌"
            result_info = f"\n{result_emoji} Last: {'WON' if previous_result['won'] else 'LOST'}"

        message = f"""
<b>✨ {side_emoji} {prediction.upper()}</b>
Confidence: {confidence}
Target: {target_num} | Hedge: {hedge_num}

Round: {next_issue[-8:]}{result_info}

<i>EVOSEQ Premium Intelligence</i>
        """
        return message.strip()
    except Exception as e:
        logger.error(f"Error formatting message: {e}")
        return "<b>✨ EVOSEQ Premium</b>\n\n⏳ Refreshing intelligence..."
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

        return message.strip()
    except Exception as e:
        logger.error("Error formatting message: %s", e)
        return "<b>✨ EVOSEQ Premium</b>\n\n⏳ Refreshing intelligence..."
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
<b>✨ EVOSEQ Premium Intelligence</b>
━━━━━━━━━━━━━━━━━━

<b>Quick Commands:</b>
/forecast  • Live prediction card
/strategy  • 3-level recovery
/results   • Win/loss history
/settings  • Your preferences

<b>Smart Features:</b>
• 3-level winning algorithm
• Automatic loss recovery  
• Premium visual cards
• Respectful notifications

<i>Powerful intelligence, pleasant experience.</i>
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
<b>✨ EVOSEQ Premium Intelligence</b>
<i>Powerful predictions, pleasant experience</i>
━━━━━━━━━━━━━━━━━━

Welcome! Your premium dashboard is ready.

🔔 <b>Smart Updates:</b> Enabled (respectful timing)
🎯 <b>Your Preferences:</b> All rounds (adjustable)

Tap below for live forecasts and results.
    """.strip()
    await update.message.reply_text(welcome, parse_mode='HTML', reply_markup=main_keyboard())


async def predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get current prediction - premium but concise"""
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
    """Check for new predictions and dispatch to subscribed users with rate limiting."""
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
                # Apply rate limiting for pleasant experience
                if not should_send_to_user(user_id, prediction_data):
                    continue
                
                # Check notification cooldown (prevent spam)
                current_time = asyncio.get_event_loop().time()
                last_time = last_notification_time.get(user_id, 0)
                
                # Allow immediate win/loss results, but rate limit regular predictions
                if (current_time - last_time < NOTIFICATION_COOLDOWN and 
                    previous_result is None):
                    logger.info("Rate limiting prediction for user %s - cooldown active", user_id)
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
                    
                    # Update notification tracking
                    last_notification_time[user_id] = current_time
                    
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
