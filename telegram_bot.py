#!/usr/bin/env python3
"""
Telegram Bot for WinGo Predictions
Provides predictions via Telegram commands and automatic updates
"""

import asyncio
from html import escape
import logging
import os
import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes
from backend.telegram_card import render_forecast_card

try:
    from dotenv import load_dotenv
    project_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(project_dir, ".env"))
    load_dotenv(os.path.join(project_dir, "backend", ".env"))
except ImportError:
    pass

# Configuration
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
API_URL = os.environ.get("PREDICTION_API_URL", "http://localhost:8000/api/state")
METRICS_API_URL = os.environ.get(
    "PREDICTION_METRICS_API_URL", f"{API_URL.rsplit('/', 1)[0]}/metrics"
)
WINGO_API_URL = os.environ.get("WINGO_API_URL", "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json")
CHECK_INTERVAL = 5  # Check for new predictions every 5 seconds

# Store for subscribed users and last prediction
subscribed_users = set()
last_prediction_issue = None
last_prediction = None  # Store last prediction data for win/loss tracking

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


async def get_prediction():
    """Fetch prediction from local API"""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(API_URL)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API returned status {response.status_code}")
                return None
    except Exception as e:
        logger.error(f"Error fetching prediction: {e}")
        return None


async def get_metrics():
    """Fetch outcome-based intelligence metrics for the Telegram dashboard."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(METRICS_API_URL)
            if response.status_code == 200:
                return response.json()
            logger.error(f"Metrics API returned status {response.status_code}")
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
    return None


async def get_wingo_results():
    """Fetch recent results from WinGo API"""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(WINGO_API_URL)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"WinGo API returned status {response.status_code}")
                return None
    except Exception as e:
        logger.error(f"Error fetching WinGo results: {e}")
        return None


async def check_win_loss(previous_prediction):
    """Check if previous prediction was correct by fetching actual results"""
    if not previous_prediction:
        return None
    
    try:
        # Get the issue we predicted for
        predicted_issue = previous_prediction.get('nextIssue', 'N/A')
        predicted_result = previous_prediction.get('prediction', '').lower()
        
        # Fetch actual results from WinGo API
        wingo_data = await get_wingo_results()
        if not wingo_data:
            return None
        
        # Find the result for our predicted issue
        # WinGo API structure: data.list contains the results
        if 'data' in wingo_data and 'list' in wingo_data['data']:
            for result in wingo_data['data']['list']:
                issue_number = result.get('issueNumber', '')
                if issue_number == predicted_issue:
                    # Get the actual outcome
                    number = result.get('number', 0)
                    actual_result = 'big' if int(number) >= 5 else 'small'
                    
                    # Check if prediction was correct
                    won = (predicted_result == actual_result)
                    
                    return {
                        'won': won,
                        'issue': predicted_issue,
                        'predicted': predicted_result,
                        'actual': actual_result,
                        'number': number
                    }
        
        logger.warning(f"Could not find result for issue {predicted_issue}")
        return None
            
    except Exception as e:
        logger.error(f"Error checking win/loss: {e}")
        return None


def main_keyboard() -> InlineKeyboardMarkup:
    """Minimalist keyboard for extraordinary intelligence predictions."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔮 Next Prediction", callback_data="forecast")],
        [InlineKeyboardButton("📊 Accuracy Stats", callback_data="metrics")],
    ])


def back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("← Back", callback_data="forecast")]])


def premium_help_message() -> str:
    return """
<b>◈ EXTRAORDINARY INTELLIGENCE</b>
━━━━━━━━━━━━━━━━━━

<b>/predict</b>  Next AI forecast
<b>/stats</b>    Win rate & accuracy

The engine evolves daily from every outcome in the database.
Confidence is calibrated from resolved predictions only.

<i>Entertainment purposes. Outcomes may be random.</i>
    """.strip()


def _text(value, fallback="—") -> str:
    """Escape dynamic API data before placing it in Telegram HTML."""
    if value is None or value == "":
        return fallback
    return escape(str(value))


def _percentage(value) -> str:
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "—"


def format_prediction_message(data, previous_result=None):
    """Render a simple, powerful extraordinary intelligence forecast."""
    if not data:
        return "<b>◈ EXTRAORDINARY INTELLIGENCE</b>\n\n⚠️ <b>Analyzing market...</b>\nPrediction engine is recalibrating."
    
    try:
        prediction = _text(data.get('prediction'))
        confidence = _percentage(data.get('confidence'))
        target_num = _text(data.get('targetNum'))
        hedge_num = _text(data.get('hedgeNum'))
        next_issue = _text(data.get('nextIssue'))
        strike_quality = _text(data.get('strikeQuality', 'CONSERVATIVE'))
        evidence = data.get('evidence', {}) or {}
        
        side_emoji = "🔵" if prediction.lower() == "big" else "🟡" if prediction.lower() == "small" else "⚪"
        
        # Strike quality emojis
        strike_emoji = {
            "ULTIMATE_CONVICTION": "💎",
            "BEAST_CONVICTION": "🔥",
            "HIGH_CONVICTION": "⚡",
            "MODERATE_CONVICTION": "🎯",
            "CONSERVATIVE_SAFE": "🛡️",
            "HOLD_RISK_TOO_HIGH": "⚠️",
            "HOLD_INSUFFICIENT_DATA": "⏳",
        }.get(strike_quality, "🧿")

        # Handle HOLD scenarios
        if strike_quality in ["HOLD_RISK_TOO_HIGH", "HOLD_INSUFFICIENT_DATA"]:
            hold_reason = "Risk too high for 3-level safety" if strike_quality == "HOLD_RISK_TOO_HIGH" else "Collecting more data"
            return f"""<b>◈ EXTRAORDINARY INTELLIGENCE</b>
━━━━━━━━━━━━━━━━━━
⚠️ <b>HOLD - NO PREDICTION</b>

Reason: {hold_reason}
Strategy: Wait for >66% accuracy edge

<i>Patience creates profit.</i>"""

        # Calculate 3-level win probability
        p_win_in_3 = data.get('calibratedPWinIn3', evidence.get('joint3_probability', None))
        p3_pct = f"{float(p_win_in_3) * 100:.1f}%" if p_win_in_3 is not None else "—"

        # Add win/loss tracking if available
        result_info = ""
        if previous_result:
            result_emoji = "✅" if previous_result['won'] else "❌"
            result_text = "WON" if previous_result['won'] else "LOST"
            result_info = f"\n<b>LAST:</b> {result_emoji} {result_text}"

        message = f"""<b>◈ EXTRAORDINARY INTELLIGENCE</b>
━━━━━━━━━━━━━━━━━━

{side_emoji} <b>{prediction.upper()}</b>
<b>Confidence:</b> {confidence}
<b>Target:</b> {target_num} | <b>Hedge:</b> {hedge_num}

{strike_emoji} <b>{strike_quality.replace('_', ' ')}</b>
<b>P(Win in 3 levels):</b> {p3_pct}
<b>Next Issue:</b> <code>{next_issue}</code>
{result_info}

<i>Evolving daily from database outcomes</i>"""
        return message.strip()
    except Exception as e:
        logger.error(f"Error formatting message: {e}")
        return "<b>◈ EXTRAORDINARY INTELLIGENCE</b>\n\n⚠️ <b>Formatting error</b>"


def format_forecast_caption(data: dict, previous_result=None) -> str:
    """Minimal caption for the forecast image."""
    evidence = data.get("evidence") or {}
    strike = _text(data.get("strikeQuality", "CONSERVATIVE")).replace("_", " ")
    p3 = data.get("calibratedPWinIn3", evidence.get("joint3_probability", None))
    p3_txt = f"{float(p3) * 100:.1f}%" if p3 is not None else "—"
    prediction = _text(data.get('prediction')).upper()
    confidence = _percentage(data.get('confidence'))
    
    caption = f"""<b>◈ EXTRAORDINARY INTELLIGENCE</b>

{prediction} • {confidence}
P(Win in 3): {p3_txt} • Strike: {strike}

Next: <code>{_text(data.get('nextIssue'))}</code>"""
    if previous_result:
        outcome = "✅ WON" if previous_result.get("won") else "❌ LOST"
        caption += f"\nLast: {outcome}"
    return caption


async def reply_with_forecast(message, data: dict | None, previous_result=None):
    """Deliver a visual card, with a rich-text fallback during package upgrades."""
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


def format_metrics_message(metrics) -> str:
    """Simple accuracy stats display."""
    if not metrics:
        return "<b>◈ ACCURACY STATS</b>\n\n⚠️ Loading..."
    if metrics.get("resolved_predictions", 0) == 0:
        return """<b>◈ ACCURACY STATS</b>
━━━━━━━━━━━━━━━━━━
<b>Collecting data...</b>

Accuracy stats appear after predictions resolve."""
    
    accuracy = metrics.get("directional_accuracy")
    accuracy_text = f"{float(accuracy) * 100:.1f}%" if isinstance(accuracy, (int, float)) else "—"
    resolved = _text(metrics.get('resolved_predictions'))
    
    return f"""<b>◈ ACCURACY STATS</b>
━━━━━━━━━━━━━━━━━━

<b>Win Rate:</b> {accuracy_text}
<b>Sample Size:</b> {resolved}

<i>Based on resolved predictions only</i>"""


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    metrics = await get_metrics()
    await update.message.reply_text(
        format_metrics_message(metrics), parse_mode="HTML", reply_markup=main_keyboard()
    )


async def dashboard_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Serve dashboard controls without asking the user to type commands."""
    query = update.callback_query
    if not query:
        return
    await query.answer("Updating dashboard…")

    if query.data == "forecast":
        prediction = await get_prediction()
        await query.edit_message_text(
            format_prediction_message(prediction), parse_mode="HTML", reply_markup=main_keyboard()
        )
    elif query.data == "metrics":
        await query.edit_message_text(
            format_metrics_message(await get_metrics()), parse_mode="HTML", reply_markup=back_keyboard()
        )
    elif query.data == "status":
        prediction = await get_prediction()
        online = prediction is not None and not prediction.get("error")
        status = "🟢 <b>All systems online</b>" if online else "🔴 <b>Forecast API unavailable</b>"
        issue = _text(prediction.get("currentIssue")) if online else "—"
        text = f"""
<b>◈ SYSTEM STATUS</b>
━━━━━━━━━━━━━━━━━━
{status}

<b>Latest issue</b>  <code>{issue}</code>
<b>Runtime</b>       AWS EC2
<b>Delivery</b>      Telegram live updates
        """.strip()
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=back_keyboard())
    elif query.data == "learn":
        await query.edit_message_text(premium_help_message(), parse_mode="HTML", reply_markup=back_keyboard())
    elif query.data == "subscribe":
        subscribed_users.add(query.message.chat.id)
        await query.edit_message_text(
            "<b>◈ UPDATES ENABLED</b>\n\n🔔 You’ll receive each new live forecast automatically.",
            parse_mode="HTML", reply_markup=back_keyboard()
        )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome message with simple keyboard."""
    user_id = update.effective_chat.id
    subscribed_users.add(user_id)
    
    welcome_message = """<b>◈ EXTRAORDINARY INTELLIGENCE</b>
━━━━━━━━━━━━━━━━━━

Welcome! Your prediction engine is ready.

<b>/predict</b> - Next AI forecast
<b>/stats</b> - Win rate & accuracy

<i>Entertainment purposes only.</i>"""
    await update.message.reply_text(
        welcome_message.strip(), parse_mode='HTML', reply_markup=main_keyboard()
    )


async def predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /predict command"""
    prediction_data = await get_prediction()
    message = format_prediction_message(prediction_data)
    await update.message.reply_text(message, parse_mode='HTML', reply_markup=main_keyboard())


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command"""
    try:
        prediction_data = await get_prediction()
        if prediction_data:
            status = "✅ Online and functioning"
            last_issue = prediction_data.get('currentIssue', 'N/A')
            confidence = prediction_data.get('confidence', 0)
            message = f"""
<b>◈ SYSTEM STATUS</b>
━━━━━━━━━━━━━━━━━━
{status}

<b>Latest issue</b>  <code>{_text(last_issue)}</code>
<b>Confidence</b>    {_percentage(confidence)}
<b>Engine</b>        EVOSEQ evolving ensemble
<b>Runtime</b>       AWS EC2
            """
        else:
            status = "❌ API Error"
            message = f"""
<b>◈ SYSTEM STATUS</b>
━━━━━━━━━━━━━━━━━━
{status}

⚠️ Unable to connect to the prediction API.
            """
    except Exception as e:
        message = f"❌ Error: {str(e)}"
    
    await update.message.reply_text(message.strip(), parse_mode='HTML', reply_markup=main_keyboard())


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    help_message = """
🎰 <b>WinGo Prediction Bot Help</b>

<b>Available Commands:</b>
/start - Start the bot and see welcome message
/predict - Get current AI prediction
/status - Check bot and API status
/subscribe - Subscribe to automatic prediction updates
/unsubscribe - Unsubscribe from automatic updates
/help - Show this help message

<b>Features:</b>
• Real-time AI predictions
• Automatic updates when new predictions available
• Continuous day-by-day confidence learning
• Pattern analysis
• Multiple model ensemble

<b>About:</b>
Powered by EVOSEQ - Evolving Intelligence Engine
Uses an evolving ensemble plus outcome-based confidence calibration.
Every resolved database outcome improves the next forecast.

    ⚠️ <b>Disclaimer</b>: For entertainment purposes only.
    """
    await update.message.reply_text(
        premium_help_message(), parse_mode='HTML', reply_markup=main_keyboard()
    )


async def subscribe_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /subscribe command"""
    user_id = update.effective_chat.id
    subscribed_users.add(user_id)
    await update.message.reply_text(
        "🔔 <b>Updates enabled</b>\nYou’ll receive the next live forecast automatically.",
        parse_mode='HTML', reply_markup=main_keyboard()
    )


async def unsubscribe_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /unsubscribe command"""
    user_id = update.effective_chat.id
    if user_id in subscribed_users:
        subscribed_users.remove(user_id)
        await update.message.reply_text(
            "🔕 <b>Updates paused</b>\nUse /subscribe whenever you want live forecasts again.",
            parse_mode="HTML", reply_markup=main_keyboard()
        )
    else:
        await update.message.reply_text("🔕 Updates are already paused.", reply_markup=main_keyboard())


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors"""
    logger.error(f"Update {update} caused error {context.error}")
    if update and update.message:
        await update.message.reply_text("❌ An error occurred. Please try again.")


async def check_and_send_predictions(context: ContextTypes.DEFAULT_TYPE):
    """Check for new predictions and send to subscribed users"""
    global last_prediction_issue, last_prediction
    
    try:
        prediction_data = await get_prediction()
        if not prediction_data:
            return
        
        current_issue = prediction_data.get('currentIssue')
        next_issue = prediction_data.get('nextIssue')
        
        # Check if this is a new prediction (based on current issue changing)
        # When current_issue changes, it means a new round has started
        if current_issue and current_issue != last_prediction_issue:
            logger.info(f"New round detected: {current_issue}")
            last_prediction_issue = current_issue
            
            # Check win/loss for previous prediction
            previous_result = None
            if last_prediction:
                previous_result = await check_win_loss(last_prediction)
                if previous_result:
                    result_text = "WON" if previous_result['won'] else "LOST"
                    logger.info(f"Previous prediction {result_text}: {previous_result['issue']}")
            
            # Send prediction for the next issue
            message = format_prediction_message(prediction_data, previous_result)
            
            for user_id in subscribed_users.copy():  # Use copy to avoid modification during iteration
                try:
                    await context.bot.send_message(
                        chat_id=user_id,
                        text=message,
                        parse_mode='HTML',
                        reply_markup=main_keyboard(),
                    )
                    logger.info(f"Sent prediction to user {user_id}")
                except Exception as e:
                    logger.error(f"Failed to send to user {user_id}: {e}")
                    # Remove user if they blocked the bot or chat doesn't exist
                    subscribed_users.discard(user_id)
            
            # Store this prediction for next win/loss check
            last_prediction = prediction_data
                    
    except Exception as e:
        logger.error(f"Error in prediction check: {e}")


async def prediction_updater(bot):
    """Background task to check for new predictions periodically"""
    while True:
        try:
            # Create a mock context with the bot
            class MockContext:
                def __init__(self, bot):
                    self.bot = bot
            
            context = MockContext(bot)
            await check_and_send_predictions(context)
        except Exception as e:
            logger.error(f"Error in prediction updater: {e}")
        
        await asyncio.sleep(CHECK_INTERVAL)


async def main():
    """Start the bot"""
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required. Put it in your environment, not source code.")
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("predict", predict_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("subscribe", subscribe_command))
    application.add_handler(CommandHandler("unsubscribe", unsubscribe_command))
    application.add_handler(CallbackQueryHandler(dashboard_callback))
    
    # Add error handler
    application.add_error_handler(error_handler)
    
    # Start the bot
    logger.info("Starting WinGo Prediction Bot...")
    
    # Send immediate prediction on startup
    try:
        initial_prediction = await get_prediction()
        if initial_prediction:
            logger.info(f"Initial prediction fetched: {initial_prediction.get('currentIssue')}")
    except Exception as e:
        logger.error(f"Error fetching initial prediction: {e}")
    
    # Run the bot before scheduling network work.  This avoids a race where
    # the background updater tries to use an uninitialized Telegram client.
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    updater_task = asyncio.create_task(prediction_updater(application.bot))
    
    # Keep the bot running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
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
