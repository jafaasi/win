#!/usr/bin/env python3
"""
Telegram Card Renderer
Creates high-definition, visual forecast cards for Telegram predictions
"""

import io
import os
from datetime import datetime

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def _load_font(size: int, bold: bool = False):
    """Attempt loading system TTF fonts across Linux, macOS, or fall back to default."""
    font_candidates = []
    if bold:
        font_candidates = [
            "/System/Library/Fonts/SFProText-Bold.ttf",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
    else:
        font_candidates = [
            "/System/Library/Fonts/SFProText-Regular.ttf",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    for path in font_candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_forecast_card(data, previous_result=None):
    """
    Render a visual forecast card for Telegram
    
    Args:
        data: Prediction data dictionary
        previous_result: Previous prediction result for win/loss display
        
    Returns:
        BytesIO object containing the card image, or None if rendering fails
    """
    if not HAS_PIL:
        return None
    try:
        # Extract prediction data
        prediction = data.get('prediction', 'Unknown')
        confidence = float(data.get('confidence', 0))
        target_num = data.get('targetNum', 0)
        hedge_num = data.get('hedgeNum', 0)
        current_issue = str(data.get('currentIssue', 'N/A'))
        next_issue = str(data.get('nextIssue', 'N/A'))
        pattern = str(data.get('patternName', 'ULTRA v1.0'))
        strike_quality = str(data.get('strikeQuality', 'CONSERVATIVE')).replace('_', ' ')
        action = str(data.get('action', 'FORECAST'))
        
        # Calibration & exploit stats
        p_win_in_3 = data.get('calibratedPWinIn3', None)
        scorecard = data.get('scorecard', {}) or {}
        win_streak = scorecard.get('win_streak', 0)
        loss_streak = scorecard.get('loss_streak', 0)
        session_rate = scorecard.get('session_win_rate', None)

        width, height = 640, 440
        img = Image.new('RGB', (width, height), color=(15, 23, 42))
        draw = ImageDraw.Draw(img)

        # Draw smooth vertical dark gradient
        for y in range(height):
            ratio = y / height
            r = int(15 * (1 - ratio) + 10 * ratio)
            g = int(23 * (1 - ratio) + 15 * ratio)
            b = int(42 * (1 - ratio) + 30 * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        # Main inner container
        card_bg = (22, 33, 62)
        border_color = (41, 74, 110)
        draw.rounded_rectangle([16, 16, width - 16, height - 16], radius=12, fill=card_bg, outline=border_color, width=2)

        # Fonts
        title_font = _load_font(22, bold=True)
        h2_font = _load_font(18, bold=True)
        big_font = _load_font(26, bold=True)
        normal_font = _load_font(15, bold=False)
        bold_font = _load_font(15, bold=True)
        small_font = _load_font(12, bold=False)

        # Header Title
        draw.text((36, 32), "◈ ULTRA INTELLIGENCE", fill=(233, 69, 96), font=title_font)
        draw.text((36, 58), "WINGO 30S • OUTCOME-CALIBRATED ENGINE", fill=(148, 163, 184), font=small_font)

        # Badge: Strike / Action
        if action == "SKIP":
            badge_text = "⏭️ SKIP / NO EDGE"
            badge_bg = (55, 65, 81)
            badge_color = (209, 213, 219)
        elif action == "STRIKE":
            badge_text = f"⚡ {strike_quality}"
            badge_bg = (30, 64, 175)
            badge_color = (191, 219, 254)
        else:
            badge_text = f"🎯 {strike_quality}"
            badge_bg = (17, 24, 39)
            badge_color = (226, 232, 240)

        draw.rounded_rectangle([width - 240, 32, width - 36, 62], radius=6, fill=badge_bg, outline=(75, 85, 99), width=1)
        draw.text((width - 225, 38), badge_text, fill=badge_color, font=bold_font)

        # Horizontal separator
        draw.line([(36, 80), (width - 36, 80)], fill=(30, 41, 59), width=1)

        # Prediction Display Box
        pred_box_x = 36
        pred_box_y = 96
        pred_box_w = 260
        pred_box_h = 100

        pred_lower = prediction.lower()
        if pred_lower == 'big':
            pred_color = (56, 189, 248)  # Cyan/Sky
            pred_box_bg = (12, 45, 72)
        elif pred_lower == 'small':
            pred_color = (251, 191, 36)  # Amber
            pred_box_bg = (60, 45, 15)
        else:
            pred_color = (203, 213, 225)
            pred_box_bg = (30, 41, 59)

        draw.rounded_rectangle([pred_box_x, pred_box_y, pred_box_x + pred_box_w, pred_box_y + pred_box_h],
                               radius=8, fill=pred_box_bg, outline=pred_color, width=1)
        draw.text((pred_box_x + 16, pred_box_y + 14), "PRIMARY CALL", fill=(148, 163, 184), font=small_font)
        draw.text((pred_box_x + 16, pred_box_y + 36), prediction.upper(), fill=pred_color, font=big_font)
        draw.text((pred_box_x + 16, pred_box_y + 70), f"Conf: {confidence:.1f}%", fill=(241, 245, 249), font=bold_font)

        # Targets Box
        tgt_box_x = 312
        tgt_box_y = 96
        tgt_box_w = width - 36 - tgt_box_x
        tgt_box_h = 100

        draw.rounded_rectangle([tgt_box_x, tgt_box_y, tgt_box_x + tgt_box_w, tgt_box_y + tgt_box_h],
                               radius=8, fill=(15, 23, 42), outline=(51, 65, 85), width=1)
        draw.text((tgt_box_x + 16, tgt_box_y + 14), "DIGIT TARGETS", fill=(148, 163, 184), font=small_font)
        draw.text((tgt_box_x + 16, tgt_box_y + 40), f"🎯 Target: {target_num}", fill=(248, 250, 252), font=h2_font)
        draw.text((tgt_box_x + 16, tgt_box_y + 68), f"🛡️ Hedge: {hedge_num}", fill=(148, 163, 184), font=normal_font)

        # 3-Level Martingale & Metrics row
        row2_y = 212
        draw.text((36, row2_y), "STRATEGY (3-LEVEL)", fill=(233, 69, 96), font=bold_font)
        if p_win_in_3 is not None:
            p3_val = float(p_win_in_3) * 100
            p3_color = (52, 211, 153) if p3_val >= 94 else (251, 191, 36) if p3_val >= 88 else (248, 113, 113)
            draw.text((36, row2_y + 24), f"P(win in 3): {p3_val:.1f}%", fill=p3_color, font=bold_font)
        else:
            draw.text((36, row2_y + 24), "P(win in 3): —", fill=(148, 163, 184), font=normal_font)

        # Rounds
        draw.text((240, row2_y), "ISSUE", fill=(148, 163, 184), font=small_font)
        draw.text((240, row2_y + 20), f"Current: {current_issue[-6:]}", fill=(203, 213, 225), font=normal_font)
        draw.text((240, row2_y + 42), f"Target:  #{next_issue[-6:]}", fill=(56, 189, 248), font=bold_font)

        # Scorecard / Session
        draw.text((440, row2_y), "SCORECARD", fill=(148, 163, 184), font=small_font)
        streak_str = f"Streak: W{win_streak}" if win_streak > 0 else f"Streak: L{loss_streak}" if loss_streak > 0 else "Streak: —"
        streak_col = (52, 211, 153) if win_streak > 0 else (248, 113, 113) if loss_streak > 0 else (148, 163, 184)
        draw.text((440, row2_y + 20), streak_str, fill=streak_col, font=bold_font)
        if session_rate is not None:
            draw.text((440, row2_y + 42), f"Session: {float(session_rate):.1f}%", fill=(203, 213, 225), font=normal_font)

        # Horizontal separator
        draw.line([(36, 280), (width - 36, 280)], fill=(30, 41, 59), width=1)

        # Last Outcome Banner
        banner_y = 296
        if previous_result:
            won = previous_result.get('won', False)
            res_bg = (6, 78, 59) if won else (127, 29, 29)
            res_border = (52, 211, 153) if won else (239, 68, 68)
            res_icon = "✅ WON" if won else "❌ LOST"
            pred_side = str(previous_result.get('predicted', 'N/A')).upper()
            act_side = str(previous_result.get('actual', 'N/A')).upper()
            act_num = previous_result.get('number', '')
            issue_id = str(previous_result.get('issue', ''))[-6:]

            draw.rounded_rectangle([36, banner_y, width - 36, banner_y + 48], radius=6, fill=res_bg, outline=res_border, width=1)
            draw.text((50, banner_y + 14), f"PREV RESULT: {res_icon}", fill=(255, 255, 255), font=bold_font)
            draw.text((250, banner_y + 14), f"Issue #{issue_id} • Pred {pred_side} | Act {act_side} ({act_num})", fill=(226, 232, 240), font=normal_font)
        else:
            draw.rounded_rectangle([36, banner_y, width - 36, banner_y + 48], radius=6, fill=(15, 23, 42), outline=(51, 65, 85), width=1)
            draw.text((50, banner_y + 14), "PREV RESULT: Reconciling with live draw…", fill=(148, 163, 184), font=normal_font)

        # Footer
        footer_y = 368
        draw.text((36, footer_y), f"Model: {pattern[:35]}", fill=(100, 116, 139), font=small_font)
        timestamp = datetime.utcnow().strftime("%H:%M:%S UTC")
        draw.text((width - 150, footer_y), timestamp, fill=(100, 116, 139), font=small_font)

        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error rendering forecast card: {e}")
        return None


def render_metrics_card(metrics):
    """Render a visual metrics card for Telegram."""
    if not HAS_PIL:
        return None
    try:
        width, height = 600, 360
        img = Image.new('RGB', (width, height), color=(15, 23, 42))
        draw = ImageDraw.Draw(img)

        # Background gradient
        for y in range(height):
            ratio = y / height
            draw.line([(0, y), (width, y)], fill=(int(15 * (1-ratio) + 8*ratio),
                                                  int(23 * (1-ratio) + 12*ratio),
                                                  int(42 * (1-ratio) + 24*ratio)))

        draw.rectangle([16, 16, width - 16, height - 16], fill=(22, 33, 62), outline=(41, 74, 110), width=2)

        title_font = _load_font(22, bold=True)
        data_font = _load_font(18, bold=True)
        small_font = _load_font(14, bold=False)

        draw.text((36, 36), "◈ LIVE ACCURACY & EVIDENCE", fill=(233, 69, 96), font=title_font)

        resolved = metrics.get('resolved_predictions', 0)
        accuracy = metrics.get('directional_accuracy', 0)
        brier = metrics.get('brier_score', 0)
        log_loss = metrics.get('log_loss', 0)

        draw.text((36, 90), f"RESOLVED FORECASTS: {resolved}", fill=(255, 255, 255), font=data_font)

        if accuracy > 0:
            acc_color = (52, 211, 153) if accuracy >= 0.6 else (251, 191, 36) if accuracy >= 0.52 else (239, 68, 68)
            draw.text((36, 130), f"DIRECTIONAL ACCURACY: {accuracy*100:.1f}%", fill=acc_color, font=data_font)

        draw.text((36, 170), f"BRIER SCORE: {brier:.4f}" if isinstance(brier, (int, float)) else "BRIER SCORE: —",
                  fill=(148, 163, 184), font=data_font)
        draw.text((36, 210), f"LOG LOSS: {log_loss:.4f}" if isinstance(log_loss, (int, float)) else "LOG LOSS: —",
                  fill=(148, 163, 184), font=data_font)

        draw.text((36, 290), "Calibrated strictly on resolved database outcomes.", fill=(100, 116, 139), font=small_font)

        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes

    except Exception as e:
        print(f"Error rendering metrics card: {e}")
        return None


def render_status_card(status_data):
    """Render a visual status card for Telegram."""
    if not HAS_PIL:
        return None
    try:
        width, height = 600, 320
        img = Image.new('RGB', (width, height), color=(15, 23, 42))
        draw = ImageDraw.Draw(img)

        # Background gradient
        for y in range(height):
            ratio = y / height
            draw.line([(0, y), (width, y)], fill=(int(15 * (1-ratio) + 8*ratio),
                                                  int(23 * (1-ratio) + 12*ratio),
                                                  int(42 * (1-ratio) + 24*ratio)))

        draw.rectangle([16, 16, width - 16, height - 16], fill=(22, 33, 62), outline=(41, 74, 110), width=2)

        title_font = _load_font(22, bold=True)
        data_font = _load_font(18, bold=True)
        small_font = _load_font(14, bold=False)

        draw.text((36, 36), "◈ SYSTEM STATUS", fill=(233, 69, 96), font=title_font)

        online = status_data.get('online', False)
        status_text = "🟢 ENGINE ACTIVE & SYNCED" if online else "🔴 ENGINE OFFLINE"
        status_color = (52, 211, 153) if online else (239, 68, 68)
        draw.text((36, 90), status_text, fill=status_color, font=data_font)

        if 'issue' in status_data:
            draw.text((36, 140), f"LATEST ISSUE: #{status_data['issue']}", fill=(255, 255, 255), font=data_font)

        draw.text((36, 190), "ENGINE: Ultra Intelligence v1.0 (8-model Hedge)", fill=(148, 163, 184), font=small_font)
        draw.text((36, 220), "STORAGE: PostgreSQL / Supabase Direct", fill=(148, 163, 184), font=small_font)
        draw.text((36, 250), "DISPATCH: Cycle-Synchronized Telegram Stream", fill=(148, 163, 184), font=small_font)

        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes

    except Exception as e:
        print(f"Error rendering status card: {e}")
        return None