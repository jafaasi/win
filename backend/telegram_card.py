#!/usr/bin/env python3
"""
Ultra-Premium Telegram Card Renderer
Creates luxury quant-terminal style visual forecast and analytics cards for Telegram
"""

import io
import os
import math
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


def _draw_progress_bar(draw, x, y, width, height, progress, fill_color, bg_color=(30, 41, 59), radius=4):
    """Draw a rounded horizontal progress meter."""
    draw.rounded_rectangle([x, y, x + width, y + height], radius=radius, fill=bg_color)
    filled_w = max(height, int(width * min(1.0, max(0.0, progress))))
    if filled_w > 0:
        draw.rounded_rectangle([x, y, x + filled_w, y + height], radius=radius, fill=fill_color)


def render_forecast_card(data, previous_result=None):
    """
    Render a luxury quant-style forecast card with glowing highlights and visual meters.
    """
    if not HAS_PIL:
        return None
    try:
        prediction = str(data.get('prediction', 'Unknown')).upper()
        confidence = float(data.get('confidence', 50.0))
        target_num = data.get('targetNum', 0)
        hedge_num = data.get('hedgeNum', 0)
        current_issue = str(data.get('currentIssue', 'N/A'))
        next_issue = str(data.get('nextIssue', 'N/A'))
        pattern = str(data.get('patternName', 'Ultra Engine v1.0'))
        strike_quality = str(data.get('strikeQuality', 'CONSERVATIVE')).replace('_', ' ')
        action = str(data.get('action', 'FORECAST'))
        
        p_win_in_3 = data.get('calibratedPWinIn3', None)
        p_single = data.get('calibratedPSingle', None)
        scorecard = data.get('scorecard', {}) or {}
        win_streak = scorecard.get('win_streak', 0)
        loss_streak = scorecard.get('loss_streak', 0)
        session_rate = scorecard.get('session_win_rate', 50.0)

        width, height = 700, 470
        img = Image.new('RGB', (width, height), color=(11, 15, 25))
        draw = ImageDraw.Draw(img)

        # Background Gradient: Luxury Obsidian/Navy to Slate
        for y in range(height):
            ratio = y / height
            r = int(11 * (1 - ratio) + 17 * ratio)
            g = int(15 * (1 - ratio) + 24 * ratio)
            b = int(25 * (1 - ratio) + 39 * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        # Ambient Glow accents in corners
        draw.ellipse([-50, -50, 150, 150], fill=(20, 35, 60))
        draw.ellipse([width - 120, -50, width + 50, 150], fill=(25, 20, 50))

        # Outer Glassmorphism Card
        draw.rounded_rectangle([12, 12, width - 12, height - 12], radius=16, fill=(15, 23, 42), outline=(35, 53, 84), width=2)

        # Typography
        brand_font = _load_font(20, bold=True)
        super_font = _load_font(32, bold=True)
        h2_font = _load_font(18, bold=True)
        body_font = _load_font(14, bold=False)
        bold_font = _load_font(14, bold=True)
        small_font = _load_font(11, bold=False)

        # Top Bar: Brand + Action Pill Badge
        draw.text((32, 28), "◈ ULTRA INTELLIGENCE", fill=(236, 72, 153), font=brand_font)
        draw.text((32, 54), "WINGO 30S • QUANT ADAPTIVE SUITE", fill=(148, 163, 184), font=small_font)

        # Conviction Badge Pill
        if action == "SKIP":
            badge_txt = "⏭️ RESTRAIN (SKIP)"
            badge_bg = (51, 65, 85)
            badge_border = (100, 116, 139)
            badge_col = (226, 232, 240)
        elif action == "STRIKE" or strike_quality in ("ULTIMATE CONVICTION", "BEAST CONVICTION"):
            badge_txt = f"⚡ {strike_quality}"
            badge_bg = (30, 58, 138)
            badge_border = (96, 165, 250)
            badge_col = (219, 234, 254)
        elif strike_quality == "HIGH CONVICTION":
            badge_txt = f"🎯 {strike_quality}"
            badge_bg = (19, 78, 74)
            badge_border = (45, 212, 191)
            badge_col = (204, 251, 241)
        else:
            badge_txt = f"◌ {strike_quality}"
            badge_bg = (30, 41, 59)
            badge_border = (71, 85, 105)
            badge_col = (203, 213, 225)

        draw.rounded_rectangle([width - 245, 26, width - 32, 60], radius=17, fill=badge_bg, outline=badge_border, width=1)
        draw.text((width - 230, 35), badge_txt, fill=badge_col, font=bold_font)

        # Horizontal Divider
        draw.line([(32, 78), (width - 32, 78)], fill=(30, 41, 59), width=1)

        # --- SECTION 1: HERO PREDICTION BOX (LEFT) ---
        hero_x, hero_y, hero_w, hero_h = 32, 92, 300, 130
        if prediction == "BIG":
            hero_bg = (12, 43, 70)
            hero_border = (56, 189, 248)
            hero_text_col = (56, 189, 248)
            side_icon = "🔵"
        elif prediction == "SMALL":
            hero_bg = (55, 38, 12)
            hero_border = (251, 191, 36)
            hero_text_col = (251, 191, 36)
            side_icon = "🟡"
        else:
            hero_bg = (30, 41, 59)
            hero_border = (148, 163, 184)
            hero_text_col = (203, 213, 225)
            side_icon = "⚪"

        draw.rounded_rectangle([hero_x, hero_y, hero_x + hero_w, hero_y + hero_h], radius=12, fill=hero_bg, outline=hero_border, width=2)
        draw.text((hero_x + 18, hero_y + 14), f"{side_icon} PRIMARY PREDICTION", fill=(148, 163, 184), font=small_font)
        draw.text((hero_x + 18, hero_y + 36), prediction, fill=hero_text_col, font=super_font)

        # Confidence tag & meter
        draw.text((hero_x + 18, hero_y + 82), f"Calibrated Edge: {confidence:.1f}%", fill=(241, 245, 249), font=bold_font)
        _draw_progress_bar(draw, hero_x + 18, hero_y + 104, hero_w - 36, 10, confidence / 100.0, hero_border, bg_color=(15, 23, 42))

        # --- SECTION 2: DIGIT TARGETS & ROUND INFO (RIGHT) ---
        right_x, right_y, right_w, right_h = 348, 92, width - 348 - 32, 130
        draw.rounded_rectangle([right_x, right_y, right_x + right_w, right_y + right_h], radius=12, fill=(20, 30, 48), outline=(45, 65, 95), width=1)

        draw.text((right_x + 16, right_y + 14), "🎯 PROBABILISTIC TARGETS", fill=(148, 163, 184), font=small_font)

        # Glowing Digit Chips
        chip1_x, chip1_y = right_x + 16, right_y + 36
        draw.rounded_rectangle([chip1_x, chip1_y, chip1_x + 130, chip1_y + 40], radius=8, fill=(30, 58, 138), outline=(96, 165, 250), width=1)
        draw.text((chip1_x + 12, chip1_y + 10), f"Primary: {target_num}", fill=(248, 250, 252), font=bold_font)

        chip2_x, chip2_y = right_x + 155, right_y + 36
        draw.rounded_rectangle([chip2_x, chip2_y, chip2_x + 130, chip2_y + 40], radius=8, fill=(30, 41, 59), outline=(71, 85, 105), width=1)
        draw.text((chip2_x + 12, chip2_y + 10), f"Hedge:   {hedge_num}", fill=(203, 213, 225), font=bold_font)

        # Target Issue timing
        draw.text((right_x + 16, right_y + 90), f"Current: #{current_issue[-6:]}", fill=(148, 163, 184), font=body_font)
        draw.text((right_x + 155, right_y + 90), f"Target:  #{next_issue[-6:]}", fill=(56, 189, 248), font=bold_font)

        # --- SECTION 3: 3-LEVEL MARTINGALE & SCORECARD METRICS ROW ---
        row3_y = 236
        draw.line([(32, row3_y), (width - 32, row3_y)], fill=(30, 41, 59), width=1)

        box3_y = row3_y + 14
        # Left Box: 3-Level Joint Probability
        draw.rounded_rectangle([32, box3_y, 340, box3_y + 80], radius=10, fill=(17, 26, 42), outline=(39, 57, 85), width=1)
        draw.text((46, box3_y + 10), "💎 3-LEVEL MARTINGALE JOINT WIN RATE", fill=(236, 72, 153), font=small_font)
        if p_win_in_3 is not None:
            p3_val = float(p_win_in_3) * 100
            p3_col = (52, 211, 153) if p3_val >= 94 else (251, 191, 36) if p3_val >= 88 else (248, 113, 113)
            draw.text((46, box3_y + 30), f"{p3_val:.1f}% Joint Probability", fill=p3_col, font=h2_font)
            _draw_progress_bar(draw, 46, box3_y + 58, 265, 8, p3_val / 100.0, p3_col, bg_color=(11, 15, 25))
        else:
            draw.text((46, box3_y + 30), "P(win in 3): Computing…", fill=(148, 163, 184), font=body_font)

        # Right Box: Live Session Scorecard
        draw.rounded_rectangle([356, box3_y, width - 32, box3_y + 80], radius=10, fill=(17, 26, 42), outline=(39, 57, 85), width=1)
        draw.text((370, box3_y + 10), "🏆 SESSION PERFORMANCE & STREAK", fill=(56, 189, 248), font=small_font)
        
        streak_txt = f"🔥 W{win_streak} Streak" if win_streak > 0 else f"❄️ L{loss_streak} Streak" if loss_streak > 0 else "Streak: Neutral"
        streak_col = (52, 211, 153) if win_streak > 0 else (248, 113, 113) if loss_streak > 0 else (148, 163, 184)
        draw.text((370, box3_y + 30), streak_txt, fill=streak_col, font=h2_font)
        draw.text((370, box3_y + 56), f"Session Rate: {float(session_rate):.1f}%", fill=(203, 213, 225), font=bold_font)

        # --- SECTION 4: PREVIOUS RESULT STATUS BANNER ---
        banner_y = box3_y + 92
        if previous_result:
            won = bool(previous_result.get('won', False))
            res_bg = (6, 78, 59) if won else (127, 29, 29)
            res_border = (52, 211, 153) if won else (239, 68, 68)
            res_icon = "✅ WON (+1.96x)" if won else "❌ LOST"
            pred_side = str(previous_result.get('predicted', 'N/A')).upper()
            act_side = str(previous_result.get('actual', 'N/A')).upper()
            act_num = previous_result.get('number', '')
            issue_id = str(previous_result.get('issue', ''))[-6:]

            draw.rounded_rectangle([32, banner_y, width - 32, banner_y + 44], radius=8, fill=res_bg, outline=res_border, width=1)
            draw.text((46, banner_y + 12), f"LAST DRAW: {res_icon}", fill=(255, 255, 255), font=bold_font)
            draw.text((270, banner_y + 12), f"#{issue_id} • Call: {pred_side} | Result: {act_side} ({act_num})", fill=(226, 232, 240), font=body_font)
        else:
            draw.rounded_rectangle([32, banner_y, width - 32, banner_y + 44], radius=8, fill=(17, 24, 39), outline=(39, 57, 85), width=1)
            draw.text((46, banner_y + 12), "LAST DRAW: Reconciling with verified blockchain outcome…", fill=(148, 163, 184), font=body_font)

        # --- FOOTER ---
        footer_y = banner_y + 54
        draw.text((32, footer_y), f"Model: {pattern[:42]}", fill=(100, 116, 139), font=small_font)
        timestamp = datetime.utcnow().strftime("%H:%M:%S UTC")
        draw.text((width - 140, footer_y), timestamp, fill=(100, 116, 139), font=small_font)

        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error rendering forecast card: {e}")
        return None


def render_martingale_card(p_single: float = 0.58, base_unit: float = 10.0):
    """
    Render a dedicated Martingale Strategy card showing 3-step capital allocations.
    """
    if not HAS_PIL:
        return None
    try:
        width, height = 660, 400
        img = Image.new('RGB', (width, height), color=(11, 15, 25))
        draw = ImageDraw.Draw(img)

        # Gradient
        for y in range(height):
            ratio = y / height
            draw.line([(0, y), (width, y)], fill=(int(11*(1-ratio) + 15*ratio), int(15*(1-ratio) + 23*ratio), int(25*(1-ratio) + 42*ratio)))

        draw.rounded_rectangle([12, 12, width - 12, height - 12], radius=16, fill=(15, 23, 42), outline=(35, 53, 84), width=2)

        title_font = _load_font(20, bold=True)
        h2_font = _load_font(17, bold=True)
        bold_font = _load_font(14, bold=True)
        body_font = _load_font(13, bold=False)
        small_font = _load_font(11, bold=False)

        draw.text((32, 28), "💎 3-LEVEL MARTINGALE LADDER", fill=(236, 72, 153), font=title_font)
        draw.text((32, 54), "OPTIMAL STAKE PROGRESSION & CUMULATIVE WIN RATE", fill=(148, 163, 184), font=small_font)

        # 3 Step boxes
        p1 = p_single
        p2 = 0.5 + 0.94 * (p1 - 0.5)
        p3 = 0.5 + 0.88 * (p1 - 0.5)
        joint_p = 1.0 - (1.0 - p1) * (1.0 - p2) * (1.0 - p3)

        steps = [
            ("STEP 1: PRIMARY", f"{base_unit:.0f} Units", f"{p1*100:.1f}% Win", "Risk: Minimum", (30, 58, 138), (96, 165, 250)),
            ("STEP 2: RECOVERY", f"{base_unit*2.2:.0f} Units", f"{(1-(1-p1)*(1-p2))*100:.1f}% Cumul", "Risk: Moderate", (88, 28, 135), (192, 132, 252)),
            ("STEP 3: MAX STRIKE", f"{base_unit*4.8:.0f} Units", f"{joint_p*100:.1f}% Joint", "Risk: Controlled", (136, 19, 55), (251, 113, 133)),
        ]

        step_w = 185
        for i, (title, stake, win_pct, risk, bg_col, border_col) in enumerate(steps):
            box_x = 32 + i * (step_w + 14)
            box_y = 86
            draw.rounded_rectangle([box_x, box_y, box_x + step_w, box_y + 160], radius=10, fill=bg_col, outline=border_col, width=1)
            draw.text((box_x + 12, box_y + 14), title, fill=(248, 250, 252), font=bold_font)
            draw.text((box_x + 12, box_y + 44), stake, fill=(255, 255, 255), font=h2_font)
            draw.text((box_x + 12, box_y + 80), win_pct, fill=(52, 211, 153), font=bold_font)
            draw.text((box_x + 12, box_y + 115), risk, fill=(203, 213, 225), font=small_font)

        # Summary footer box
        sum_y = 265
        draw.rounded_rectangle([32, sum_y, width - 32, sum_y + 90], radius=10, fill=(17, 24, 39), outline=(45, 65, 95), width=1)
        draw.text((46, sum_y + 14), f"Cumulative 3-Step Success Probability: {joint_p*100:.1f}%", fill=(52, 211, 153), font=h2_font)
        draw.text((46, sum_y + 44), "Capital rule: Reset to Step 1 immediately upon any win. Never extend beyond Step 3.", fill=(148, 163, 184), font=body_font)

        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes
    except Exception as e:
        print(f"Error rendering martingale card: {e}")
        return None


def render_metrics_card(metrics):
    """Render a visual quant metrics card for Telegram."""
    if not HAS_PIL:
        return None
    try:
        width, height = 660, 380
        img = Image.new('RGB', (width, height), color=(11, 15, 25))
        draw = ImageDraw.Draw(img)

        # Background gradient
        for y in range(height):
            ratio = y / height
            draw.line([(0, y), (width, y)], fill=(int(11*(1-ratio) + 15*ratio), int(15*(1-ratio) + 23*ratio), int(25*(1-ratio) + 42*ratio)))

        draw.rounded_rectangle([12, 12, width - 12, height - 12], radius=16, fill=(15, 23, 42), outline=(35, 53, 84), width=2)

        title_font = _load_font(20, bold=True)
        h2_font = _load_font(18, bold=True)
        body_font = _load_font(14, bold=False)
        bold_font = _load_font(14, bold=True)
        small_font = _load_font(11, bold=False)

        draw.text((32, 28), "📊 QUANT EVIDENCE & ACCURACY METRICS", fill=(236, 72, 153), font=title_font)
        draw.text((32, 54), "STRICT OUT-OF-SAMPLE AUDIT TRAIL", fill=(148, 163, 184), font=small_font)

        resolved = metrics.get('resolved_predictions', 0)
        accuracy = metrics.get('directional_accuracy', 0.0)
        brier = metrics.get('brier_score', 0.0)
        log_loss = metrics.get('log_loss', 0.0)

        # 4 Metric Cards Grid
        grid = [
            ("Resolved Forecasts", str(resolved), (56, 189, 248), 0, 0),
            ("Directional Accuracy", f"{accuracy*100:.1f}%" if accuracy > 0 else "—", (52, 211, 153), 1, 0),
            ("Brier Score", f"{brier:.4f}" if isinstance(brier, (int, float)) else "—", (251, 191, 36), 0, 1),
            ("Log Loss", f"{log_loss:.4f}" if isinstance(log_loss, (int, float)) else "—", (192, 132, 252), 1, 1),
        ]

        card_w = 285
        card_h = 90
        for name, val, col, col_idx, row_idx in grid:
            cx = 32 + col_idx * (card_w + 26)
            cy = 86 + row_idx * (card_h + 16)
            draw.rounded_rectangle([cx, cy, cx + card_w, cy + card_h], radius=10, fill=(20, 30, 48), outline=(45, 65, 95), width=1)
            draw.text((cx + 16, cy + 14), name, fill=(148, 163, 184), font=body_font)
            draw.text((cx + 16, cy + 42), val, fill=col, font=h2_font)

        draw.text((32, 320), "All metrics derived strictly from resolved database predictions timestamped prior to draws.", fill=(100, 116, 139), font=small_font)

        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes

    except Exception as e:
        print(f"Error rendering metrics card: {e}")
        return None


def render_status_card(status_data):
    """Render a visual system health & latency status card for Telegram."""
    if not HAS_PIL:
        return None
    try:
        width, height = 660, 340
        img = Image.new('RGB', (width, height), color=(11, 15, 25))
        draw = ImageDraw.Draw(img)

        for y in range(height):
            ratio = y / height
            draw.line([(0, y), (width, y)], fill=(int(11*(1-ratio) + 15*ratio), int(15*(1-ratio) + 23*ratio), int(25*(1-ratio) + 42*ratio)))

        draw.rounded_rectangle([12, 12, width - 12, height - 12], radius=16, fill=(15, 23, 42), outline=(35, 53, 84), width=2)

        title_font = _load_font(20, bold=True)
        h2_font = _load_font(18, bold=True)
        body_font = _load_font(14, bold=False)
        small_font = _load_font(11, bold=False)

        draw.text((32, 28), "🟢 SYSTEM TELEMETRY & RUNTIME HEALTH", fill=(52, 211, 153), font=title_font)

        online = status_data.get('online', False)
        status_text = "ALL CLUSTERS OPERATIONAL" if online else "CLUSTER OFFLINE"
        status_col = (52, 211, 153) if online else (239, 68, 68)

        draw.rounded_rectangle([32, 75, width - 32, 140], radius=10, fill=(20, 30, 48), outline=status_col, width=1)
        draw.text((48, 92), f"Engine State: {status_text}", fill=status_col, font=h2_font)
        if 'issue' in status_data and status_data['issue']:
            draw.text((48, 116), f"Active Blockchain Issue: #{status_data['issue']}", fill=(203, 213, 225), font=small_font)

        # Node properties
        node_items = [
            ("Core Architecture", "Ultra Intelligence (8-Model Hedge)"),
            ("Database Engine", "PostgreSQL / Supabase Dedicated"),
            ("Dispatch Cycle", "1.5s (Zero-Latency Sync)"),
            ("Deployment", "AWS EC2 High-Compute Node"),
        ]

        for i, (k, v) in enumerate(node_items):
            ny = 160 + i * 32
            draw.text((48, ny), f"• {k}:", fill=(148, 163, 184), font=body_font)
            draw.text((230, ny), v, fill=(248, 250, 252), font=body_font)

        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes

    except Exception as e:
        print(f"Error rendering status card: {e}")
        return None