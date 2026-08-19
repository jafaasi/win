#!/usr/bin/env python3
"""
Telegram Card Renderer
Creates visual forecast cards for Telegram predictions
"""

import io
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import os

def render_forecast_card(data, previous_result=None):
    """
    Render a visual forecast card for Telegram
    
    Args:
        data: Prediction data dictionary
        previous_result: Previous prediction result for win/loss display
        
    Returns:
        BytesIO object containing the card image, or None if rendering fails
    """
    try:
        # Extract prediction data
        prediction = data.get('prediction', 'Unknown')
        confidence = data.get('confidence', 0)
        target_num = data.get('targetNum', 0)
        hedge_num = data.get('hedgeNum', 0)
        current_issue = data.get('currentIssue', 'N/A')
        next_issue = data.get('nextIssue', 'N/A')
        pattern = data.get('patternName', 'EVOSEQ')
        
        # Create image
        width, height = 600, 400
        img = Image.new('RGB', (width, height), color='#1a1a2e')
        draw = ImageDraw.Draw(img)
        
        # Colors
        bg_color = '#1a1a2e'
        card_bg = '#16213e'
        accent_color = '#0f3460'
        text_color = '#e94560'
        white = '#ffffff'
        gray = '#a0a0a0'
        
        # Draw background gradient effect
        for i in range(height):
            alpha = int(255 * (i / height))
            draw.rectangle([(0, i), (width, i+1)], fill=f'#{alpha:02x}1a1a2e')
        
        # Draw main card background
        draw.rectangle([20, 20, width-20, height-20], fill=card_bg, outline=accent_color, width=2)
        
        # Try to load fonts, fall back to default
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            data_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            data_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Title
        draw.text((40, 40), "◈ EVOSEQ", fill=text_color, font=title_font)
        draw.text((40, 70), "LIVE INTELLIGENCE", fill=gray, font=subtitle_font)
        
        # Prediction display
        pred_color = '#4a90e2' if prediction.lower() == 'big' else '#f5a623' if prediction.lower() == 'small' else '#ffffff'
        draw.text((40, 120), f"PREDICTION: {prediction.upper()}", fill=pred_color, font=data_font)
        
        # Confidence display
        conf_color = '#2ecc71' if confidence >= 80 else '#f39c12' if confidence >= 60 else '#e74c3c'
        draw.text((40, 150), f"CONFIDENCE: {confidence:.1f}%", fill=conf_color, font=data_font)
        
        # Numbers
        draw.text((40, 190), f"TARGET: {target_num}", fill=white, font=subtitle_font)
        draw.text((40, 220), f"HEDGE: {hedge_num}", fill=white, font=subtitle_font)
        
        # Issue numbers
        draw.text((40, 260), f"CURRENT: {current_issue[-8:]}", fill=gray, font=small_font)
        draw.text((40, 280), f"NEXT: {next_issue[-8:]}", fill=gray, font=small_font)
        
        # Pattern/model info
        draw.text((40, 320), f"MODEL: {pattern[:20]}", fill=gray, font=small_font)
        
        # Win/Loss result if available
        if previous_result:
            result_text = "✅ WON" if previous_result['won'] else "❌ LOST"
            result_color = '#2ecc71' if previous_result['won'] else '#e74c3c'
            draw.text((40, 350), f"LAST: {result_text}", fill=result_color, font=subtitle_font)
        
        # Timestamp
        timestamp = datetime.utcnow().strftime("%H:%M UTC")
        draw.text((width-150, height-40), timestamp, fill=gray, font=small_font)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return img_bytes
        
    except Exception as e:
        print(f"Error rendering forecast card: {e}")
        return None


def render_metrics_card(metrics):
    """
    Render a visual metrics card for Telegram
    
    Args:
        metrics: Metrics data dictionary
        
    Returns:
        BytesIO object containing the metrics card image, or None if rendering fails
    """
    try:
        # Create image
        width, height = 600, 350
        img = Image.new('RGB', (width, height), color='#1a1a2e')
        draw = ImageDraw.Draw(img)
        
        # Colors
        card_bg = '#16213e'
        accent_color = '#0f3460'
        text_color = '#e94560'
        white = '#ffffff'
        gray = '#a0a0a0'
        
        # Draw background
        draw.rectangle([20, 20, width-20, height-20], fill=card_bg, outline=accent_color, width=2)
        
        # Try to load fonts
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            data_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            title_font = ImageFont.load_default()
            data_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Title
        draw.text((40, 40), "◈ LIVE METRICS", fill=text_color, font=title_font)
        
        # Metrics data
        resolved = metrics.get('resolved_predictions', 0)
        accuracy = metrics.get('directional_accuracy', 0)
        brier = metrics.get('brier_score', 0)
        log_loss = metrics.get('log_loss', 0)
        
        # Display metrics
        draw.text((40, 100), f"RESOLVED: {resolved}", fill=white, font=data_font)
        
        if accuracy > 0:
            acc_text = f"ACCURACY: {accuracy*100:.1f}%"
            acc_color = '#2ecc71' if accuracy >= 0.6 else '#f39c12' if accuracy >= 0.5 else '#e74c3c'
            draw.text((40, 140), acc_text, fill=acc_color, font=data_font)
        
        draw.text((40, 180), f"BRIER: {brier:.4f}", fill=gray, font=data_font)
        draw.text((40, 220), f"LOG LOSS: {log_loss:.4f}", fill=gray, font=data_font)
        
        # Footer
        draw.text((40, 300), "Outcome-calibrated metrics", fill=gray, font=small_font)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return img_bytes
        
    except Exception as e:
        print(f"Error rendering metrics card: {e}")
        return None


def render_status_card(status_data):
    """
    Render a visual status card for Telegram
    
    Args:
        status_data: Status data dictionary
        
    Returns:
        BytesIO object containing the status card image, or None if rendering fails
    """
    try:
        # Create image
        width, height = 600, 300
        img = Image.new('RGB', (width, height), color='#1a1a2e')
        draw = ImageDraw.Draw(img)
        
        # Colors
        card_bg = '#16213e'
        accent_color = '#0f3460'
        text_color = '#e94560'
        green = '#2ecc71'
        red = '#e74c3c'
        white = '#ffffff'
        gray = '#a0a0a0'
        
        # Draw background
        draw.rectangle([20, 20, width-20, height-20], fill=card_bg, outline=accent_color, width=2)
        
        # Try to load fonts
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            data_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            title_font = ImageFont.load_default()
            data_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Title
        draw.text((40, 40), "◈ SYSTEM STATUS", fill=text_color, font=title_font)
        
        # Status indicator
        online = status_data.get('online', False)
        status_text = "🟢 ONLINE" if online else "🔴 OFFLINE"
        status_color = green if online else red
        draw.text((40, 100), status_text, fill=status_color, font=data_font)
        
        # Additional info
        if 'issue' in status_data:
            draw.text((40, 150), f"ISSUE: {status_data['issue']}", fill=white, font=data_font)
        
        draw.text((40, 200), "RUNTIME: AWS EC2", fill=gray, font=small_font)
        draw.text((40, 230), "DELIVERY: Telegram", fill=gray, font=small_font)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return img_bytes
        
    except Exception as e:
        print(f"Error rendering status card: {e}")
        return None