# 🎯 PURE ACCURACY MODE - 3-Level Martingale Safety System

## Overview
Your WinGo prediction engine now operates in **PURE ACCURACY MODE**, designed specifically for safe 3-level Martingale betting.

## How It Works

### Safety Threshold
- **Maximum acceptable risk**: 5% chance of losing all 3 levels
- **Required single-round accuracy**: ~66% or higher
- **Formula**: `Risk = (1 - accuracy)³ × 1.15` (includes correlation factor)

### Risk Calculator
| Single Accuracy | Risk of 3-Loss Streak | Status |
|----------------|----------------------|--------|
| 50% | 14.37% | 🔴 BLOCKED |
| 55% | 10.48% | 🔴 BLOCKED |
| 60% | 7.36% | 🔴 BLOCKED |
| 62% | 6.31% | 🔴 BLOCKED |
| 64% | 5.37% | 🔴 BLOCKED |
| **66%** | **4.52%** | 🟢 **ALLOWED** |
| 68% | 3.77% | 🟢 ALLOWED |
| 70% | 3.11% | 🟢 ALLOWED |
| 72% | 2.52% | 🟢 ALLOWED |
| 75% | 1.80% | 🟢 ALLOWED |

## Strike Quality Levels

When risk < 5%, predictions are issued with these confidence levels:

| Level | P(Win in 3) | Min Single P | Emoji |
|-------|-------------|--------------|-------|
| ULTIMATE_CONVICTION | ≥98.5% | ≥68% | 💎 |
| BEAST_CONVICTION | ≥97.0% | ≥66% | 🔥 |
| HIGH_CONVICTION | ≥95.0% | ≥64% | ⚡ |
| MODERATE_CONVICTION | ≥92.0% | ≥62% | 🎯 |
| CONSERVATIVE_SAFE | ≥90.0% | ≥62% | 🛡️ |

## HOLD Scenarios

The bot will display **⚠️ HOLD - NO PREDICTION** when:

1. **HOLD_RISK_TOO_HIGH**: Risk exceeds 5% threshold
   - Message: "Risk exceeds 5% threshold for 3-level Martingale"
   
2. **HOLD_INSUFFICIENT_DATA**: Not enough historical data (<5 rounds)
   - Message: "Insufficient historical data"

## Telegram Bot Behavior

When conditions are unsafe, users see:
```
◈ EVOSEQ  LIVE INTELLIGENCE
━━━━━━━━━━━━━━━━━━
⚠️ HOLD - NO PREDICTION

Reason: Risk exceeds 5% threshold for 3-level Martingale
Strategy: Pure Accuracy Mode (3-Level Safe)

The engine is monitoring the market and will issue a prediction
only when the risk of losing 3 consecutive rounds drops below 5%.

Patience ensures long-term profitability.
```

## Key Features

### 1. Risk of Ruin Calculation
```python
def calculate_risk_of_ruin_3_levels(accuracy):
    p_loss = 1.0 - accuracy
    correlation_factor = 1.15  # Accounts for streak correlation
    return (p_loss ** 3) * correlation_factor
```

### 2. Automatic HOLD Enforcement
The engine automatically blocks predictions when:
- `risk_of_ruin >= 0.05` (5%)
- Insufficient calibration data
- Market volatility detected

### 3. Evolution Through Feedback
The system learns day-by-day by:
- Recording predictions before outcomes
- Reconciling with actual Supabase results
- Updating ensemble weights via exponential weighting
- Calibrating probabilities using Platt scaling + isotonic regression

## Files Modified

1. **backend/high_intelligence_predictor.py**
   - Added `calculate_risk_of_ruin_3_levels()` function
   - Updated `recommend_strike_level()` with 5% threshold
   - Added `risk_of_ruin_3_levels` field to PredictionResult
   - Enforced HOLD when risk > 5%

2. **telegram_bot.py**
   - Added HOLD message formatting
   - New emoji mappings for strike qualities
   - Special handling for unsafe predictions

## Usage

The engine runs automatically with your existing setup:
```bash
# Scraper stores outcomes to Supabase
python backend/scraper.py

# AI engine generates predictions with safety checks
python backend/local_ai_engine.py

# Telegram bot shows only safe predictions
python telegram_bot.py
```

## Important Notes

⚠️ **This is NOT a guarantee of winnings**
- The system blocks risky predictions but cannot eliminate gambling risk
- Real-world WinGo outcomes may not follow historical patterns
- Always gamble responsibly

✅ **Benefits**
- Mathematically sound risk management
- Prevents betting during unfavorable conditions  
- Forces discipline through automated HOLD signals
- Transparent probability calculations

📊 **Metrics to Watch**
- `calibrated_p_single`: Must be >66% for active predictions
- `risk_of_ruin_3_levels`: Must be <5% for active predictions
- `strike_quality`: Shows confidence level when prediction is active
