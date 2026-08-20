# Deploy Premium UI to AWS

## Quick Deployment

### Option 1: Manual Deployment (Recommended)

**Step 1: SSH into AWS**
```bash
ssh ubuntu@3.7.65.149
```

**Step 2: Pull latest changes**
```bash
cd ~/win
git pull
```

**Step 3: Restart Telegram bot**
```bash
sudo systemctl restart win-telegram
```

**Step 4: Check status**
```bash
sudo systemctl status win-telegram
```

**Step 5: View logs**
```bash
sudo journalctl -u win-telegram -f
```

### Option 2: Use Deployment Script

**Make script executable and run:**
```bash
chmod +x deploy_premium_ui.sh
./deploy_premium_ui.sh
```

## What Changed

**Premium UI Features:**
- Cleaner keyboard layout (8 organized buttons)
- Simplified prediction messages
- Less frequent notifications (3s interval vs 1.5s)
- 15-second notification cooldown
- Maximum 20 notifications/hour
- Concise help message
- Pleasant welcome message

**New Keyboard Layout:**
- ✨ Live Forecast
- 📊 Performance  
- 💎 Strategy
- 🏆 Results
- ⚙️ Preferences
- 🟢 Status
- 🔔 Updates

**New Message Format:**
```
✨ 🔵 BIG
Confidence: 88.5%
Target: 9 | Hedge: 6

Round: 52473

EVOSEQ Premium Intelligence
```

## Testing

**In Telegram:**
1. Send `/start` to see new welcome message
2. Use keyboard buttons to see new layout
3. Send `/forecast` to see new message format
4. Wait for automatic updates (less frequent now)

## Troubleshooting

**If bot doesn't start:**
```bash
# Check if TELEGRAM_BOT_TOKEN is set
echo $TELEGRAM_BOT_TOKEN

# Set it if missing
export TELEGRAM_BOT_TOKEN='your_token_here'

# Edit service file
sudo nano /etc/systemd/system/win-telegram.service
```

**If UI still looks the same:**
```bash
# Check if latest code is pulled
cd ~/win
git log --oneline -1

# Force pull latest
git fetch --all
git reset --hard origin/main
sudo systemctl restart win-telegram
```