# Telegram Bot Setup Guide

## Overview
Telegram bot integration for WinGo predictions - provides predictions via Telegram commands without web connectivity issues.

## Prerequisites
- Set `TELEGRAM_BOT_TOKEN` in your environment before starting the bot. Do not
  put a Telegram token in this file or commit it to the repository.
- AWS EC2 instance with API running on port 8000
- Python 3.8+

## Installation on AWS

### Step 1: Update Requirements
```bash
cd ~/win
git pull
source .venv/bin/activate
pip install python-telegram-bot
```

### Step 2: Test the Bot
```bash
python telegram_bot.py
```

### Step 3: Set Up Systemd Service
```bash
sudo nano /etc/systemd/system/win-telegram.service
```

Add this content:
```ini
[Unit]
Description=WinGo Telegram Bot
After=network.target win-api.service

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/win
Environment="PATH=/home/ubuntu/win/.venv/bin"
ExecStart=/home/ubuntu/win/.venv/bin/python /home/ubuntu/win/telegram_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Step 4: Enable and Start Service
```bash
sudo systemctl daemon-reload
sudo systemctl enable win-telegram
sudo systemctl start win-telegram
sudo systemctl status win-telegram
```

## Bot Commands

- `/start` - Start the bot and auto-subscribe to automatic updates
- `/predict` - Get current AI prediction
- `/status` - Check bot and API status
- `/subscribe` - Subscribe to automatic prediction updates
- `/unsubscribe` - Unsubscribe from automatic updates
- `/help` - Show help message

## Architecture

```
Telegram User
    ↓
Telegram Bot (on AWS)
    ↓
Local API (localhost:8000)
    ↓
EVOSEQ AI Engine
```

## Benefits

- **No SSL issues**: Telegram handles all security
- **Mobile access**: Get predictions on phone
- **Reliable**: Uses Telegram's infrastructure
- **Free**: No additional costs
- **Simple**: No proxy/tunnel configuration needed

## Troubleshooting

### Check Bot Status
```bash
sudo systemctl status win-telegram
```

### View Bot Logs
```bash
sudo journalctl -u win-telegram -n 50 --no-pager
```

### Restart Bot
```bash
sudo systemctl restart win-telegram
```

### Test API Connection
```bash
curl http://localhost:8000/api/state
```

## Security Notes

- Bot token is stored in the script (consider using environment variables for production)
- Only authorized users should have access to the bot
- Consider adding user whitelisting for production use

## Features

- Real-time AI predictions
- **Automatic prediction updates** - subscribers get new predictions automatically
- Confidence scores
- Pattern analysis
- Multiple model ensemble information
- Status monitoring
- Issue tracking
- Mobile-friendly access

## Automatic Updates

The bot automatically checks for new predictions every 30 seconds and sends them to subscribed users:

- **Auto-subscribe**: Users are automatically subscribed when they use `/start`
- **New predictions**: When a new prediction is generated (detected by issue number change), all subscribers receive it
- **Unsubscribe**: Users can unsubscribe with `/unsubscribe` command
- **Reliable**: Failed sends are logged and users are removed if they block the bot
