# 🛠️ AWS Service Management Guide — WinGo Ultra Intelligence

This document provides commands and workflows for managing the **WinGo 30s Ultra Intelligence** system on AWS EC2.

---

## 🚀 1. The 4 System Services

The system runs 4 background daemons via `systemd`:

| Service | Daemon Script | Role |
|---|---|---|
| **`win-ai.service`** | `backend/local_ai_engine.py` | Core Ultra Intelligence Engine (8-model Hedge ensemble, exploit gating, online learning) |
| **`win-telegram.service`** | `telegram_bot.py` | Luxury Telegram Bot dispatcher (cycle-synchronized 1.5s push, card rendering) |
| **`win-scraper.service`** | `backend/scraper.py` | 24/7 WinGo 30s draw collector syncing directly to Supabase |
| **`win-api.service`** | `backend.server:app` | FastAPI REST gateway (port 8000) for web dashboard & telemetry |

---

## ⚡ 2. Instant Shortcut Commands

The installer configures these aliases in `/etc/profile.d/win_aliases.sh`:

```bash
# Check status of all 4 services
win-status

# Live stream combined logs from all services
win-logs

# Stream AI Engine logs only
win-ai-logs

# Stream Telegram Bot logs only
win-bot-logs

# Restart all services
win-restart

# Stop all services
win-stop
```

---

## 🔄 3. Updating to Latest Code from GitHub

When you push new changes to GitHub, update your AWS server in 1 step:

```bash
cd ~/win && git pull && win-restart
```

If dependencies in `requirements.txt` changed:
```bash
cd ~/win
git pull
source .venv/bin/activate
pip install -r requirements.txt
win-restart
```

---

## 📊 4. Standard `systemctl` Commands

### Check Status
```bash
# All services:
sudo systemctl status win-ai win-telegram win-scraper win-api --no-pager

# Individual services:
sudo systemctl status win-ai
sudo systemctl status win-telegram
sudo systemctl status win-scraper
sudo systemctl status win-api
```

### Restart Services
```bash
# Restart all:
sudo systemctl restart win-*

# Restart individual:
sudo systemctl restart win-ai
sudo systemctl restart win-telegram
sudo systemctl restart win-scraper
sudo systemctl restart win-api
```

### Stop / Start Services
```bash
# Stop all:
sudo systemctl stop win-ai win-telegram win-scraper win-api

# Start all:
sudo systemctl start win-ai win-telegram win-scraper win-api
```

---

## 📜 5. Viewing Logs with `journalctl`

```bash
# Live stream AI Engine log:
sudo journalctl -u win-ai -f

# Live stream Telegram Bot log:
sudo journalctl -u win-telegram -f

# Live stream Scraper log:
sudo journalctl -u win-scraper -f

# View last 100 log lines:
sudo journalctl -u win-ai -n 100 --no-pager
```

---

## ⚙️ 6. Environment Configuration

All services share the unified environment configuration at:
```bash
/etc/win/win.env
```

### To Edit:
```bash
sudo nano /etc/win/win.env
```

### Expected Environment Variables:
```ini
# Supabase PostgreSQL connection string
DATABASE_URL=postgresql://postgres.zyryxnifpduwsulglhdq:JafAasi1517@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres

# Telegram Bot Token from @BotFather
TELEGRAM_BOT_TOKEN=8486018151:AAEgqW2jE5W1u1E2x6qD7yZ8_example

# Prediction & Game APIs
PREDICTION_API_URL=http://127.0.0.1:8000/api/state
WINGO_API_URL=https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json
```

*After editing, always run `win-restart`.*

---

## 🔍 7. Troubleshooting Common Issues

### 1. Telegram Bot is not sending messages:
- Check token in `/etc/win/win.env`
- Verify bot logs: `win-bot-logs`
- Test network connectivity: `curl -s https://api.telegram.org`

### 2. AI Engine is waiting for draws:
- Check if scraper is running: `sudo systemctl status win-scraper`
- Verify scraper logs: `sudo journalctl -u win-scraper -f`
- Verify Supabase DB connection: `cd ~/win && .venv/bin/python test_db.py`

### 3. Out of Memory on `t2.micro` / `t3.micro`:
- Check memory: `free -h`
- Enable 2GB swap space:
  ```bash
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  ```