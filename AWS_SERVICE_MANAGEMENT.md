# AWS Deployment and Service Management Guide

## 🚀 How to Run Enhanced System on AWS

### Step 1: Pull Latest Changes
```bash
cd ~/win
git pull
```

### Step 2: Install Dependencies
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Restart Services

**Option A: Restart Individual Services**
```bash
# Restart the main API service
sudo systemctl restart win-api

# Restart the scraper service
sudo systemctl restart win-scraper

# Restart the Telegram bot service
sudo systemctl restart win-telegram
```

**Option B: Restart All Services**
```bash
sudo systemctl restart win-api win-scraper win-telegram
```

### Step 4: Check Service Status
```bash
# Check all win services
sudo systemctl status win-api win-scraper win-telegram

# Check individual service
sudo systemctl status win-api
```

### Step 5: View Service Logs
```bash
# View API logs
sudo journalctl -u win-api -f

# View scraper logs
sudo journalctl -u win-scraper -f

# View Telegram bot logs
sudo journalctl -u win-telegram -f
```

## 🛑 How to Stop Unwanted Services

### Check All Running Services
```bash
# List all active services
sudo systemctl list-units --type=service --state=running

# List all win-related services
sudo systemctl list-units --type=service | grep win
```

### Stop Specific Services
```bash
# Stop API service
sudo systemctl stop win-api

# Stop scraper service
sudo systemctl stop win-scraper

# Stop Telegram bot service
sudo systemctl stop win-telegram
```

### Disable Services (prevent auto-start)
```bash
# Disable from starting on boot
sudo systemctl disable win-api
sudo systemctl disable win-scraper
sudo systemctl disable win-telegram
```

### Enable Services (allow auto-start)
```bash
# Enable to start on boot
sudo systemctl enable win-api
sudo systemctl enable win-scraper
sudo systemctl enable win-telegram
```

## 🔧 Complete Service Management

### Stop All Win Services
```bash
sudo systemctl stop win-api win-scraper win-telegram
```

### Start All Win Services
```bash
sudo systemctl start win-api win-scraper win-telegram
```

### Restart All Win Services
```bash
sudo systemctl restart win-api win-scraper win-telegram
```

### Check All Win Services Status
```bash
sudo systemctl status win-api win-scraper win-telegram
```

## 📋 Service Files Location

Service files are located at:
```bash
/etc/systemd/system/win-api.service
/etc/systemd/system/win-scraper.service
/etc/systemd/system/win-telegram.service
```

### View Service Configuration
```bash
# View API service configuration
sudo cat /etc/systemd/system/win-api.service

# View scraper service configuration
sudo cat /etc/systemd/system/win-scraper.service

# View Telegram bot service configuration
sudo cat /etc/systemd/system/win-telegram.service
```

## 🧹 Remove Unwanted Services

### Stop and Disable Service
```bash
# Stop the service
sudo systemctl stop win-api

# Disable from auto-start
sudo systemctl disable win-api

# Remove service file
sudo rm /etc/systemd/system/win-api.service

# Reload systemd
sudo systemctl daemon-reload
```

## 🔍 Monitor System Resources

### Check CPU and Memory Usage
```bash
# Check overall system resources
htop

# Or use top
top

# Check specific process
ps aux | grep python
```

### Check Disk Usage
```bash
# Check disk space
df -h

# Check directory size
du -sh ~/win
```

## 🎯 Recommended Setup for Your System

### Start Only Essential Services
```bash
# If you only want Telegram predictions (no API, no scraper)
sudo systemctl stop win-api win-scraper
sudo systemctl disable win-api win-scraper
sudo systemctl start win-telegram
sudo systemctl enable win-telegram

# If you want full system (API + scraper + Telegram)
sudo systemctl start win-api win-scraper win-telegram
sudo systemctl enable win-api win-scraper win-telegram
```

### Run Scraper Separately (if needed)
```bash
# If you want to run scraper manually instead of as service
cd ~/win
source .venv/bin/activate
python backend/scraper.py
```

### Run Telegram Bot Separately (if needed)
```bash
# If you want to run bot manually instead of as service
cd ~/win
source .venv/bin/activate
python telegram_bot.py
```

## 🔄 Service Dependency Order

Services have dependencies:
- **win-scraper**: Runs independently (collects data)
- **win-api**: Depends on scraper data (provides predictions)
- **win-telegram**: Depends on API (sends predictions via Telegram)

**Recommended startup order:**
```bash
sudo systemctl start win-scraper
sleep 5
sudo systemctl start win-api
sleep 5
sudo systemctl start win-telegram
```

## 📊 Enhanced System Features (Now Active)

Your enhanced system now automatically:
- ✅ Fetches data from Supabase for training
- ✅ Cleans up 2-day old data automatically
- ✅ Uses sophisticated existing intelligence
- ✅ Integrates with daily Supabase data

**No additional setup needed - just pull and restart services!**

## 🚨 Troubleshooting

### Service Won't Start
```bash
# Check service status for errors
sudo systemctl status win-api

# View detailed logs
sudo journalctl -u win-api -n 50 --no-pager

# Check if port is already in use
sudo netstat -tlnp | grep 8000
```

### Service Keeps Restarting
```bash
# Check logs for crash reasons
sudo journalctl -u win-api -n 100 --no-pager

# Test manually
cd ~/win
source .venv/bin/activate
python backend/server.py
```

### Database Connection Issues
```bash
# Test database connection
cd ~/win
source .venv/bin/activate
python -c "from backend.database import engine; print('Connected')"
```

## 📝 Quick Reference

**Start all services:**
```bash
sudo systemctl start win-api win-scraper win-telegram
```

**Stop all services:**
```bash
sudo systemctl stop win-api win-scraper win-telegram
```

**Check status:**
```bash
sudo systemctl status win-api win-scraper win-telegram
```

**View logs:**
```bash
sudo journalctl -u win-api -f
```

**Restart after deployment:**
```bash
cd ~/win && git pull && sudo systemctl restart win-api win-scraper win-telegram
```