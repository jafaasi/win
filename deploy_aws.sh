#!/bin/bash

# ==============================================================================
# AWS EC2 Automated Deployment Script for WinGo Ultra Intelligence & Telegram Bot
# ==============================================================================
# Run this script directly on your Ubuntu EC2 instance:
#   curl -sSL https://raw.githubusercontent.com/jafaasi/win/main/deploy_aws.sh | bash
# Or clone and run:
#   ./deploy_aws.sh
# ==============================================================================

set -e

echo "🚀 Starting AWS EC2 Deployment for WinGo Ultra Intelligence..."

# Step 1: Update and install system dependencies & fonts
echo "📦 Installing system dependencies, fonts, and build tools..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git build-essential \
    libssl-dev libffi-dev python3-dev libpq-dev fonts-dejavu fonts-liberation \
    curl ufw htop

# Step 2: Clone or update repository
echo "📥 Setting up project directory at /home/ubuntu/win..."
cd /home/ubuntu
if [ -d "win" ]; then
    echo "Repository exists, pulling latest main branch..."
    cd win
    git fetch origin
    git reset --hard origin/main
else
    git clone https://github.com/jafaasi/win.git
    cd win
fi

# Step 3: Python Virtual Environment Setup
echo "🐍 Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

echo "📚 Installing Python requirements..."
pip install -r requirements.txt

# Step 4: System configuration directory
echo "⚙️ Setting up environment configuration in /etc/win..."
sudo mkdir -p /etc/win
sudo chown ubuntu:ubuntu /etc/win

if [ ! -f "/etc/win/win.env" ]; then
    echo "Creating default /etc/win/win.env..."
    cat > /tmp/win.env << 'EOF'
# Supabase PostgreSQL connection URL (Pooler)
DATABASE_URL=postgresql://postgres.zyryxnifpduwsulglhdq:JafAasi1517@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres

# Telegram Bot API Token from @BotFather
TELEGRAM_BOT_TOKEN=8486018151:AAEgqW2jE5W1u1E2x6qD7yZ8_example

# Prediction & Game APIs
PREDICTION_API_URL=http://127.0.0.1:8000/api/state
WINGO_API_URL=https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json
EOF
    sudo mv /tmp/win.env /etc/win/win.env
    sudo chmod 600 /etc/win/win.env
fi

# Also link to local .env if missing
if [ ! -f "/home/ubuntu/win/.env" ]; then
    ln -sf /etc/win/win.env /home/ubuntu/win/.env
fi

# Step 5: Install and configure systemd services
echo "🔌 Registering systemd daemons..."

# 1. Scraper Daemon
sudo tee /etc/systemd/system/win-scraper.service > /dev/null << 'EOF'
[Unit]
Description=WinGo 30s Scraper Daemon (Supabase Sync)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/win
EnvironmentFile=/etc/win/win.env
ExecStart=/home/ubuntu/win/.venv/bin/python3 /home/ubuntu/win/backend/scraper.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# 2. Ultra Intelligence AI Engine
sudo tee /etc/systemd/system/win-ai.service > /dev/null << 'EOF'
[Unit]
Description=WinGo Ultra Intelligence Prediction Engine
After=network-online.target win-scraper.service
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/win
EnvironmentFile=/etc/win/win.env
ExecStart=/home/ubuntu/win/.venv/bin/python3 /home/ubuntu/win/backend/local_ai_engine.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# 3. Telegram Bot Service
sudo tee /etc/systemd/system/win-telegram.service > /dev/null << 'EOF'
[Unit]
Description=WinGo Ultra Quant Telegram Bot
After=network-online.target win-ai.service
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/win
EnvironmentFile=/etc/win/win.env
ExecStart=/home/ubuntu/win/.venv/bin/python3 /home/ubuntu/win/telegram_bot.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# 4. FastAPI Server Service (Port 8000)
sudo tee /etc/systemd/system/win-api.service > /dev/null << 'EOF'
[Unit]
Description=WinGo FastAPI Web Gateway
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/win
EnvironmentFile=/etc/win/win.env
ExecStart=/home/ubuntu/win/.venv/bin/uvicorn backend.server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Step 6: Reload and activate all services
echo "▶️ Enabling and launching services..."
sudo systemctl daemon-reload
sudo systemctl enable win-scraper win-ai win-telegram win-api
sudo systemctl restart win-scraper win-ai win-telegram win-api

# Step 7: Convenience management aliases
sudo tee /etc/profile.d/win_aliases.sh > /dev/null << 'EOF'
alias win-status="sudo systemctl status win-scraper win-ai win-telegram win-api --no-pager"
alias win-restart="sudo systemctl restart win-scraper win-ai win-telegram win-api"
alias win-stop="sudo systemctl stop win-scraper win-ai win-telegram win-api"
alias win-logs="sudo journalctl -u win-scraper -u win-ai -u win-telegram -u win-api -f"
alias win-ai-logs="sudo journalctl -u win-ai -f"
alias win-bot-logs="sudo journalctl -u win-telegram -f"
EOF

# Check service status
sleep 3
echo ""
echo "=========================================================================="
echo "✅ WinGo Ultra Intelligence AWS Deployment Complete!"
echo "=========================================================================="
echo ""
sudo systemctl status win-scraper win-ai win-telegram win-api --no-pager || true

echo ""
echo "📋 Quick Maintenance Guide:"
echo "1. Edit environment variables:  sudo nano /etc/win/win.env"
echo "2. Restart all services:       sudo systemctl restart win-*"
echo "3. View live stream logs:      sudo journalctl -u win-ai -u win-telegram -f"
echo "4. Pull updates from GitHub:   cd ~/win && git pull && sudo systemctl restart win-*"
echo ""

