#!/bin/bash

# AWS Deployment Script for EVOSEQ
# Run this on your AWS EC2 instance after SSH connection

set -e

echo "🚀 Starting AWS Deployment for EVOSEQ..."

# Step 1: Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Step 2: Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y python3 python3-pip python3-venv git build-essential libssl-dev libffi-dev python3-dev

# Step 3: Clone repository
echo "📥 Cloning repository..."
cd ~
if [ -d "win" ]; then
    echo "Repository already exists, pulling latest changes..."
    cd win
    git pull
else
    git clone https://github.com/jafaasi/win.git
    cd win
fi

# Step 4: Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Step 5: Install Python dependencies
echo "📚 Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Installing core dependencies..."
    pip install fastapi uvicorn sqlalchemy psycopg2-binary numpy pandas scikit-learn torch httpx python-dotenv
fi

# Step 6: Create .env file if it doesn't exist
echo "⚙️ Setting up environment variables..."
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# Add your Supabase DATABASE_URL here
DATABASE_URL=postgresql://user:password@host:port/database
EOF
    echo "⚠️  Please edit .env file with your Supabase credentials!"
    echo "   Run: nano .env"
fi

# Step 7: Create systemd services
echo "🔌 Creating systemd services..."

# Scraper service
sudo tee /etc/systemd/system/win-scraper.service > /dev/null << EOF
[Unit]
Description=WinGo Scraper Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/win
Environment="PATH=/home/ubuntu/win/.venv/bin"
ExecStart=/home/ubuntu/win/.venv/bin/python3 /home/ubuntu/win/run_local_scraper.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# AI Engine service
sudo tee /etc/systemd/system/win-ai.service > /dev/null << EOF
[Unit]
Description=WinGo AI Engine Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/win
Environment="PATH=/home/ubuntu/win/.venv/bin"
ExecStart=/home/ubuntu/win/.venv/bin/python3 /home/ubuntu/win/backend/local_ai_engine.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# API Server service
sudo tee /etc/systemd/system/win-api.service > /dev/null << EOF
[Unit]
Description=WinGo API Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/win
Environment="PATH=/home/ubuntu/win/.venv/bin"
ExecStart=/home/ubuntu/win/.venv/bin/uvicorn backend.server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Step 8: Enable and start services
echo "▶️  Enabling and starting services..."
sudo systemctl daemon-reload
sudo systemctl enable win-scraper
sudo systemctl enable win-ai
sudo systemctl enable win-api

# Start services
sudo systemctl start win-scraper
sudo systemctl start win-ai
sudo systemctl start win-api

# Step 9: Check service status
echo "📊 Checking service status..."
sleep 5
sudo systemctl status win-scraper --no-pager || true
sudo systemctl status win-ai --no-pager || true
sudo systemctl status win-api --no-pager || true

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your Supabase credentials: nano .env"
echo "2. Restart services after editing .env: sudo systemctl restart win-*"
echo "3. Check logs: sudo journalctl -u win-scraper -f"
echo "4. Get your instance public IP from AWS console"
echo "5. Update Vercel frontend to use: http://<your-aws-ip>:8000"
echo ""
echo "🔍 Useful commands:"
echo "  View logs:    sudo journalctl -u win-* -f"
echo "  Restart all: sudo systemctl restart win-*"
echo "  Check status: sudo systemctl status win-*"
echo "  Update code:  git pull && sudo systemctl restart win-*"
