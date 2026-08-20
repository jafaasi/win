#!/bin/bash
# Deploy Premium UI Changes to AWS Telegram Bot

echo "🚀 Deploying Premium UI Changes to AWS Telegram Bot..."

# Step 1: SSH into AWS and pull latest changes
echo "Step 1: Connecting to AWS and pulling latest changes..."
ssh ubuntu@3.7.65.149 << 'ENDSSH'
cd ~/win
git pull
ENDSSH

# Step 2: Restart the Telegram bot service
echo "Step 2: Restarting Telegram bot service..."
ssh ubuntu@3.7.65.149 << 'ENDSSH'
sudo systemctl restart win-telegram
sudo systemctl status win-telegram
ENDSSH

# Step 3: Check logs
echo "Step 3: Checking Telegram bot logs..."
ssh ubuntu@3.7.65.149 << 'ENDSSH'
sudo journalctl -u win-telegram -n 20 --no-pager
ENDSSH

echo "✅ Premium UI deployment completed!"
echo ""
echo "The Telegram bot should now show:"
echo "• Cleaner keyboard layout"
echo "• Simplified messages"
echo "• Less frequent notifications"
echo "• Premium but pleasant experience"
echo ""
echo "Test in Telegram with /start command"