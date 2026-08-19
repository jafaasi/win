#!/bin/bash
# Setup script for daily evolving intelligence cron job

echo "Setting up daily evolving intelligence cron job..."

# Create log directory
mkdir -p ~/win/logs

# Make scripts executable
chmod +x ~/win/backend/daily_learning.py
chmod +x ~/win/backend/data_cleanup.py
chmod +x ~/win/backend/evolving_intelligence.py
chmod +x ~/win/backend/daily_evolution.py

# Add cron job to run daily at 2 AM UTC
(crontab -l 2>/dev/null; echo "0 2 * * * cd /home/ubuntu/win && /home/ubuntu/win/.venv/bin/python backend/daily_evolution.py >> /home/ubuntu/win/logs/daily_evolution.log 2>&1") | crontab -

echo "Cron job set up to run daily at 2 AM UTC"
echo "Logs will be saved to ~/win/logs/daily_evolution.log"
echo ""
echo "To view logs: tail -f ~/win/logs/daily_evolution.log"
echo "To edit cron: crontab -e"
echo "To test manually: cd ~/win && source .venv/bin/activate && python backend/daily_evolution.py"