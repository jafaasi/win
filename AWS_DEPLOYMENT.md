# AWS Deployment Guide

This guide shows how to deploy your EVOSEQ prediction system to AWS EC2 and run it continuously.

## Architecture

```
AWS EC2 (m7i-flex.large)
├── Ubuntu 22.04 LTS
├── Python 3.10+
├── EVOSEQ Engine (scraper + AI + API + Telegram bot)
├── Systemd (keep processes running)
└── Uvicorn (FastAPI server)

Vercel (Frontend)
└── Calls AWS API endpoints

Supabase (Database)
└── Stores outcomes and predictions
```

## Prerequisites

- AWS EC2 instance running (Ubuntu 22.04)
- SSH access to instance
- Code repository cloned
- Supabase credentials (DATABASE_URL)
- Git repository URL

## Step 1: Connect to AWS Instance

```bash
ssh -i ~/.ssh/win-aws-key.pem ubuntu@<your-public-ip>
```

## Step 2: Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+ and pip
sudo apt install -y python3 python3-pip python3-venv git

# Install other dependencies
sudo apt install -y build-essential libssl-dev libffi-dev python3-dev
```

## Step 3: Clone Repository

```bash
# Navigate to home directory
cd ~

# Clone your repository
git clone https://github.com/jafaasi/win.git

# Navigate to project directory
cd win
```

## Step 4: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## Step 5: Install Python Dependencies

```bash
# Install requirements
pip install -r requirements.txt

# If requirements.txt doesn't exist, install manually:
pip install fastapi uvicorn sqlalchemy psycopg2-binary numpy pandas scikit-learn torch httpx python-dotenv
```

## Step 6: Configure Environment Variables

```bash
# Create the systemd-only environment file
sudo install -d -m 700 /etc/win
sudo nano /etc/win/win.env
```

Add the following:
```env
DATABASE_URL=postgresql://user:password@host:port/database
TELEGRAM_BOT_TOKEN=replace-with-a-new-token-from-botfather
PREDICTION_API_URL=http://127.0.0.1:8000/api/state
OUTCOME_RETENTION_DAYS=30
EVOSEQ_LOOKBACK_DAYS=30
EVOSEQ_MAX_TRAINING_OUTCOMES=50000
```

Save and exit, then protect it:

```bash
sudo chmod 600 /etc/win/win.env
```

Do not put this file in Git. Rotate the Telegram token if it was previously committed.

## Step 7: Create Systemd Service for Scraper

The repository includes production service definitions in `deploy/systemd/`.
Install all four instead of manually copying old service examples:

```bash
sudo cp deploy/systemd/win-scraper.service /etc/systemd/system/
sudo cp deploy/systemd/win-ai.service /etc/systemd/system/
sudo cp deploy/systemd/win-api.service /etc/systemd/system/
sudo cp deploy/systemd/win-telegram.service /etc/systemd/system/
sudo systemctl daemon-reload
```

The remainder of this guide explains what each service does.

```bash
sudo nano /etc/systemd/system/win-scraper.service
```

Add the following:
```ini
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
```

## Step 8: Create Systemd Service for AI Engine

```bash
sudo nano /etc/systemd/system/win-ai.service
```

Add the following:
```ini
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
```

## Step 9: Create Systemd Service for API Server

```bash
sudo nano /etc/systemd/system/win-api.service
```

Add the following:
```ini
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
```

## Step 10: Enable and Start Services

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable services (start on boot)
sudo systemctl enable win-scraper
sudo systemctl enable win-ai
sudo systemctl enable win-api
sudo systemctl enable win-telegram

# Start services
sudo systemctl start win-scraper
sudo systemctl start win-ai
sudo systemctl start win-api
sudo systemctl start win-telegram
```

## Step 11: Check Service Status

```bash
# Check all services
sudo systemctl status win-scraper
sudo systemctl status win-ai
sudo systemctl status win-api
sudo systemctl status win-telegram

# View logs
sudo journalctl -u win-scraper -f
sudo journalctl -u win-ai -f
sudo journalctl -u win-api -f
sudo journalctl -u win-telegram -f
```

## Step 12: Update Vercel Frontend API URL

Update your Vercel frontend to call the AWS API instead of local:

```javascript
// In your frontend code
const API_URL = 'http://<your-aws-public-ip>:8000';
```

## Step 13: Set Up Nginx (Optional but Recommended)

For better SSL/HTTPS support:

```bash
# Install Nginx
sudo apt install -y nginx

# Configure Nginx reverse proxy
sudo nano /etc/nginx/sites-available/win
```

Add:
```nginx
server {
    listen 80;
    server_name <your-domain-or-ip>;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/win /etc/nginx/sites-enabled/

# Test and restart Nginx
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx
```

## Step 14: Set Up SSL with Let's Encrypt (Optional)

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d <your-domain>

# Auto-renewal is already configured
```

## Monitoring and Management

### View Logs
```bash
# All services
sudo journalctl -u win-* -f

# Specific service
sudo journalctl -u win-scraper -n 100
```

### Restart Services
```bash
sudo systemctl restart win-scraper
sudo systemctl restart win-ai
sudo systemctl restart win-api
```

### Stop Services
```bash
sudo systemctl stop win-scraper
sudo systemctl stop win-ai
sudo systemctl stop win-api
```

### Update Code
```bash
cd ~/win
git pull
sudo systemctl restart win-scraper win-ai win-api
```

## Cost Optimization

### Current Setup
- Instance: m7i-flex.large (~$47/month)
- Storage: 16 GiB gp3 (~$1.28/month)
- Data transfer: ~$0.09/GB
- **Total**: ~$50-60/month

### Reduce Costs Further
- Use t3.large instead (burstable, ~$60/month but more predictable)
- Schedule instance start/stop if not needed 24/7
- Use reserved instances for long-term commitment

## Troubleshooting

### Service Won't Start
```bash
# Check service status
sudo systemctl status win-scraper

# View detailed logs
sudo journalctl -u win-scraper -n 50 --no-pager
```

### Database Connection Issues
- Check .env file has correct DATABASE_URL
- Ensure Supabase allows connections from AWS IP
- Check security group allows outbound traffic

### API Not Accessible
- Check security group allows HTTP/HTTPS from anywhere
- Check if API service is running: `sudo systemctl status win-api`
- Check if port 8000 is listening: `sudo netstat -tlnp | grep 8000`

## Security Best Practices

1. **Keep system updated**: `sudo apt update && sudo apt upgrade -y`
2. **Use SSH key authentication** (already configured)
3. **Restrict SSH to your IP** in security group
4. **Use SSL/HTTPS** for API calls
5. **Keep dependencies updated**: `pip install --upgrade -r requirements.txt`
6. **Monitor logs** for suspicious activity
7. **Backup database** regularly (Supabase handles this)

## Next Steps

1. Deploy to AWS using this guide
2. Update Vercel frontend to use AWS API URL
3. Monitor services for first few days
4. Set up alerts for service failures
5. Consider setting up monitoring (CloudWatch, etc.)
