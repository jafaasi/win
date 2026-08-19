# ☁️ AWS EC2 Deployment Guide — WinGo Ultra Intelligence

This guide walks you through deploying the complete **WinGo 30s Ultra Intelligence & Telegram Bot** system on an AWS EC2 instance with 24/7 background `systemd` daemon automation.

---

## 1. 🖥️ Recommended AWS EC2 Instance Specs

| Spec | Recommended | Minimum (Free Tier) |
|---|---|---|
| **AMI (OS)** | **Ubuntu 24.04 LTS** or **22.04 LTS** (64-bit x86 or ARM) | Ubuntu 22.04 LTS |
| **Instance Type** | **`t3.medium`** (2 vCPU, 4GB RAM) or **`c6i.large`** | `t2.micro` / `t3.micro` (with swap) |
| **Storage (EBS)** | **20 GB gp3** SSD | 15 GB gp2/gp3 |
| **Region** | Singapore (`ap-southeast-1`) or nearest to Supabase | Any |

---

## 2. 🛡️ Security Group Inbound Rules

In your AWS EC2 Console, configure your Security Group with the following inbound rules:

| Type | Port Range | Protocol | Source | Purpose |
|---|---|---|---|---|
| **SSH** | `22` | TCP | `My IP` (or `0.0.0.0/0`) | Remote SSH Access |
| **Custom TCP** | `8000` | TCP | `0.0.0.0/0` | FastAPI Web Gateway (Optional) |

*(Outbound rules: Leave default `All Traffic` so the instance can connect to Supabase, Telegram, and WinGo API).*

---

## 3. 🚀 1-Command Automated Installation

Connect to your EC2 instance via SSH:
```bash
ssh -i your-key.pem ubuntu@<your-ec2-public-ip>
```

Run the one-line deployment script:
```bash
curl -sSL https://raw.githubusercontent.com/jafaasi/win/main/deploy_aws.sh | bash
```

### What the installer does automatically:
1. Updates Ubuntu packages and installs build tools, Python 3, `libpq-dev`, and TrueType font libraries (`fonts-dejavu`, `fonts-liberation`).
2. Clones the latest repository from `https://github.com/jafaasi/win.git`.
3. Creates an isolated Python virtual environment (`.venv`) and installs all PyTorch, Telegram Bot, Pillow, and Quant packages.
4. Registers and starts **4 independent systemd services**:
   - `win-scraper.service`: Scrapes 30s draws into Supabase.
   - `win-ai.service`: Runs the Ultra Intelligence engine with continuous Hedge online learning.
   - `win-telegram.service`: Runs the cycle-synchronized Telegram bot with luxury cards.
   - `win-api.service`: Runs the FastAPI server on port 8000.

---

## 4. 🔑 Configure Your API Keys

Open the global environment configuration file:
```bash
sudo nano /etc/win/win.env
```

Verify and edit your credentials:
```ini
# Supabase PostgreSQL connection string
DATABASE_URL=postgresql://postgres.zyryxnifpduwsulglhdq:JafAasi1517@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres

# Your Telegram Bot Token from @BotFather
TELEGRAM_BOT_TOKEN=8486018151:AAEgqW2jE5W1u1E2x6qD7yZ8_example

# Prediction & Game APIs
PREDICTION_API_URL=http://127.0.0.1:8000/api/state
WINGO_API_URL=https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json
```

Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

Restart the services to apply changes:
```bash
sudo systemctl restart win-scraper win-ai win-telegram win-api
```

---

## 5. 🛠️ Daily Management & Monitoring Commands

We included pre-configured alias commands for effortless server management:

### Check Status of All Services:
```bash
win-status
```
*(or `sudo systemctl status win-*`)*

### Live Stream Logs:
```bash
# All services combined:
win-logs

# AI Engine only:
win-ai-logs

# Telegram Bot only:
win-bot-logs
```

### Restart All Services:
```bash
win-restart
```

### Update to Latest GitHub Code:
```bash
cd ~/win && git pull && win-restart
```

---

## 6. 🛡️ Adding Swap Space (Recommended for t2/t3.micro Free Tier)

If using a `t2.micro` or `t3.micro` instance, enable 2GB of virtual swap memory to ensure PyTorch and Mamba inference never exceed RAM:

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```
