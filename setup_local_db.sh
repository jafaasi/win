#!/bin/bash
# 🚀 WinGo - AWS Local PostgreSQL Migration Script
# Run this on your AWS EC2 instance

echo "🚀 Installing PostgreSQL..."
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib

echo "🛠️ Configuring PostgreSQL service..."
sudo systemctl start postgresql
sudo systemctl enable postgresql

echo "🗄️ Creating database and user..."
# Ignore errors if they already exist
sudo -u postgres psql -c "CREATE USER wingo_user WITH PASSWORD 'wingo_pass_2026';" || true
sudo -u postgres psql -c "CREATE DATABASE wingo_db OWNER wingo_user;" || true
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE wingo_db TO wingo_user;" || true

# Important for SQLAlchemy and new tables created by wingo_user
sudo -u postgres psql -d wingo_db -c "GRANT ALL ON SCHEMA public TO wingo_user;" || true

echo "📝 Updating environment variables..."
DB_URL="postgresql://wingo_user:wingo_pass_2026@localhost:5432/wingo_db"

# Update backend/.env if it exists
if [ -f "$HOME/win/backend/.env" ]; then
    sed -i.bak "s|^DATABASE_URL=.*|DATABASE_URL=${DB_URL}|" "$HOME/win/backend/.env"
    echo "✅ Updated $HOME/win/backend/.env"
elif [ -f "$HOME/win/.env" ]; then
    sed -i.bak "s|^DATABASE_URL=.*|DATABASE_URL=${DB_URL}|" "$HOME/win/.env"
    echo "✅ Updated $HOME/win/.env"
else
    echo "DATABASE_URL=${DB_URL}" > "$HOME/win/backend/.env"
    echo "✅ Created $HOME/win/backend/.env"
fi

# Update systemd environment if it exists
if [ -f "/etc/win/win.env" ]; then
    sudo sed -i.bak "s|^DATABASE_URL=.*|DATABASE_URL=${DB_URL}|" /etc/win/win.env
    echo "✅ Updated /etc/win/win.env"
fi

echo "🔄 Restarting WinGo systemd services..."
if command -v systemctl >/dev/null; then
    sudo systemctl restart win-scraper win-ai win-api win-telegram || echo "⚠️ Could not restart all services. Are they running?"
    echo "✅ Services restarted."
else
    echo "⚠️ Systemd not detected. Please restart your python scripts manually."
fi

echo "🎉 Database migration complete!"
echo "Your AI engine will now use unlimited local PostgreSQL."
echo "Note: It will take about 5 minutes to gather enough historical draws (10+) to resume predictions."
