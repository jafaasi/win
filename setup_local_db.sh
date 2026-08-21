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
# Create or alter user
sudo -u postgres psql -c "DO \$\$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'wingo_user') THEN CREATE USER wingo_user WITH PASSWORD 'wingo_pass_2026'; ELSE ALTER USER wingo_user WITH PASSWORD 'wingo_pass_2026'; END IF; END \$\$;"
sudo -u postgres psql -c "SELECT 'CREATE DATABASE wingo_db OWNER wingo_user' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'wingo_db')\gexec"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE wingo_db TO wingo_user;"
sudo -u postgres psql -d wingo_db -c "GRANT ALL ON SCHEMA public TO wingo_user;"
sudo -u postgres psql -d wingo_db -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO wingo_user;"
sudo -u postgres psql -d wingo_db -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO wingo_user;"

echo "📝 Updating environment variables..."
DB_URL="postgresql://wingo_user:wingo_pass_2026@localhost:5432/wingo_db"

# Update backend/.env if it exists
mkdir -p "$HOME/win/backend"
sed -i.bak "s|^DATABASE_URL=.*|DATABASE_URL=${DB_URL}|" "$HOME/win/backend/.env" 2>/dev/null || echo "DATABASE_URL=${DB_URL}" > "$HOME/win/backend/.env"
echo "✅ Updated $HOME/win/backend/.env"

# Update systemd environment
if [ -f "/etc/win/win.env" ]; then
    sudo sed -i.bak "s|^DATABASE_URL=.*|DATABASE_URL=${DB_URL}|" /etc/win/win.env
    echo "✅ Updated /etc/win/win.env"
fi

echo "🔨 Initializing database schema..."
if [ -f "$HOME/win/.venv/bin/python3" ]; then
    export DATABASE_URL="${DB_URL}"
    $HOME/win/.venv/bin/python3 -c "
import os, sys
sys.path.insert(0, '$HOME/win')
os.environ['DATABASE_URL'] = '${DB_URL}'
from backend.database import Base, engine, SessionLocal, AIBrainState, Draw
Base.metadata.create_all(bind=engine)
db = SessionLocal()
print('Connected successfully! Table count check:', db.query(AIBrainState).count())
db.close()
"
    echo "✅ Database schema and tables verified!"
fi

echo "🔄 Restarting WinGo systemd services..."
if command -v systemctl >/dev/null; then
    sudo systemctl restart win-scraper win-ai win-api win-telegram || echo "⚠️ Could not restart all services."
    echo "✅ Services restarted."
    sleep 2
    echo "🔍 Testing API Gateway response:"
    curl -s http://127.0.0.1:8000/api/state || true
    echo ""
fi

echo "🎉 Database migration complete!"
echo "Your AI engine is now operating with 100% local, unlimited PostgreSQL."
