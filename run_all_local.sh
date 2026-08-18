#!/bin/bash
# Start both local components for free tier optimized setup

echo "🚀 Starting Local Components for Free Tier Setup"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install -r backend/requirements.txt
fi

# Activate virtual environment
source .venv/bin/activate

echo "📦 Installing/updating dependencies..."
pip install -q -r requirements.txt 2>/dev/null || true
pip install -q -r backend/requirements.txt 2>/dev/null || true

echo "✅ Dependencies ready"
echo ""
echo "🏠 Starting Local Scraper (Terminal 1)..."
python3 run_local_scraper.py &
SCRAPER_PID=$!

echo "🧠 Starting Local AI Engine (Terminal 2)..."
python3 backend/local_ai_engine.py &
ENGINE_PID=$!

echo ""
echo "✅ Both components started!"
echo "🏠 Scraper PID: $SCRAPER_PID"
echo "🧠 AI Engine PID: $ENGINE_PID"
echo ""
echo "Press Ctrl+C to stop both components"
echo "Or kill individually: kill $SCRAPER_PID $ENGINE_PID"

# Wait for both processes
wait $SCRAPER_PID $ENGINE_PID