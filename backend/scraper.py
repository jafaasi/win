import asyncio
import httpx
import time
from datetime import datetime
import sys
import os
from fastapi import FastAPI
import uvicorn

# Ensure we can import from root modules when run via github actions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import SessionLocal, save_live_draws

API_ENDPOINT = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"

async def fetch_wingo_draws():
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{API_ENDPOINT}?ts={datetime.utcnow().timestamp()}")
            if response.status_code == 200:
                data = response.json()
                if "data" in data and "list" in data["data"]:
                    return data["data"]["list"]
        except Exception as e:
            print(f"Fetch Note: {e}")
    return []

async def run_scraper_daemon(max_duration_seconds=18000):
    """
    🏠 LOCAL SCRAPER: Runs on your local machine to collect and store outcomes
    - Polls WinGo API every 1.5 seconds
    - Stores outcomes in Supabase/PostgreSQL database
    - Does NOT run ML or make predictions
    - Leaves all intelligence to the local AI engine
    - Optimized for free tier: runs locally, stores to cloud database
    """
    start_time = time.time()
    print(f"🏠 Starting LOCAL Scraper (Historical Data Collection)")
    print(f"📊 Session limit: {max_duration_seconds // 3600} hours")
    print(f"⚠️  NOTE: Prediction engine runs separately via local_ai_engine.py")
    print(f"💾 Stores outcomes to cloud database (Supabase/PostgreSQL)")
    
    last_processed_issue = None
    draws_collected = 0
    
    while time.time() - start_time < max_duration_seconds:
        try:
            draws = await fetch_wingo_draws()
            if draws:
                latest_issue = str(draws[0]["issueNumber"])
                
                if latest_issue != last_processed_issue:
                    last_processed_issue = latest_issue
                    draws_collected += 1
                    
                    db = SessionLocal()
                    try:
                        # 1. ONLY sync draws to database - no prediction logic
                        new_draws = save_live_draws(db, draws)
                        print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] 🏠 Outcome #{latest_issue}: {draws[0]['number']} | {draws_collected} collected | Database sync complete")
                    finally:
                        db.close()
        except Exception as e:
            print(f"Daemon Cycle Note: {e}")
            
        # Poll every 1.5 seconds to catch every 30-second draw within 1 second of drawing
        await asyncio.sleep(1.5)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Launch the infinite scraper daemon in the background of the FastAPI event loop
    task = asyncio.create_task(run_scraper_daemon(max_duration_seconds=999999999))
    yield
    task.cancel()

app = FastAPI(title="WinGo Local Scraper (Historical Data Only)")

# Allow all origins for Vercel + Localhost
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # Spawn local scraper daemon
    print("🏠 Spawning Local Scraper...")
    from backend.scraper import run_scraper_daemon
    asyncio.create_task(run_scraper_daemon(max_duration_seconds=999999999))

@app.get("/")
@app.get("/healthz")
def health_check():
    return {
        "status": "online",
        "service": "WinGo Local Scraper",
        "role": "Historical Data Collection Only",
        "note": "Stores outcomes to cloud database. Predictions via local_ai_engine.py"
    }

@app.get("/api/health")
def api_health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    # If run with --once argument, run a single shot, otherwise run continuous daemon
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        asyncio.run(run_scraper_daemon(max_duration_seconds=15))
    else:
        port = int(os.environ.get("PORT", 10000))
        print(f"🚀 Starting Local Scraper on 0.0.0.0:{port}...", flush=True)
        print("💾 Free Tier Optimized: Stores to cloud database, runs locally", flush=True)
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

