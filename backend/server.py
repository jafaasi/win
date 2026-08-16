from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
import sys
import httpx
from datetime import datetime

# Ensure local imports work in all environments
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.scraper import run_scraper_daemon
from backend.database import SessionLocal, Draw, PredictionLog
from api.index import compute_state

app = FastAPI(title="WinGo 24/7 Deep Learning AI Server")

# Allow all origins for Vercel + Localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def keep_alive_self_pinger():
    """
    Prevents Render free tier from sleeping by sending an external HTTP request
    to its own public domain every 8 minutes (well before the 15-minute timeout).
    """
    await asyncio.sleep(20) # Wait for server boot
    
    external_url = os.environ.get("RENDER_EXTERNAL_URL")
    if not external_url:
        host = os.environ.get("RENDER_EXTERNAL_HOSTNAME")
        if host:
            external_url = f"https://{host}"
            
    print(f"💓 Keep-Alive Daemon Online (Target: {external_url or 'Local Loopback'})")
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        while True:
            try:
                # Refresh external URL in case it was assigned after boot
                url = os.environ.get("RENDER_EXTERNAL_URL") or external_url
                if not url:
                    host = os.environ.get("RENDER_EXTERNAL_HOSTNAME")
                    if host:
                        url = f"https://{host}"

                if url:
                    target = f"{url.rstrip('/')}/healthz"
                    res = await client.get(target)
                    print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] 💓 Self-pinged {target} (Status: {res.status_code}) -> Render sleep timer reset.")
                else:
                    port = os.environ.get("PORT", 8080)
                    res = await client.get(f"http://127.0.0.1:{port}/healthz")
                    print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] 💓 Local loopback ping (Status: {res.status_code})")
            except Exception as e:
                print(f"Keep-Alive ping note: {e}")
                
            # Ping every 8 minutes (480 seconds)
            await asyncio.sleep(480)

@app.on_event("startup")
async def startup_event():
    # 1. Spawn continuous 24/7 scraper daemon
    print("🚀 Starting 24/7 Autonomous AI Worker Background Task...")
    asyncio.create_task(run_scraper_daemon(max_duration_seconds=999999999))
    
    # 2. Spawn keep-alive self-pinger
    print("💓 Spawning Auto Keep-Alive Self-Pinger...")
    asyncio.create_task(keep_alive_self_pinger())

@app.get("/")
@app.get("/healthz")
def health_check():
    return {
        "status": "online",
        "service": "WinGo 24/7 Quantum AI Engine",
        "architecture": "Render Web Service + Supabase PostgreSQL",
        "keep_alive": "Self-Pinging Active"
    }

@app.get("/api/state")
def get_api_state():
    return compute_state(init=True)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(f"📡 Starting WinGo Web Service on port {port}...")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
