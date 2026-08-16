from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
import sys

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

@app.on_event("startup")
async def startup_event():
    # Spawn continuous 24/7 scraper daemon in the background
    print("🚀 Starting 24/7 Autonomous AI Worker Background Task...")
    asyncio.create_task(run_scraper_daemon(max_duration_seconds=999999999))

@app.get("/")
@app.get("/healthz")
def health_check():
    return {
        "status": "online",
        "service": "WinGo 24/7 Quantum AI Engine",
        "architecture": "Render Web Service + Supabase PostgreSQL"
    }

@app.get("/api/state")
def get_api_state():
    return compute_state(init=True)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(f"📡 Starting WinGo Web Service on port {port}...")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
