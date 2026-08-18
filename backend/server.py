from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
import sys
import httpx
from datetime import datetime

# Force unbuffered output for Render logging
sys.stdout.reconfigure(line_buffering=True)

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

import subprocess

@app.on_event("startup")
async def startup_event():
    # 1. Spawn isolated daemon process
    print("🚀 Spawning completely isolated Deep Learning Daemon process...")
    try:
        subprocess.Popen([sys.executable, "backend/daemon_runner.py"])
    except Exception as e:
        print(f"Failed to spawn daemon: {e}")

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

@app.post("/api/state")
def post_api_state(payload: dict | None = None):
    try:
        if payload is None:
            return compute_state(init=True)
        return compute_state(payload, init=True)
    except Exception:
        return compute_state(init=True)

@app.get("/api/models")
def get_models_population():
    db = SessionLocal()
    try:
        from backend.database import ModelVersion
        models = db.query(ModelVersion).order_by(ModelVersion.id.desc()).limit(50).all()
        return {
            "count": len(models),
            "models": [
                {
                    "id": m.id,
                    "model_name": m.model_name,
                    "version": m.version,
                    "validation_score": m.validation_score,
                    "log_loss": m.log_loss,
                    "brier_score": m.brier_score,
                    "status": m.status,
                    "created_at": m.created_at.isoformat() if m.created_at else None
                }
                for m in models
            ]
        }
    finally:
        db.close()

@app.get("/api/models/champion")
def get_champion_model():
    state = compute_state(init=True)
    stats = state.get("evolutionStats", {})
    return {
        "champion": stats.get("championModel", "SSM-Mamba-v1"),
        "generation": stats.get("modelGeneration", 1),
        "predictive_score": stats.get("predictiveScore", 0.54),
        "log_loss": stats.get("logLoss", 0.65),
        "brier_score": stats.get("brierScore", 0.20),
        "null_advantage": stats.get("nullAdvantage", 0.04)
    }

@app.get("/api/ensemble")
def get_ensemble_details():
    state = compute_state(init=True)
    pred = state.get("activePrediction", {})
    return {
        "familyWeights": pred.get("familyWeights", {"statistical": 0.35, "recurrent": 0.35, "neural": 0.30}),
        "modelDisagreement": pred.get("modelDisagreement", 0.045),
        "aleatoricEntropy": pred.get("aleatoricEntropy", 3.22),
        "h1": pred.get("h1", [0.1]*10),
        "h2": pred.get("h2", [0.1]*10),
        "h3": pred.get("h3", [0.1]*10)
    }

@app.get("/api/environment")
def get_environment_state():
    state = compute_state(init=True)
    stats = state.get("evolutionStats", {})
    pred = state.get("activePrediction", {})
    return {
        "entropy": stats.get("entropy", 3.22),
        "driftLevel": stats.get("driftLevel", "LOW"),
        "driftScore": stats.get("driftScore", 0.02),
        "regime": stats.get("regimeProbabilities", {}),
        "environmentVector": pred.get("environmentVector", [3.22, 0.08, 0.03, 0.02, 0.12, 0.34, 0.045])
    }

@app.get("/api/evolution/genealogy")
def get_evolution_genealogy():
    state = compute_state(init=True)
    stats = state.get("evolutionStats", {})
    return {
        "generation": stats.get("modelGeneration", 1),
        "champion": stats.get("championModel", "SSM-Mamba-v1"),
        "modelsTested": stats.get("modelsTested", 128),
        "activeChallengers": stats.get("activeChallengers", 5),
        "retiredModels": stats.get("retiredModels", 122),
        "familySurvivalRates": {
            "Statistical": 0.33,
            "Recurrent": 0.33,
            "Neural": 0.34,
            "StateSpace": 0.50
        }
    }

@app.get("/api/experiments")
def get_experiments_list():
    db = SessionLocal()
    try:
        from backend.database import ExperimentResult, ModelCandidate
        exp_list = db.query(ExperimentResult).order_by(ExperimentResult.id.desc()).limit(20).all()
        return {
            "count": len(exp_list),
            "experiments": [
                {
                    "id": e.id,
                    "candidate_id": e.candidate_id,
                    "fold": e.fold,
                    "seed": e.seed,
                    "log_loss": e.log_loss,
                    "brier_score": e.brier_score,
                    "accuracy": e.accuracy,
                    "null_p_value": e.null_p_value
                }
                for e in exp_list
            ]
        }
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(f"📡 Starting WinGo Web Service on port {port}...")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
