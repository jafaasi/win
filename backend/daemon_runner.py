import sys
import os
import asyncio

# STRICT SINGLE-THREADING FOR RENDER FREE TIER TO PREVENT LIVELOCKS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Force unbuffered output for Render logging
sys.stdout.reconfigure(line_buffering=True)

# Ensure local imports work in all environments
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.scraper import run_scraper_daemon
from backend.database import SessionLocal, AIBrainState
from datetime import datetime

def remote_log(msg):
    try:
        db = SessionLocal()
        brain = db.query(AIBrainState).filter(AIBrainState.model_name == "DAEMON_LOG").first()
        full_msg = f"[{datetime.utcnow().isoformat()}] {msg}\n"
        if not brain:
            db.add(AIBrainState(model_name="DAEMON_LOG", synaptic_weights=full_msg))
        else:
            brain.synaptic_weights = (brain.synaptic_weights or "") + full_msg
            # keep only last 5000 chars
            brain.synaptic_weights = brain.synaptic_weights[-5000:]
            brain.updated_at = datetime.utcnow()
        db.commit()
        db.close()
    except Exception as e:
        print(f"Failed to remote log: {e}")

if __name__ == "__main__":
    # Lower CPU priority to absolute minimum so uvicorn web server can instantly answer health checks
    try:
        os.nice(19)
    except Exception as e:
        print(f"Note: Could not set nice value: {e}")

    msg = "🚀 Starting Isolated Daemon Process for PyTorch Engine..."
    print(msg)
    remote_log(msg)
    try:
        asyncio.run(run_scraper_daemon(max_duration_seconds=999999999))
    except Exception as e:
        remote_log(f"CRASH: {str(e)}")
        raise e
