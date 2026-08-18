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

from backend.database import SessionLocal, Outcome, Draw, save_live_draws, save_prediction
from api.index import exploit_all_loopholes

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

async def run_scraper_daemon(max_duration_seconds=18000): # 5 hours per job
    start_time = time.time()
    print(f"🚀 Starting 24/7 WinGo AI Deep Learning Daemon (Session limit: {max_duration_seconds // 3600} hours)...")
    
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
                        # 1. Sync draws & verify pending prediction logs
                        new_draws = save_live_draws(db, draws)
                        
                        # 2. Extract full deep sequence history from Supabase (up to 50,000 outcomes)
                        outcomes_list = db.query(Outcome).order_by(Outcome.sequence_no.desc()).limit(50000).all()
                        if outcomes_list:
                            history = [int(o.digit) for o in reversed(outcomes_list)]
                        else:
                            db_draws = db.query(Draw).order_by(Draw.issue_number.desc()).limit(50000).all()
                            if db_draws:
                                history = [int(d.number) for d in reversed(db_draws)]
                            else:
                                history = [int(d["number"]) for d in reversed(draws)]
                        
                        # 3. EVOSEQ Continuous Evolution Loop (Non-blocking worker thread)
                        from backend.evoseq_loop import run_evoseq_cycle
                        import json
                        registry_state = await asyncio.to_thread(run_evoseq_cycle, history, db)

                        
                        # 4. Fast Edge Inference (Reads EVOSEQ_Registry)
                        ai_result = exploit_all_loopholes(history, db=db)
                        next_issue = str(int(latest_issue) + 1)
                        
                        # --- MERGE EVOSEQ PYTORCH INFERENCE ---
                        if registry_state and registry_state.get("live_inference"):
                            li = registry_state["live_inference"]
                            is_big = li["prediction"] == "Big"
                            prob = li["probability_big"] if is_big else li["probability_small"]
                            
                            ai_result["prediction"] = li["prediction"]
                            ai_result["confidence"] = li.get("confidence", round(prob * 100, 1))
                            ai_result["targetNum"] = li["targetNum"]
                            ai_result["hedgeNum"] = li["hedgeNum"]
                            ai_result["patternName"] = f"🧬 {registry_state.get('champion_id', 'SSM')} Deep Neural Engine"
                            ai_result["loopholeInsight"] = f"PyTorch EVOSEQ Champion deployed. Validated Walk-Forward Backtest with {registry_state.get('calibration_quality', 0.99)} calibration quality."
                            
                        ai_result["currentIssue"] = latest_issue
                        ai_result["nextIssue"] = next_issue
                        ai_result["latestIssue"] = latest_issue

                        # Save prediction for the upcoming draw before the issue window expires
                        save_prediction(
                            db=db,
                            issue_number=next_issue,
                            prediction=ai_result["prediction"],
                            confidence=ai_result["confidence"],
                            pattern_name=ai_result["patternName"]
                        )
                        
                        # --- NEW: SAVE FULL STATE FOR VERCEL ---
                        from backend.database import save_ai_brain_state
                        save_ai_brain_state(
                            db=db,
                            model_name="Live_UI_State",
                            generation=registry_state.get("generation", 1) if registry_state else 1,
                            total_samples=len(history),
                            weights_json=json.dumps(ai_result),
                            win_rate=registry_state.get("fitness", 50.0) if registry_state else 50.0
                        )
                        print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] Completed Draw #{latest_issue} (Num: {draws[0]['number']}) -> 🎯 PREDICTED FOR CURRENT ISSUE #{next_issue}: {ai_result['prediction']} ({ai_result['confidence']}%) | {ai_result['patternName']}")
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
    task = asyncio.create_task(run_scraper_daemon(max_duration_seconds=18000))
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "AI Engine Daemon is running 24/7", "service": "EVOSEQ WinGo Brain"}

@app.get("/health")
@app.get("/api/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    # If run with --once argument, run a single shot, otherwise run continuous daemon
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        asyncio.run(run_scraper_daemon(max_duration_seconds=15))
    else:
        port = int(os.environ.get("PORT", 10000))
        print(f"🚀 Starting FastAPI Server on 0.0.0.0:{port} for Render compatibility...", flush=True)
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
