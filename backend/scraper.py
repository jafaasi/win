import asyncio
import httpx
import time
from datetime import datetime
import sys
import os

# Ensure we can import from root modules when run via github actions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import SessionLocal, Draw, save_live_draws, save_prediction
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
                        
                        # 2. Extract full deep sequence history from Supabase (up to 50,000 draws)
                        db_draws = db.query(Draw).order_by(Draw.issue_number.desc()).limit(50000).all()
                        if db_draws:
                            history = [int(d.number) for d in reversed(db_draws)]
                        else:
                            history = [int(d["number"]) for d in reversed(draws)]
                        
                        # 3. EVOSEQ Continuous Evolution Loop (The heavy lifting)
                        from backend.evoseq_loop import run_evoseq_cycle
                        run_evoseq_cycle(history, db)
                        
                        # 4. Fast Edge Inference (Reads EVOSEQ_Registry)
                        ai_result = exploit_all_loopholes(history, db=db)
                        next_issue = str(int(latest_issue) + 1)
                        
                        # 4. Save prediction for the upcoming draw
                        save_prediction(
                            db=db,
                            issue_number=next_issue,
                            prediction=ai_result["prediction"],
                            confidence=ai_result["confidence"],
                            pattern_name=ai_result["patternName"]
                        )
                        print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] Draw #{latest_issue} (Number: {draws[0]['number']}) -> Next #{next_issue} Predicted: {ai_result['prediction']} ({ai_result['confidence']}%) | Loophole: {ai_result['patternName']} | Gen: {ai_result.get('generation')}")
                    finally:
                        db.close()
        except Exception as e:
            print(f"Daemon Cycle Note: {e}")
            
        # Poll every 10 seconds to catch every 30-second draw instantly
        await asyncio.sleep(10)

    print(f"✅ Daemon session finished smoothly. Processed {draws_collected} draws.")

if __name__ == "__main__":
    # If run with --once argument, run a single shot, otherwise run continuous daemon
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        asyncio.run(run_scraper_daemon(max_duration_seconds=15))
    else:
        asyncio.run(run_scraper_daemon(max_duration_seconds=18000))
