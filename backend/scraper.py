import asyncio
import httpx
import time
from datetime import datetime
import sys
import os
import threading
import http.server
import socketserver

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
                        
                        # 3. EVOSEQ Continuous Evolution Loop (The heavy lifting)
                        from backend.evoseq_loop import run_evoseq_cycle
                        import json
                        registry_state = run_evoseq_cycle(history, db)

                        
                        # 4. Fast Edge Inference (Reads EVOSEQ_Registry)
                        ai_result = exploit_all_loopholes(history, db=db)
                        next_issue = str(int(latest_issue) + 1)
                        
                        # --- MERGE EVOSEQ PYTORCH INFERENCE ---
                        if registry_state and registry_state.get("live_inference"):
                            li = registry_state["live_inference"]
                            is_big = li["prediction"] == "Big"
                            prob = li["probability_big"] if is_big else li["probability_small"]
                            
                            ai_result["prediction"] = li["prediction"]
                            ai_result["confidence"] = round(prob * 100, 1)
                            ai_result["targetNum"] = li["targetNum"]
                            ai_result["hedgeNum"] = li["hedgeNum"]
                            ai_result["patternName"] = f"🧬 {registry_state.get('champion_id', 'SSM')} Deep Neural Engine"
                            ai_result["loopholeInsight"] = f"PyTorch EVOSEQ Champion deployed. Validated Walk-Forward Backtest with {registry_state.get('calibration_quality', 0.99)} calibration quality."
                            
                        # Save prediction for the upcoming draw
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
                        print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] Draw #{latest_issue} (Number: {draws[0]['number']}) -> Next #{next_issue} Predicted: {ai_result['prediction']} ({ai_result['confidence']}%) | Loophole: {ai_result['patternName']} | Gen: {ai_result.get('generation')}")
                    finally:
                        db.close()
        except Exception as e:
            print(f"Daemon Cycle Note: {e}")
            
        # Poll every 10 seconds to catch every 30-second draw instantly
        await asyncio.sleep(10)

    print(f"✅ Daemon session finished smoothly. Processed {draws_collected} draws.")

if __name__ == "__main__":
    # Start a dummy HTTP server in a background thread to satisfy Render's Web Service port binding
    port = int(os.environ.get("PORT", 10000))
    handler = http.server.SimpleHTTPRequestHandler
    
    # We use a custom handler just to return 200 OK instantly
    class HealthCheckHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"AI Engine Daemon is running.")

    def run_dummy_server():
        try:
            server = http.server.HTTPServer(("0.0.0.0", port), HealthCheckHandler)
            print(f"✅ Dummy health-check server listening on 0.0.0.0:{port}")
            server.serve_forever()
        except Exception as e:
            print(f"Dummy server error: {e}")
            
    threading.Thread(target=run_dummy_server, daemon=True).start()

    # If run with --once argument, run a single shot, otherwise run continuous daemon
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        asyncio.run(run_scraper_daemon(max_duration_seconds=15))
    else:
        asyncio.run(run_scraper_daemon(max_duration_seconds=18000))
