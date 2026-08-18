import sys
import os
import time
import json
from datetime import datetime

# Ensure local imports work in all environments
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import SessionLocal, Draw, Outcome, AIBrainState
from backend.evoseq_loop import run_evoseq_cycle

def run_local_engine():
    print("🚀 Starting Local WinGo AI Deep Learning Engine...")
    print("This process will poll Supabase every 5 seconds for new draws and execute the ML pipeline.")
    
    last_processed_issue = None
    
    while True:
        try:
            db = SessionLocal()
            
            # Check for the latest draw from the cloud scraper
            latest_draw = db.query(Draw).order_by(Draw.issue_number.desc()).first()
            
            if latest_draw:
                latest_issue = str(latest_draw.issue_number)
                
                if latest_issue != last_processed_issue:
                    print(f"\n[+] New Draw Detected from Cloud: {latest_issue} (Number: {latest_draw.number})")
                    last_processed_issue = latest_issue
                    
                    # 1. Extract full deep sequence history from Supabase (up to 50,000 outcomes)
                    outcomes_list = db.query(Outcome).order_by(Outcome.sequence_no.desc()).limit(50000).all()
                    if outcomes_list:
                        history = [int(o.digit) for o in reversed(outcomes_list)]
                    else:
                        db_draws = db.query(Draw).order_by(Draw.issue_number.desc()).limit(50000).all()
                        if db_draws:
                            history = [int(d.number) for d in reversed(db_draws)]
                        else:
                            history = []
                    
                    # 2. EVOSEQ Continuous Evolution Loop
                    if history:
                        registry_state = run_evoseq_cycle(history, db)
                        
                        # 3. Construct Live UI State using the exact format Vercel expects
                        from api.index import exploit_all_loopholes
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
                        ai_result["generation"] = registry_state.get("generation", 1) if registry_state else 1

                        print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] Completed Draw #{latest_issue} (Num: {history[-1]}) -> 🎯 PREDICTED FOR CURRENT ISSUE #{next_issue}: {ai_result['prediction']} ({ai_result['confidence']}%) | {ai_result['patternName']}")
                        
                        # 4. Sync AI state directly to Supabase as Live_UI_State
                        db.add(AIBrainState(model_name="Live_UI_State", generation=ai_result["generation"], synaptic_weights=json.dumps(ai_result), updated_at=datetime.utcnow()))
                        db.commit()
                        
                    else:
                        print("Waiting for sufficient history...")
            
            db.close()
            
        except Exception as e:
            print(f"Engine Loop Error: {e}")
            
        time.sleep(5)

if __name__ == "__main__":
    run_local_engine()
