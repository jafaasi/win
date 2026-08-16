from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from scraper import start_background_scraper
from database import SessionLocal, Draw, PredictionLog
from ai_engine import predict_next_outcome, is_model_trained

app = FastAPI(title="WinGo 24/7 Deep Learning Backend")

# Allow React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for Vercel + Localhost
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import threading

def run_scraper_sync():
    # Run the async scraper in a new event loop in a dedicated thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_background_scraper())

@app.on_event("startup")
async def startup_event():
    # Start the continuous 24/7 scraper in a dedicated thread to prevent blocking
    thread = threading.Thread(target=run_scraper_sync, daemon=True)
    thread.start()

@app.get("/api/state")
def get_state():
    db = SessionLocal()
    
    # 1. Get History (Last 20 draws for the UI)
    history_draws = db.query(Draw).order_by(Draw.issue_number.desc()).limit(20).all()
    history = [d.number for d in history_draws][::-1] # chronological for frontend

    # 2. Get the latest Issue Number
    latest_issue = None
    if history_draws:
        latest_issue = history_draws[0].issue_number
        
    # 3. Get Prediction Logs (Last 50 for the UI Win/Loss table)
    logs = db.query(PredictionLog).filter(PredictionLog.actual_size.isnot(None)).order_by(PredictionLog.id.desc()).limit(50).all()
    round_logs = []
    
    wins = 0
    losses = 0
    
    for log in logs:
        if log.is_win:
            wins += 1
        else:
            losses += 1
            
        round_logs.append({
            "id": log.id,
            "issue": f"#{str(log.issue_number)[-5:]}",
            "targetBS": log.predicted_size,
            "targetNum": 7 if log.predicted_size == 'Big' else 2, # simplified for ui
            "actualBS": log.actual_size,
            "isWin": log.is_win,
            "level": log.martingale_level,
            "pattern": log.pattern_detected,
            "time": log.created_at.strftime("%H:%M:%S")
        })

    # 4. Get active prediction for the upcoming draw
    active_pred = None
    if latest_issue:
        next_issue = str(int(latest_issue) + 1)
        pred_record = db.query(PredictionLog).filter(PredictionLog.issue_number == next_issue).first()
        if pred_record:
            active_pred = {
                "prediction": pred_record.predicted_size,
                "confidence": pred_record.confidence,
                "level": pred_record.martingale_level,
                "patternName": pred_record.pattern_detected,
                "targetNum": 7 if pred_record.predicted_size == 'Big' else 2,
                "hedgeNum": 8 if pred_record.predicted_size == 'Big' else 3,
                "nextIssue": next_issue
            }
            
    db.close()
    
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    
    return {
        "history": history,
        "roundLogs": round_logs,
        "latestIssue": latest_issue,
        "activePrediction": active_pred,
        "stats": {
            "totalVerified": wins + losses,
            "wins": wins,
            "losses": losses,
            "winRate": round(win_rate, 1),
            "isModelTrained": is_model_trained
        }
    }

if __name__ == "__main__":
    import uvicorn
    # Start the server on port 8080
    uvicorn.run(app, host="0.0.0.0", port=8080)
