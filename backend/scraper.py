import asyncio
import httpx
from datetime import datetime
from database import SessionLocal, Draw, PredictionLog
from ai_engine import predict_next_outcome, train_deep_learning_model

API_ENDPOINT = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"

async def fetch_wingo_draws():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_ENDPOINT}?ts={datetime.utcnow().timestamp()}")
            if response.status_code == 200:
                data = response.json()
                if "data" in data and "list" in data["data"]:
                    return data["data"]["list"]
        except Exception as e:
            print(f"Scraper Error: {e}")
    return []

def to_big_small(num):
    return 'Big' if num >= 5 else 'Small'

async def start_background_scraper():
    print("🚀 24/7 Background Deep Learning Scraper Started...")
    
    # Run an initial training cycle if DB already has data
    train_deep_learning_model()
    
    while True:
        draws = await fetch_wingo_draws()
        
        if len(draws) > 0:
            db = SessionLocal()
            
            # Sort newest first, but iterate oldest first to insert chronologically
            chronological_draws = draws[::-1]
            
            new_draws_added = False
            for draw in chronological_draws:
                issue = str(draw["issueNumber"])
                num = int(draw["number"])
                color = draw["color"]
                size = to_big_small(num)
                
                # Check if this draw already exists
                existing_draw = db.query(Draw).filter(Draw.issue_number == issue).first()
                if not existing_draw:
                    # 1. VERIFY PREVIOUS PREDICTION BEFORE SAVING DRAW
                    # If there was a prediction for THIS issue, mark it as Win/Loss!
                    pending_prediction = db.query(PredictionLog).filter(PredictionLog.issue_number == issue).first()
                    if pending_prediction and pending_prediction.actual_size is None:
                        pending_prediction.actual_size = size
                        is_win = pending_prediction.predicted_size == size
                        pending_prediction.is_win = is_win
                        
                        # Calculate next level if loss
                        if not is_win:
                            next_level = pending_prediction.martingale_level + 1
                            if next_level > 3:
                                next_level = 1 # Reset after level 3
                        else:
                            next_level = 1
                            
                    # 2. SAVE THE NEW DRAW TO DB
                    new_draw = Draw(
                        issue_number=issue,
                        number=num,
                        color=color,
                        size=size
                    )
                    db.add(new_draw)
                    db.commit()
                    new_draws_added = True
                    print(f"💾 Logged New Draw: #{issue} -> {num} ({size})")
                    
                    # 3. GENERATE AND SAVE PREDICTION FOR THE *NEXT* ISSUE
                    next_issue = str(int(issue) + 1)
                    # Use Deep Learning to predict next
                    ai_result = predict_next_outcome()
                    
                    # Determine martingale level for the next bet
                    last_pred = db.query(PredictionLog).filter(PredictionLog.issue_number == issue).first()
                    current_level = 1
                    if last_pred and last_pred.actual_size:
                        if not last_pred.is_win:
                            current_level = last_pred.martingale_level + 1
                            if current_level > 3:
                                current_level = 1
                                
                    new_pred = PredictionLog(
                        issue_number=next_issue,
                        predicted_size=ai_result["prediction"],
                        confidence=ai_result["confidence"],
                        pattern_detected=ai_result["ai_mode"],
                        martingale_level=current_level
                    )
                    db.add(new_pred)
                    db.commit()
                    print(f"🧠 Deep Learning Prediction for #{next_issue}: {ai_result['prediction']} ({ai_result['confidence']}%)")

            # Retrain model if we added new draws and have enough data
            if new_draws_added:
                # We can retrain occasionally, e.g., every 10 draws, or every draw for maximum learning
                train_deep_learning_model()
                
            db.close()
            
        # Poll every 4 seconds (WinGo 30S is very fast)
        await asyncio.sleep(4)
