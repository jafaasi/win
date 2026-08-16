import asyncio
import httpx
from datetime import datetime
import sys
import os

# Ensure we can import from root modules when run via github actions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import SessionLocal, save_live_draws, save_prediction
from api.index import exploit_all_loopholes

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

async def run_scraper_once():
    print("🚀 Running 24/7 Background Deep Learning Scraper...")
    draws = await fetch_wingo_draws()
    if len(draws) > 0:
        db = SessionLocal()
        try:
            # Save the draws and update any pending prediction win/loss records
            new_draws = save_live_draws(db, draws)
            print(f"💾 Synced {new_draws} new draws to Supabase.")
            
            # Now, generate the prediction for the NEXT issue
            latest_issue = str(draws[0]["issueNumber"])
            next_issue = str(int(latest_issue) + 1)
            
            # Build history from draws
            history = [int(d["number"]) for d in reversed(draws)]
            
            # Predict using deterministic DeepMLP
            ai_result = exploit_all_loopholes(history)
            
            save_prediction(
                db=db,
                issue_number=next_issue,
                prediction=ai_result["prediction"],
                confidence=ai_result["confidence"],
                pattern_name=ai_result["patternName"]
            )
            print(f"🧠 Logged Prediction for #{next_issue}: {ai_result['prediction']} ({ai_result['confidence']}%)")
        finally:
            db.close()
    else:
        print("⚠️ Failed to fetch live draws.")

if __name__ == "__main__":
    asyncio.run(run_scraper_once())
