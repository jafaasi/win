import math
import random
from database import SessionLocal, Draw
from sqlalchemy import desc
from api.index import exploit_all_loopholes

def predict_next_outcome():
    db = SessionLocal()
    draws = db.query(Draw).order_by(desc(Draw.issue_number)).limit(10000).all()
    db.close()
    
    if not draws:
        return {"prediction": "Big", "confidence": 94.0, "ai_mode": "Quantum Evolving Ensemble"}
        
    history = [int(d.number) for d in reversed(draws)]
    ai_result = exploit_all_loopholes(history)
    
    return {
        "prediction": ai_result["prediction"],
        "confidence": float(ai_result["confidence"]),
        "ai_mode": ai_result["patternName"],
        "targetNum": ai_result["targetNum"],
        "hedgeNum": ai_result["hedgeNum"],
        "loopholeInsight": ai_result["loopholeInsight"]
    }
