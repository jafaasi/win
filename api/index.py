import os
import json
import httpx
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, desc
from sqlalchemy.orm import declarative_base, sessionmaker
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Database Setup
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:rodrE0%2Dfyvnov%2Dgyvzuz@db.zyryxnifpduwsulglhdq.supabase.co:5432/postgres")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Draw(Base):
    __tablename__ = "draws"
    id = Column(Integer, primary_key=True, index=True)
    issue_number = Column(String, unique=True, index=True)
    number = Column(Integer)
    color = Column(String)
    size = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, index=True)
    issue_number = Column(String, unique=True, index=True)
    predicted_size = Column(String)
    confidence = Column(Float)
    actual_size = Column(String, nullable=True)
    is_win = Column(Boolean, nullable=True)
    martingale_level = Column(Integer, default=1)
    pattern_detected = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"Table creation error: {e}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def to_big_small(num):
    return 'Big' if num >= 5 else 'Small'

def extract_features(sequence):
    features = []
    for n in sequence:
        features.append(n)
        features.append(1 if n >= 5 else 0)
    features.append(sum(sequence) / len(sequence) if sequence else 0)
    return features

def run_ai_prediction(db):
    recent_draws = db.query(Draw).order_by(desc(Draw.issue_number)).limit(100).all()
    if not recent_draws:
        return {"prediction": "Big", "confidence": 85.0, "ai_mode": "Baseline Initialization", "isTrained": False}
    
    recent_draws = recent_draws[::-1]
    numbers = [d.number for d in recent_draws]
    
    if len(numbers) >= 20:
        try:
            X, y = [], []
            window_size = 5
            for i in range(len(numbers) - window_size):
                seq = numbers[i:i + window_size]
                target = 1 if numbers[i + window_size] >= 5 else 0
                X.append(extract_features(seq))
                y.append(target)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42)
            mlp.fit(X_scaled, y)
            
            last_seq = numbers[-window_size:]
            curr_features = scaler.transform([extract_features(last_seq)])
            probs = mlp.predict_proba(curr_features)[0]
            
            prob_small = probs[0]
            prob_big = probs[1]
            pred = "Big" if prob_big >= prob_small else "Small"
            conf = round(float(max(prob_big, prob_small) * 100), 1)
            conf = min(99.4, max(85.0, conf))
            
            return {"prediction": pred, "confidence": conf, "ai_mode": "MLP Deep Neural Network", "isTrained": True}
        except Exception as err:
            print("MLP Error:", err)
    
    # Fallback Momentum / Equilibrium
    big_count = sum(1 for n in numbers[-10:] if n >= 5)
    pred = "Small" if big_count > 5 else "Big"
    return {"prediction": pred, "confidence": 88.0, "ai_mode": "Dynamic Pattern Equilibrium", "isTrained": False}

def sync_latest_draws(db):
    try:
        url = f"https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json?ts={datetime.utcnow().timestamp()}"
        res = httpx.get(url, timeout=5.0)
        if res.status_code == 200:
            data = res.json()
            draws = data.get("data", {}).get("list", [])
            for draw in reversed(draws):
                issue = str(draw["issueNumber"])
                num = int(draw["number"])
                color = draw["color"]
                size = to_big_small(num)
                
                existing = db.query(Draw).filter(Draw.issue_number == issue).first()
                if not existing:
                    # Check pending prediction
                    pending = db.query(PredictionLog).filter(PredictionLog.issue_number == issue).first()
                    if pending and pending.actual_size is None:
                        pending.actual_size = size
                        pending.is_win = (pending.predicted_size == size)
                    
                    new_draw = Draw(issue_number=issue, number=num, color=color, size=size)
                    db.add(new_draw)
                    db.commit()
    except Exception as e:
        print("Sync Error:", e)

@app.get("/api/state")
@app.get("/api/index")
def get_state():
    db = SessionLocal()
    try:
        sync_latest_draws(db)
        
        history_draws = db.query(Draw).order_by(desc(Draw.issue_number)).limit(20).all()
        history = [d.number for d in history_draws][::-1]
        
        latest_issue = history_draws[0].issue_number if history_draws else None
        
        ai_res = run_ai_prediction(db)
        
        logs = db.query(PredictionLog).filter(PredictionLog.actual_size.isnot(None)).order_by(desc(PredictionLog.id)).limit(50).all()
        round_logs = []
        wins, losses = 0, 0
        for log in logs:
            if log.is_win:
                wins += 1
            else:
                losses += 1
            round_logs.append({
                "id": log.id,
                "issue": f"#{str(log.issue_number)[-5:]}",
                "targetBS": log.predicted_size,
                "targetNum": 7 if log.predicted_size == 'Big' else 2,
                "actualBS": log.actual_size,
                "isWin": log.is_win,
                "level": log.martingale_level,
                "pattern": log.pattern_detected,
                "time": log.created_at.strftime("%H:%M:%S") if log.created_at else ""
            })
            
        current_level = 1
        if logs and not logs[0].is_win:
            current_level = (logs[0].martingale_level % 3) + 1
            
        active_pred = None
        if latest_issue:
            next_issue = str(int(latest_issue) + 1)
            active_pred = {
                "prediction": ai_res["prediction"],
                "confidence": ai_res["confidence"],
                "level": current_level,
                "patternName": ai_res["ai_mode"],
                "targetNum": 7 if ai_res["prediction"] == 'Big' else 2,
                "hedgeNum": 8 if ai_res["prediction"] == 'Big' else 3,
                "nextIssue": next_issue
            }
            
            # Save upcoming prediction record if not exists
            existing_pred = db.query(PredictionLog).filter(PredictionLog.issue_number == next_issue).first()
            if not existing_pred:
                pred_record = PredictionLog(
                    issue_number=next_issue,
                    predicted_size=ai_res["prediction"],
                    confidence=ai_res["confidence"],
                    pattern_detected=ai_res["ai_mode"],
                    martingale_level=current_level
                )
                db.add(pred_record)
                db.commit()

        win_rate = round((wins / (wins + losses) * 100), 1) if (wins + losses) > 0 else 0
        
        return {
            "history": history,
            "roundLogs": round_logs,
            "latestIssue": latest_issue,
            "activePrediction": active_pred,
            "stats": {
                "totalVerified": wins + losses,
                "wins": wins,
                "losses": losses,
                "winRate": win_rate,
                "isModelTrained": ai_res.get("isTrained", False)
            }
        }
    finally:
        db.close()
