import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from dotenv import load_dotenv

load_dotenv() # Load variables from .env file

# Use cloud DATABASE_URL if provided, else fallback to local SQLite
DATABASE_URL = os.environ.get("DATABASE_URL")

if DATABASE_URL:
    # Use standard SQLAlchemy connection for Postgres/MySQL
    # Fix for Heroku/Render postgres:// -> postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(DATABASE_URL)
else:
    # Fallback to local SQLite
    DB_PATH = os.path.join(os.path.dirname(__file__), 'wingo_history.db')
    engine = create_engine(f'sqlite:///{DB_PATH}', connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Draw(Base):
    """
    Stores every single raw outcome pulled from the WinGo 30S API.
    """
    __tablename__ = "draws"

    id = Column(Integer, primary_key=True, index=True)
    issue_number = Column(String, unique=True, index=True)
    number = Column(Integer)
    color = Column(String)
    size = Column(String)  # 'Big' or 'Small'
    created_at = Column(DateTime, default=datetime.utcnow)

class PredictionLog(Base):
    """
    Stores the AI's prediction for a given issue, and whether it won or lost.
    """
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    issue_number = Column(String, unique=True, index=True)
    predicted_size = Column(String)  # 'Big' or 'Small'
    confidence = Column(Float)
    actual_size = Column(String, nullable=True) # filled after the draw happens
    is_win = Column(Boolean, nullable=True)
    martingale_level = Column(Integer, default=1)
    pattern_detected = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create all tables in the database
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def to_big_small(num):
    return 'Big' if int(num) >= 5 else 'Small'

def save_live_draws(db, live_draws):
    """
    Safely saves a list of live draws from the WinGo API to Supabase.
    Skips duplicates based on issue_number.
    """
    new_draws = 0
    for item in reversed(live_draws):
        issue = str(item.get("issueNumber"))
        num = int(item.get("number"))
        
        # Check if exists
        existing = db.query(Draw).filter(Draw.issue_number == issue).first()
        if not existing:
            new_draw = Draw(
                issue_number=issue,
                number=num,
                color="green" if num in [1,3,7,9] else "violet" if num in [0,5] else "red",
                size=to_big_small(num)
            )
            db.add(new_draw)
            new_draws += 1
            
            # Also update pending PredictionLog if it exists
            pending_log = db.query(PredictionLog).filter(PredictionLog.issue_number == issue).first()
            if pending_log and pending_log.actual_size is None:
                pending_log.actual_size = new_draw.size
                pending_log.is_win = (pending_log.predicted_size == new_draw.size)
    if new_draws > 0:
        db.commit()
    return new_draws

def save_prediction(db, issue_number, prediction, confidence, pattern_name):
    """
    Saves a prediction for a future issue.
    """
    existing = db.query(PredictionLog).filter(PredictionLog.issue_number == issue_number).first()
    if not existing:
        log = PredictionLog(
            issue_number=issue_number,
            predicted_size=prediction,
            confidence=confidence,
            pattern_detected=pattern_name
        )
        db.add(log)
        db.commit()
