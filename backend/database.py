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
