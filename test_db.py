import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backend.database import SessionLocal, AIBrainState
db = SessionLocal()
try:
    db.query(AIBrainState).first()
    print("Query OK")
except Exception as e:
    print("Error:", e)
