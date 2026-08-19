import os

if not os.environ.get("DATABASE_URL"):
    raise RuntimeError("DATABASE_URL is required to run this database check.")

from backend.database import SessionLocal, Draw, PredictionLog

db = SessionLocal()
draws = db.query(Draw).order_by(Draw.issue_number.desc()).limit(10).all()
print(f"Fetched {len(draws)} draws from Supabase.")
for d in draws:
    print(f"Issue {d.issue_number}: {d.number}")
db.close()
