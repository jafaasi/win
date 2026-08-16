import os
os.environ["DATABASE_URL"] = "postgresql://postgres:rodrE0%2Dfyvnov%2Dgyvzuz@db.zyryxnifpduwsulglhdq.supabase.co:5432/postgres"

from backend.database import SessionLocal, Draw, PredictionLog

db = SessionLocal()
draws = db.query(Draw).order_by(Draw.issue_number.desc()).limit(10).all()
print(f"Fetched {len(draws)} draws from Supabase.")
for d in draws:
    print(f"Issue {d.issue_number}: {d.number}")
db.close()
