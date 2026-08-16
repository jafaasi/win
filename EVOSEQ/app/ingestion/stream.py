from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import text
from ..database import SessionLocal
from ..schemas import Outcome

def get_latest_sequence() -> int:
    """Returns the highest sequence_no currently recorded in the database."""
    with SessionLocal() as session:
        result = session.execute(
            text(
                """
                SELECT COALESCE(MAX(sequence_no), 0)
                FROM outcomes
                """
            )
        )
        val = result.scalar()
        return int(val) if val is not None else 0

def fetch_after(sequence_no: int, limit: int = 500) -> List[Dict[str, Any]]:
    """Streams observations after a given sequence_no cursor."""
    with SessionLocal() as session:
        result = session.execute(
            text(
                """
                SELECT
                    sequence_no,
                    timestamp_utc,
                    digit,
                    size,
                    color,
                    parity
                FROM outcomes
                WHERE sequence_no > :sequence_no
                ORDER BY sequence_no ASC
                LIMIT :limit
                """
            ),
            {
                "sequence_no": sequence_no,
                "limit": limit,
            },
        )
        return [dict(row) for row in result.mappings().all()]

def ingest_outcomes_batch(raw_items: List[Dict[str, Any]]) -> int:
    """Safely saves a batch of raw outcomes to the database, ignoring duplicates."""
    if not raw_items:
        return 0
        
    inserted = 0
    with SessionLocal() as session:
        for item in raw_items:
            seq = int(item["sequence_no"])
            digit = int(item["digit"])
            size = 1 if digit >= 5 else 0
            # Color: 1 = green (1,3,7,9), 2 = violet (0,5), 0 = red (2,4,6,8)
            color = 1 if digit in [1, 3, 7, 9] else 2 if digit in [0, 5] else 0
            parity = 1 if digit % 2 != 0 else 0
            
            existing = session.query(Outcome).filter(Outcome.sequence_no == seq).first()
            if not existing:
                ts = item.get("timestamp_utc") or datetime.utcnow()
                out = Outcome(
                    sequence_no=seq,
                    timestamp_utc=ts,
                    digit=digit,
                    size=size,
                    color=color,
                    parity=parity
                )
                session.add(out)
                inserted += 1
        try:
            session.commit()
        except Exception as e:
            session.rollback()
            print("Ingest batch commit note:", e)
            
    return inserted
