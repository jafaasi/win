import sys
import os
import numpy as np

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.schemas import Outcome, FeatureVectorRecord
from app.features.builder import extract_causal_feature_vector
from app.features.entropy import categorical_entropy
from app.features.conditional_entropy import conditional_entropy
from app.features.information import information_gain
from app.features.lz import lz_complexity

def backfill(window_size: int = 128):
    print("🔄 Backfilling causal feature vectors...")
    with SessionLocal() as session:
        outcomes = session.query(Outcome).order_by(Outcome.sequence_no.asc()).all()
        if not outcomes:
            print("⚠️ No outcomes found in database. Ingesting synthetic seed stream...")
            from app.ingestion.stream import ingest_outcomes_batch
            seed_batch = [{"sequence_no": i + 1, "digit": int(np.random.randint(0, 10))} for i in range(200)]
            ingest_outcomes_batch(seed_batch)
            outcomes = session.query(Outcome).order_by(Outcome.sequence_no.asc()).all()
            
        digits = [o.digit for o in outcomes]
        inserted = 0
        
        for i in range(10, len(digits)):
            seq_no = outcomes[i].sequence_no
            existing = session.query(FeatureVectorRecord).filter(FeatureVectorRecord.sequence_no == seq_no).first()
            if existing:
                continue
                
            hist = digits[max(0, i - window_size + 1): i + 1]
            vec = extract_causal_feature_vector(hist, window_size=window_size)
            h = categorical_entropy(hist)
            h1 = conditional_entropy(hist, order=1)
            ig1 = information_gain(hist, order=1)
            lz = lz_complexity(hist)
            
            rec = FeatureVectorRecord(
                sequence_no=seq_no,
                feature_vector=vec.tolist(),
                digit_entropy=float(h),
                conditional_entropy_1=float(h1),
                information_gain_1=float(ig1),
                lz_complexity=float(lz),
                window_size=window_size
            )
            session.add(rec)
            inserted += 1
            
        session.commit()
        print(f"✅ Backfill completed: {inserted} feature vectors saved.")

if __name__ == "__main__":
    backfill()
