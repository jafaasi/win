from typing import Optional, List
import numpy as np
from .base import BaseWorker
from ..database import SessionLocal
from ..schemas import SystemEventRecord, Outcome, FeatureVectorRecord
from ..features.builder import build_features

class FeatureWorker(BaseWorker):
    """
    Feature Worker: Consumes OUTCOME_ARRIVED events, builds causal features, stores in feature_vectors,
    and publishes FEATURES_UPDATED events.
    """

    def __init__(self, poll_interval: float = 1.0):
        super().__init__(worker_name="feature_worker", poll_interval=poll_interval)

    def process_cycle(self) -> int:
        state = self.get_state()
        last_seq = state.last_processed_sequence or 0
        
        with SessionLocal() as session:
            events = session.query(SystemEventRecord).filter(
                SystemEventRecord.event_type == "OUTCOME_ARRIVED",
                SystemEventRecord.sequence_no > last_seq
            ).order_by(SystemEventRecord.sequence_no.asc()).limit(50).all()
            
            if not events:
                self.update_state(last_seq, status="idle")
                return 0
                
            for evt in events:
                seq_no = evt.sequence_no
                # Load context up to seq_no
                past_rows = session.query(Outcome).filter(Outcome.sequence_no <= seq_no).order_by(Outcome.sequence_no.desc()).limit(128).all()
                past_rows.reverse()
                
                digits = [r.digit for r in past_rows]
                sizes = [r.size for r in past_rows]
                colors = [r.color for r in past_rows]
                parities = [r.parity for r in past_rows]
                
                feat = build_features(digits, sizes, colors, parities)
                
                # Check if feature vector already exists
                existing_feat = session.query(FeatureVectorRecord).filter(FeatureVectorRecord.sequence_no == seq_no).first()
                if not existing_feat:
                    fv_rec = FeatureVectorRecord(
                        sequence_no=seq_no,
                        window_size=len(past_rows),
                        feature_vector=feat.vector.tolist(),
                        digit_entropy=feat.entropy_digit,
                        conditional_entropy_1=feat.conditional_entropy_1,
                        conditional_entropy_2=feat.conditional_entropy_2,
                        conditional_entropy_3=feat.conditional_entropy_3,
                        information_gain_1=feat.information_gain_1,
                        information_gain_2=feat.information_gain_2,
                        information_gain_3=feat.information_gain_3,
                        lz_complexity=feat.lz_complexity
                    )
                    session.add(fv_rec)
                    
                self.publish_event("FEATURES_UPDATED", seq_no, {"dim": len(feat.vector)})
                last_seq = seq_no
                
            session.commit()
            self.update_state(last_seq, status="running")
            return len(events)

if __name__ == "__main__":
    FeatureWorker().run()
