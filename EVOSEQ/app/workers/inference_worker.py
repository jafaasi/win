from typing import Optional, List, Dict, Any
import numpy as np
from .base import BaseWorker
from ..database import SessionLocal
from ..schemas import SystemEventRecord, Outcome, PredictionRecord, ModelVersionRecord
from ..models.markov import MarkovModel
from ..models.uniform import UniformModel

class InferenceWorker(BaseWorker):
    """
    Inference Worker: Consumes FEATURES_UPDATED events, runs active champion / models,
    and stores probabilities in predictions table while publishing PREDICTION_GENERATED.
    """

    def __init__(self, poll_interval: float = 1.0):
        super().__init__(worker_name="inference_worker", poll_interval=poll_interval)

    def process_cycle(self) -> int:
        state = self.get_state()
        last_seq = state.last_processed_sequence or 0
        
        with SessionLocal() as session:
            events = session.query(SystemEventRecord).filter(
                SystemEventRecord.event_type == "FEATURES_UPDATED",
                SystemEventRecord.sequence_no > last_seq
            ).order_by(SystemEventRecord.sequence_no.asc()).limit(50).all()
            
            if not events:
                self.update_state(last_seq, status="idle")
                return 0
                
            # Get active champion
            champ = session.query(ModelVersionRecord).filter(ModelVersionRecord.status == "champion").order_by(ModelVersionRecord.id.desc()).first()
            
            for evt in events:
                seq_no = evt.sequence_no
                past_rows = session.query(Outcome).filter(Outcome.sequence_no <= seq_no).order_by(Outcome.sequence_no.desc()).limit(64).all()
                past_rows.reverse()
                digits = [r.digit for r in past_rows]
                
                # Predict
                if champ and champ.model_name == "Markov":
                    mkv = MarkovModel(order=champ.parameters.get("order", 2), smoothing=champ.parameters.get("smoothing", 0.5))
                    mkv.fit(digits)
                    probs = mkv.predict_proba(digits)
                    model_id = champ.id
                else:
                    probs = np.full(10, 0.1, dtype=np.float64)
                    model_id = champ.id if champ else 1
                    
                target_seq = seq_no + 1
                existing_pred = session.query(PredictionRecord).filter(
                    PredictionRecord.model_version_id == model_id,
                    PredictionRecord.sequence_no == target_seq
                ).first()
                
                if not existing_pred:
                    pred_rec = PredictionRecord(
                        model_version_id=model_id,
                        sequence_no=target_seq,
                        probability_vector=probs.tolist(),
                        predicted_class=int(np.argmax(probs))
                    )
                    session.add(pred_rec)
                    
                self.publish_event("PREDICTION_GENERATED", seq_no, {"model_id": model_id, "target_seq": target_seq}, model_version_id=model_id)
                last_seq = seq_no
                
            session.commit()
            self.update_state(last_seq, status="running")
            return len(events)

if __name__ == "__main__":
    InferenceWorker().run()
