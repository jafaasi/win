import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from ..database import SessionLocal
from ..schemas import ModelVersionRecord, ModelEvent

class ModelRegistry:
    """Manages model lifecycle, genealogy, promotion, and retirement."""

    def register_candidate(
        self,
        model_name: str,
        version: str,
        parameters: Dict[str, Any],
        training_start: Optional[int] = None,
        training_end: Optional[int] = None,
        validation_accuracy: Optional[float] = None,
        validation_log_loss: Optional[float] = None,
        validation_brier: Optional[float] = None,
        status: str = "candidate"
    ) -> int:
        with SessionLocal() as session:
            existing = session.query(ModelVersionRecord).filter(
                ModelVersionRecord.model_name == model_name,
                ModelVersionRecord.version == version
            ).first()
            
            if existing:
                existing.parameters = parameters
                existing.validation_accuracy = validation_accuracy
                existing.validation_log_loss = validation_log_loss
                existing.validation_brier = validation_brier
                existing.status = status
                session.commit()
                return existing.id
            else:
                record = ModelVersionRecord(
                    model_name=model_name,
                    version=version,
                    parameters=parameters,
                    training_start_sequence=training_start,
                    training_end_sequence=training_end,
                    validation_accuracy=validation_accuracy,
                    validation_log_loss=validation_log_loss,
                    validation_brier=validation_brier,
                    status=status
                )
                session.add(record)
                session.commit()
                session.refresh(record)
                
                # Log model event
                event = ModelEvent(
                    event_type="REGISTERED",
                    model_version_id=record.id,
                    details={"name": model_name, "version": version, "status": status}
                )
                session.add(event)
                session.commit()
                return record.id

    def get_champion(self) -> Optional[Dict[str, Any]]:
        with SessionLocal() as session:
            champion = session.query(ModelVersionRecord).filter(
                ModelVersionRecord.status == "champion"
            ).order_by(ModelVersionRecord.id.desc()).first()
            
            if not champion:
                return None
            return {
                "id": champion.id,
                "model_name": champion.model_name,
                "version": champion.version,
                "parameters": champion.parameters,
                "validation_accuracy": champion.validation_accuracy,
                "validation_brier": champion.validation_brier,
                "status": champion.status
            }

    def promote(self, model_id: int, reason: str = "Outperformed champion on walk-forward audit") -> bool:
        with SessionLocal() as session:
            new_champ = session.query(ModelVersionRecord).filter(ModelVersionRecord.id == model_id).first()
            if not new_champ:
                return False
                
            # Retire previous champions
            current_champs = session.query(ModelVersionRecord).filter(ModelVersionRecord.status == "champion").all()
            for c in current_champs:
                if c.id != model_id:
                    c.status = "retired"
                    event = ModelEvent(
                        event_type="RETIRED",
                        model_version_id=c.id,
                        details={"reason": f"Replaced by model #{model_id} ({new_champ.version})"}
                    )
                    session.add(event)
                    
            new_champ.status = "champion"
            promo_event = ModelEvent(
                event_type="PROMOTED_TO_CHAMPION",
                model_version_id=new_champ.id,
                details={"reason": reason}
            )
            session.add(promo_event)
            session.commit()
            return True

    def retire(self, model_id: int, reason: str = "Underperformed on out-of-sample test") -> bool:
        with SessionLocal() as session:
            model = session.query(ModelVersionRecord).filter(ModelVersionRecord.id == model_id).first()
            if not model:
                return False
            model.status = "retired"
            event = ModelEvent(
                event_type="RETIRED",
                model_version_id=model.id,
                details={"reason": reason}
            )
            session.add(event)
            session.commit()
            return True

    def get_population_summary(self) -> Dict[str, int]:
        with SessionLocal() as session:
            total = session.query(ModelVersionRecord).count()
            challengers = session.query(ModelVersionRecord).filter(ModelVersionRecord.status.in_(["candidate", "challenger"])).count()
            retired = session.query(ModelVersionRecord).filter(ModelVersionRecord.status == "retired").count()
            return {
                "models_tested": total,
                "active_challengers": challengers,
                "retired_models": retired
            }
