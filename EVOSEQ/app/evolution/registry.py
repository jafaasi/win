import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from ..database import SessionLocal
from ..schemas import ModelVersionRecord, ModelEvent, ModelGenealogyRecord

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
        status: str = "candidate",
        parent_model_id: Optional[int] = None,
        generation: int = 1,
        mutation_details: Optional[Dict[str, Any]] = None
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
                
                # Log model genealogy
                genealogy = ModelGenealogyRecord(
                    model_version_id=record.id,
                    parent_model_version_id=parent_model_id,
                    generation=generation,
                    mutation=mutation_details or {},
                    selection_reason=f"Candidate generated for generation {generation}"
                )
                session.add(genealogy)
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

    def put_on_probation(self, model_id: int, reason: str = "Candidate passed referee, entering probation") -> bool:
        with SessionLocal() as session:
            model = session.query(ModelVersionRecord).filter(ModelVersionRecord.id == model_id).first()
            if not model:
                return False
            model.status = "probation"
            event = ModelEvent(
                event_type="ENTERED_PROBATION",
                model_version_id=model.id,
                details={"reason": reason}
            )
            session.add(event)
            session.commit()
            return True

    def rollback(self, current_champ_id: int, fallback_champ_id: int, reason: str = "Probation degraded") -> bool:
        with SessionLocal() as session:
            cur = session.query(ModelVersionRecord).filter(ModelVersionRecord.id == current_champ_id).first()
            prev = session.query(ModelVersionRecord).filter(ModelVersionRecord.id == fallback_champ_id).first()
            if not cur or not prev:
                return False
            cur.status = "rollback"
            prev.status = "champion"
            session.add(ModelEvent(event_type="ROLLBACK_TRIGGERED", model_version_id=cur.id, details={"reason": reason}))
            session.add(ModelEvent(event_type="REINSTATED_AS_CHAMPION", model_version_id=prev.id, details={"reason": "Reinstated after rollback"}))
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

    def get_genealogy_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        with SessionLocal() as session:
            records = session.query(ModelGenealogyRecord).order_by(ModelGenealogyRecord.generation.desc()).limit(limit).all()
            return [
                {
                    "id": r.id,
                    "model_version_id": r.model_version_id,
                    "parent_model_version_id": r.parent_model_version_id,
                    "generation": r.generation,
                    "mutation": r.mutation,
                    "selection_reason": r.selection_reason
                }
                for r in records
            ]

    def ensure_initial_population(self) -> None:
        """Ensures baseline default models exist in the registry."""
        with SessionLocal() as session:
            existing = session.query(ModelVersionRecord).first()
            if not existing:
                self.register_candidate(
                    model_name="Uniform",
                    version="v1.0",
                    parameters={},
                    status="champion"
                )
                self.register_candidate(
                    model_name="Markov",
                    version="mkv-ord1-v1.0",
                    parameters={"order": 1, "smoothing": 0.5},
                    status="challenger"
                )
                self.register_candidate(
                    model_name="Markov",
                    version="mkv-ord2-v1.0",
                    parameters={"order": 2, "smoothing": 0.5},
                    status="challenger"
                )



