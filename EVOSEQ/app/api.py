from fastapi import FastAPI, HTTPException
from typing import Dict, Any, List, Optional
from .database import SessionLocal
from .schemas import (
    ModelVersionRecord,
    Outcome,
    PredictionRecord,
    ResearchExperimentRecord,
    SystemEventRecord,
    WorkerStateRecord,
    ChampionHealthRecord,
    ModelEvent
)

app = FastAPI(title="EVOSEQ Autonomous Research Platform API", version="1.0.0")

@app.get("/health")
def health() -> Dict[str, Any]:
    with SessionLocal() as session:
        workers = session.query(WorkerStateRecord).all()
        w_dict = {w.worker_name: w.status for w in workers}
        last_seq = session.query(Outcome.sequence_no).order_by(Outcome.sequence_no.desc()).first()
        champ = session.query(ModelVersionRecord).filter(ModelVersionRecord.status == "champion").order_by(ModelVersionRecord.id.desc()).first()
        
    return {
        "status": "healthy",
        "database": "connected",
        "workers": w_dict,
        "last_sequence": last_seq[0] if last_seq else 0,
        "champion": champ.version if champ else "None"
    }

@app.get("/system/status")
def system_status() -> Dict[str, Any]:
    with SessionLocal() as session:
        obs_count = session.query(Outcome).count()
        champ = session.query(ModelVersionRecord).filter(ModelVersionRecord.status == "champion").order_by(ModelVersionRecord.id.desc()).first()
        evts = session.query(SystemEventRecord).filter(SystemEventRecord.event_type == "DRIFT_EVALUATED").order_by(SystemEventRecord.id.desc()).first()
        drift_state = evts.payload.get("state", "stable") if evts else "stable"
        
    return {
        "observations": obs_count,
        "champion": champ.version if champ else "None",
        "champion_id": champ.id if champ else None,
        "drift": drift_state
    }

@app.get("/models")
def list_models(limit: int = 50) -> List[Dict[str, Any]]:
    with SessionLocal() as session:
        models = session.query(ModelVersionRecord).order_by(ModelVersionRecord.id.desc()).limit(limit).all()
        return [
            {
                "id": m.id,
                "model_name": m.model_name,
                "version": m.version,
                "status": m.status,
                "validation_accuracy": m.validation_accuracy,
                "validation_brier": m.validation_brier,
                "parameters": m.parameters,
                "created_at": m.created_at.isoformat() if m.created_at else None
            }
            for m in models
        ]

@app.get("/models/champion")
def get_champion_model() -> Dict[str, Any]:
    with SessionLocal() as session:
        champ = session.query(ModelVersionRecord).filter(ModelVersionRecord.status == "champion").order_by(ModelVersionRecord.id.desc()).first()
        if not champ:
            raise HTTPException(status_code=404, detail="No active champion found")
        return {
            "id": champ.id,
            "model_name": champ.model_name,
            "version": champ.version,
            "status": champ.status,
            "validation_accuracy": champ.validation_accuracy,
            "validation_brier": champ.validation_brier,
            "parameters": champ.parameters
        }

@app.get("/models/{model_id}")
def get_model_details(model_id: int) -> Dict[str, Any]:
    with SessionLocal() as session:
        m = session.query(ModelVersionRecord).filter(ModelVersionRecord.id == model_id).first()
        if not m:
            raise HTTPException(status_code=404, detail="Model not found")
        events = session.query(ModelEvent).filter(ModelEvent.model_version_id == model_id).all()
        return {
            "id": m.id,
            "model_name": m.model_name,
            "version": m.version,
            "status": m.status,
            "validation_accuracy": m.validation_accuracy,
            "validation_brier": m.validation_brier,
            "parameters": m.parameters,
            "events": [{"event_type": e.event_type, "details": e.details} for e in events]
        }

@app.get("/drift/latest")
def get_latest_drift() -> Dict[str, Any]:
    with SessionLocal() as session:
        evt = session.query(SystemEventRecord).filter(SystemEventRecord.event_type == "DRIFT_EVALUATED").order_by(SystemEventRecord.id.desc()).first()
        if not evt:
            return {"composite_drift": 0.0, "state": "stable"}
        return evt.payload

@app.get("/experiments")
def list_experiments(limit: int = 50) -> List[Dict[str, Any]]:
    with SessionLocal() as session:
        exps = session.query(ResearchExperimentRecord).order_by(ResearchExperimentRecord.id.desc()).limit(limit).all()
        return [
            {
                "id": e.id,
                "experiment_type": e.experiment_type,
                "model_version_id": e.model_version_id,
                "null_model": e.null_model,
                "sample_size": e.sample_size,
                "observed_score": e.observed_score,
                "null_mean": e.null_mean,
                "p_value": e.p_value,
                "correction_method": e.correction_method,
                "created_at": e.created_at.isoformat() if e.created_at else None
            }
            for e in exps
        ]

@app.get("/events")
def list_system_events(limit: int = 50) -> List[Dict[str, Any]]:
    with SessionLocal() as session:
        events = session.query(SystemEventRecord).order_by(SystemEventRecord.id.desc()).limit(limit).all()
        return [
            {
                "id": evt.id,
                "event_type": evt.event_type,
                "sequence_no": evt.sequence_no,
                "model_version_id": evt.model_version_id,
                "payload": evt.payload,
                "status": evt.status,
                "created_at": evt.created_at.isoformat() if evt.created_at else None
            }
            for evt in events
        ]
