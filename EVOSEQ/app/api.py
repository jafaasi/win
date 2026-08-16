from fastapi import FastAPI, HTTPException
from typing import Dict, Any, List, Optional
import numpy as np
from .database import SessionLocal
from .schemas import (
    ModelVersionRecord,
    Outcome,
    PredictionRecord,
    ResearchExperimentRecord,
    SystemEventRecord,
    WorkerStateRecord,
    ChampionHealthRecord,
    ModelEvent,
    HiddenStateExperimentRecord
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

@app.get("/meta/questions")
def list_research_questions() -> List[Dict[str, Any]]:
    from .meta.questions import ResearchQuestionManager
    qm = ResearchQuestionManager()
    return qm.get_agenda()

@app.get("/meta/insights")
def get_meta_insights() -> Dict[str, Any]:
    from .meta.knowledge_graph import ModelKnowledgeGraph
    kg = ModelKnowledgeGraph()
    return {
        "robust_families_under_drift": kg.query_robust_families_under_drift(),
        "supported_hypotheses": kg.get_supported_scientific_hypotheses()
    }

@app.get("/dynamical/change-points")
def get_dynamical_change_points() -> Dict[str, Any]:
    from .dynamical.change_point import OnlineChangeDetector
    with SessionLocal() as session:
        rows = session.query(Outcome).order_by(Outcome.sequence_no.desc()).limit(250).all()
        rows.reverse()
        digits = [r.digit for r in rows]
    detector = OnlineChangeDetector(reference_size=150, recent_size=30)
    score = detector.score(digits)
    state = detector.classify_state(score, info_gain=0.02)
    return {"change_score": score, "state": state}

@app.get("/dynamical/recurrence")
def get_dynamical_recurrence() -> Dict[str, Any]:
    from .dynamical.recurrence import recurrence_matrix, recurrence_quantification_analysis
    with SessionLocal() as session:
        rows = session.query(Outcome).order_by(Outcome.sequence_no.desc()).limit(60).all()
        rows.reverse()
        digits = [r.digit for r in rows]
    R = recurrence_matrix(np.array(digits), epsilon=0.5)
    rqa = recurrence_quantification_analysis(R)
    return rqa

@app.get("/dynamical/symbolic")
def get_dynamical_symbolic() -> Dict[str, Any]:
    from .dynamical.symbolic import symbolic_complexity_curve
    with SessionLocal() as session:
        rows = session.query(Outcome).order_by(Outcome.sequence_no.desc()).limit(100).all()
        rows.reverse()
        digits = [r.digit for r in rows]
    return symbolic_complexity_curve(digits, max_k=4)

@app.get("/dynamical/memory-depth")
def get_dynamical_memory_depth() -> Dict[str, Any]:
    from .dynamical.bottleneck import MemoryDepthEstimator
    with SessionLocal() as session:
        rows = session.query(Outcome).order_by(Outcome.sequence_no.desc()).limit(100).all()
        rows.reverse()
        digits = [r.digit for r in rows]
    return MemoryDepthEstimator.estimate_depth_curve(digits)

@app.get("/dynamical/benchmarks")
def get_dynamical_benchmarks() -> List[Dict[str, Any]]:
    with SessionLocal() as session:
        exps = session.query(HiddenStateExperimentRecord).order_by(HiddenStateExperimentRecord.id.desc()).limit(20).all()
        return [
            {
                "id": e.id,
                "generator_type": e.generator_type,
                "observation_count": e.observation_count,
                "state_dimension": e.state_dimension,
                "state_recovery_score": e.state_recovery_score,
                "parameter_recovery_score": e.parameter_recovery_score,
                "runtime_seconds": e.runtime_seconds
            }
            for e in exps
        ]


