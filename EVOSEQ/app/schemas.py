from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, Integer, BigInteger, SmallInteger, String, Float, Boolean, DateTime, JSON
from .database import Base

class Outcome(Base):
    __tablename__ = "outcomes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    sequence_no = Column(BigInteger, unique=True, index=True, nullable=False)
    timestamp_utc = Column(DateTime, nullable=False, default=datetime.utcnow)
    digit = Column(SmallInteger, nullable=False)
    size = Column(SmallInteger, nullable=False)
    color = Column(SmallInteger, nullable=False)
    parity = Column(SmallInteger, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelVersionRecord(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_name = Column(String, index=True, nullable=False)
    version = Column(String, index=True, nullable=False)
    parameters = Column(JSON, default=dict)
    training_start_sequence = Column(BigInteger, nullable=True)
    training_end_sequence = Column(BigInteger, nullable=True)
    validation_accuracy = Column(Float, nullable=True)
    validation_log_loss = Column(Float, nullable=True)
    validation_brier = Column(Float, nullable=True)
    status = Column(String, default="candidate") # 'candidate', 'challenger', 'champion', 'retired'
    created_at = Column(DateTime, default=datetime.utcnow)

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_version_id = Column(BigInteger, index=True, nullable=True)
    sequence_no = Column(BigInteger, index=True, nullable=False)
    probability_vector = Column(JSON, nullable=False) # JSON encoded list for wide cross-DB compat
    predicted_class = Column(SmallInteger, nullable=True)
    actual_class = Column(SmallInteger, nullable=True)
    log_loss = Column(Float, nullable=True)
    brier_score = Column(Float, nullable=True)
    entropy = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelEvent(Base):
    __tablename__ = "model_events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    event_type = Column(String, nullable=False)
    model_version_id = Column(BigInteger, nullable=True)
    details = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

class FeatureSnapshot(Base):
    __tablename__ = "feature_snapshots"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    sequence_no = Column(BigInteger, index=True, nullable=False)
    features = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class FeatureVectorRecord(Base):
    __tablename__ = "feature_vectors"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    sequence_no = Column(BigInteger, unique=True, index=True, nullable=False)
    window_size = Column(Integer, nullable=False)
    feature_vector = Column(JSON, nullable=False)
    digit_entropy = Column(Float, nullable=True)
    conditional_entropy_1 = Column(Float, nullable=True)
    conditional_entropy_2 = Column(Float, nullable=True)
    conditional_entropy_3 = Column(Float, nullable=True)
    information_gain_1 = Column(Float, nullable=True)
    information_gain_2 = Column(Float, nullable=True)
    information_gain_3 = Column(Float, nullable=True)
    lz_complexity = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelGenealogyRecord(Base):
    __tablename__ = "model_genealogy"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_version_id = Column(BigInteger, index=True, nullable=False)
    parent_model_version_id = Column(BigInteger, nullable=True)
    generation = Column(Integer, index=True, nullable=False)
    mutation = Column(JSON, default=dict)
    selection_reason = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelRuntimeStateRecord(Base):
    __tablename__ = "model_runtime_state"

    model_version_id = Column(BigInteger, primary_key=True, index=True)
    last_sequence_no = Column(BigInteger, index=True, nullable=False)
    state_path = Column(String, nullable=True)
    optimizer_state_path = Column(String, nullable=True)
    feature_state = Column(JSON, default=dict)
    runtime_metadata = Column(JSON, default=dict)
    updated_at = Column(DateTime, default=datetime.utcnow)

class ResearchExperimentRecord(Base):
    __tablename__ = "research_experiments"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    experiment_type = Column(String, index=True, nullable=False)
    model_version_id = Column(BigInteger, index=True, nullable=True)
    null_model = Column(String, nullable=True)
    sample_size = Column(BigInteger, nullable=True)
    observed_score = Column(Float, nullable=True)
    null_mean = Column(Float, nullable=True)
    null_std = Column(Float, nullable=True)
    p_value = Column(Float, nullable=True)
    correction_method = Column(String, nullable=True)
    test_range_start = Column(BigInteger, nullable=True)
    test_range_end = Column(BigInteger, nullable=True)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

class SystemEventRecord(Base):
    __tablename__ = "system_events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    event_type = Column(String, index=True, nullable=False)
    sequence_no = Column(BigInteger, index=True, nullable=True)
    model_version_id = Column(BigInteger, index=True, nullable=True)
    payload = Column(JSON, default=dict)
    status = Column(String, default="completed")
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class WorkerStateRecord(Base):
    __tablename__ = "worker_state"

    worker_name = Column(String, primary_key=True)
    last_processed_sequence = Column(BigInteger, default=0)
    status = Column(String, default="idle")
    records_processed = Column(BigInteger, default=0)
    last_success_at = Column(DateTime, nullable=True)
    last_failure_at = Column(DateTime, nullable=True)
    last_error = Column(String, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow)

class ChampionHealthRecord(Base):
    __tablename__ = "champion_health"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_version_id = Column(BigInteger, index=True, nullable=False)
    sequence_no = Column(BigInteger, index=True, nullable=False)
    health_score = Column(Float, nullable=True)
    drift_score = Column(Float, nullable=True)
    calibration_error = Column(Float, nullable=True)
    disagreement = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class MetaExperimentRecord(Base):
    __tablename__ = "meta_experiments"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_version_id = Column(BigInteger, index=True, nullable=True)
    environment = Column(JSON, nullable=False)
    model_descriptor = Column(JSON, nullable=False)
    log_loss = Column(Float, nullable=True)
    brier_score = Column(Float, nullable=True)
    calibration_error = Column(Float, nullable=True)
    stability_score = Column(Float, nullable=True)
    null_advantage = Column(Float, nullable=True)
    inference_latency = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ResearchQuestionRecord(Base):
    __tablename__ = "research_questions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    question_code = Column(String, unique=True, index=True, nullable=False)
    question = Column(String, nullable=False)
    hypothesis = Column(String, nullable=True)
    status = Column(String, default="OPEN") # OPEN, ACTIVE, RESOLVED, CLOSED
    priority = Column(Float, default=0.5)
    created_at = Column(DateTime, default=datetime.utcnow)

class ResearchHypothesisRecord(Base):
    __tablename__ = "research_hypotheses"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    question_id = Column(BigInteger, index=True, nullable=True)
    title = Column(String, nullable=False)
    description = Column(String, nullable=True)
    status = Column(String, default="PRE_REGISTERED") # PRE_REGISTERED, SUPPORTED, WEAK_EVIDENCE, INCONCLUSIVE, REJECTED
    evidence_score = Column(Float, default=0.0)
    evidence_summary = Column(JSON, default=dict)
    updated_at = Column(DateTime, default=datetime.utcnow)

class HiddenStateExperimentRecord(Base):
    __tablename__ = "hidden_state_experiments"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_version_id = Column(BigInteger, index=True, nullable=True)
    generator_type = Column(String, index=True, nullable=False)
    observation_count = Column(BigInteger, nullable=True)
    state_dimension = Column(Integer, nullable=True)
    state_recovery_score = Column(Float, nullable=True)
    parameter_recovery_score = Column(Float, nullable=True)
    predictive_score = Column(Float, nullable=True)
    runtime_seconds = Column(Float, nullable=True)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

class EnvironmentFingerprintRecord(Base):
    __tablename__ = "environment_fingerprints"

    sequence_no = Column(BigInteger, primary_key=True, index=True)
    entropy = Column(Float, nullable=True)
    conditional_entropy_1 = Column(Float, nullable=True)
    conditional_entropy_2 = Column(Float, nullable=True)
    information_gain_1 = Column(Float, nullable=True)
    information_gain_2 = Column(Float, nullable=True)
    lz_complexity = Column(Float, nullable=True)
    lz_zscore = Column(Float, nullable=True)
    autocorrelation_1 = Column(Float, nullable=True)
    autocorrelation_2 = Column(Float, nullable=True)
    drift_score = Column(Float, nullable=True)
    recurrence_rate = Column(Float, nullable=True)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)







