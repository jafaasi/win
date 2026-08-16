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
