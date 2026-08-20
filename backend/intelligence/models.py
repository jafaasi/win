import sys
import os
import json
from datetime import datetime
from sqlalchemy import Column, Integer, BigInteger, String, Float, Boolean, DateTime, JSON, Text, Index

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.database import Base


class StateFingerprintRecord(Base):
    __tablename__ = "state_fingerprints"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    sequence_no = Column(BigInteger, index=True, nullable=False)
    sequence_hash = Column(String(64), index=True)

    short_window_size = Column(Integer, default=10)
    medium_window_size = Column(Integer, default=50)
    long_window_size = Column(Integer, default=200)

    recent_sequence = Column(String(256))
    short_big_rate = Column(Float)
    medium_big_rate = Column(Float)
    long_big_rate = Column(Float)

    current_streak = Column(Integer)
    streak_value = Column(Integer)

    entropy = Column(Float)
    transition_entropy = Column(Float)
    autocorr_lag1 = Column(Float)
    autocorr_lag2 = Column(Float)
    autocorr_lag3 = Column(Float)

    conditional_entropy_1 = Column(Float)
    conditional_entropy_2 = Column(Float)

    information_gain_1 = Column(Float)
    information_gain_2 = Column(Float)

    lz_complexity = Column(Float)
    drift_score = Column(Float)
    regime_id = Column(String(32))

    feature_vector_json = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_fp_seq_regime", "sequence_no", "regime_id"),
    )


class SimilarStateRecord(Base):
    __tablename__ = "similar_state_memory"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    fingerprint_id = Column(BigInteger, index=True)
    source_sequence_no = Column(BigInteger, index=True)

    matched_state_hash = Column(String(64), index=True)
    cosine_similarity = Column(Float)
    euclidean_distance = Column(Float)
    jaccard_similarity = Column(Float)

    next_outcome_size = Column(Integer)
    next_outcome_digit = Column(Integer)

    horizon_1_correct = Column(Boolean)
    horizon_2_correct = Column(Boolean)
    horizon_3_correct = Column(Boolean)

    created_at = Column(DateTime, default=datetime.utcnow)


class ModelPerformanceRecord(Base):
    __tablename__ = "model_performance"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    model_name = Column(String(64), index=True, nullable=False)
    model_version = Column(String(64), index=True)
    generation = Column(Integer, index=True, default=1)
    family = Column(String(32), index=True)

    regime_id = Column(String(32), index=True)

    total_predictions = Column(BigInteger, default=0)
    correct_predictions = Column(BigInteger, default=0)
    accuracy = Column(Float)

    log_loss_sum = Column(Float, default=0.0)
    brier_sum = Column(Float, default=0.0)
    mean_log_loss = Column(Float)
    mean_brier = Column(Float)

    recent_50_accuracy = Column(Float)
    recent_200_accuracy = Column(Float)

    reliability_buckets_json = Column(JSON, default=dict)
    calibration_error = Column(Float)

    last_updated_sequence = Column(BigInteger)
    updated_at = Column(DateTime, default=datetime.utcnow)


class GenerationRecord(Base):
    __tablename__ = "generations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    generation = Column(Integer, unique=True, index=True, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    training_cutoff_sequence = Column(BigInteger)
    validation_cutoff_sequence = Column(BigInteger)
    test_sequence_start = Column(BigInteger)

    total_training_samples = Column(BigInteger, default=0)
    total_validation_samples = Column(BigInteger, default=0)
    total_test_samples = Column(BigInteger, default=0)

    champion_model_name = Column(String(64))
    champion_model_version = Column(String(64))
    champion_oos_accuracy = Column(Float)
    champion_oos_log_loss = Column(Float)
    champion_oos_brier = Column(Float)

    previous_champion_accuracy = Column(Float)
    accuracy_delta = Column(Float)
    statistically_significant = Column(Boolean, default=False)

    model_weights_json = Column(JSON, default=dict)
    hyperparameters_json = Column(JSON, default=dict)
    calibration_parameters_json = Column(JSON, default=dict)

    baseline_uniform_accuracy = Column(Float)
    baseline_majority_accuracy = Column(Float)
    baseline_frequency_accuracy = Column(Float)
    baseline_markov_accuracy = Column(Float)

    rejected_models_json = Column(JSON, default=list)
    promoted_models_json = Column(JSON, default=list)

    status = Column(String(32), default="CANDIDATE")
    notes = Column(Text)


class CalibrationRecord(Base):
    __tablename__ = "calibration_memory"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    generation = Column(Integer, index=True)

    bucket_start = Column(Float)
    bucket_end = Column(Float)
    predicted_count = Column(BigInteger, default=0)
    actual_win_count = Column(BigInteger, default=0)
    observed_win_rate = Column(Float)
    expected_calibrated_rate = Column(Float)

    last_updated_sequence = Column(BigInteger)
    updated_at = Column(DateTime, default=datetime.utcnow)


class DriftRecord(Base):
    __tablename__ = "drift_history"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    sequence_no = Column(BigInteger, index=True)

    ks_drift_score = Column(Float)
    js_divergence = Column(Float)
    wasserstein_distance = Column(Float)
    composite_drift = Column(Float)

    drift_detected = Column(Boolean, default=False)
    severity = Column(String(16))
    regime_before = Column(String(32))
    regime_after = Column(String(32))

    created_at = Column(DateTime, default=datetime.utcnow)


class AbstentionRecord(Base):
    __tablename__ = "abstention_log"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    sequence_no = Column(BigInteger, unique=True, index=True)

    action = Column(String(16), default="PREDICT")
    reason = Column(String(64))

    entropy = Column(Float)
    model_disagreement = Column(Float)
    state_sample_size = Column(BigInteger)
    calibration_error = Column(Float)
    drift_score = Column(Float)
    adversarial_contradiction = Column(Float)

    baseline_beat = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)


class DailyReportRecord(Base):
    __tablename__ = "daily_reports"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    report_date = Column(String(16), unique=True, index=True)

    generation = Column(Integer)
    total_historical_samples = Column(BigInteger, default=0)
    new_samples_today = Column(BigInteger, default=0)

    current_regime = Column(String(32))
    champion_model = Column(String(64))
    strongest_model = Column(String(64))
    weakest_model = Column(String(64))

    model_disagreement = Column(Float)
    calibration_ece = Column(Float)
    mean_brier = Column(Float)
    mean_log_loss = Column(Float)

    baseline_accuracy = Column(Float)
    oos_accuracy = Column(Float)

    drift_detected_today = Column(Boolean, default=False)
    models_rejected = Column(Integer, default=0)
    models_promoted = Column(Integer, default=0)

    conclusion = Column(String(64))
    report_json = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)


def ensure_intelligence_tables(engine):
    try:
        Base.metadata.create_all(bind=engine)
        print("[IntelligenceDB] All intelligence memory tables ready.")
    except Exception as e:
        print(f"[IntelligenceDB] Table creation note: {e}")
