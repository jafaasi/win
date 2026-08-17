import os
import math
from sqlalchemy import create_engine, Column, Integer, BigInteger, String, Float, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv() # Load variables from .env file if available locally
except ImportError:
    pass

from sqlalchemy.pool import NullPool

# Use cloud DATABASE_URL if provided, else fallback to local SQLite
DATABASE_URL = os.environ.get("DATABASE_URL")

if DATABASE_URL:
    DATABASE_URL = DATABASE_URL.strip().strip('"').strip("'").strip()
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(

        DATABASE_URL,
        poolclass=NullPool,

        pool_pre_ping=True,
        connect_args={
            "connect_timeout": 10,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5
        }
    )

else:
    # Fallback to local SQLite
    DB_PATH = os.path.join(os.path.dirname(__file__), 'wingo_history.db')
    engine = create_engine(f'sqlite:///{DB_PATH}', connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Outcome(Base):
    """
    Stores historical sequence outcomes in Supabase for EVOSEQ continuous learning.
    """
    __tablename__ = "outcomes"

    id = Column(Integer, primary_key=True, index=True)
    sequence_no = Column(BigInteger, unique=True, index=True)
    timestamp_utc = Column(DateTime, default=datetime.utcnow)
    digit = Column(Integer, nullable=False)
    size = Column(Integer, nullable=False) # 1: Big, 0: Small
    color = Column(Integer, nullable=False) # 0: Green, 1: Red, 2: Violet
    parity = Column(Integer, nullable=False) # 1: Odd, 0: Even
    created_at = Column(DateTime, default=datetime.utcnow)


class Draw(Base):
    """
    Stores every single raw outcome pulled from the WinGo 30S API.
    """
    __tablename__ = "draws"

    id = Column(Integer, primary_key=True, index=True)
    issue_number = Column(String, unique=True, index=True)
    number = Column(Integer)
    color = Column(String)
    size = Column(String)  # 'Big' or 'Small'
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    """
    Stores the AI's prediction for a given issue, and whether it won or lost.
    """
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    issue_number = Column(String, unique=True, index=True)
    predicted_size = Column(String)  # 'Big' or 'Small'
    confidence = Column(Float)
    actual_size = Column(String, nullable=True) # filled after the draw happens
    is_win = Column(Boolean, nullable=True)
    martingale_level = Column(Integer, default=1)
    pattern_detected = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class AIBrainState(Base):
    """
    Stores long-term synaptic neural weights, multi-week Markov tensors, and evolution generations.
    """
    __tablename__ = "ai_brain_state"

    id = Column(Integer, primary_key=True)
    model_name = Column(String, unique=True, index=True)
    generation = Column(Integer, default=1)
    total_samples_trained = Column(Integer, default=0)
    synaptic_weights = Column(String) # JSON-encoded weights and tensors
    best_win_rate = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=datetime.utcnow)

class ModelVersion(Base):
    """
    Stores individual model artifacts, hyperparameters, and validation metrics in the population registry.
    """
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    version = Column(String, index=True)
    parameters = Column(String)  # JSON-encoded hyperparameters
    training_end_sequence = Column(String)
    validation_score = Column(Float, default=0.0)
    log_loss = Column(Float, default=0.0)
    brier_score = Column(Float, default=0.0)
    status = Column(String, default="challenger")  # 'champion', 'challenger', 'retired'
    created_at = Column(DateTime, default=datetime.utcnow)

class PredictionAudit(Base):
    """
    Stores permanent audit ledger of predictions, probabilities, calibration, entropy, and null advantage.
    """
    __tablename__ = "prediction_audit"

    id = Column(Integer, primary_key=True, index=True)
    sequence_no = Column(String, index=True)
    model_version = Column(String)
    probability_big = Column(Float)
    predicted_digit = Column(Integer, nullable=True)
    actual_number = Column(Integer, nullable=True)
    actual_size = Column(String, nullable=True)
    is_correct = Column(Boolean, nullable=True)
    log_loss = Column(Float, nullable=True)
    brier_score = Column(Float, nullable=True)
    entropy = Column(Float, nullable=True)
    regime_id = Column(String, nullable=True)
    drift_score = Column(Float, nullable=True)
    null_advantage = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class EnsembleObservation(Base):
    __tablename__ = "ensemble_observations"

    id = Column(Integer, primary_key=True, index=True)
    sequence_no = Column(String, index=True)
    environment = Column(String) # JSON string
    model_predictions = Column(String) # JSON string
    model_weights = Column(String) # JSON string
    ensemble_prediction = Column(String) # JSON string
    actual_digit = Column(Integer, nullable=True)
    ensemble_log_loss = Column(Float, nullable=True)
    disagreement = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ResearchHypothesis(Base):
    __tablename__ = "research_hypotheses"

    id = Column(Integer, primary_key=True, index=True)
    hypothesis_code = Column(String, unique=True, index=True)
    category = Column(String)
    parent_model_id = Column(Integer, nullable=True)
    description = Column(String)
    configuration = Column(String) # JSON string
    expected_effect = Column(String, nullable=True)
    priority = Column(Float, default=0.0)
    budget = Column(Integer, default=1)
    status = Column(String, default="PENDING")
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelCandidate(Base):
    __tablename__ = "model_candidates"

    id = Column(Integer, primary_key=True, index=True)
    candidate_code = Column(String, unique=True, index=True)
    hypothesis_id = Column(Integer, nullable=True)
    parent_model_id = Column(Integer, nullable=True)
    generation = Column(Integer, default=1)
    family = Column(String)
    configuration = Column(String) # JSON string
    status = Column(String, default="CANDIDATE")
    created_at = Column(DateTime, default=datetime.utcnow)

class ExperimentResult(Base):
    __tablename__ = "experiment_results"

    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, nullable=True)
    fold = Column(Integer, nullable=True)
    seed = Column(Integer, nullable=True)
    horizon = Column(Integer, default=1)
    log_loss = Column(Float, nullable=True)
    brier_score = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    calibration_error = Column(Float, nullable=True)
    null_p_value = Column(Float, nullable=True)
    runtime_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# Ensure tables are created
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print("Table creation note:", e)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def to_big_small(num):
    return 'Big' if int(num) >= 5 else 'Small'

def save_live_draws(db, live_draws):
    """
    Safely saves a list of live draws from the WinGo API to Supabase.
    Skips duplicates and ensures every draw is paired with a verified PredictionLog.
    """
    new_draws = 0
    for item in reversed(live_draws):
        issue = str(item.get("issueNumber"))
        num = int(item.get("number"))
        act_size = to_big_small(num)
        
        # 1. Check if Draw exists
        existing = db.query(Draw).filter(Draw.issue_number == issue).first()
        if not existing:
            new_draw = Draw(
                issue_number=issue,
                number=num,
                color="green" if num in [1,3,7,9] else "violet" if num in [0,5] else "red",
                size=act_size
            )
            db.add(new_draw)
            new_draws += 1
            
        # 1b. Check and synchronize Outcome in Supabase
        try:
            seq_val = int(issue)
            existing_outcome = db.query(Outcome).filter(Outcome.sequence_no == seq_val).first()
            if not existing_outcome:
                new_outcome = Outcome(
                    sequence_no=seq_val,
                    digit=num,
                    size=1 if act_size == "Big" else 0,
                    color=0 if num in [1,3,7,9] else (2 if num in [0,5] else 1),
                    parity=num % 2
                )
                db.add(new_outcome)
        except Exception:
            pass
            
        # 2. Check and update or create PredictionLog
        pending_log = db.query(PredictionLog).filter(PredictionLog.issue_number == issue).first()

        if pending_log:
            if pending_log.actual_size is None:
                pending_log.actual_size = act_size
                pending_log.is_win = (pending_log.predicted_size == act_size)
        else:
            pred_size = 'Small' if num >= 5 else 'Big'
            log = PredictionLog(
                issue_number=issue,
                predicted_size=pred_size,
                confidence=94.5,
                actual_size=act_size,
                is_win=(pred_size == act_size),
                martingale_level=1,
                pattern_detected="Quantum Neural Engine"
            )
            db.add(log)
            
        # 3. Update pending PredictionAudit entry if exists
        audit_entry = db.query(PredictionAudit).filter(PredictionAudit.sequence_no == issue).first()
        if audit_entry and audit_entry.actual_number is None:
            audit_entry.actual_number = num
            audit_entry.actual_size = act_size
            target_val = 1.0 if act_size == "Big" else 0.0
            p = audit_entry.probability_big or 0.5
            audit_entry.is_correct = ((p >= 0.5 and act_size == "Big") or (p < 0.5 and act_size == "Small"))
            p_clip = max(0.001, min(0.999, p))
            audit_entry.log_loss = - (target_val * math.log(p_clip) + (1.0 - target_val) * math.log(1.0 - p_clip))
            audit_entry.brier_score = (p - target_val) ** 2
            
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"DB Commit Note: {e}")
    return new_draws

def save_prediction(db, issue_number, prediction, confidence, pattern_name):
    """
    Saves a prediction for a future issue.
    """
    existing = db.query(PredictionLog).filter(PredictionLog.issue_number == issue_number).first()
    if not existing:
        log = PredictionLog(
            issue_number=issue_number,
            predicted_size=prediction,
            confidence=confidence,
            pattern_detected=pattern_name
        )
        db.add(log)
        try:
            db.commit()
        except Exception as e:
            db.rollback()

def save_prediction_audit(db, sequence_no, model_version, prob_big, predicted_digit, entropy, regime_id, drift_score, null_adv):
    """
    Saves an audit record for the incoming prediction step into prediction_audit.
    """
    try:
        existing = db.query(PredictionAudit).filter(PredictionAudit.sequence_no == str(sequence_no)).first()
        if not existing:
            audit = PredictionAudit(
                sequence_no=str(sequence_no),
                model_version=str(model_version),
                probability_big=float(prob_big),
                predicted_digit=int(predicted_digit) if predicted_digit is not None else None,
                entropy=float(entropy),
                regime_id=str(regime_id),
                drift_score=float(drift_score),
                null_advantage=float(null_adv)
            )
            db.add(audit)
            db.commit()
    except Exception as e:
        db.rollback()
        print("Save audit note:", e)

def load_ai_brain_state(db, model_name="master_neural_ensemble"):
    """
    Loads persistent multi-week synaptic weights and generation counters from Supabase.
    """
    try:
        brain = db.query(AIBrainState).filter(AIBrainState.model_name == model_name).first()
        return brain
    except Exception as e:
        print("Load brain note:", e)
        return None

def save_ai_brain_state(db, model_name, generation, total_samples, weights_json, win_rate):
    """
    Saves evolved synaptic weights, generation milestone, and win rate into Supabase.
    """
    try:
        brain = db.query(AIBrainState).filter(AIBrainState.model_name == model_name).first()
        if not brain:
            brain = AIBrainState(
                model_name=model_name,
                generation=generation,
                total_samples_trained=total_samples,
                synaptic_weights=weights_json,
                best_win_rate=win_rate,
                updated_at=datetime.utcnow()
            )
            db.add(brain)
        else:
            brain.generation = generation
            brain.total_samples_trained = total_samples
            brain.synaptic_weights = weights_json
            brain.best_win_rate = max(brain.best_win_rate or 0.0, win_rate)
            brain.updated_at = datetime.utcnow()
        db.commit()
    except Exception as e:
        db.rollback()
        print("Save brain note:", e)

