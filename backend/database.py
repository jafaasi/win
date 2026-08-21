import os
import math
import json
from sqlalchemy import create_engine, Column, Integer, BigInteger, String, Float, Boolean, DateTime, JSON, Text, text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv() # Load variables from .env file if available locally
except ImportError:
    pass



# Use the supplied environment configuration.  On AWS this is injected by
# systemd from /etc/win/win.env; local development may use backend/.env.
DATABASE_URL = os.environ.get("DATABASE_URL")

# If no DATABASE_URL env var, try to load from .env file
if not DATABASE_URL:
    try:
        from dotenv import load_dotenv
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        env_file = os.path.join(backend_dir, '.env')
        if os.path.exists(env_file):
            load_dotenv(env_file)
            DATABASE_URL = os.environ.get("DATABASE_URL")
    except:
        pass

if DATABASE_URL:
    DATABASE_URL = DATABASE_URL.strip().strip('"').strip("'").strip()
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,

        pool_pre_ping=True,
        connect_args={
            "connect_timeout": 10,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5
        }
    )
    print("[DB] Connected using DATABASE_URL")

else:
    # Fallback to local SQLite
    DB_PATH = os.path.join(os.path.dirname(__file__), 'wingo_history.db')
    engine = create_engine(f'sqlite:///{DB_PATH}', connect_args={"check_same_thread": False})
    print(f"[DB] Using local SQLite: {DB_PATH}")

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

    Extended with EVOSEQ v3 fields: generation, state fingerprint, similarity, adversarial,
    calibration, OOS scores, edge status, etc. All new columns are nullable for backward compat.
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

    # ===== EVOSEQ v3 new fields (all nullable) =====
    generation = Column(Integer, nullable=True)
    state_fingerprint_json = Column(JSON, nullable=True)
    input_window_hash = Column(String(128), nullable=True)
    state_similarity = Column(Float, nullable=True)
    state_sample_size = Column(BigInteger, nullable=True)
    regime = Column(String(64), nullable=True)
    adversarial_score = Column(Float, nullable=True)
    support_score = Column(Float, nullable=True)
    contradiction_score = Column(Float, nullable=True)
    uncertainty_score = Column(Float, nullable=True)
    calibrated_probability = Column(Float, nullable=True)
    calibration_error = Column(Float, nullable=True)
    expected_calibration_error = Column(Float, nullable=True)
    oos_score = Column(Float, nullable=True)
    baseline_score = Column(Float, nullable=True)
    edge_status = Column(String(32), nullable=True)
    learning_status = Column(String(32), nullable=True)
    model_reliability_json = Column(JSON, nullable=True)
    knowledge_version = Column(String(64), nullable=True)
    action = Column(String(32), nullable=True)
    model_predictions_json = Column(JSON, nullable=True)
    model_weights_json = Column(JSON, nullable=True)
    ensemble_probability = Column(Float, nullable=True)
    meta_probability = Column(Float, nullable=True)
    p_success_l1 = Column(Float, nullable=True)
    p_success_l2 = Column(Float, nullable=True)
    p_success_l3 = Column(Float, nullable=True)
    model_consensus = Column(Float, nullable=True)
    strike_quality = Column(String(32), nullable=True)
    target_num = Column(Integer, nullable=True)
    hedge_num = Column(Integer, nullable=True)
    notes = Column(Text, nullable=True)

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


class DecisionMemoryRow(Base):
    """
    EVOSEQ v3 — Full reasoning-state ledger for every prediction.
    Supports OOS temporal splits, adversarial audit, and condition analysis.
    Never modified after initial insert (immutable prediction record).
    Outcome fields backfilled once the draw resolves.
    """
    __tablename__ = "decision_memory"

    id                    = Column(Integer, primary_key=True, index=True)
    issue_number          = Column(String, unique=True, index=True)
    timestamp_utc         = Column(DateTime, default=datetime.utcnow, index=True)

    # Core prediction
    prediction            = Column(String)       # "Big" | "Small"
    confidence            = Column(Float)
    probability_big       = Column(Float)

    # TAIE decision tier
    taie_tier             = Column(String(32))   # STRONG|MODERATE|WEAK|ABSTAIN
    action                = Column(String(32))   # STRIKE|FORECAST|CAUTION|SKIP

    # 12-model feature vectors (JSON arrays)
    model_p_big_vector    = Column(Text)         # JSON list[float], len=12
    model_weights         = Column(Text)         # JSON list[float], len=12
    model_consensus       = Column(Float)

    # Signal agreement
    signal_agreement      = Column(Float)
    engines_agree         = Column(Integer)
    reject_iid            = Column(Boolean)

    # Adversarial summary
    adversarial_score     = Column(Float)
    adversarial_verdict   = Column(String(32))   # HOLD|CAUTION|OVERRIDE|ABSTAIN
    adversarial_net_score = Column(Float)

    # Statistical context at decision time
    entropy               = Column(Float)
    exploit_score         = Column(Float)
    drift_level           = Column(String(64))
    drift_score           = Column(Float)
    change_probability    = Column(Float)
    streak_run_length     = Column(Integer)

    # Martingale position
    martingale_level      = Column(Integer)
    martingale_loss_streak= Column(Integer)

    # Evidence gate
    validated_edge        = Column(Boolean)
    three_level_lower_bound = Column(Float)
    brier_improvement     = Column(Float)

    # Human-readable reasoning chain
    prediction_reason     = Column(Text)

    # Outcome — filled once draw resolves (NULL until then)
    actual_size           = Column(String, nullable=True)
    actual_digit          = Column(Integer, nullable=True)
    is_correct            = Column(Boolean, nullable=True)
    log_loss              = Column(Float, nullable=True)
    brier_score_val       = Column(Float, nullable=True)

    created_at            = Column(DateTime, default=datetime.utcnow)


class StateMemoryRow(Base):
    """
    EVOSEQ v3 — State fingerprint memory for similar-state nearest-neighbour retrieval.
    Each row is one historical state fingerprint compressed to a fixed float vector,
    plus the outcome that followed it. Used by StateMemory to answer:
    'Have I seen a state like this before, and what happened next?'
    """
    __tablename__ = "state_memory"

    id               = Column(Integer, primary_key=True, index=True)
    sequence_no      = Column(String, index=True)
    timestamp_utc    = Column(DateTime, default=datetime.utcnow, index=True)

    # 12-dim fingerprint vector (JSON list[float])
    fingerprint_vec  = Column(Text, nullable=False)

    # Context stored for diagnostics
    entropy          = Column(Float)
    big_rate_short   = Column(Float)    # last-10 Big rate
    big_rate_medium  = Column(Float)    # last-50 Big rate
    big_rate_long    = Column(Float)    # last-200 Big rate
    streak_len       = Column(Integer)
    streak_side      = Column(Integer)  # 1=Big, 0=Small
    drift_level      = Column(String(64))

    # Outcome that followed this state
    actual_side      = Column(Integer, nullable=True)   # 1=Big, 0=Small

    created_at       = Column(DateTime, default=datetime.utcnow)


def migrate_decision_memory_table(db):
    """Idempotent: create decision_memory and state_memory if they don't exist yet."""
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"[DB Migrate] decision_memory/state_memory: {e}")


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
        
        # 1. Check if Draw exists (with better error handling)
        try:
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
        except Exception as e:
            # If there's any database error, skip this draw and continue
            print(f"DB Note: Skipping draw {issue} due to error: {e}")
            continue
            
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
            
        # 2. Reconcile a prediction only when it was made before this outcome.
        # Never create a synthetic inverse prediction here: that corrupts the
        # measured win rate and makes calibration meaningless.
        pending_log = db.query(PredictionLog).filter(PredictionLog.issue_number == issue).first()

        if pending_log:
            if pending_log.actual_size is None:
                pending_log.actual_size = act_size
                pending_log.is_win = (pending_log.predicted_size == act_size)
            
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

def migrate_prediction_audit_columns(db):
    """
    Idempotent migration: add any missing EVOSEQ v3 columns to prediction_audit table
    if they don't already exist. Safe for both PostgreSQL and SQLite.
    """
    new_columns_sql = [
        ("generation", "INTEGER"),
        ("state_fingerprint_json", "TEXT"),
        ("input_window_hash", "VARCHAR(128)"),
        ("state_similarity", "FLOAT"),
        ("state_sample_size", "BIGINT"),
        ("regime", "VARCHAR(64)"),
        ("adversarial_score", "FLOAT"),
        ("support_score", "FLOAT"),
        ("contradiction_score", "FLOAT"),
        ("uncertainty_score", "FLOAT"),
        ("calibrated_probability", "FLOAT"),
        ("calibration_error", "FLOAT"),
        ("expected_calibration_error", "FLOAT"),
        ("oos_score", "FLOAT"),
        ("baseline_score", "FLOAT"),
        ("edge_status", "VARCHAR(32)"),
        ("learning_status", "VARCHAR(32)"),
        ("model_reliability_json", "TEXT"),
        ("knowledge_version", "VARCHAR(64)"),
        ("action", "VARCHAR(32)"),
        ("model_predictions_json", "TEXT"),
        ("model_weights_json", "TEXT"),
        ("ensemble_probability", "FLOAT"),
        ("meta_probability", "FLOAT"),
        ("p_success_l1", "FLOAT"),
        ("p_success_l2", "FLOAT"),
        ("p_success_l3", "FLOAT"),
        ("model_consensus", "FLOAT"),
        ("strike_quality", "VARCHAR(32)"),
        ("target_num", "INTEGER"),
        ("hedge_num", "INTEGER"),
        ("notes", "TEXT"),
    ]
    try:
        for col_name, col_type in new_columns_sql:
            try:
                db.execute(text(f"ALTER TABLE prediction_audit ADD COLUMN {col_name} {col_type}"))
                db.commit()
                print(f"[DB Migrate] Added prediction_audit.{col_name}")
            except Exception:
                # Column already exists → ignore
                db.rollback()
    except Exception as e:
        print(f"[DB Migrate] Note: {e}")
        try:
            db.rollback()
        except Exception:
            pass


def save_prediction_audit(db, sequence_no, model_version, prob_big, predicted_digit, entropy, regime_id, drift_score, null_adv):
    """
    Legacy wrapper for backward compatibility — calls the extended save.
    """
    return save_full_prediction_audit(
        db=db,
        sequence_no=sequence_no,
        model_version=model_version,
        probability_big=prob_big,
        predicted_digit=predicted_digit,
        entropy=entropy,
        regime_id=regime_id,
        drift_score=drift_score,
        null_advantage=null_adv,
    )


def save_full_prediction_audit(
    db,
    sequence_no,
    model_version,
    probability_big,
    predicted_digit=None,
    entropy=None,
    regime_id=None,
    drift_score=None,
    null_advantage=None,
    # === EVOSEQ v3 extended fields ===
    generation=None,
    state_fingerprint=None,
    input_window_hash=None,
    state_similarity=None,
    state_sample_size=None,
    regime=None,
    adversarial_score=None,
    support_score=None,
    contradiction_score=None,
    uncertainty_score=None,
    calibrated_probability=None,
    calibration_error=None,
    expected_calibration_error=None,
    oos_score=None,
    baseline_score=None,
    edge_status=None,
    learning_status=None,
    model_reliability=None,
    knowledge_version=None,
    action=None,
    model_predictions=None,
    model_weights=None,
    ensemble_probability=None,
    meta_probability=None,
    p_success_l1=None,
    p_success_l2=None,
    p_success_l3=None,
    model_consensus=None,
    strike_quality=None,
    target_num=None,
    hedge_num=None,
):
    """
    Saves a FULL audit record with all EVOSEQ v3 fields.
    Never modifies an existing record — predictions are immutable.
    """
    try:
        existing = db.query(PredictionAudit).filter(PredictionAudit.sequence_no == str(sequence_no)).first()
        if existing:
            return existing  # Immutable: do not overwrite prior record

        def _to_json(obj):
            if obj is None:
                return None
            if isinstance(obj, str):
                return obj
            try:
                return json.dumps(obj)
            except Exception:
                return None

        audit = PredictionAudit(
            sequence_no=str(sequence_no),
            model_version=str(model_version) if model_version else "unknown",
            probability_big=float(probability_big) if probability_big is not None else 0.5,
            predicted_digit=int(predicted_digit) if predicted_digit is not None else None,
            entropy=float(entropy) if entropy is not None else None,
            regime_id=str(regime_id) if regime_id is not None else None,
            drift_score=float(drift_score) if drift_score is not None else None,
            null_advantage=float(null_advantage) if null_advantage is not None else None,
            # Extended v3
            generation=int(generation) if generation is not None else None,
            state_fingerprint_json=state_fingerprint if isinstance(state_fingerprint, (dict, list)) and False else None,  # stored via raw JSON column handler
            input_window_hash=str(input_window_hash) if input_window_hash is not None else None,
            state_similarity=float(state_similarity) if state_similarity is not None else None,
            state_sample_size=int(state_sample_size) if state_sample_size is not None else None,
            regime=str(regime) if regime is not None else None,
            adversarial_score=float(adversarial_score) if adversarial_score is not None else None,
            support_score=float(support_score) if support_score is not None else None,
            contradiction_score=float(contradiction_score) if contradiction_score is not None else None,
            uncertainty_score=float(uncertainty_score) if uncertainty_score is not None else None,
            calibrated_probability=float(calibrated_probability) if calibrated_probability is not None else None,
            calibration_error=float(calibration_error) if calibration_error is not None else None,
            expected_calibration_error=float(expected_calibration_error) if expected_calibration_error is not None else None,
            oos_score=float(oos_score) if oos_score is not None else None,
            baseline_score=float(baseline_score) if baseline_score is not None else None,
            edge_status=str(edge_status) if edge_status is not None else None,
            learning_status=str(learning_status) if learning_status is not None else None,
            action=str(action) if action is not None else None,
            ensemble_probability=float(ensemble_probability) if ensemble_probability is not None else None,
            meta_probability=float(meta_probability) if meta_probability is not None else None,
            p_success_l1=float(p_success_l1) if p_success_l1 is not None else None,
            p_success_l2=float(p_success_l2) if p_success_l2 is not None else None,
            p_success_l3=float(p_success_l3) if p_success_l3 is not None else None,
            model_consensus=float(model_consensus) if model_consensus is not None else None,
            strike_quality=str(strike_quality) if strike_quality is not None else None,
            target_num=int(target_num) if target_num is not None else None,
            hedge_num=int(hedge_num) if hedge_num is not None else None,
        )
        # JSON columns (stored as TEXT if dialect doesn't natively support JSON)
        try:
            audit.state_fingerprint_json = state_fingerprint if isinstance(state_fingerprint, (dict, list)) else None
        except Exception:
            audit.state_fingerprint_json = None
        try:
            audit.model_reliability_json = model_reliability if isinstance(model_reliability, (dict, list)) else None
        except Exception:
            audit.model_reliability_json = None
        try:
            audit.model_predictions_json = model_predictions if isinstance(model_predictions, (dict, list)) else None
        except Exception:
            audit.model_predictions_json = None
        try:
            audit.model_weights_json = model_weights if isinstance(model_weights, (dict, list)) else None
        except Exception:
            audit.model_weights_json = None
        db.add(audit)
        db.commit()
        return audit
    except Exception as e:
        db.rollback()
        print("Save full audit note:", e)
        return None

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
