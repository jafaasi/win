from datetime import datetime
from typing import Optional, Dict, Any, List
import numpy as np

from ..ingestion.stream import get_latest_sequence, fetch_after
from ..features.basic import digit_entropy, digit_distribution
from ..features.runs import run_statistics
from ..models.uniform import UniformModel
from ..models.markov import MarkovModel
from ..evaluation.walk_forward import evaluate_model_walk_forward
from ..database import SessionLocal, Base, engine
from ..schemas import PredictionRecord, FeatureSnapshot
from .registry import ModelRegistry
from .drift import calculate_drift

# Ensure all tables exist on startup
Base.metadata.create_all(bind=engine)

def spawn_challengers(sequence: List[int], current_gen: int = 1) -> List[Dict[str, Any]]:
    """Spawns diverse challenger configurations (varying Markov orders and smoothing)."""
    challengers = []
    configs = [
        {"name": "Markov", "order": 1, "smoothing": 1.0},
        {"name": "Markov", "order": 2, "smoothing": 0.5},
        {"name": "Markov", "order": 3, "smoothing": 0.25},
        {"name": "Markov", "order": 4, "smoothing": 0.1},
        {"name": "UniformNull", "order": 0, "smoothing": 0.0}
    ]
    
    for cfg in configs:
        if cfg["name"] == "UniformNull":
            model = UniformModel()
            ver = f"null-v{current_gen}"
        else:
            model = MarkovModel(order=cfg["order"], smoothing=cfg["smoothing"])
            ver = f"mkv-ord{cfg['order']}-s{cfg['smoothing']}-v{current_gen}"
            
        challengers.append({
            "model_name": cfg["name"],
            "version": ver,
            "model_instance": model,
            "parameters": cfg
        })
        
    return challengers

def daily_evolution(last_seq_cursor: int = 0) -> Dict[str, Any]:
    """
    Executes the full EVOSEQ Daily Evolution Lifecycle:
    Ingest -> Feature Extraction -> Drift Detection -> Walk-Forward Audit -> Challenger Selection -> Champion Registration.
    """
    print("=== 🧠 EVOSEQ EVOLUTION CYCLE STARTED ===")
    
    # 1. Ingest new streaming observations
    observations = fetch_after(sequence_no=last_seq_cursor, limit=5000)
    if not observations:
        print("No new sequence observations to process.")
        return {"status": "NO_NEW_DATA"}
        
    digits = [int(o["digit"]) for o in observations]
    latest_seq = int(observations[-1]["sequence_no"])
    print(f"Ingested {len(digits)} observations (Cursor #{last_seq_cursor} -> #{latest_seq})")
    
    # 2. Extract and Snapshot Feature State
    ent = digit_entropy(digits)
    runs_stat = run_statistics(digits)
    feature_dict = {
        "entropy": ent,
        "run_stats": runs_stat,
        "sample_count": len(digits),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    with SessionLocal() as session:
        snap = FeatureSnapshot(sequence_no=latest_seq, features=feature_dict)
        session.add(snap)
        session.commit()
        
    # 3. Calculate Distribution Drift
    drift = calculate_drift(digits, recent_window=min(200, len(digits)//2), historical_window=min(1000, len(digits)))
    print(f"Drift Analysis: Level={drift.level}, JSD={drift.js_divergence}")
    
    # 4. Spawn & Evaluate Model Population
    registry = ModelRegistry()
    current_champion = registry.get_champion()
    gen_counter = (int(current_champion["id"]) + 1) if current_champion else 1
    
    challengers = spawn_challengers(digits, current_gen=gen_counter)
    initial_train_size = max(50, int(len(digits) * 0.5))
    
    best_challenger = None
    best_score = -float("inf")
    
    for c in challengers:
        eval_metrics = evaluate_model_walk_forward(
            c["model_instance"],
            digits,
            initial_train_size=initial_train_size
        )
        
        # Rigorous multi-objective score: S = - (20 * Brier) - (8 * LogLoss) + NullAdvantage
        score = eval_metrics["null_advantage"] - (eval_metrics["mean_brier_score"] * 5.0) - (eval_metrics["calibration_error"] * 2.0)
        
        cand_id = registry.register_candidate(
            model_name=c["model_name"],
            version=c["version"],
            parameters=c["parameters"],
            training_start=int(observations[0]["sequence_no"]),
            training_end=latest_seq,
            validation_accuracy=eval_metrics["accuracy"],
            validation_log_loss=eval_metrics["mean_log_loss"],
            validation_brier=eval_metrics["mean_brier_score"],
            status="challenger"
        )
        c["id"] = cand_id
        c["eval_metrics"] = eval_metrics
        c["score"] = score
        
        if score > best_score:
            best_score = score
            best_challenger = c
            
    # 5. Champion Selection & Promotion
    if best_challenger:
        registry.promote(
            best_challenger["id"],
            reason=f"Won walk-forward evaluation (Score: {best_score:.4f}, NullAdv: +{best_challenger['eval_metrics']['null_advantage']:.4f})"
        )
        print(f"🏆 CROWNED CHAMPION: {best_challenger['version']} (Score: {best_score:.4f})")
        
    pop_summary = registry.get_population_summary()
    print(f"Model Registry Summary: Tested={pop_summary['models_tested']}, Challengers={pop_summary['active_challengers']}, Retired={pop_summary['retired_models']}")
    print("=== 🧠 EVOSEQ EVOLUTION CYCLE COMPLETE ===")
    
    return {
        "status": "SUCCESS",
        "latest_sequence": latest_seq,
        "champion": best_challenger["version"] if best_challenger else "None",
        "drift": drift.level,
        "population": pop_summary
    }

def run_streaming_evolution_cycle():
    """Helper to run a single streaming step starting from latest sequence."""
    latest_cursor = get_latest_sequence()
    # If starting fresh, seed 0
    start_cursor = max(0, latest_cursor - 2000)
    return daily_evolution(last_seq_cursor=start_cursor)
