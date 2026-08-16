from datetime import datetime
from typing import Optional, Dict, Any, List
import numpy as np

from ..ingestion.stream import get_latest_sequence, fetch_after
from ..features.basic import digit_entropy, digit_distribution
from ..features.runs import run_statistics
from ..models.uniform import UniformModel
from ..models.markov import MarkovModel
from ..models.hmm_model import DiscreteHMM
from ..models.esn import EchoStateNetwork
from ..models.transformer import TransformerSequenceModel
from ..models.ensemble import MetaEnsemble
from ..evaluation.walk_forward import evaluate_model_walk_forward
from ..database import SessionLocal, Base, engine
from ..schemas import PredictionRecord, FeatureSnapshot
from .registry import ModelRegistry
from .drift import calculate_drift

# Ensure all tables exist on startup
Base.metadata.create_all(bind=engine)

def spawn_challengers(sequence: List[int], current_gen: int = 1, parent_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Spawns a diverse population of candidate sequence architectures:
    Markov, Discrete HMM, Echo State Network, Transformer, and Meta-Ensemble.
    """
    challengers = []
    
    # 1. Markov Models
    for ord_val in [1, 2, 3]:
        mkv = MarkovModel(order=ord_val, smoothing=0.5, version=f"markov-ord{ord_val}-gen{current_gen}")
        challengers.append({
            "model_name": "Markov",
            "version": mkv.metadata.version,
            "model_instance": mkv,
            "parameters": mkv.metadata.parameters,
            "mutation": {"order": ord_val, "smoothing": 0.5}
        })
        
    # 2. Discrete Hidden Markov Model
    for n_st in [2, 4]:
        hmm = DiscreteHMM(n_states=n_st, smoothing=1e-3, version=f"hmm-st{n_st}-gen{current_gen}")
        challengers.append({
            "model_name": "DiscreteHMM",
            "version": hmm.metadata.version,
            "model_instance": hmm,
            "parameters": hmm.metadata.parameters,
            "mutation": {"n_states": n_st, "smoothing": 1e-3}
        })
        
    # 3. Echo State Network Reservoir
    for (res_sz, rho) in [(64, 0.85), (128, 0.95)]:
        esn = EchoStateNetwork(
            input_size=10,
            reservoir_size=res_sz,
            spectral_radius=rho,
            leak_rate=0.3,
            version=f"esn-res{res_sz}-rho{rho}-gen{current_gen}"
        )
        challengers.append({
            "model_name": "EchoStateNetwork",
            "version": esn.metadata.version,
            "model_instance": esn,
            "parameters": esn.metadata.parameters,
            "mutation": {"reservoir_size": res_sz, "spectral_radius": rho}
        })
        
    # 4. PyTorch Transformer
    trans = TransformerSequenceModel(
        input_size=10,
        hidden_size=32,
        heads=2,
        layers=1,
        temperature=1.1,
        version=f"transformer-h32-gen{current_gen}"
    )
    challengers.append({
        "model_name": "Transformer",
        "version": trans.metadata.version,
        "model_instance": trans,
        "parameters": trans.metadata.parameters,
        "mutation": {"hidden_size": 32, "heads": 2, "temperature": 1.1}
    })
    
    # 5. Null Baseline
    null_mod = UniformModel(version=f"null-gen{current_gen}")
    challengers.append({
        "model_name": "UniformNull",
        "version": null_mod.metadata.version,
        "model_instance": null_mod,
        "parameters": null_mod.metadata.parameters,
        "mutation": {"type": "null"}
    })
    
    return challengers

def daily_evolution(last_seq_cursor: int = 0) -> Dict[str, Any]:
    """
    Executes the full EVOSEQ Daily Evolution Lifecycle:
    Ingest -> Feature Extraction -> Drift Detection -> Locked-Future Walk-Forward Audit ->
    Multi-Architecture Evaluation -> Champion Selection -> Model Genealogy Logging.
    """
    print("=== 🧠 EVOSEQ EVOLUTION CYCLE STARTED ===")
    
    # 1. Ingest streaming observations
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
    
    # 4. Anti-Overfitting Mechanism: Locked Future Buffer
    # Reserve last 25% of data strictly for out-of-sample statistical audit
    total_len = len(digits)
    locked_split = int(total_len * 0.75) if total_len >= 40 else total_len
    dev_sequence = digits[:locked_split]
    locked_future = digits[locked_split:] if total_len >= 40 else digits
    
    registry = ModelRegistry()
    current_champion = registry.get_champion()
    current_champ_id = current_champion["id"] if current_champion else None
    gen_counter = (int(current_champion["id"]) + 1) if current_champion else 1
    
    challengers = spawn_challengers(digits, current_gen=gen_counter, parent_id=current_champ_id)
    initial_train_size = max(20, int(len(dev_sequence) * 0.5))
    
    best_challenger = None
    best_score = -float("inf")
    evaluated_models = []
    evaluated_scores = []
    
    for c in challengers:
        eval_metrics = evaluate_model_walk_forward(
            c["model_instance"],
            digits,
            initial_train_size=initial_train_size
        )
        
        # Rigorous research score S = NullAdvantage - (5 * Brier) - (2 * Calibration)
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
            status="challenger",
            parent_model_id=current_champ_id,
            generation=gen_counter,
            mutation_details=c.get("mutation")
        )
        c["id"] = cand_id
        c["eval_metrics"] = eval_metrics
        c["score"] = score
        
        evaluated_models.append(c["model_instance"])
        evaluated_scores.append(score)
        
        if score > best_score:
            best_score = score
            best_challenger = c
            
    # 5. Build and Evaluate Dynamic Meta-Ensemble
    meta_ens = MetaEnsemble(models=evaluated_models, scores=evaluated_scores, version=f"ensemble-gen{gen_counter}")
    ens_metrics = evaluate_model_walk_forward(meta_ens, digits, initial_train_size=initial_train_size)
    ens_score = ens_metrics["null_advantage"] - (ens_metrics["mean_brier_score"] * 5.0) - (ens_metrics["calibration_error"] * 2.0)
    
    ens_id = registry.register_candidate(
        model_name="MetaEnsemble",
        version=meta_ens.metadata.version,
        parameters={"n_models": len(evaluated_models)},
        training_start=int(observations[0]["sequence_no"]),
        training_end=latest_seq,
        validation_accuracy=ens_metrics["accuracy"],
        validation_log_loss=ens_metrics["mean_log_loss"],
        validation_brier=ens_metrics["mean_brier_score"],
        status="challenger",
        parent_model_id=current_champ_id,
        generation=gen_counter,
        mutation_details={"type": "meta_ensemble"}
    )
    
    if ens_score > best_score:
        best_score = ens_score
        best_challenger = {
            "id": ens_id,
            "version": meta_ens.metadata.version,
            "eval_metrics": ens_metrics
        }
        
    # 6. Champion Selection & Promotion
    if best_challenger:
        registry.promote(
            best_challenger["id"],
            reason=f"Won out-of-sample audit (Score: {best_score:.4f}, NullAdv: +{best_challenger['eval_metrics']['null_advantage']:.4f})"
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
    latest_cursor = get_latest_sequence()
    start_cursor = max(0, latest_cursor - 2000)
    return daily_evolution(last_seq_cursor=start_cursor)
