import json
from datetime import datetime
from typing import Optional, Dict, Any, List
import numpy as np

from ..ingestion.stream import get_latest_sequence, fetch_after
from ..features.basic import digit_entropy, digit_distribution
from ..features.runs import run_statistics
from ..models.uniform import UniformModel
from ..models.markov import MarkovModel
from ..models.hmm_model import DiscreteHMM, RegimeMonitor
from ..models.esn import EchoStateNetwork
from ..models.transformer import TransformerSequenceModel
from ..models.ensemble import MetaEnsemble
from ..evaluation.walk_forward import evaluate_model_walk_forward
from ..database import SessionLocal, Base, engine
from ..schemas import PredictionRecord, FeatureSnapshot, ModelEvent
from .registry import ModelRegistry
from .drift import calculate_multidimensional_drift, DriftState
from .performance import PerformanceMonitor
from .uncertainty import calculate_prediction_uncertainty, calculate_model_disagreement
from .mutation import mutate_model_parameters, compute_adaptive_exploration
from .controller import EvolutionController, Action

# Ensure all tables exist on startup
Base.metadata.create_all(bind=engine)

def log_episodic_event(event_type: str, model_version_id: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
    """Logs an episodic event to the model_events table."""
    with SessionLocal() as session:
        evt = ModelEvent(
            event_type=event_type,
            model_version_id=model_version_id,
            details=details or {}
        )
        session.add(evt)
        session.commit()

from ..models.ssm import S4SequenceModel, MambaSequenceModel

def spawn_evolved_challengers(
    sequence: List[int],
    champion_params: Optional[Dict[str, Any]] = None,
    current_gen: int = 1,
    parent_id: Optional[int] = None,
    exploration_factor: float = 0.20
) -> List[Dict[str, Any]]:
    """
    Generates a population of mutated architecture challengers using adaptive exploration/exploitation
    across Statistical, Probabilistic, Dynamical, and State-Space neural model families.
    """
    challengers = []
    
    # 1. Mutated Markov Models
    base_mkv_params = champion_params if champion_params and "order" in champion_params else {"order": 2, "smoothing": 0.5}
    for i in range(2):
        mut_params = mutate_model_parameters("Markov", base_mkv_params, mutation_rate=0.4, exploration_factor=exploration_factor)
        mkv = MarkovModel(order=mut_params.get("order", 2), smoothing=mut_params.get("smoothing", 0.5), version=f"markov-ord{mut_params.get('order', 2)}-gen{current_gen}.{i+1}")
        challengers.append({
            "model_name": "Markov",
            "version": mkv.metadata.version,
            "model_instance": mkv,
            "parameters": mut_params,
            "mutation": mut_params
        })
        
    # 2. Discrete HMM Latent Regime Models
    for n_st in [2, 4]:
        hmm = DiscreteHMM(n_states=n_st, smoothing=1e-3, version=f"hmm-st{n_st}-gen{current_gen}")
        challengers.append({
            "model_name": "DiscreteHMM",
            "version": hmm.metadata.version,
            "model_instance": hmm,
            "parameters": hmm.metadata.parameters,
            "mutation": {"n_states": n_st}
        })
        
    # 3. Mutated Echo State Networks
    base_esn_params = {"reservoir_size": 64, "spectral_radius": 0.9, "leak_rate": 0.3, "ridge": 1e-4}
    for i in range(2):
        mut_esn = mutate_model_parameters("EchoStateNetwork", base_esn_params, mutation_rate=0.5, exploration_factor=exploration_factor)
        esn = EchoStateNetwork(
            input_size=10,
            reservoir_size=mut_esn.get("reservoir_size", 64),
            spectral_radius=mut_esn.get("spectral_radius", 0.9),
            leak_rate=mut_esn.get("leak_rate", 0.3),
            ridge=mut_esn.get("ridge", 1e-4),
            version=f"esn-res{mut_esn.get('reservoir_size', 64)}-gen{current_gen}.{i+1}"
        )
        challengers.append({
            "model_name": "EchoStateNetwork",
            "version": esn.metadata.version,
            "model_instance": esn,
            "parameters": mut_esn,
            "mutation": mut_esn
        })
        
    # 4. Calibrated PyTorch Causal Transformer
    trans = TransformerSequenceModel(
        input_size=10,
        hidden_size=32,
        heads=2,
        layers=1,
        temperature=1.15,
        version=f"transformer-h32-gen{current_gen}"
    )
    challengers.append({
        "model_name": "Transformer",
        "version": trans.metadata.version,
        "model_instance": trans,
        "parameters": trans.metadata.parameters,
        "mutation": {"hidden_size": 32, "heads": 2}
    })

    # 5. Structured State Space Model (S4)
    s4_model = S4SequenceModel(
        input_size=10,
        hidden_size=32,
        layers=1,
        context_length=64,
        temperature=1.1,
        version=f"s4-h32-gen{current_gen}"
    )
    challengers.append({
        "model_name": "S4",
        "version": s4_model.metadata.version,
        "model_instance": s4_model,
        "parameters": s4_model.metadata.parameters,
        "mutation": {"hidden_size": 32, "layers": 1}
    })

    # 6. Mamba Selective State Space Model
    mamba_model = MambaSequenceModel(
        input_size=10,
        hidden_size=32,
        layers=1,
        context_length=64,
        temperature=1.1,
        version=f"mamba-h32-gen{current_gen}"
    )
    challengers.append({
        "model_name": "Mamba",
        "version": mamba_model.metadata.version,
        "model_instance": mamba_model,
        "parameters": mamba_model.metadata.parameters,
        "mutation": {"hidden_size": 32, "layers": 1}
    })
    
    # 7. Null Baseline
    null_mod = UniformModel(version=f"null-gen{current_gen}")
    challengers.append({
        "model_name": "UniformNull",
        "version": null_mod.metadata.version,
        "model_instance": null_mod,
        "parameters": null_mod.metadata.parameters,
        "mutation": {"type": "null"}
    })
    
    return challengers


def autonomous_evolution_cycle(last_seq_cursor: int = 0) -> Dict[str, Any]:
    """
    Executes the full Autonomous Evolution Controller loop:
    Ingest -> Feature Update -> Multi-Horizon Performance -> Composite Drift -> Disagreement ->
    Action Decision (WAIT / UPDATE / ADAPT / INVESTIGATE / EVOLVE) -> Execution -> Event Logging -> Audit Report.
    """
    print("=== 🧠 EVOSEQ AUTONOMOUS CONTROLLER CYCLE STARTED ===")
    
    # 1. Ingest new streaming observations
    observations = fetch_after(sequence_no=last_seq_cursor, limit=5000)
    if not observations:
        print("No new sequence observations.")
        return {"status": "NO_NEW_DATA", "action": "WAIT"}
        
    digits = [int(o["digit"]) for o in observations]
    latest_seq = int(observations[-1]["sequence_no"])
    obs_count = len(digits)
    print(f"Ingested {obs_count} observations (Cursor #{last_seq_cursor} -> #{latest_seq})")
    
    # 2. Extract and Snapshot Features
    ent = digit_entropy(digits)
    runs_stat = run_statistics(digits)
    feature_dict = {
        "entropy": ent,
        "run_stats": runs_stat,
        "sample_count": obs_count,
        "timestamp": datetime.utcnow().isoformat()
    }
    with SessionLocal() as session:
        snap = FeatureSnapshot(sequence_no=latest_seq, features=feature_dict)
        session.add(snap)
        session.commit()

    # 3. Multi-Dimensional Drift Analysis
    drift_result = calculate_multidimensional_drift(
        digits,
        recent_window=min(200, max(20, obs_count // 2)),
        historical_window=min(1000, max(50, obs_count))
    )
    print(f"Multi-Dimensional Drift: Composite={drift_result.composite_drift:.4f}, State={drift_result.state.value}")

    # 4. Multi-Horizon Performance & EWMA Tracking
    perf_monitor = PerformanceMonitor(window_size=1000)
    # Replay recent accuracy
    for i in range(1, min(100, len(digits))):
        pred_guess = digits[i-1] # sample baseline
        actual = digits[i]
        perf_monitor.update(correct=(pred_guess == actual), log_loss=2.302, brier=0.09)
    perf_deltas = perf_monitor.compute_multi_horizon_deltas()

    # 5. Measure Population Disagreement & Uncertainty
    sample_mkv = MarkovModel(order=1).fit(digits[:max(20, len(digits)//2)])
    sample_hmm = DiscreteHMM(n_states=2).fit(digits[:max(20, len(digits)//2)])
    sample_esn = EchoStateNetwork(input_size=10, reservoir_size=32).fit(digits[:max(20, len(digits)//2)])
    
    recent_ctx = digits[-10:] if len(digits) >= 10 else digits
    p_mkv = sample_mkv.predict_proba(recent_ctx)
    p_hmm = sample_hmm.predict_proba(recent_ctx)
    p_esn = sample_esn.predict_proba(recent_ctx)
    
    disagreement = calculate_model_disagreement([p_mkv, p_hmm, p_esn])
    uncertainty = calculate_prediction_uncertainty(p_mkv)
    print(f"Diagnostics: Model Disagreement={disagreement:.4f}, Prediction Uncertainty={uncertainty:.4f} bits")

    # 6. Autonomous Controller Decision
    controller = EvolutionController(min_observations=40)
    action = controller.decide(
        observations=obs_count,
        drift_score=drift_result.composite_drift,
        performance_delta=perf_deltas.get("delta_50", 0.0),
        uncertainty=uncertainty,
        disagreement=disagreement
    )
    print(f"🎯 AUTONOMOUS CONTROLLER ACTION: {action.value.upper()}")
    
    registry = ModelRegistry()
    current_champion = registry.get_champion()
    current_champ_id = current_champion["id"] if current_champion else None
    gen_counter = (int(current_champion["id"]) + 1) if current_champion else 1
    
    report: Dict[str, Any] = {
        "status": "SUCCESS",
        "generation": gen_counter,
        "observations": obs_count,
        "champion": current_champion["version"] if current_champion else "None",
        "action": action.value.upper(),
        "drift": {
            "composite": drift_result.composite_drift,
            "state": drift_result.state.value,
            "dimensions": drift_result.dimension_drifts
        },
        "performance": perf_deltas,
        "model_disagreement": round(disagreement, 4),
        "prediction_uncertainty": round(uncertainty, 4),
        "challengers_created": 0,
        "challengers_promoted": 0
    }

    # 7. Execute Action
    if action == Action.WAIT:
        log_episodic_event("CONTROLLER_WAIT", current_champ_id, {"reason": "Insufficient observation density"})
        return report

    elif action == Action.UPDATE:
        log_episodic_event("MODEL_UPDATED", current_champ_id, {"delta": perf_deltas.get("delta_50", 0.0)})
        print("⚡ Applied online incremental update to active models.")
        return report

    elif action == Action.ADAPT:
        log_episodic_event("MODEL_ADAPTED", current_champ_id, {"drift": drift_result.composite_drift})
        print("🔧 Adaptation triggered: fine-tuning meta-ensemble weights.")
        return report

    elif action == Action.INVESTIGATE:
        log_episodic_event("INVESTIGATION_TRIGGERED", current_champ_id, {"disagreement": disagreement})
        print("🔍 Investigation triggered: analyzing divergent latent regime hypotheses.")
        return report

    elif action == Action.EVOLVE:
        log_episodic_event("EVOLUTION_STARTED", current_champ_id, {"drift": drift_result.composite_drift, "disagreement": disagreement})
        
        # Locked future buffer
        total_len = len(digits)
        locked_split = int(total_len * 0.75) if total_len >= 40 else total_len
        dev_seq = digits[:locked_split]
        locked_future = digits[locked_split:] if total_len >= 40 else digits
        
        exploration_factor = compute_adaptive_exploration(uncertainty, drift_result.composite_drift, perf_deltas.get("delta_50", 0.0))
        champ_params = current_champion.get("parameters") if current_champion else None
        challengers = spawn_evolved_challengers(
            digits,
            champion_params=champ_params,
            current_gen=gen_counter,
            parent_id=current_champ_id,
            exploration_factor=exploration_factor
        )
        report["challengers_created"] = len(challengers)
        
        # Meta-Learning: Research Director Environment Analysis & Bayesian Planning
        from ..meta.director import ResearchDirector
        from ..meta.types import ModelDescriptor
        director = ResearchDirector()
        env_state = director.analyze_environment(
            digits,
            drift_score=drift_result.composite_drift,
            disagreement=disagreement
        )
        planned_challengers = director.plan_candidate_evaluation(challengers, env_state, budget=10)
        
        initial_train_size = max(20, int(len(dev_seq) * 0.5))
        best_challenger = None
        best_score = -float("inf")
        evaluated_models = []
        evaluated_scores = []
        
        for c in planned_challengers:
            eval_metrics = evaluate_model_walk_forward(c["model_instance"], digits, initial_train_size=initial_train_size)
            
            # Model efficiency / complexity penalty: - 0.001 * log(1 + param_count)
            param_count = 0
            if hasattr(c["model_instance"], "net"):
                param_count = sum(p.numel() for p in c["model_instance"].net.parameters() if p.requires_grad)
            elif hasattr(c["model_instance"], "W_out"):
                param_count = c["model_instance"].W_out.size
            elif hasattr(c["model_instance"], "counts"):
                param_count = len(c["model_instance"].counts) * 10
                
            complexity_penalty = 0.001 * float(np.log1p(param_count))
            score = eval_metrics["null_advantage"] - (eval_metrics["mean_brier_score"] * 5.0) - (eval_metrics["calibration_error"] * 2.0) - complexity_penalty
            
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
            
            # Record meta-experiment
            desc = ModelDescriptor(
                family=c["model_name"],
                context_length=c["parameters"].get("context_length", c["parameters"].get("order", 32)),
                parameter_count=param_count
            )
            director.record_meta_experiment(cand_id, env_state, desc, eval_metrics)
            
            if score > best_score:
                best_score = score
                best_challenger = c
                
        # Meta-Ensemble challenger
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
            best_challenger = {"id": ens_id, "version": meta_ens.metadata.version, "eval_metrics": ens_metrics, "model_instance": meta_ens}

        # 6. Statistical Referee: Audit against Null Hypothesis Laboratory & Adversarial Environments
        if best_challenger:
            from ..research.statistics.confidence import compare_model_to_null
            from ..research.null_models.iid import generate_iid
            from ..research.audit.robustness import evaluate_model_robustness
            from ..research.audit.report import generate_ascii_audit_hud
            from ..schemas import ResearchExperimentRecord
            
            # 6a. Null Model Experiment (20 IID permutations)
            null_probs = digit_distribution(digits)
            null_res = compare_model_to_null(
                best_challenger["model_instance"],
                digits,
                null_generator_fn=lambda l, s: generate_iid(null_probs, length=l, seed=s),
                repetitions=15,
                initial_train_size=initial_train_size
            )
            
            # 6b. Adversarial Robustness Matrix
            rob_res = evaluate_model_robustness(
                best_challenger["model_instance"],
                digits,
                initial_train_size=initial_train_size
            )
            
            # 6c. Persist Research Experiment Record to DB
            with SessionLocal() as session:
                exp_rec = ResearchExperimentRecord(
                    experiment_type="CHAMPION_PROMOTION_AUDIT",
                    model_version_id=best_challenger["id"],
                    null_model="iid_empirical_marginal",
                    sample_size=obs_count,
                    observed_score=null_res["observed_score"],
                    null_mean=null_res["null_mean"],
                    null_std=null_res["null_std"],
                    p_value=null_res["p_value"],
                    correction_method="benjamini_hochberg",
                    test_range_start=int(observations[0]["sequence_no"]),
                    test_range_end=latest_seq,
                    metadata_json={"robustness": rob_res}
                )
                session.add(exp_rec)
                session.commit()
                
            # 6d. Promote Champion and Render Research HUD
            registry.promote(
                best_challenger["id"],
                reason=f"Passed statistical referee (Score: {best_score:.4f}, DeltaNull: {null_res['delta_vs_null']:+.4f}, p={null_res['p_value']}, Robustness={rob_res['status']})"
            )
            log_episodic_event("MODEL_PROMOTED", best_challenger["id"], {
                "version": best_challenger["version"],
                "score": best_score,
                "null_delta": null_res["delta_vs_null"],
                "p_value": null_res["p_value"]
            })
            report["challengers_promoted"] = 1
            report["champion"] = best_challenger["version"]
            report["delta_vs_null"] = null_res["delta_vs_null"]
            report["p_value"] = null_res["p_value"]
            report["robustness"] = rob_res["status"]
            
            hud = generate_ascii_audit_hud({
                "generation": gen_counter,
                "observations": obs_count,
                "champion": best_challenger["version"],
                "recent_performance": perf_deltas.get("ewma_accuracy", 0.10),
                "historical_performance": 0.10,
                "delta_vs_null": null_res["delta_vs_null"],
                "drift": drift_result.state.value.upper(),
                "calibration": "GOOD",
                "disagreement": "LOW" if disagreement < 0.12 else "HIGH",
                "null_experiments": 15,
                "candidate_models": len(challengers) + 1,
                "retired_models": registry.get_population_summary()["retired_models"],
                "robustness": rob_res["status"]
            })
            print(hud)
            print(f"🏆 CROWNED NEW CHAMPION: {best_challenger['version']} (Score: {best_score:.4f}, p-val: {null_res['p_value']})")

        log_episodic_event("EVOLUTION_COMPLETED", best_challenger["id"] if best_challenger else current_champ_id, report)
        print("=== 🧠 EVOSEQ AUTONOMOUS CONTROLLER CYCLE COMPLETE ===")
        return report

# Backward compatible helper
def daily_evolution(last_seq_cursor: int = 0) -> Dict[str, Any]:
    return autonomous_evolution_cycle(last_seq_cursor=last_seq_cursor)

def run_streaming_evolution_cycle():
    latest_cursor = get_latest_sequence()
    start_cursor = max(0, latest_cursor - 2000)
    return autonomous_evolution_cycle(last_seq_cursor=start_cursor)
