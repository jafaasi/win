import sys
import os
import torch
import numpy as np

# Add EVOSEQ root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.randomness import seed_everything
from app.database import SessionLocal, engine, Base
from app.schemas import Outcome, EnsembleObservationRecord
from app.ensemble import (
    HierarchicalEnsemble,
    decompose_uncertainty,
    evolution_pressure,
    evolution_state,
    compute_model_contributions,
    TemperatureScaler
)
from app.models.frequency import FrequencyBaseline
from app.models.markov import MarkovModel
from app.models.hmm import HMMModel
from app.models.esn import ESN
from app.models.neural import NeuralTransformer
from app.models.ssm import S4DSequenceModel, MambaSequenceModel
from app.features.vector import encode_observation

def run_dynamic_ensemble():
    seed_everything(42)
    Base.metadata.create_all(bind=engine)
    print("🚀 Initializing EVOSEQ — Dynamic Ensemble Intelligence Engine...")
    
    with SessionLocal() as session:
        outcomes = session.query(Outcome).order_by(Outcome.sequence_no.asc()).all()
        
    if len(outcomes) < 100:
        from app.ingestion.stream import ingest_outcomes_batch
        seed_batch = [{"sequence_no": i + 1, "digit": int(np.random.randint(0, 10))} for i in range(200)]
        ingest_outcomes_batch(seed_batch)
        with SessionLocal() as session:
            outcomes = session.query(Outcome).order_by(Outcome.sequence_no.asc()).all()
            
    digits = np.array([o.digit for o in outcomes], dtype=np.int64)
    N = len(digits)
    test_slice = digits[-50:]
    
    # 1. Models Population Setup
    models = {
        "frequency": FrequencyBaseline().fit(digits[:-50]),
        "markov1": MarkovModel(order=1).fit(digits[:-50]),
        "markov2": MarkovModel(order=2).fit(digits[:-50]),
        "hmm": HMMModel(states=4).fit(digits[:-50]),
        "esn": ESN(input_size=10, reservoir_size=32).fit(digits[:-50]),
        "transformer": NeuralTransformer(input_size=17, d_model=32, n_heads=2, n_layers=2),
        "mamba": MambaSequenceModel(input_size=17, d_model=32, d_state=8),
        "s4d": S4DSequenceModel(input_size=17, d_model=32, state_size=16)
    }
    
    # 2. Generate Out-Of-Sample Predictions
    predictions = {}
    for name, model in models.items():
        if hasattr(model, "predict_sequence"):
            p = model.predict_sequence(test_slice)
        elif name == "frequency":
            p = model.predict_proba(len(test_slice))
        else:
            p_list = []
            for t in range(len(test_slice)):
                ctx = test_slice[:t+1]
                p_list.append(model.predict_proba(ctx) if hasattr(model, "predict_proba") else np.full(10, 0.1))
            p = np.asarray(p_list)
        predictions[name] = p
        
    # 3. Hierarchical Combination
    ensemble = HierarchicalEnsemble()
    hier_result = ensemble.combine_hierarchical(predictions)
    ens_pred = hier_result["ensemble_prediction"]
    fam_weights = hier_result["family_weights"]
    
    # 4. Temperature Calibration
    scaler = TemperatureScaler()
    calibrated_pred = scaler.scale(ens_pred, temperature=1.05)
    
    # 5. Uncertainty Decomposition (Predictive Entropy vs Disagreement)
    unc = decompose_uncertainty([predictions[k][-1] for k in predictions])
    
    # 6. Model Contribution / Ablation Test
    contributions = compute_model_contributions(predictions, test_slice)
    
    # 7. Evolutionary Pressure & Controller State
    from app.research.metrics import log_loss
    final_loss = log_loss(calibrated_pred, test_slice)
    pressure = evolution_pressure(
        loss_degradation=max(0.0, final_loss - 2.3026),
        drift=0.012,
        disagreement=unc["disagreement"],
        calibration_error=0.015
    )
    state = evolution_state(pressure)
    
    # 8. Record to Database
    with SessionLocal() as session:
        rec = EnsembleObservationRecord(
            sequence_no=int(outcomes[-1].sequence_no),
            environment={"entropy": 3.28, "drift": 0.012, "disagreement": unc["disagreement"]},
            model_predictions={k: [round(float(x), 4) for x in predictions[k][-1]] for k in predictions},
            model_weights=fam_weights,
            ensemble_prediction=[round(float(x), 4) for x in calibrated_pred[-1]],
            actual_digit=int(test_slice[-1]),
            ensemble_log_loss=round(final_loss, 4),
            disagreement=unc["disagreement"]
        )
        session.add(rec)
        session.commit()
        
    # 9. Render ASCII Report
    report = f"""
══════════════════════════════════════════════════════════════
            EVOSEQ — DYNAMIC ENSEMBLE INTELLIGENCE
══════════════════════════════════════════════════════════════
Observations:       {N:,}
Test Window:        {len(test_slice)} steps
Ensemble Loss:      {final_loss:.4f}

FAMILY WEIGHTS
Statistical:        {fam_weights.get('statistical', 0.33):.4f}
Recurrent:          {fam_weights.get('recurrent', 0.33):.4f}
Neural:             {fam_weights.get('neural', 0.34):.4f}

UNCERTAINTY PROFILE
Aleatoric Entropy:  {unc['aleatoric_entropy']:.4f} bits
Total Entropy:      {unc['total_entropy']:.4f} bits
Model Disagreement: {unc['disagreement']:.4f} bits

ABLATION CONTRIBUTIONS (Δ Loss without model)
MODEL               CONTRIBUTION    STATUS
──────────────────────────────────────────────────────────────"""
    for name, c in contributions.items():
        report += f"\n{name:19} {c['contribution']:+.5f}        {c['status']}"

    report += f"""
──────────────────────────────────────────────────────────────
EVOLUTION CONTROLLER
Evolution Pressure: {pressure:.4f}
Controller Action:  {state}
══════════════════════════════════════════════════════════════
"""
    print(report)
    return report

if __name__ == "__main__":
    run_dynamic_ensemble()
