import sys
import os
import numpy as np

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.schemas import Outcome, ModelVersionRecord
from app.models.markov import MarkovModel
from app.models.hmm import DiscreteHMM
from app.models.esn import EchoStateNetwork
from app.models.transformer import CausalTransformer
from app.models.ssm import S4SequenceModel, MambaSequenceModel
from app.evaluation.walk_forward import evaluate_model_walk_forward
from app.evolution.registry import ModelRegistry

def train_population():
    print("🧠 Training EVOSEQ Model Population...")
    with SessionLocal() as session:
        outcomes = session.query(Outcome).order_by(Outcome.sequence_no.asc()).all()
        digits = [o.digit for o in outcomes]
        
    if len(digits) < 50:
        print("⚠️ Insufficient historical records (<50). Run backfill_features.py first.")
        return
        
    registry = ModelRegistry()
    models = [
        ("Markov", MarkovModel(order=2, smoothing=0.5), {"order": 2, "smoothing": 0.5}),
        ("DiscreteHMM", DiscreteHMM(n_states=3), {"n_states": 3}),
        ("EchoStateNetwork", EchoStateNetwork(reservoir_size=64, spectral_radius=0.9), {"reservoir_size": 64, "spectral_radius": 0.9}),
        ("Transformer", CausalTransformer(d_model=32, nhead=2, layers=2), {"d_model": 32, "nhead": 2, "layers": 2}),
        ("S4", S4SequenceModel(d_input=10, d_state=16, d_model=32), {"d_state": 16, "d_model": 32}),
        ("Mamba", MambaSequenceModel(d_input=10, d_state=16, d_model=32), {"d_state": 16, "d_model": 32})
    ]
    
    print(f"📊 Running walk-forward validation across {len(models)} model families...")
    for name, model_inst, params in models:
        metrics = evaluate_model_walk_forward(model_inst, digits, initial_train_size=max(20, len(digits) // 2))
        cand_id = registry.register_candidate(
            model_name=name,
            version=f"{name.lower()}-v1.0",
            parameters=params,
            training_start=1,
            training_end=len(digits),
            validation_accuracy=metrics["accuracy"],
            validation_log_loss=metrics["mean_log_loss"],
            validation_brier=metrics["mean_brier_score"],
            status="challenger"
        )
        print(f"  - {name:18}: Acc={metrics['accuracy']:.4f} | LogLoss={metrics['mean_log_loss']:.4f} | Brier={metrics['mean_brier_score']:.4f} (ID: {cand_id})")
        
    print("✅ Model Population training complete.")

if __name__ == "__main__":
    train_population()
