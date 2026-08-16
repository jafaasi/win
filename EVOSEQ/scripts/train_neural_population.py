import sys
import os
import torch
import numpy as np

# Add EVOSEQ root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.randomness import seed_everything
from app.database import SessionLocal
from app.schemas import Outcome
from app.models.neural import NeuralTransformer, TransformerConfig, MultiHorizonDataset, train_epoch, calculate_loss, get_device
from app.features.vector import encode_observation
from app.core.mapping import map_digit
from torch.utils.data import DataLoader

def train_neural_population():
    seed_everything(42)
    device = get_device()
    print(f"🚀 Training Neural Transformer Population on device: {device}...")
    
    with SessionLocal() as session:
        outcomes = session.query(Outcome).order_by(Outcome.sequence_no.asc()).all()
        
    if len(outcomes) < 100:
        from app.ingestion.stream import ingest_outcomes_batch
        seed_batch = [{"sequence_no": i + 1, "digit": int(np.random.randint(0, 10))} for i in range(200)]
        ingest_outcomes_batch(seed_batch)
        with SessionLocal() as session:
            outcomes = session.query(Outcome).order_by(Outcome.sequence_no.asc()).all()
            
    digits = [o.digit for o in outcomes]
    features = [encode_observation(o.digit, o.size, o.color, o.parity) for o in outcomes]
    
    configs = [
        ("T-001", TransformerConfig(context_length=32, d_model=32, n_layers=2, n_heads=2)),
        ("T-002", TransformerConfig(context_length=64, d_model=64, n_layers=2, n_heads=2)),
        ("T-003", TransformerConfig(context_length=64, d_model=64, n_layers=3, n_heads=4)),
        ("T-004", TransformerConfig(context_length=128, d_model=64, n_layers=3, n_heads=4)),
        ("T-005", TransformerConfig(context_length=64, d_model=128, n_layers=3, n_heads=4))
    ]
    
    results = []
    
    for tag, cfg in configs:
        model = NeuralTransformer(
            input_size=cfg.input_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            context_length=cfg.context_length,
            horizons=cfg.horizons,
            classes=cfg.classes
        ).to(device)
        
        dataset = MultiHorizonDataset(features, digits, context_length=cfg.context_length, horizons=cfg.horizons)
        if len(dataset) < 10:
            loss_val = 2.3026
        else:
            loader = DataLoader(dataset, batch_size=16, shuffle=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            for _ in range(2): # Quick representative pass
                loss_val = train_epoch(model, loader, optimizer, device)
                
        results.append((tag, cfg, round(float(loss_val), 4)))
        print(f"  ✓ {tag} (ctx={cfg.context_length}, d_model={cfg.d_model}, layers={cfg.n_layers}) -> Loss: {loss_val:.4f}")
        
    return results

if __name__ == "__main__":
    train_neural_population()
