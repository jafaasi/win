import sys
import os
import unittest
import torch
import numpy as np

# Add EVOSEQ root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.models.neural import (
    SinusoidalPositionEncoding,
    NeuralTransformer,
    TransformerConfig,
    MultiHorizonDataset,
    calculate_loss,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    ReplayBuffer,
    recency_weights
)

class TestNeuralPopulation(unittest.TestCase):

    def test_sinusoidal_position_encoding(self):
        pe = SinusoidalPositionEncoding(d_model=32, max_length=128)
        x = torch.zeros(2, 64, 32)
        out = pe(x)
        self.assertEqual(out.shape, (2, 64, 32))
        self.assertFalse(torch.all(out == 0))

    def test_neural_transformer_causal_forward(self):
        model = NeuralTransformer(
            input_size=17,
            d_model=32,
            n_heads=2,
            n_layers=2,
            context_length=64,
            horizons=3,
            classes=10
        )
        x = torch.randn(4, 64, 17)
        logits = model(x)
        self.assertEqual(len(logits), 3) # H1, H2, H3
        for lg in logits:
            self.assertEqual(lg.shape, (4, 10))
            
        probs = model.predict_proba(x)
        self.assertEqual(len(probs), 3)
        self.assertAlmostEqual(float(torch.sum(probs[0][0]).item()), 1.0, places=4)

    def test_multi_horizon_dataset(self):
        feats = np.random.randn(100, 17).astype(np.float32)
        digits = np.random.randint(0, 10, size=100)
        ds = MultiHorizonDataset(feats, digits, context_length=32, horizons=3)
        self.assertEqual(len(ds), 100 - 32 - 3 + 1)
        
        X, y = ds[0]
        self.assertEqual(X.shape, (32, 17))
        self.assertEqual(y.shape, (3,))

    def test_calculate_loss_and_early_stopping(self):
        logits = [torch.randn(4, 10), torch.randn(4, 10), torch.randn(4, 10)]
        targets = torch.randint(0, 10, (4, 3))
        loss, individual = calculate_loss(logits, targets)
        self.assertGreater(float(loss.item()), 0.0)
        self.assertEqual(len(individual), 3)
        
        # Early Stopping
        es = EarlyStopping(patience=3, min_delta=1e-3)
        self.assertFalse(es.update(2.5))
        self.assertFalse(es.update(2.4))
        self.assertFalse(es.update(2.4))
        self.assertFalse(es.update(2.4))
        self.assertTrue(es.update(2.4))

    def test_checkpointing_and_replay(self):
        model = NeuralTransformer(input_size=17, d_model=16, n_heads=2, n_layers=1, horizons=3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        ckpt_path = "/tmp/test_evoseq_transformer.pt"
        save_checkpoint(ckpt_path, model, optimizer, epoch=1, validation_loss=2.29)
        
        new_model = NeuralTransformer(input_size=17, d_model=16, n_heads=2, n_layers=1, horizons=3)
        data = load_checkpoint(ckpt_path, model=new_model)
        self.assertEqual(data["epoch"], 1)
        
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
            
        # Replay buffer
        buf = ReplayBuffer(recent_capacity=10, historical_capacity=10)
        for i in range(25):
            buf.add_recent(i)
        sampled = buf.sample(batch_size=8)
        self.assertGreater(len(sampled), 0)
        
        weights = recency_weights(10)
        self.assertEqual(len(weights), 10)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=4)

if __name__ == "__main__":
    unittest.main()
