import sys
import os
import unittest
import torch
import numpy as np

# Add EVOSEQ root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.models.ssm import (
    S4DLayer,
    S4DSequenceModel,
    PyTorchSelectiveSSM,
    MambaSequenceModel,
    Mamba2SequenceModel,
    SSMAdapter,
    StateNormMonitor
)

class TestSSMPopulation(unittest.TestCase):

    def test_stateful_equals_full_context(self):
        """
        Mandatory CI Equivalence Test:
        Verifies that recurrent single-step state forwarding h_t = f(h_{t-1}, x_t)
        equals parallel full-context sequence forward pass.
        """
        torch.manual_seed(42)
        d_model = 16
        state_size = 32
        L = 10
        B = 2
        
        layer = S4DLayer(d_model=d_model, state_size=state_size)
        layer.eval()
        
        u = torch.randn(B, L, d_model)
        
        # Mode 1: Parallel full sequence forward
        full_output = layer(u) # [B, L, d_model]
        
        # Mode 2: Sequential step-by-step state forwarding
        step_outputs = []
        state = None
        for t in range(L):
            u_t = u[:, t, :]
            y_t, state = layer.step(u_t, state)
            step_outputs.append(y_t.unsqueeze(1))
            
        sequential_output = torch.cat(step_outputs, dim=1) # [B, L, d_model]
        
        # Check numerical equivalence
        diff = torch.max(torch.abs(full_output - sequential_output)).item()
        self.assertLess(diff, 1e-4, f"Stateful output diverged from full-context by {diff}")

    def test_s4d_sequence_model_multi_horizon(self):
        model = S4DSequenceModel(input_size=17, d_model=32, state_size=16, horizons=3, classes=10)
        x = torch.randn(4, 32, 17)
        logits = model(x)
        self.assertEqual(len(logits), 3)
        for lg in logits:
            self.assertEqual(lg.shape, (4, 10))

    def test_mamba_selective_ssm_and_mamba2(self):
        model = MambaSequenceModel(input_size=17, d_model=32, d_state=8, horizons=3)
        x = torch.randn(2, 32, 17)
        logits = model(x)
        self.assertEqual(len(logits), 3)
        
        m2 = Mamba2SequenceModel(input_size=17, d_model=32, d_state=16, horizons=3)
        logits_m2 = m2(x)
        self.assertEqual(len(logits_m2), 3)

    def test_stability_monitor(self):
        monitor = StateNormMonitor(max_norm_threshold=50.0)
        
        # Normal tensor
        t1 = torch.randn(4, 16)
        res1 = monitor.check_tensor(t1, "normal_state")
        self.assertTrue(res1["is_finite"])
        self.assertEqual(res1["status"], "HEALTHY")
        
        # Nan/Inf tensor
        t2 = torch.tensor([1.0, float("nan"), 3.0])
        res2 = monitor.check_tensor(t2, "bad_state")
        self.assertFalse(res2["is_finite"])
        self.assertEqual(res2["status"], "UNSTABLE_NAN_INF")

    def test_ssm_adapter(self):
        raw_s4 = S4DSequenceModel(input_size=17, d_model=16, state_size=8, horizons=3)
        adapter = SSMAdapter(model=raw_s4, context_length=16, epochs=1)
        
        seq = [1, 2, 3, 4, 5, 6, 7, 8] * 4
        adapter.fit(seq)
        
        probs = adapter.predict_proba(seq[:16])
        self.assertEqual(len(probs), 10)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4)
        
        multi_p = adapter.predict_multi_horizon(seq[:16])
        self.assertEqual(len(multi_p), 3)

if __name__ == "__main__":
    unittest.main()
