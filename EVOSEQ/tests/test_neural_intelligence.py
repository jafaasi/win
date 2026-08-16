import sys
import os
import unittest
import numpy as np
import torch

# Add EVOSEQ root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.dataset import SequenceDataset
from app.models.transformer import CausalTransformer, TransformerSequenceModel, temperature_scale
from app.models.ssm import StateSpaceLayer, S4SequenceModel, MambaSequenceModel
from app.models.distillation import KnowledgeDistiller
from app.models.ensemble import MetaEnsemble
from app.models.markov import MarkovModel
from app.evolution.orchestrator import autonomous_evolution_cycle
from app.ingestion.stream import ingest_outcomes_batch
from app.database import SessionLocal
from app.schemas import ModelRuntimeStateRecord

class TestNeuralIntelligence(unittest.TestCase):

    def test_causal_dataset(self):
        seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10
        dataset = SequenceDataset(seq[:-1], seq[1:], context_length=16, input_size=10)
        self.assertGreater(len(dataset), 50)
        
        X, y = dataset[0]
        self.assertEqual(X.shape, (16, 10))
        self.assertEqual(y.shape, ())
        self.assertEqual(int(y), seq[17]) # target is element at index 17

    def test_causal_transformer(self):
        model = CausalTransformer(input_size=10, hidden_size=16, layers=1, heads=2, output_size=10)
        x = torch.randn(2, 20, 10)
        out = model(x)
        self.assertEqual(out.shape, (2, 10))
        
        # Test wrapper SequenceModel
        seq = [1, 2, 3, 4, 1, 2, 3, 4] * 10
        trans_wrapper = TransformerSequenceModel(input_size=10, hidden_size=16, heads=2, layers=1, context_length=16)
        trans_wrapper.fit(seq, epochs=3)
        probs = trans_wrapper.predict_proba([1, 2, 3])
        self.assertEqual(len(probs), 10)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

    def test_state_space_layer_and_s4(self):
        ssm_layer = StateSpaceLayer(input_size=10, hidden_size=16)
        x = torch.randn(2, 15, 10)
        out, state = ssm_layer(x)
        self.assertEqual(out.shape, (2, 15, 16))
        self.assertEqual(state.shape, (2, 16))
        
        # Test S4 Sequence Model wrapper
        seq = [0, 5, 0, 5, 0, 5, 0, 5] * 10
        s4 = S4SequenceModel(input_size=10, hidden_size=16, layers=1, context_length=16)
        s4.fit(seq, epochs=3)
        probs = s4.predict_proba([0, 5, 0])
        self.assertEqual(len(probs), 10)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

    def test_mamba_sequence_model(self):
        seq = [2, 4, 6, 8, 2, 4, 6, 8] * 10
        mamba = MambaSequenceModel(input_size=10, hidden_size=16, layers=1, context_length=16)
        mamba.fit(seq, epochs=3)
        probs = mamba.predict_proba([2, 4, 6])
        self.assertEqual(len(probs), 10)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

    def test_knowledge_distillation(self):
        seq = [1, 2, 1, 2, 1, 2, 1, 2] * 10
        teacher = MarkovModel(order=1).fit(seq)
        student = TransformerSequenceModel(input_size=10, hidden_size=16, heads=2, layers=1, context_length=8)
        
        distiller = KnowledgeDistiller(alpha=0.5, temperature=1.5, lr=1e-3)
        distilled_student = distiller.distill(teacher, student, seq, epochs=2)
        
        probs = distilled_student.predict_proba([1, 2, 1])
        self.assertEqual(len(probs), 10)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

    def test_model_runtime_state_persistence(self):
        with SessionLocal() as session:
            rec = ModelRuntimeStateRecord(
                model_version_id=999,
                last_sequence_no=50000,
                state_path="/tmp/test_state.pt",
                feature_state={"entropy": 3.32},
                runtime_metadata={"epochs": 10}
            )
            session.merge(rec)
            session.commit()
            
            saved = session.query(ModelRuntimeStateRecord).filter(ModelRuntimeStateRecord.model_version_id == 999).first()
            self.assertIsNotNone(saved)
            self.assertEqual(saved.last_sequence_no, 50000)

    def test_full_neural_population_evolution(self):
        synthetic_batch = []
        for i in range(120):
            synthetic_batch.append({
                "sequence_no": 990000 + i,
                "digit": (i % 4) * 2
            })
        ingest_outcomes_batch(synthetic_batch)
        
        report = autonomous_evolution_cycle(last_seq_cursor=989999)
        self.assertEqual(report["status"], "SUCCESS")
        self.assertIn("action", report)

if __name__ == "__main__":
    unittest.main()
