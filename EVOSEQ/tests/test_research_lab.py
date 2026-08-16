import sys
import os
import unittest
import numpy as np

# Add EVOSEQ root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.research.null_models import (
    generate_iid,
    estimate_transition,
    generate_markov,
    block_shuffle,
    marginal_preserving_shuffle,
    NullModelType,
    SurrogateHierarchy
)
from app.research.statistics import (
    block_bootstrap,
    compute_bootstrap_ci,
    empirical_p_value,
    compare_model_to_null,
    bonferroni_correction,
    benjamini_hochberg_fdr
)
from app.research.information import (
    calculate_entropy_rate_profile,
    information_gain_curve,
    lz_null_z_score
)
from app.research.audit import (
    evaluate_model_robustness,
    generate_ascii_audit_hud
)
from app.models.markov import MarkovModel
from app.database import SessionLocal
from app.schemas import ResearchExperimentRecord

class TestResearchLaboratory(unittest.TestCase):

    def test_null_models(self):
        # 1. IID null generator
        probs = [0.1] * 10
        iid_seq = generate_iid(probs, length=100, seed=42)
        self.assertEqual(len(iid_seq), 100)
        self.assertTrue((iid_seq >= 0).all() and (iid_seq < 10).all())
        
        # 2. Markov null generator
        seq = [0, 1, 0, 1, 0, 1] * 10
        trans = estimate_transition(seq, n_symbols=10)
        self.assertEqual(trans.shape, (10, 10))
        mkv_syn = generate_markov(trans, length=50, seed=42)
        self.assertEqual(len(mkv_syn), 50)
        
        # 3. Block shuffle
        blk_shuf = block_shuffle(seq, block_size=6, seed=42)
        self.assertEqual(len(blk_shuf), len(seq))
        
        # 4. Surrogate hierarchy
        surr = SurrogateHierarchy.generate_surrogate(seq, NullModelType.BLOCK_SHUFFLE, block_size=4, seed=42)
        self.assertEqual(len(surr), len(seq))

    def test_statistics_and_bootstrap(self):
        seq = [1, 2, 3, 4, 5] * 20
        boot_samples = block_bootstrap(seq, block_size=10, samples=20, seed=42)
        self.assertEqual(len(boot_samples), 20)
        self.assertEqual(len(boot_samples[0]), len(seq))
        
        med, low, high = compute_bootstrap_ci([0.50, 0.52, 0.48, 0.55, 0.51])
        self.assertGreater(high, low)
        
        # Empirical p-value
        p_val = empirical_p_value(observed=0.55, null_scores=[0.50, 0.51, 0.52, 0.53, 0.54])
        self.assertLess(p_val, 0.3)
        
        # Multiple testing corrections
        p_vals = [0.001, 0.02, 0.04, 0.06, 0.80]
        bonf_sig, adj_alpha = bonferroni_correction(p_vals, alpha=0.05)
        self.assertTrue(bonf_sig[0])
        self.assertFalse(bonf_sig[4])
        
        bh_sig = benjamini_hochberg_fdr(p_vals, q=0.05)
        self.assertTrue(bh_sig[0])

    def test_information_and_complexity(self):
        # Deterministic sequence should have decaying conditional entropy
        seq = [1, 2, 3, 1, 2, 3] * 20
        profile = calculate_entropy_rate_profile(seq, max_order=3)
        self.assertIn("H_0_marginal", profile)
        self.assertIn("H_1_conditional", profile)
        
        ig_curve = information_gain_curve(seq, max_order=3)
        self.assertIn("IG_1", ig_curve)
        self.assertGreater(ig_curve["IG_1"], 0.0)
        
        # LZ complexity Z-score
        lz_res = lz_null_z_score(seq, repetitions=10, seed=42)
        self.assertIn("z_score", lz_res)
        self.assertIn("algorithmic_compression_ratio", lz_res)

    def test_model_vs_null_comparison(self):
        seq = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4] * 10
        model = MarkovModel(order=1)
        res = compare_model_to_null(
            model,
            seq,
            null_generator_fn=lambda l, s: np.random.default_rng(s).integers(0, 10, size=l),
            repetitions=10,
            initial_train_size=30
        )
        self.assertIn("observed_score", res)
        self.assertIn("null_mean", res)
        self.assertIn("p_value", res)

    def test_robustness_audit_and_reporting(self):
        seq = [1, 2, 1, 2, 1, 2, 1, 2] * 10
        model = MarkovModel(order=1)
        rob = evaluate_model_robustness(model, seq, initial_train_size=20)
        self.assertIn("robustness_score", rob)
        self.assertIn("status", rob)
        
        hud = generate_ascii_audit_hud({
            "generation": 5,
            "observations": 80,
            "champion": "markov-v1",
            "recent_performance": 0.52,
            "historical_performance": 0.50,
            "delta_vs_null": 0.02,
            "robustness": "HIGH"
        })
        self.assertIn("EVOSEQ RESEARCH AUDIT", hud)

    def test_research_experiment_persistence(self):
        with SessionLocal() as session:
            rec = ResearchExperimentRecord(
                experiment_type="NULL_HYPOTHESIS_TEST",
                model_version_id=1,
                null_model="iid_marginal",
                sample_size=1000,
                observed_score=0.54,
                null_mean=0.50,
                null_std=0.01,
                p_value=0.001,
                correction_method="bonferroni"
            )
            session.add(rec)
            session.commit()
            
            saved = session.query(ResearchExperimentRecord).filter(ResearchExperimentRecord.experiment_type == "NULL_HYPOTHESIS_TEST").first()
            self.assertIsNotNone(saved)
            self.assertEqual(saved.p_value, 0.001)

if __name__ == "__main__":
    unittest.main()
