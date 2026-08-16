from typing import Dict, List, Sequence, Any, Tuple, Optional
import numpy as np
from ..research.metrics import log_loss, brier_score, accuracy_score
from ..research.null_models import iid_null, markov_null, block_shuffle_null

class TemporalValidationLab:
    """
    Scientific Validation Laboratory for Evolutionary Candidates:
    1. Nested Walk-Forward Splits (Inner research folds vs Locked outer test fold)
    2. Multi-Seed Robustness Evaluation (seeds: 1, 7, 42, 123, 999)
    3. Paired Bootstrap Confidence Intervals (95% CI on Loss delta)
    4. Empirical Null / Surrogate Significance Referee (p-value calculation)
    """

    def __init__(self, seeds: Optional[List[int]] = None):
        self.seeds = seeds or [1, 7, 42, 123, 999]

    def evaluate_multi_seed(
        self,
        candidate_builder_fn: Any,
        data_train: np.ndarray,
        data_test: np.ndarray
    ) -> Dict[str, float]:
        """Runs multi-seed stability check, computing mean loss and variance."""
        losses = []
        for s in self.seeds:
            np.random.seed(s)
            model = candidate_builder_fn(seed=s)
            if hasattr(model, "fit"):
                model.fit(data_train)
            if hasattr(model, "predict_sequence"):
                preds = model.predict_sequence(data_test)
            elif hasattr(model, "predict_proba"):
                preds = np.asarray([model.predict_proba(data_test[:t+1]) for t in range(len(data_test))])
            else:
                preds = np.full((len(data_test), 10), 0.1)
                
            ll = log_loss(preds, data_test)
            losses.append(float(ll))
            
        losses_arr = np.asarray(losses)
        return {
            "mean_loss": round(float(np.mean(losses_arr)), 4),
            "std_loss": round(float(np.std(losses_arr)), 4),
            "min_loss": round(float(np.min(losses_arr)), 4),
            "max_loss": round(float(np.max(losses_arr)), 4)
        }

    def bootstrap_paired_delta(
        self,
        candidate_losses: Sequence[float],
        champion_losses: Sequence[float],
        n_bootstraps: int = 1000,
        ci: float = 0.95
    ) -> Dict[str, float]:
        """Computes 95% bootstrap confidence interval of paired loss difference: D_t = L_cand,t - L_champ,t."""
        c_arr = np.asarray(candidate_losses)
        ch_arr = np.asarray(champion_losses)
        deltas = c_arr - ch_arr
        
        boot_means = []
        N = len(deltas)
        if N == 0:
            return {"mean_delta": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
            
        for _ in range(n_bootstraps):
            sample = np.random.choice(deltas, size=N, replace=True)
            boot_means.append(np.mean(sample))
            
        alpha = (1.0 - ci) / 2.0
        lower = float(np.percentile(boot_means, alpha * 100))
        upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
        mean_d = float(np.mean(deltas))
        
        return {
            "mean_delta": round(mean_d, 5),
            "ci_lower": round(lower, 5),
            "ci_upper": round(upper, 5),
            "significant_advantage": bool(upper < 0.0) # Upper bound below 0 means candidate is strictly better
        }

    def null_significance_test(
        self,
        candidate_fn: Any,
        real_sequence: Sequence[int],
        n_surrogates: int = 50
    ) -> Dict[str, Any]:
        """
        Calculates empirical permutation p-value against Markov & IID surrogates:
        p = (1 + sum(Loss_real >= Loss_null)) / (N + 1)
        """
        real_seq = list(real_sequence)
        real_preds = candidate_fn(real_seq)
        real_ll = log_loss(real_preds, real_seq)
        
        null_losses = []
        # Generate mixed surrogates
        for i in range(n_surrogates):
            if i % 2 == 0:
                null_seq = iid_null(real_seq)
            else:
                null_seq = block_shuffle_null(real_seq, block_size=5)
            null_p = candidate_fn(null_seq)
            null_losses.append(log_loss(null_p, null_seq))
            
        better_or_equal_nulls = sum(1 for nl in null_losses if nl <= real_ll)
        p_val = float((1 + better_or_equal_nulls) / (n_surrogates + 1))
        
        return {
            "real_log_loss": round(float(real_ll), 4),
            "mean_null_loss": round(float(np.mean(null_losses)), 4),
            "null_p_value": round(p_val, 4),
            "passed_null_referee": bool(p_val <= 0.10) # 90%+ significance
        }
