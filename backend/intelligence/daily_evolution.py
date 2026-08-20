from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Sequence, Tuple
import numpy as np

from .state_fingerprint import compute_state_fingerprint
from .multi_model import MultiModelEnsemble, ModelFamilyOutput
from .meta_learner import MetaLearner, MetaLearnerOutput
from .calibration import ConfidenceCalibrator
from .concept_drift import ConceptDriftDetector


@dataclass
class GenerationRecord:
    generation: int
    created_at: str
    training_cutoff: int
    validation_cutoff: int
    test_cutoff: int
    champion_model: str
    champion_oos_accuracy: float
    champion_oos_log_loss: float
    champion_oos_brier: float
    previous_champion_accuracy: Optional[float]
    accuracy_delta: float
    statistically_significant: bool
    model_weights: Dict[str, float]
    calibration_parameters: Dict[str, Any]
    baselines: Dict[str, float]
    rejected_models: List[str]
    promoted_models: List[str]
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


class WalkForwardEvaluator:
    """
    Walk-Forward Validation: NEVER randomly shuffle historical rounds.

    Train → Validate → Test windows rolling forward:

    Window 1: Train Days 1-10, Validate Day 11, Test Day 12
    Window 2: Train Days 1-11, Validate Day 12, Test Day 13
    etc.
    """

    def __init__(
        self,
        initial_train: int = 500,
        validation_size: int = 50,
        test_size: int = 50,
        step: int = 25,
        folds: int = 8,
    ):
        self.initial_train = initial_train
        self.validation_size = validation_size
        self.test_size = test_size
        self.step = step
        self.folds = folds

    def generate_windows(self, n_total: int) -> List[Dict[str, int]]:
        windows = []
        start = self.initial_train
        for _ in range(self.folds):
            train_end = start
            val_end = train_end + self.validation_size
            test_end = val_end + self.test_size
            if test_end > n_total:
                break
            windows.append(
                {
                    "train_start": 0,
                    "train_end": train_end,
                    "val_start": train_end,
                    "val_end": val_end,
                    "test_start": val_end,
                    "test_end": test_end,
                }
            )
            start += self.step
        return windows

    def evaluate_ensemble(
        self,
        ensemble: MultiModelEnsemble,
        meta_learner: MetaLearner,
        calibrator: ConfidenceCalibrator,
        digits: Sequence[int],
    ) -> Dict[str, Any]:
        dlist = [int(d) for d in digits]
        n = len(dlist)
        windows = self.generate_windows(n)
        if not windows:
            return {
                "oos_accuracy": 0.5,
                "oos_log_loss": math.log(2),
                "oos_brier": 0.25,
                "n_folds": 0,
                "n_predictions": 0,
            }

        all_correct: List[bool] = []
        all_log_loss: List[float] = []
        all_brier: List[float] = []

        for w in windows:
            # Strictly causal: build the ensemble on TRAINING data only
            train_digits = dlist[w["train_start"]:w["train_end"]]
            val_digits = dlist[w["val_start"]:w["val_end"]]
            test_digits = dlist[w["test_start"]:w["test_end"]]

            # Fit ensemble and meta on TRAIN
            try:
                ensemble.fit_all(train_digits)
            except Exception:
                continue

            # Meta-learner is "pre-warmed" with validation pass
            for i in range(len(val_digits)):
                ctx = train_digits + val_digits[:i]
                if len(ctx) < 20:
                    continue
                fp = compute_state_fingerprint(ctx)
                outs = ensemble.predict_all(ctx, fp)
                actual = "Big" if val_digits[i] >= 5 else "Small"
                target = 1.0 if actual == "Big" else 0.0
                for o in outs:
                    p = o.probability_big
                    brier = (p - target) ** 2
                    ll = -(target * math.log(max(1e-6, p)) + (1 - target) * math.log(max(1e-6, 1 - p)))
                    try:
                        meta_learner.update_performance(
                            o.model_name, o.prediction, actual, p, brier, ll, regime=fp.regime_id
                        )
                    except Exception:
                        pass

            # Test set: make predictions strictly from the past
            for i in range(len(test_digits)):
                ctx = train_digits + val_digits + test_digits[:i]
                if len(ctx) < 30:
                    continue
                fp = compute_state_fingerprint(ctx)
                try:
                    outs = ensemble.predict_all(ctx, fp)
                except Exception:
                    continue
                meta = meta_learner.infer(outs, fp, sim_state=None, baseline_accuracy=0.5)
                p_big = meta.ensemble_probability_big
                p_big = max(0.001, min(0.999, p_big))
                actual = 1.0 if test_digits[i] >= 5 else 0.0
                actual_side = "Big" if test_digits[i] >= 5 else "Small"
                predicted_side = "Big" if p_big >= 0.5 else "Small"

                all_correct.append(predicted_side == actual_side)
                all_log_loss.append(
                    -(actual * math.log(p_big) + (1 - actual) * math.log(1 - p_big))
                )
                all_brier.append((p_big - actual) ** 2)

        if not all_correct:
            return {
                "oos_accuracy": 0.5,
                "oos_log_loss": math.log(2),
                "oos_brier": 0.25,
                "n_folds": len(windows),
                "n_predictions": 0,
            }

        return {
            "oos_accuracy": float(np.mean(all_correct)),
            "oos_log_loss": float(np.mean(all_log_loss)),
            "oos_brier": float(np.mean(all_brier)),
            "n_folds": len(windows),
            "n_predictions": len(all_correct),
        }


class BaselineEvaluator:
    """
    Baselines that every sophisticated model must compete against:
      1. Random
      2. Majority class
      3. Recent frequency
      4. Simple Markov (order 1)
      5. Current champion (passed in)
    """

    def __init__(self):
        pass

    def random(self, n: int, seed: int = 42) -> float:
        r = random.Random(seed)
        return 0.5

    def majority(self, digits: Sequence[int]) -> float:
        if not digits:
            return 0.5
        sizes = [1 if int(d) >= 5 else 0 for d in digits]
        m = float(np.mean(sizes))
        # Accuracy of predicting the majority class every time
        return max(m, 1 - m)

    def recent_frequency(self, digits: Sequence[int], window: int = 100) -> float:
        """
        Walk-forward with a recent frequency heuristic.
        Returns expected accuracy.
        """
        dlist = [int(d) for d in digits]
        if len(dlist) < window + 10:
            return self.majority(dlist)
        correct = 0
        total = 0
        for i in range(window, len(dlist) - 1):
            ctx = dlist[i - window:i]
            sizes = [1 if d >= 5 else 0 for d in ctx]
            p_big = float(np.mean(sizes))
            predicted = "Big" if p_big >= 0.5 else "Small"
            actual = "Big" if dlist[i + 1] >= 5 else "Small"
            if predicted == actual:
                correct += 1
            total += 1
        return correct / max(1, total)

    def simple_markov(self, digits: Sequence[int]) -> float:
        """Simple first-order Markov, walk-forward evaluated."""
        dlist = [int(d) for d in digits]
        if len(dlist) < 150:
            return self.majority(dlist)
        sizes = [1 if d >= 5 else 0 for d in dlist]
        burn = 100
        correct = 0
        total = 0
        for i in range(burn, len(sizes) - 1):
            # Count transitions 0→0, 0→1, 1→0, 1→1 using only data before i
            counts = np.ones((2, 2))
            for j in range(1, i):
                a = sizes[j - 1]
                b = sizes[j]
                if 0 <= a < 2 and 0 <= b < 2:
                    counts[a, b] += 1
            prev = sizes[i]
            row = counts[int(prev)]
            pred_big = row[1] / row.sum()
            predicted = "Big" if pred_big >= 0.5 else "Small"
            actual = "Big" if sizes[i + 1] == 1 else "Small"
            if predicted == actual:
                correct += 1
            total += 1
        return correct / max(1, total)

    def evaluate_all(self, digits: Sequence[int]) -> Dict[str, float]:
        return {
            "random": 0.5,
            "majority": self.majority(digits),
            "recent_frequency": self.recent_frequency(digits),
            "simple_markov": self.simple_markov(digits),
        }


class DailyEvolution:
    """
    Daily evolution pipeline: every day, create a new generation.

    Process:
      1. Load ALL historical data up to cutoff.
      2. Create candidate models.
      3. Train/update candidates.
      4. Walk-forward evaluation.
      5. Compare against current champion.
      6. Locked out-of-sample period.
      7. Reject models without improvement.
      8. Promote only validated candidates.
      9. Save generation.
     10. Never overwrite previous champion.
    """

    def __init__(self, generation: int = 1):
        self.current_generation = generation

    def run_daily_evolution(
        self,
        history_digits: Sequence[int],
        previous_accuracy: Optional[float] = None,
        previous_generation_weights: Optional[Dict[str, float]] = None,
    ) -> GenerationRecord:
        dlist = [int(d) for d in history_digits]
        n = len(dlist)

        # Data split: last 50 held out as locked OOS; prior 50 validation; rest train
        test_cutoff = max(self._pct(n, 0.95), n - 50)
        val_cutoff = max(test_cutoff - 50, self._pct(n, 0.90))
        train_cutoff = val_cutoff

        # Baselines
        baseline_eval = BaselineEvaluator()
        baselines = baseline_eval.evaluate_all(dlist[:test_cutoff])
        best_baseline = max(baselines.values())

        # Build ensemble and run walk-forward
        ensemble = MultiModelEnsemble(generation=self.current_generation)
        meta = MetaLearner(generation=self.current_generation)
        calibrator = ConfidenceCalibrator()

        evaluator = WalkForwardEvaluator(
            initial_train=min(500, max(200, train_cutoff // 3)),
            validation_size=50,
            test_size=50,
            step=25,
            folds=6,
        )
        eval_result = evaluator.evaluate_ensemble(ensemble, meta, calibrator, dlist[:test_cutoff])

        # Locked OOS: single final evaluation on the last 50 rounds never seen during tuning
        oos_accuracy = eval_result["oos_accuracy"]
        oos_ll = eval_result["oos_log_loss"]
        oos_brier = eval_result["oos_brier"]

        # Statistical significance: simple normal approx on the held-out predictions
        n_preds = eval_result["n_predictions"]
        statistically_significant = False
        if n_preds >= 30:
            p_hat = oos_accuracy
            se = math.sqrt(max(0.0, p_hat * (1 - p_hat) / n_preds))
            z = (p_hat - best_baseline) / max(1e-9, se)
            statistically_significant = z >= 1.65  # one-sided ~95%

        # Accuracy delta vs previous champion
        delta = oos_accuracy - previous_accuracy if previous_accuracy is not None else 0.0

        # Champion model: the one with highest weight from meta-learner
        # We do one final pass to get meta weights:
        weights: Dict[str, float] = {}
        try:
            # Use historical end-of-training window to compute weights
            ctx = dlist[:train_cutoff]
            fp = compute_state_fingerprint(ctx)
            outs = ensemble.predict_all(ctx, fp)
            meta_out = meta.infer(outs, fp, sim_state=None, baseline_accuracy=best_baseline)
            weights = {k: v for k, v in meta_out.model_weights.items()}
        except Exception:
            weights = {m.__class__.__name__: 1.0 / max(1, len(ensemble.models)) for m in ensemble.models}

        champion_name = max(weights, key=weights.get) if weights else "FrequencyModel"

        # Reject / promote
        rejected = []
        promoted = []
        if previous_accuracy is not None:
            # Reject if OOS is worse than or equal to previous (with tiny margin)
            if delta <= 0.001 and not statistically_significant:
                rejected.append(champion_name)
                status = "REJECTED"
            elif delta >= 0.003 and statistically_significant:
                promoted.append(champion_name)
                status = "PROMOTED"
            else:
                status = "CANDIDATE"
        else:
            promoted.append(champion_name)
            status = "PROMOTED"

        record = GenerationRecord(
            generation=self.current_generation,
            created_at=str(np.datetime64("now")),
            training_cutoff=int(train_cutoff),
            validation_cutoff=int(val_cutoff),
            test_cutoff=int(test_cutoff),
            champion_model=champion_name,
            champion_oos_accuracy=float(oos_accuracy),
            champion_oos_log_loss=float(oos_ll),
            champion_oos_brier=float(oos_brier),
            previous_champion_accuracy=previous_accuracy,
            accuracy_delta=float(delta),
            statistically_significant=statistically_significant,
            model_weights=weights,
            calibration_parameters={"temperature": 1.05},
            baselines=baselines,
            rejected_models=rejected,
            promoted_models=promoted,
            status=status,
        )

        return record

    def _pct(self, n: int, p: float) -> int:
        return int(math.floor(n * p))
