from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Sequence, Tuple
import numpy as np

from .state_fingerprint import StateFingerprint
from .similar_state import SimilarStateMemory, SimilarStateResult


@dataclass
class ModelFamilyOutput:
    model_name: str
    family: str
    version: str

    prediction: str
    probability_big: float
    probability_small: float
    confidence: float
    sample_size: int
    regime: str
    generation: int = 1

    probability_vector: List[float] = field(default_factory=lambda: [0.1] * 10)
    target_digit: int = 5
    hedge_digit: int = 4

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Abstract Base
# ---------------------------------------------------------------------------

class BasePredictionModel(ABC):
    family: str = "base"
    name: str = "BaseModel"
    version: str = "v1.0"

    def __init__(self, generation: int = 1):
        self.generation = generation
        self._total_updates = 0
        self._trained = False

    @abstractmethod
    def fit(self, digits: Sequence[int]) -> None: ...

    @abstractmethod
    def partial_fit(self, digits: Sequence[int]) -> None: ...

    @abstractmethod
    def predict(self, digits: Sequence[int], fp: Optional[StateFingerprint] = None) -> ModelFamilyOutput: ...


# =====================================================================
# A. STATISTICAL FAMILY
# =====================================================================

class FrequencyModel(BasePredictionModel):
    family = "statistical"
    name = "Frequency"
    version = "v1"

    def __init__(self, window: int = 200, generation: int = 1):
        super().__init__(generation)
        self.window = window
        self._counts = np.ones(10, dtype=np.float64)

    def fit(self, digits: Sequence[int]) -> None:
        self._counts = np.ones(10, dtype=np.float64)
        for d in digits[-min(self.window * 5, len(digits)):]:
            if 0 <= int(d) < 10:
                self._counts[int(d)] += 1
        self._trained = True
        self._total_updates += 1

    def partial_fit(self, digits: Sequence[int]) -> None:
        for d in digits[-self.window:]:
            if 0 <= int(d) < 10:
                self._counts[int(d)] += 0.5
        self._counts *= 0.998
        self._total_updates += 1

    def predict(self, digits: Sequence[int], fp: Optional[StateFingerprint] = None) -> ModelFamilyOutput:
        if not self._trained:
            self.fit(digits)
        # Apply recency weighting via window override
        recent = digits[-min(self.window, len(digits)):]
        recent_counts = np.ones(10, dtype=np.float64) * 0.1
        for d in recent:
            if 0 <= int(d) < 10:
                recent_counts[int(d)] += 1.0
        blended = 0.6 * (self._counts / self._counts.sum()) + 0.4 * (recent_counts / recent_counts.sum())
        p_vec = blended / blended.sum()

        p_big = float(p_vec[5:].sum())
        p_small = float(p_vec[:5].sum())
        total = p_big + p_small
        p_big /= total
        p_small /= total

        best = int(np.argmax(p_vec))
        sorted_idx = p_vec.argsort()[::-1]
        hedge = int(sorted_idx[1]) if len(sorted_idx) > 1 else 0

        confidence = max(p_big, p_small)
        regime = fp.regime_id if fp is not None else "UNKNOWN"

        return ModelFamilyOutput(
            model_name=self.name,
            family=self.family,
            version=self.version,
            prediction="Big" if p_big >= 0.5 else "Small",
            probability_big=p_big,
            probability_small=p_small,
            confidence=confidence,
            sample_size=len(recent),
            regime=regime,
            generation=self.generation,
            probability_vector=p_vec.tolist(),
            target_digit=best,
            hedge_digit=hedge,
        )


class BayesianModel(BasePredictionModel):
    family = "statistical"
    name = "Bayesian"
    version = "v1"

    def __init__(self, prior_strength: float = 5.0, generation: int = 1):
        super().__init__(generation)
        self.prior_strength = prior_strength
        self.alpha = np.ones(10, dtype=np.float64) * prior_strength / 10

    def fit(self, digits: Sequence[int]) -> None:
        self.alpha = np.ones(10, dtype=np.float64) * self.prior_strength / 10
        for d in digits:
            if 0 <= int(d) < 10:
                self.alpha[int(d)] += 1.0
        self._trained = True

    def partial_fit(self, digits: Sequence[int]) -> None:
        for d in digits[-50:]:
            if 0 <= int(d) < 10:
                self.alpha[int(d)] += 0.5

    def predict(self, digits: Sequence[int], fp: Optional[StateFingerprint] = None) -> ModelFamilyOutput:
        if not self._trained:
            self.fit(digits)
        alpha_sum = self.alpha.sum()
        p_vec = self.alpha / alpha_sum
        # Laplace smoothing already handled by prior
        p_vec = np.clip(p_vec, 1e-4, None)
        p_vec = p_vec / p_vec.sum()

        p_big = float(p_vec[5:].sum())
        p_small = float(p_vec[:5].sum())
        total = p_big + p_small
        p_big /= total
        p_small /= total

        best = int(np.argmax(p_vec))
        sorted_idx = p_vec.argsort()[::-1]
        hedge = int(sorted_idx[1]) if len(sorted_idx) > 1 else 0

        regime = fp.regime_id if fp is not None else "UNKNOWN"
        confidence = max(p_big, p_small)

        return ModelFamilyOutput(
            model_name=self.name,
            family=self.family,
            version=self.version,
            prediction="Big" if p_big >= 0.5 else "Small",
            probability_big=p_big,
            probability_small=p_small,
            confidence=confidence,
            sample_size=int(alpha_sum),
            regime=regime,
            generation=self.generation,
            probability_vector=p_vec.tolist(),
            target_digit=best,
            hedge_digit=hedge,
        )


class MarkovModel(BasePredictionModel):
    family = "statistical"
    name = "VariableMarkov"
    version = "v1"

    def __init__(self, order: int = 3, generation: int = 1):
        super().__init__(generation)
        self.order = order
        self.transitions: Dict[Tuple[int, ...], np.ndarray] = {}
        self.smoothing = 0.05

    def _get_key(self, seq: Sequence[int]) -> Tuple[int, ...]:
        return tuple(int(x) for x in seq[-self.order:])

    def fit(self, digits: Sequence[int]) -> None:
        self.transitions = {}
        dlist = [int(d) for d in digits]
        for i in range(self.order, len(dlist)):
            ctx = tuple(dlist[i - self.order:i])
            nxt = dlist[i]
            if ctx not in self.transitions:
                self.transitions[ctx] = np.ones(10, dtype=np.float64) * self.smoothing
            if 0 <= nxt < 10:
                self.transitions[ctx][nxt] += 1.0
        self._trained = True

    def partial_fit(self, digits: Sequence[int]) -> None:
        dlist = [int(d) for d in digits[-max(self.order + 1, 20):]]
        for i in range(self.order, len(dlist)):
            ctx = tuple(dlist[i - self.order:i])
            nxt = dlist[i]
            if ctx not in self.transitions:
                self.transitions[ctx] = np.ones(10, dtype=np.float64) * self.smoothing
            if 0 <= nxt < 10:
                self.transitions[ctx][nxt] += 0.5

    def predict(self, digits: Sequence[int], fp: Optional[StateFingerprint] = None) -> ModelFamilyOutput:
        if not self._trained:
            self.fit(digits)

        dlist = [int(d) for d in digits]
        if len(dlist) < self.order:
            p_vec = np.ones(10) / 10
            sample_size = 0
        else:
            key = self._get_key(dlist)
            # Try shorter context if exact not found
            found = None
            if key in self.transitions:
                found = self.transitions[key]
            else:
                for shorten in range(1, self.order):
                    sub_key = key[shorten:]
                    if sub_key in self.transitions:
                        found = self.transitions[sub_key]
                        break
            if found is None:
                found = np.ones(10, dtype=np.float64) * self.smoothing
            p_vec = np.array(found, dtype=np.float64)
            sample_size = int(p_vec.sum())
            p_vec = p_vec / p_vec.sum()

        p_big = float(p_vec[5:].sum())
        p_small = float(p_vec[:5].sum())
        total = p_big + p_small
        if total > 0:
            p_big /= total
            p_small /= total

        best = int(np.argmax(p_vec))
        sorted_idx = p_vec.argsort()[::-1]
        hedge = int(sorted_idx[1]) if len(sorted_idx) > 1 else 0

        regime = fp.regime_id if fp is not None else "UNKNOWN"
        confidence = max(p_big, p_small)

        return ModelFamilyOutput(
            model_name=f"{self.name}_ord{self.order}",
            family=self.family,
            version=self.version,
            prediction="Big" if p_big >= 0.5 else "Small",
            probability_big=p_big,
            probability_small=p_small,
            confidence=confidence,
            sample_size=sample_size,
            regime=regime,
            generation=self.generation,
            probability_vector=p_vec.tolist(),
            target_digit=best,
            hedge_digit=hedge,
        )


# =====================================================================
# B. SEQUENCE FAMILY
# =====================================================================

class NgramModel(BasePredictionModel):
    family = "sequence"
    name = "Ngram"
    version = "v1"

    def __init__(self, n: int = 5, generation: int = 1):
        super().__init__(generation)
        self.n = n
        self.counts: Dict[str, List[int]] = {}

    def _encode_size_ngram(self, digits: Sequence[int]) -> str:
        return "".join(["B" if int(d) >= 5 else "S" for d in digits])

    def fit(self, digits: Sequence[int]) -> None:
        self.counts = {}
        sizes = ["B" if int(d) >= 5 else "S" for d in digits]
        for i in range(self.n, len(sizes)):
            ctx = "".join(sizes[i - self.n:i])
            nxt = sizes[i]
            if ctx not in self.counts:
                self.counts[ctx] = [0, 0]
            self.counts[ctx][0 if nxt == "B" else 1] += 1
        self._trained = True

    def partial_fit(self, digits: Sequence[int]) -> None:
        sizes = ["B" if int(d) >= 5 else "S" for d in digits[-max(self.n + 1, 50):]]
        for i in range(self.n, len(sizes)):
            ctx = "".join(sizes[i - self.n:i])
            nxt = sizes[i]
            if ctx not in self.counts:
                self.counts[ctx] = [0, 0]
            self.counts[ctx][0 if nxt == "B" else 1] += 0.5

    def predict(self, digits: Sequence[int], fp: Optional[StateFingerprint] = None) -> ModelFamilyOutput:
        if not self._trained:
            self.fit(digits)

        sizes = ["B" if int(d) >= 5 else "S" for d in digits]
        ctx = "".join(sizes[-self.n:]) if len(sizes) >= self.n else "".join(sizes)
        found = None
        if ctx in self.counts:
            found = self.counts[ctx]
        else:
            # Try shorter prefixes
            for k in range(self.n - 1, 1, -1):
                sub = ctx[-k:]
                if sub in self.counts:
                    found = self.counts[sub]
                    break
        if found is None:
            found = [1, 1]

        a, b = found[0] + 0.5, found[1] + 0.5
        total = a + b
        p_big = a / total
        p_small = b / total

        # Digit-level: use simple frequency
        digit_counts = np.ones(10) * 0.1
        for d in digits[-200:]:
            if 0 <= int(d) < 10:
                digit_counts[int(d)] += 1
        digit_probs = digit_counts / digit_counts.sum()
        # Bias toward big/small
        if p_big >= p_small:
            digit_probs[5:] *= p_big / 0.5
        else:
            digit_probs[:5] *= p_small / 0.5
        digit_probs = digit_probs / digit_probs.sum()

        best = int(np.argmax(digit_probs))
        sorted_idx = digit_probs.argsort()[::-1]
        hedge = int(sorted_idx[1]) if len(sorted_idx) > 1 else 0

        regime = fp.regime_id if fp is not None else "UNKNOWN"
        sample_size = int(a + b)

        return ModelFamilyOutput(
            model_name=f"{self.name}_n{self.n}",
            family=self.family,
            version=self.version,
            prediction="Big" if p_big >= 0.5 else "Small",
            probability_big=p_big,
            probability_small=p_small,
            confidence=max(p_big, p_small),
            sample_size=sample_size,
            regime=regime,
            generation=self.generation,
            probability_vector=digit_probs.tolist(),
            target_digit=best,
            hedge_digit=hedge,
        )


class SequenceSimilarityModel(BasePredictionModel):
    family = "sequence"
    name = "SequenceSimilarity"
    version = "v1"

    def __init__(self, k: int = 10, top_k: int = 30, generation: int = 1):
        super().__init__(generation)
        self.k = k
        self.top_k = top_k
        self.history_sizes: List[List[int]] = []

    def fit(self, digits: Sequence[int]) -> None:
        sizes = [1 if int(d) >= 5 else 0 for d in digits]
        self.history_sizes = sizes
        self._trained = True

    def partial_fit(self, digits: Sequence[int]) -> None:
        for d in digits[-10:]:
            self.history_sizes.append(1 if int(d) >= 5 else 0)
        if len(self.history_sizes) > 50000:
            self.history_sizes = self.history_sizes[-50000:]

    def predict(self, digits: Sequence[int], fp: Optional[StateFingerprint] = None) -> ModelFamilyOutput:
        if not self._trained:
            self.fit(digits)

        current_sizes = [1 if int(d) >= 5 else 0 for d in digits[-self.k:]]
        if len(current_sizes) < self.k:
            current_sizes = current_sizes + [0] * (self.k - len(current_sizes))

        # Find matching positions in history
        matches = []
        hist = self.history_sizes
        if len(hist) <= self.k + 1:
            p_big, p_small = 0.5, 0.5
            sample_size = 0
        else:
            for i in range(len(hist) - self.k - 1):
                window = hist[i:i + self.k]
                # Hamming similarity
                matches_count = sum(1 for a, b in zip(window, current_sizes) if a == b)
                sim = matches_count / self.k
                if sim >= 0.7:
                    next_val = hist[i + self.k]
                    matches.append((sim, next_val))
            matches.sort(key=lambda x: x[0], reverse=True)
            matches = matches[:self.top_k]
            sample_size = len(matches)
            if sample_size == 0:
                p_big, p_small = 0.5, 0.5
            else:
                total_weight = sum(s for s, _ in matches)
                if total_weight == 0:
                    p_big, p_small = 0.5, 0.5
                else:
                    big_w = sum(s for s, v in matches if v == 1)
                    small_w = sum(s for s, v in matches if v == 0)
                    p_big = (big_w + 0.5) / (total_weight + 1.0)
                    p_small = (small_w + 0.5) / (total_weight + 1.0)

        digit_counts = np.ones(10) * 0.1
        for d in digits[-200:]:
            if 0 <= int(d) < 10:
                digit_counts[int(d)] += 1
        digit_probs = digit_counts / digit_counts.sum()
        if p_big >= p_small:
            digit_probs[5:] *= max(1.01, p_big / 0.5)
        else:
            digit_probs[:5] *= max(1.01, p_small / 0.5)
        digit_probs = digit_probs / digit_probs.sum()

        best = int(np.argmax(digit_probs))
        sorted_idx = digit_probs.argsort()[::-1]
        hedge = int(sorted_idx[1]) if len(sorted_idx) > 1 else 0

        regime = fp.regime_id if fp is not None else "UNKNOWN"

        return ModelFamilyOutput(
            model_name=self.name,
            family=self.family,
            version=self.version,
            prediction="Big" if p_big >= 0.5 else "Small",
            probability_big=p_big,
            probability_small=p_small,
            confidence=max(p_big, p_small),
            sample_size=sample_size,
            regime=regime,
            generation=self.generation,
            probability_vector=digit_probs.tolist(),
            target_digit=best,
            hedge_digit=hedge,
        )


# =====================================================================
# C. TEMPORAL / RECENCY FAMILY
# =====================================================================

class RecencyWeightedModel(BasePredictionModel):
    family = "temporal"
    name = "RecencyWeighted"
    version = "v1"

    def __init__(self, decay: float = 0.985, generation: int = 1):
        super().__init__(generation)
        self.decay = decay
        self.weighted_counts = np.ones(10, dtype=np.float64) * 0.1
        self.total_weight = 1.0

    def fit(self, digits: Sequence[int]) -> None:
        self.weighted_counts = np.ones(10, dtype=np.float64) * 0.1
        self.total_weight = 1.0
        w = 1.0
        for d in reversed(digits[-5000:]):
            if 0 <= int(d) < 10:
                self.weighted_counts[int(d)] += w
                self.total_weight += w
            w *= self.decay
        self._trained = True

    def partial_fit(self, digits: Sequence[int]) -> None:
        for d in digits[-10:]:
            if 0 <= int(d) < 10:
                self.weighted_counts[int(d)] += 1.0
                self.total_weight += 1.0
        self.weighted_counts *= self.decay
        self.total_weight *= self.decay

    def predict(self, digits: Sequence[int], fp: Optional[StateFingerprint] = None) -> ModelFamilyOutput:
        if not self._trained:
            self.fit(digits)
        p_vec = self.weighted_counts / self.weighted_counts.sum()
        p_vec = np.clip(p_vec, 1e-5, None)
        p_vec = p_vec / p_vec.sum()

        p_big = float(p_vec[5:].sum())
        p_small = float(p_vec[:5].sum())
        total = p_big + p_small
        p_big /= total
        p_small /= total

        best = int(np.argmax(p_vec))
        sorted_idx = p_vec.argsort()[::-1]
        hedge = int(sorted_idx[1]) if len(sorted_idx) > 1 else 0

        regime = fp.regime_id if fp is not None else "UNKNOWN"

        return ModelFamilyOutput(
            model_name=self.name,
            family=self.family,
            version=self.version,
            prediction="Big" if p_big >= 0.5 else "Small",
            probability_big=p_big,
            probability_small=p_small,
            confidence=max(p_big, p_small),
            sample_size=int(self.total_weight),
            regime=regime,
            generation=self.generation,
            probability_vector=p_vec.tolist(),
            target_digit=best,
            hedge_digit=hedge,
        )


# =====================================================================
# D. MEMORY FAMILY (Similar State backed)
# =====================================================================

class SimilarStateModel(BasePredictionModel):
    family = "memory"
    name = "SimilarState"
    version = "v1"

    def __init__(self, sim_threshold: float = 0.80, generation: int = 1):
        super().__init__(generation)
        self.memory = SimilarStateMemory(
            similarity_threshold=sim_threshold,
            min_sample_size=30,
        )
        self.last_result: Optional[SimilarStateResult] = None

    def fit(self, digits: Sequence[int]) -> None:
        self.memory = SimilarStateMemory(similarity_threshold=0.80, min_sample_size=30)
        self.memory.bulk_remember_from_history(digits, start_offset=30, stride=2)
        self._trained = True

    def partial_fit(self, digits: Sequence[int]) -> None:
        # Incremental: remember only recent additions
        # (bulk remember handles startup; partial_fit is minimal here)
        if len(digits) >= 40:
            tail = list(digits[-40:])
            ctx = tail[:-3]
            if len(ctx) >= 30:
                try:
                    from .state_fingerprint import compute_state_fingerprint
                    fp = compute_state_fingerprint(ctx)
                    next_d = tail[-3]
                    next_s = 1 if next_d >= 5 else 0
                    self.memory.remember(fp, next_s, next_d, (False, False, False))
                except Exception:
                    pass

    def predict(self, digits: Sequence[int], fp: Optional[StateFingerprint] = None) -> ModelFamilyOutput:
        if not self._trained:
            self.fit(digits)

        if fp is None:
            from .state_fingerprint import compute_state_fingerprint
            fp = compute_state_fingerprint(digits)

        result = self.memory.query(fp)
        self.last_result = result

        # Blend with uniform if insufficient evidence
        p_big_raw = result.empirical_p_big
        p_small_raw = result.empirical_p_small
        sample_weight = min(1.0, result.sample_size / 500.0)
        p_big = sample_weight * p_big_raw + (1 - sample_weight) * 0.5
        p_small = sample_weight * p_small_raw + (1 - sample_weight) * 0.5
        total = p_big + p_small
        p_big /= total
        p_small /= total

        # Digit vector from frequency with state bias
        digit_counts = np.ones(10) * 0.1
        for d in digits[-200:]:
            if 0 <= int(d) < 10:
                digit_counts[int(d)] += 1
        digit_probs = digit_counts / digit_counts.sum()
        if p_big >= p_small:
            digit_probs[5:] *= 1.0 + 0.5 * (p_big - 0.5) * 2
        else:
            digit_probs[:5] *= 1.0 + 0.5 * (p_small - 0.5) * 2
        digit_probs = digit_probs / digit_probs.sum()

        best = int(np.argmax(digit_probs))
        sorted_idx = digit_probs.argsort()[::-1]
        hedge = int(sorted_idx[1]) if len(sorted_idx) > 1 else 0

        return ModelFamilyOutput(
            model_name=self.name,
            family=self.family,
            version=self.version,
            prediction="Big" if p_big >= 0.5 else "Small",
            probability_big=p_big,
            probability_small=p_small,
            confidence=max(p_big, p_small),
            sample_size=result.sample_size,
            regime=fp.regime_id,
            generation=self.generation,
            probability_vector=digit_probs.tolist(),
            target_digit=best,
            hedge_digit=hedge,
        )


# =====================================================================
# E. BASELINE FAMILY
# =====================================================================

class RandomBaseline(BasePredictionModel):
    family = "baseline"
    name = "RandomBaseline"
    version = "v1"

    def fit(self, digits: Sequence[int]) -> None:
        self._trained = True

    def partial_fit(self, digits: Sequence[int]) -> None:
        pass

    def predict(self, digits: Sequence[int], fp: Optional[StateFingerprint] = None) -> ModelFamilyOutput:
        r = random.Random(hash(tuple(list(digits)[-5:])) % (2**32))
        choice = r.random()
        p_big = 0.5 + (choice - 0.5) * 0.02  # near-uniform
        p_small = 1 - p_big
        regime = fp.regime_id if fp is not None else "UNKNOWN"
        return ModelFamilyOutput(
            model_name=self.name,
            family=self.family,
            version=self.version,
            prediction="Big" if p_big >= 0.5 else "Small",
            probability_big=p_big,
            probability_small=p_small,
            confidence=0.5,
            sample_size=0,
            regime=regime,
            generation=self.generation,
            probability_vector=([0.1] * 10),
            target_digit=r.randint(0, 9),
            hedge_digit=r.randint(0, 9),
        )


class MajorityBaseline(BasePredictionModel):
    family = "baseline"
    name = "MajorityBaseline"
    version = "v1"

    def __init__(self, generation: int = 1):
        super().__init__(generation)
        self._overall_p_big = 0.5

    def fit(self, digits: Sequence[int]) -> None:
        sizes = [1 if int(d) >= 5 else 0 for d in digits]
        self._overall_p_big = float(np.mean(sizes)) if sizes else 0.5
        self._trained = True

    def partial_fit(self, digits: Sequence[int]) -> None:
        for d in digits[-100:]:
            s = 1 if int(d) >= 5 else 0
            self._overall_p_big = 0.99 * self._overall_p_big + 0.01 * s

    def predict(self, digits: Sequence[int], fp: Optional[StateFingerprint] = None) -> ModelFamilyOutput:
        if not self._trained:
            self.fit(digits)
        p_big = 0.5 + 0.01 * (1 if self._overall_p_big >= 0.5 else -1)
        # Shrink very close to 0.5: it's a trivial baseline
        p_big = 0.5 + (p_big - 0.5) * 0.2
        p_small = 1 - p_big
        regime = fp.regime_id if fp is not None else "UNKNOWN"
        return ModelFamilyOutput(
            model_name=self.name,
            family=self.family,
            version=self.version,
            prediction="Big" if self._overall_p_big >= 0.5 else "Small",
            probability_big=p_big,
            probability_small=p_small,
            confidence=max(p_big, p_small),
            sample_size=len(digits),
            regime=regime,
            generation=self.generation,
            probability_vector=([0.1] * 10),
            target_digit=5 if self._overall_p_big >= 0.5 else 4,
            hedge_digit=6 if self._overall_p_big >= 0.5 else 3,
        )


# =====================================================================
# ENSEMBLE ASSEMBLY
# =====================================================================

class MultiModelEnsemble:
    """Assembles all model families and runs them together."""

    def __init__(self, generation: int = 1):
        self.generation = generation
        self.models: List[BasePredictionModel] = []
        self._build_default_population()

    def _build_default_population(self) -> None:
        g = self.generation
        # Statistical
        self.models.append(FrequencyModel(window=100, generation=g))
        self.models.append(FrequencyModel(window=500, generation=g))
        self.models.append(BayesianModel(prior_strength=10.0, generation=g))
        self.models.append(MarkovModel(order=1, generation=g))
        self.models.append(MarkovModel(order=2, generation=g))
        self.models.append(MarkovModel(order=3, generation=g))
        # Sequence
        self.models.append(NgramModel(n=3, generation=g))
        self.models.append(NgramModel(n=5, generation=g))
        self.models.append(SequenceSimilarityModel(k=8, top_k=20, generation=g))
        # Temporal
        self.models.append(RecencyWeightedModel(decay=0.99, generation=g))
        self.models.append(RecencyWeightedModel(decay=0.97, generation=g))
        # Memory
        self.models.append(SimilarStateModel(sim_threshold=0.80, generation=g))
        # Baselines
        self.models.append(RandomBaseline(generation=g))
        self.models.append(MajorityBaseline(generation=g))

    def fit_all(self, digits: Sequence[int]) -> None:
        for m in self.models:
            try:
                m.fit(digits)
            except Exception:
                pass

    def partial_fit_all(self, digits: Sequence[int]) -> None:
        for m in self.models:
            try:
                m.partial_fit(digits)
            except Exception:
                pass

    def predict_all(self, digits: Sequence[int], fp: Optional[StateFingerprint] = None) -> List[ModelFamilyOutput]:
        results = []
        for m in self.models:
            try:
                results.append(m.predict(digits, fp))
            except Exception:
                # Fallback: uniform prediction so the ensemble is never empty
                results.append(ModelFamilyOutput(
                    model_name=m.name,
                    family=getattr(m, "family", "unknown"),
                    version=getattr(m, "version", "v1"),
                    prediction="Big",
                    probability_big=0.5,
                    probability_small=0.5,
                    confidence=0.5,
                    sample_size=0,
                    regime=fp.regime_id if fp is not None else "UNKNOWN",
                    generation=self.generation,
                    probability_vector=[0.1] * 10,
                    target_digit=5,
                    hedge_digit=4,
                ))
        return results
