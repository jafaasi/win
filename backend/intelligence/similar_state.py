from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence, Tuple
import numpy as np

from .state_fingerprint import StateFingerprint, compute_state_fingerprint


@dataclass
class SimilarStateResult:
    sample_size: int = 0
    matched_states: int = 0

    empirical_p_big: float = 0.5
    empirical_p_small: float = 0.5

    mean_similarity: float = 0.0
    min_similarity: float = 0.0

    horizon_1_win_rate: float = 0.0
    horizon_2_win_rate: float = 0.0
    horizon_3_win_rate: float = 0.0

    log_uncertainty: float = 0.0
    std_error: float = 0.0

    sufficient_evidence: bool = False
    evidence_ratio: float = 0.0

    matched_outcomes_big: int = 0
    matched_outcomes_small: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def is_statistically_reliable(self, min_samples: int = 100, min_edge: float = 0.005) -> bool:
        if self.sample_size < min_samples:
            return False
        edge = abs(self.empirical_p_big - 0.5)
        return edge >= min_edge


class SimilarStateMemory:
    """
    Similar-State Memory: when a new fingerprint arrives, find historically
    similar fingerprints and measure the empirical conditional outcome
    distribution that followed them.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.82,
        min_sample_size: int = 50,
        max_memory_size: int = 50000,
    ):
        self.similarity_threshold = similarity_threshold
        self.min_sample_size = min_sample_size
        self.max_memory_size = max_memory_size

        self._fingerprints: List[StateFingerprint] = []
        self._next_outcome_sizes: List[int] = []
        self._next_outcome_digits: List[int] = []
        self._horizon_correct: List[Tuple[bool, bool, bool]] = []

    # ------------------------------------------------------------------
    # Storage APIs
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._fingerprints)

    def remember(
        self,
        fp: StateFingerprint,
        next_size: int,
        next_digit: int = -1,
        horizon_correct: Tuple[bool, bool, bool] = (False, False, False),
    ) -> None:
        """Append a fingerprint -> outcome pair to memory."""
        self._fingerprints.append(fp)
        self._next_outcome_sizes.append(int(next_size))
        self._next_outcome_digits.append(int(next_digit))
        self._horizon_correct.append(horizon_correct)

        if len(self._fingerprints) > self.max_memory_size:
            trim = len(self._fingerprints) - self.max_memory_size
            self._fingerprints = self._fingerprints[trim:]
            self._next_outcome_sizes = self._next_outcome_sizes[trim:]
            self._next_outcome_digits = self._next_outcome_digits[trim:]
            self._horizon_correct = self._horizon_correct[trim:]

    def bulk_remember_from_history(
        self,
        history_digits: Sequence[int],
        start_offset: int = 60,
        stride: int = 1,
    ) -> int:
        """
        Scan historical digit sequence and remember fingerprints + next outcomes.
        Ensures each memory entry uses only causal information.
        """
        digits = [int(d) for d in history_digits]
        n = len(digits)
        added = 0
        for i in range(start_offset, n - 3, stride):
            ctx = digits[:i]  # strictly causal
            next_size = 1 if digits[i] >= 5 else 0
            next_digit = digits[i]
            # Evaluate horizons: was the predicted side correct within 1/2/3 rounds?
            predicted_side = 1 if (np.mean(ctx[-10:]) if len(ctx) >= 10 else 0) >= 0.5 else 0
            h1 = (1 if digits[i] >= 5 else 0) == predicted_side
            h2 = h1 or ((1 if digits[i + 1] >= 5 else 0) == predicted_side) if i + 1 < n else h1
            h3 = h2 or ((1 if digits[i + 2] >= 5 else 0) == predicted_side) if i + 2 < n else h2

            fp = compute_state_fingerprint(ctx, sequence_no=i)
            self.remember(fp, next_size, next_digit, (h1, h2, h3))
            added += 1
        return added

    # ------------------------------------------------------------------
    # Similarity measures
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def jaccard_sequence(fp_a: StateFingerprint, fp_b: StateFingerprint) -> float:
        sa = fp_a.recent_sequence
        sb = fp_b.recent_sequence
        if not sa or not sb:
            return 0.0
        set_a = set(sa[i : i + 3] for i in range(len(sa) - 2))
        set_b = set(sb[i : i + 3] for i in range(len(sb) - 2))
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return inter / union if union > 0 else 0.0

    def _combined_similarity(self, fp_a: StateFingerprint, fp_b: StateFingerprint) -> float:
        vec_a = fp_a.to_numpy()
        vec_b = fp_b.to_numpy()
        cos = self.cosine_sim(vec_a, vec_b)
        jac = self.jaccard_sequence(fp_a, fp_b)
        regime_match = 1.0 if fp_a.regime_id == fp_b.regime_id else 0.0
        # Weighted combination: vector structure + pattern overlap + regime
        score = 0.45 * cos + 0.35 * jac + 0.20 * regime_match
        return max(0.0, min(1.0, float(score)))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        target_fp: StateFingerprint,
        top_k: Optional[int] = None,
    ) -> SimilarStateResult:
        """Find similar states and compute empirical conditional statistics."""
        result = SimilarStateResult()
        if len(self._fingerprints) == 0:
            return result

        similarities = []
        for fp in self._fingerprints:
            sim = self._combined_similarity(target_fp, fp)
            if sim >= self.similarity_threshold:
                similarities.append(sim)
            else:
                similarities.append(-1.0)  # mark as below threshold

        idxs = [i for i, s in enumerate(similarities) if s >= self.similarity_threshold]
        if len(idxs) < self.min_sample_size and top_k is None:
            # Fall back to top-K if threshold filter is too strict
            order = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
            idxs = order[: max(self.min_sample_size, 200)]
        elif top_k is not None and len(idxs) > top_k:
            idxs_with_sim = [(i, similarities[i]) for i in idxs]
            idxs_with_sim.sort(key=lambda x: x[1], reverse=True)
            idxs = [i for i, _ in idxs_with_sim[:top_k]]

        if not idxs:
            return result

        result.sample_size = len(idxs)
        result.matched_states = len(idxs)

        big_count = 0
        small_count = 0
        h1_wins = 0
        h2_wins = 0
        h3_wins = 0
        sims = []

        for i in idxs:
            s = self._next_outcome_sizes[i]
            if s == 1:
                big_count += 1
            else:
                small_count += 1
            sims.append(similarities[i])
            h1, h2, h3 = self._horizon_correct[i]
            if h1:
                h1_wins += 1
            if h2:
                h2_wins += 1
            if h3:
                h3_wins += 1

        total = big_count + small_count
        if total > 0:
            result.empirical_p_big = big_count / total
            result.empirical_p_small = small_count / total
            result.horizon_1_win_rate = h1_wins / total
            result.horizon_2_win_rate = h2_wins / total
            result.horizon_3_win_rate = h3_wins / total

        result.mean_similarity = float(np.mean(sims)) if sims else 0.0
        result.min_similarity = float(np.min(sims)) if sims else 0.0

        # Standard error on the binary proportion (Clopper-Pearson approximate)
        if total >= 2:
            p = result.empirical_p_big
            result.std_error = math.sqrt(max(0.0, p * (1 - p) / total))
            result.log_uncertainty = -p * math.log2(max(1e-9, p)) - (1 - p) * math.log2(max(1e-9, 1 - p))
        else:
            result.std_error = 0.5
            result.log_uncertainty = 1.0

        result.matched_outcomes_big = big_count
        result.matched_outcomes_small = small_count

        # Evidence ratio: how much evidence above chance
        edge = abs(result.empirical_p_big - 0.5)
        se = max(result.std_error, 1e-9)
        result.evidence_ratio = edge / se
        result.sufficient_evidence = (
            result.sample_size >= self.min_sample_size and result.evidence_ratio >= 1.5
        )

        return result
