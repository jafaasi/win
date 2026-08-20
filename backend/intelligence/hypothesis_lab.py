from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Sequence, Tuple
import numpy as np

from .state_fingerprint import StateFingerprint
from .similar_state import SimilarStateMemory, SimilarStateResult


@dataclass
class Hypothesis:
    """A discovered pattern that may have predictive value."""
    
    hypothesis_id: str
    description: str
    category: str  # "sequence", "transition", "regime", "temporal", "rare_pattern"
    
    # Pattern definition
    pattern_features: Dict[str, Any]
    pattern_signature: str
    
    # Performance tracking
    creation_generation: int
    sample_size: int = 0
    correct_predictions: int = 0
    total_predictions: int = 0
    
    # Validation metrics
    training_score: float = 0.0
    validation_score: float = 0.0
    oos_score: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    
    # Status
    status: str = "CANDIDATE"  # CANDIDATE, VALIDATED, WEAK, REJECTED, DEPRECATED
    
    # Metadata
    created_at: str = ""
    last_updated: str = ""
    baseline_comparison: float = 0.0
    priority: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @property
    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions
    
    @property
    def is_statistically_significant(self) -> bool:
        """Wilson lower bound > 0.5 with sufficient samples."""
        if self.total_predictions < 30:
            return False
        p = self.accuracy
        z = 1.96  # 95% confidence
        denominator = 1 + z * z / self.total_predictions
        centre = p + z * z / (2 * self.total_predictions)
        spread = z * math.sqrt((p * (1 - p) + z * z / (4 * self.total_predictions)) / self.total_predictions)
        lower_bound = max(0.0, (centre - spread) / denominator)
        return lower_bound > 0.52  # Slight edge over random


class HypothesisLab:
    """
    Autonomous hypothesis laboratory for pattern discovery and validation.
    
    Pipeline:
      DISCOVER → FORM HYPOTHESIS → TEST HISTORICALLY → 
      WALK-FORWARD VALIDATION → OOS TEST → COMPARE TO BASELINE → 
      CONFIRM / REJECT → STORE RESULT
    """
    
    def __init__(self, generation: int = 1):
        self.generation = generation
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.false_pattern_memory: Dict[str, Hypothesis] = {}
        self.similarity_threshold = 0.90  # Threshold for pattern similarity
        
    # ------------------------------------------------------------------
    # Pattern Discovery
    # ------------------------------------------------------------------
    
    def discover_sequence_patterns(self, history_digits: Sequence[int]) -> List[Hypothesis]:
        """Discover recurring sequence patterns."""
        hypotheses = []
        digits = [int(d) for d in history_digits]
        
        # Look for repeating n-grams
        for n in [2, 3, 4, 5]:
            if len(digits) < n + 10:
                continue
                
            pattern_counts = {}
            for i in range(len(digits) - n):
                pattern = tuple(digits[i:i+n])
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # Find patterns that occur frequently
            for pattern, count in pattern_counts.items():
                if count >= 5:  # Minimum occurrence threshold
                    # Analyze what follows this pattern
                    next_counts = {}
                    for i in range(len(digits) - n - 1):
                        if tuple(digits[i:i+n]) == pattern:
                            next_val = digits[i+n]
                            next_counts[next_val] = next_counts.get(next_val, 0) + 1
                    
                    if next_counts:
                        # Check if there's a bias in what follows
                        total = sum(next_counts.values())
                        if total >= 3:
                            max_next = max(next_counts, key=next_counts.get)
                            max_prob = next_counts[max_next] / total
                            
                            if max_prob > 0.65:  # Strong bias
                                # Check if this pattern is already in false memory
                                sig = self._pattern_signature(pattern, "sequence")
                                if sig in self.false_pattern_memory:
                                    continue  # Skip known false patterns
                                
                                hypothesis = Hypothesis(
                                    hypothesis_id=str(uuid.uuid4()),
                                    description=f"Sequence pattern {pattern} → {max_next} (p={max_prob:.2f})",
                                    category="sequence",
                                    pattern_features={"pattern": list(pattern), "next_value": max_next, "probability": max_prob},
                                    pattern_signature=sig,
                                    creation_generation=self.generation,
                                    sample_size=total,
                                    training_score=max_prob,
                                    created_at=datetime.utcnow().isoformat(),
                                    last_updated=datetime.utcnow().isoformat(),
                                    priority=min(1.0, (max_prob - 0.5) * 2.0)
                                )
                                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def discover_transition_patterns(self, history_digits: Sequence[int]) -> List[Hypothesis]:
        """Discover transition patterns (state changes)."""
        hypotheses = []
        digits = [int(d) for d in history_digits]
        sizes = [1 if d >= 5 else 0 for d in digits]
        
        if len(sizes) < 20:
            return hypotheses
        
        # Analyze transition probabilities
        transition_counts = {}
        for i in range(len(sizes) - 1):
            current = sizes[i]
            next_val = sizes[i+1]
            key = (current, next_val)
            transition_counts[key] = transition_counts.get(key, 0) + 1
        
        # Find unusual transitions
        for (current, next_val), count in transition_counts.items():
            if count >= 10:
                # Calculate probability of this transition
                total_from_current = sum(c for (c, _), cnt in transition_counts.items() if c == current)
                if total_from_current > 0:
                    prob = count / total_from_current
                    
                    # Flag if probability is significantly different from baseline
                    baseline_prob = 0.5  # Expected for random
                    if abs(prob - baseline_prob) > 0.15:
                        sig = self._pattern_signature((current, next_val), "transition")
                        if sig in self.false_pattern_memory:
                            continue
                        
                        hypothesis = Hypothesis(
                            hypothesis_id=str(uuid.uuid4()),
                            description=f"Transition {current}→{next_val} (p={prob:.2f})",
                            category="transition",
                            pattern_features={"from_state": current, "to_state": next_val, "probability": prob},
                            pattern_signature=sig,
                            creation_generation=self.generation,
                            sample_size=count,
                            training_score=prob,
                            created_at=datetime.utcnow().isoformat(),
                            last_updated=datetime.utcnow().isoformat(),
                            priority=min(1.0, abs(prob - 0.5) * 2.0)
                        )
                        hypotheses.append(hypothesis)
        
        return hypotheses
    
    def discover_regime_patterns(self, history_digits: Sequence[int]) -> List[Hypothesis]:
        """Discover regime-specific patterns."""
        hypotheses = []
        digits = [int(d) for d in history_digits]
        
        if len(digits) < 100:
            return hypotheses
        
        # Divide into windows and analyze
        window_size = 50
        for i in range(0, len(digits) - window_size, window_size // 2):
            window = digits[i:i+window_size]
            sizes = [1 if d >= 5 else 0 for d in window]
            big_rate = sum(sizes) / len(sizes)
            
            # Classify regime
            if big_rate > 0.65:
                regime = "BIG_MOMENTUM"
            elif big_rate < 0.35:
                regime = "SMALL_MOMENTUM"
            else:
                regime = "EQUILIBRIUM"
            
            # Check next window for continuation
            if i + window_size + window_size <= len(digits):
                next_window = digits[i+window_size:i+2*window_size]
                next_sizes = [1 if d >= 5 else 0 for d in next_window]
                next_big_rate = sum(next_sizes) / len(next_sizes)
                
                # If regime persists, it's a pattern
                if regime == "BIG_MOMENTUM" and next_big_rate > 0.60:
                    sig = self._pattern_signature(f"{regime}_{i}", "regime")
                    if sig not in self.false_pattern_memory:
                        hypothesis = Hypothesis(
                            hypothesis_id=str(uuid.uuid4()),
                            description=f"Regime {regime} persistence",
                            category="regime",
                            pattern_features={"regime": regime, "persistence_rate": next_big_rate},
                            pattern_signature=sig,
                            creation_generation=self.generation,
                            sample_size=window_size,
                            training_score=next_big_rate,
                            created_at=datetime.utcnow().isoformat(),
                            last_updated=datetime.utcnow().isoformat(),
                            priority=0.7
                        )
                        hypotheses.append(hypothesis)
        
        return hypotheses
    
    # ------------------------------------------------------------------
    # Hypothesis Testing
    # ------------------------------------------------------------------
    
    def test_hypothesis_historically(
        self, 
        hypothesis: Hypothesis, 
        history_digits: Sequence[int],
        validation_split: float = 0.2
    ) -> Hypothesis:
        """Test a hypothesis against historical data with train/validation split."""
        digits = [int(d) for d in history_digits]
        split_idx = int(len(digits) * (1 - validation_split))
        
        train_data = digits[:split_idx]
        val_data = digits[split_idx:]
        
        if hypothesis.category == "sequence":
            hypothesis = self._test_sequence_hypothesis(hypothesis, train_data, val_data)
        elif hypothesis.category == "transition":
            hypothesis = self._test_transition_hypothesis(hypothesis, train_data, val_data)
        elif hypothesis.category == "regime":
            hypothesis = self._test_regime_hypothesis(hypothesis, train_data, val_data)
        
        hypothesis.last_updated = datetime.utcnow().isoformat()
        return hypothesis
    
    def _test_sequence_hypothesis(self, hypothesis: Hypothesis, train_data: List[int], val_data: List[int]) -> Hypothesis:
        """Test a sequence-based hypothesis."""
        pattern = tuple(hypothesis.pattern_features["pattern"])
        expected_next = hypothesis.pattern_features["next_value"]
        
        def test_on_data(data):
            correct = 0
            total = 0
            for i in range(len(data) - len(pattern)):
                if tuple(data[i:i+len(pattern)]) == pattern:
                    if i + len(pattern) < len(data):
                        actual = data[i + len(pattern)]
                        if actual == expected_next:
                            correct += 1
                        total += 1
            return correct, total
        
        train_correct, train_total = test_on_data(train_data)
        val_correct, val_total = test_on_data(val_data)
        
        hypothesis.sample_size = train_total + val_total
        hypothesis.correct_predictions = train_correct + val_correct
        hypothesis.total_predictions = train_total + val_total
        
        hypothesis.training_score = train_correct / train_total if train_total > 0 else 0.0
        hypothesis.validation_score = val_correct / val_total if val_total > 0 else 0.0
        
        # Calculate confidence interval
        if hypothesis.total_predictions > 0:
            p = hypothesis.accuracy
            se = math.sqrt(p * (1 - p) / hypothesis.total_predictions)
            z = 1.96
            ci = (max(0.0, p - z * se), min(1.0, p + z * se))
            hypothesis.confidence_interval = ci
        
        return hypothesis
    
    def _test_transition_hypothesis(self, hypothesis: Hypothesis, train_data: List[int], val_data: List[int]) -> Hypothesis:
        """Test a transition-based hypothesis."""
        from_state = hypothesis.pattern_features["from_state"]
        to_state = hypothesis.pattern_features["to_state"]
        
        def test_on_data(data):
            sizes = [1 if d >= 5 else 0 for d in data]
            correct = 0
            total = 0
            for i in range(len(sizes) - 1):
                if sizes[i] == from_state:
                    if sizes[i+1] == to_state:
                        correct += 1
                    total += 1
            return correct, total
        
        train_correct, train_total = test_on_data(train_data)
        val_correct, val_total = test_on_data(val_data)
        
        hypothesis.sample_size = train_total + val_total
        hypothesis.correct_predictions = train_correct + val_correct
        hypothesis.total_predictions = train_total + val_total
        
        hypothesis.training_score = train_correct / train_total if train_total > 0 else 0.0
        hypothesis.validation_score = val_correct / val_total if val_total > 0 else 0.0
        
        return hypothesis
    
    def _test_regime_hypothesis(self, hypothesis: Hypothesis, train_data: List[int], val_data: List[int]) -> Hypothesis:
        """Test a regime-based hypothesis."""
        # Simplified regime testing
        sizes = [1 if d >= 5 else 0 for d in train_data + val_data]
        big_rate = sum(sizes) / len(sizes)
        
        hypothesis.sample_size = len(sizes)
        hypothesis.training_score = big_rate
        hypothesis.validation_score = big_rate  # Simplified
        
        return hypothesis
    
    # ------------------------------------------------------------------
    # Walk-Forward Validation
    # ------------------------------------------------------------------
    
    def walk_forward_validation(
        self, 
        hypothesis: Hypothesis, 
        history_digits: Sequence[int],
        window_size: int = 200,
        step_size: int = 50
    ) -> Dict[str, Any]:
        """Perform walk-forward validation on historical data."""
        digits = [int(d) for d in history_digits]
        
        if len(digits) < window_size * 2:
            return {"status": "INSUFFICIENT_DATA", "oos_score": 0.0}
        
        results = []
        for i in range(0, len(digits) - window_size, step_size):
            train_window = digits[:i + window_size]
            test_window = digits[i + window_size:i + window_size + step_size]
            
            if len(test_window) == 0:
                continue
            
            # Train on train_window
            hypothesis_copy = Hypothesis(**hypothesis.to_dict())
            hypothesis_copy = self.test_hypothesis_historically(hypothesis_copy, train_window, validation_split=0.0)
            
            # Test on test_window
            test_correct, test_total = 0, 0
            if hypothesis.category == "sequence":
                pattern = tuple(hypothesis.pattern_features["pattern"])
                expected_next = hypothesis.pattern_features["next_value"]
                for j in range(len(test_window) - len(pattern)):
                    if tuple(test_window[j:j+len(pattern)]) == pattern:
                        if j + len(pattern) < len(test_window):
                            if test_window[j + len(pattern)] == expected_next:
                                test_correct += 1
                            test_total += 1
            
            if test_total > 0:
                results.append(test_correct / test_total)
        
        if not results:
            return {"status": "NO_VALID_WINDOWS", "oos_score": 0.0}
        
        oos_score = np.mean(results)
        hypothesis.oos_score = oos_score
        
        return {
            "status": "COMPLETED",
            "oos_score": oos_score,
            "num_windows": len(results),
            "std_score": np.std(results) if len(results) > 1 else 0.0
        }
    
    # ------------------------------------------------------------------
    # Baseline Comparison
    # ------------------------------------------------------------------
    
    def compare_to_baseline(self, hypothesis: Hypothesis, baseline_accuracy: float = 0.5) -> float:
        """Compare hypothesis performance to simple baseline."""
        improvement = hypothesis.validation_score - baseline_accuracy
        hypothesis.baseline_comparison = improvement
        
        # Statistical significance test
        if hypothesis.total_predictions >= 30:
            p = hypothesis.accuracy
            n = hypothesis.total_predictions
            se = math.sqrt(p * (1 - p) / n)
            z_score = improvement / se if se > 0 else 0
            
            # If improvement is statistically significant (z > 1.96)
            if z_score > 1.96:
                hypothesis.priority = min(1.0, hypothesis.priority + 0.3)
            elif z_score < -1.96:
                hypothesis.priority = max(0.0, hypothesis.priority - 0.3)
        
        return improvement
    
    # ------------------------------------------------------------------
    # Hypothesis Management
    # ------------------------------------------------------------------
    
    def evaluate_and_classify(self, hypothesis: Hypothesis) -> str:
        """Evaluate hypothesis and assign status."""
        # Check if statistically significant
        if not hypothesis.is_statistically_significant:
            hypothesis.status = "WEAK"
            return "WEAK"
        
        # Check OOS performance
        if hypothesis.oos_score > 0.55 and hypothesis.validation_score > 0.55:
            hypothesis.status = "VALIDATED"
            return "VALIDATED"
        elif hypothesis.oos_score < 0.48:
            hypothesis.status = "REJECTED"
            return "REJECTED"
        elif hypothesis.baseline_comparison < 0.0:
            hypothesis.status = "WEAK"
            return "WEAK"
        else:
            hypothesis.status = "CANDIDATE"
            return "CANDIDATE"
    
    def add_to_false_memory(self, hypothesis: Hypothesis, reason: str = "Failed validation") -> None:
        """Add a hypothesis to false pattern memory."""
        hypothesis.status = "REJECTED"
        hypothesis.description = f"[FALSE_PATTERN] {hypothesis.description} - {reason}"
        self.false_pattern_memory[hypothesis.pattern_signature] = hypothesis
        print(f"[HypothesisLab] Added to false memory: {hypothesis.hypothesis_id}")
    
    def discover_and_test_all(self, history_digits: Sequence[int]) -> List[Hypothesis]:
        """Run complete discovery and testing pipeline."""
        all_hypotheses = []
        
        # Discover patterns
        sequence_hypotheses = self.discover_sequence_patterns(history_digits)
        transition_hypotheses = self.discover_transition_patterns(history_digits)
        regime_hypotheses = self.discover_regime_patterns(history_digits)
        
        all_hypotheses.extend(sequence_hypotheses)
        all_hypotheses.extend(transition_hypotheses)
        all_hypotheses.extend(regime_hypotheses)
        
        print(f"[HypothesisLab] Discovered {len(all_hypotheses)} candidate hypotheses")
        
        # Test each hypothesis
        validated_hypotheses = []
        for hypothesis in all_hypotheses:
            # Historical testing
            hypothesis = self.test_hypothesis_historically(hypothesis, history_digits)
            
            # Walk-forward validation
            wf_result = self.walk_forward_validation(hypothesis, history_digits)
            
            # Baseline comparison
            self.compare_to_baseline(hypothesis)
            
            # Classify
            status = self.evaluate_and_classify(hypothesis)
            
            if status == "VALIDATED":
                self.hypotheses[hypothesis.hypothesis_id] = hypothesis
                validated_hypotheses.append(hypothesis)
                print(f"[HypothesisLab] Validated: {hypothesis.description}")
            elif status == "REJECTED":
                self.add_to_false_memory(hypothesis, "Failed validation")
            else:
                self.hypotheses[hypothesis.hypothesis_id] = hypothesis
        
        print(f"[HypothesisLab] Final: {len(validated_hypotheses)} validated, "
              f"{len(self.false_pattern_memory)} in false memory")
        
        return validated_hypotheses
    
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    
    def _pattern_signature(self, pattern: Any, category: str) -> str:
        """Generate a unique signature for a pattern."""
        import hashlib
        pattern_str = f"{category}:{str(pattern)}"
        return hashlib.md5(pattern_str.encode()).hexdigest()
    
    def get_hypothesis_report(self) -> Dict[str, Any]:
        """Generate a report on current hypotheses."""
        total = len(self.hypotheses)
        validated = sum(1 for h in self.hypotheses.values() if h.status == "VALIDATED")
        weak = sum(1 for h in self.hypotheses.values() if h.status == "WEAK")
        rejected = len(self.false_pattern_memory)
        
        return {
            "total_hypotheses": total,
            "validated": validated,
            "weak": weak,
            "rejected_in_memory": rejected,
            "generation": self.generation,
            "top_hypotheses": [
                h.to_dict() for h in sorted(
                    self.hypotheses.values(), 
                    key=lambda x: x.priority, 
                    reverse=True
                )[:5]
            ]
        }
