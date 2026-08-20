from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Sequence, Tuple
import numpy as np

from .state_fingerprint import StateFingerprint, compute_state_fingerprint


@dataclass
class FeatureEvaluation:
    """Evaluation result for a single feature."""
    
    feature_name: str
    importance_score: float
    stability_score: float
    conditional_value: float
    redundancy_score: float
    degradation_score: float
    oos_improvement: float
    overall_value: float
    is_useful: bool
    
    def to_dict(self) -> dict:
        return asdict(self)


class InformationValueEngine:
    """
    Information Value Engine: measures whether each feature actually improves prediction.
    
    For every feature, measure:
      - feature importance
      - feature stability
      - feature conditional value
      - feature redundancy
      - feature degradation
      - OOS improvement
    
    If a feature provides no reliable OOS improvement: REMOVE or DOWNWEIGHT IT.
    """
    
    def __init__(self):
        self.feature_history: Dict[str, List[float]] = {}
        self.feature_evaluations: Dict[str, FeatureEvaluation] = {}
        
    def extract_features(self, history_digits: Sequence[int]) -> Dict[str, float]:
        """Extract all available features from historical data."""
        digits = [int(d) for d in history_digits]
        sizes = [1 if d >= 5 else 0 for d in digits]
        
        features = {}
        
        # Basic statistical features
        if len(digits) >= 10:
            features['mean_digit'] = float(np.mean(digits[-10:]))
            features['std_digit'] = float(np.std(digits[-10:]))
            features['mean_size'] = float(np.mean(sizes[-10:]))
            features['std_size'] = float(np.std(sizes[-10:]))
        
        # Window-based features
        for window in [10, 20, 50, 100]:
            if len(digits) >= window:
                window_data = digits[-window:]
                features[f'mean_{window}'] = float(np.mean(window_data))
                features[f'std_{window}'] = float(np.std(window_data))
                features[f'min_{window}'] = float(np.min(window_data))
                features[f'max_{window}'] = float(np.max(window_data))
                
                # Trend
                if len(window_data) >= 3:
                    features[f'trend_{window}'] = float(np.polyfit(range(len(window_data)), window_data, 1)[0])
        
        # Frequency features
        for window in [10, 20, 50, 100]:
            if len(sizes) >= window:
                window_sizes = sizes[-window:]
                features[f'big_rate_{window}'] = float(np.mean(window_sizes))
        
        # Streak features
        if len(sizes) >= 2:
            current_streak = 1
            for i in range(len(sizes) - 2, -1, -1):
                if sizes[i] == sizes[-1]:
                    current_streak += 1
                else:
                    break
            features['current_streak'] = float(current_streak)
            features['streak_value'] = float(sizes[-1])
        
        # Entropy features
        if len(sizes) >= 10:
            value_counts = np.bincount(sizes[-10:], minlength=2)
            probs = value_counts / value_counts.sum()
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            features['entropy_10'] = float(entropy)
        
        if len(sizes) >= 50:
            value_counts = np.bincount(sizes[-50:], minlength=2)
            probs = value_counts / value_counts.sum()
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            features['entropy_50'] = float(entropy)
        
        # Autocorrelation features
        for lag in [1, 2, 3, 5]:
            if len(sizes) > lag:
                autocorr = np.corrcoef(sizes[:-lag], sizes[lag:])[0, 1]
                features[f'autocorr_{lag}'] = float(autocorr if not np.isnan(autocorr) else 0.0)
        
        # Transition features
        if len(sizes) >= 10:
            transition_counts = np.zeros((2, 2))
            for i in range(1, len(sizes)):
                transition_counts[sizes[i-1], sizes[i]] += 1
            row_sums = transition_counts.sum(axis=1)
            transition_probs = transition_counts / (row_sums[:, None] + 1e-10)
            features['transition_0_to_1'] = float(transition_probs[0, 1])
            features['transition_1_to_0'] = float(transition_probs[1, 0])
        
        # Pattern features
        if len(sizes) >= 4:
            # Alternation rate
            alternations = sum(1 for i in range(1, len(sizes)) if sizes[i] != sizes[i-1])
            features['alternation_rate'] = float(alternations / (len(sizes) - 1))
        
        return features
    
    def evaluate_feature_importance(
        self, 
        feature_name: str, 
        feature_values: Sequence[float], 
        target_values: Sequence[int]
    ) -> float:
        """Evaluate feature importance using mutual information."""
        if len(feature_values) != len(target_values) or len(feature_values) < 10:
            return 0.0
        
        # Discretize feature values
        n_bins = min(10, len(set(feature_values)))
        if n_bins < 2:
            return 0.0
        
        # Create bins
        feature_discrete = np.digitize(feature_values, np.percentile(feature_values, np.linspace(0, 100, n_bins)))
        
        # Calculate mutual information
        from sklearn.metrics import mutual_info_score
        mi = mutual_info_score(feature_discrete, target_values)
        
        # Normalize by entropy of target
        target_entropy = mutual_info_score(target_values, target_values)
        if target_entropy > 0:
            normalized_mi = mi / target_entropy
        else:
            normalized_mi = 0.0
        
        return float(normalized_mi)
    
    def evaluate_feature_stability(self, feature_name: str, feature_values: Sequence[float]) -> float:
        """Evaluate how stable a feature is over time."""
        if len(feature_values) < 20:
            return 0.0
        
        # Split into two halves and compare distributions
        mid = len(feature_values) // 2
        first_half = feature_values[:mid]
        second_half = feature_values[mid:]
        
        # Use Kolmogorov-Smirnov test
        from scipy.stats import ks_2samp
        ks_stat, p_value = ks_2samp(first_half, second_half)
        
        # High p-value = stable (distributions similar)
        stability = float(p_value)
        
        return stability
    
    def evaluate_conditional_value(
        self,
        feature_name: str,
        feature_values: Sequence[float],
        target_values: Sequence[int]
    ) -> float:
        """Evaluate conditional predictive value of feature."""
        if len(feature_values) != len(target_values) or len(feature_values) < 20:
            return 0.0
        
        # Discretize feature
        n_bins = min(5, len(set(feature_values)))
        if n_bins < 2:
            return 0.0
        
        feature_discrete = np.digitize(feature_values, np.percentile(feature_values, np.linspace(0, 100, n_bins)))
        
        # Calculate conditional probabilities
        conditional_probs = {}
        for bin_val in range(n_bins):
            mask = feature_discrete == bin_val
            if mask.sum() >= 5:
                bin_targets = target_values[mask]
                conditional_probs[bin_val] = float(np.mean(bin_targets))
        
        if not conditional_probs:
            return 0.0
        
        # Measure variance in conditional probabilities
        probs = list(conditional_probs.values())
        conditional_variance = np.var(probs)
        
        # High variance = feature is conditionally informative
        return float(conditional_variance)
    
    def evaluate_redundancy(
        self,
        feature_name: str,
        feature_values: Sequence[float],
        other_features: Dict[str, Sequence[float]]
    ) -> float:
        """Evaluate how redundant a feature is with other features."""
        if not other_features:
            return 0.0
        
        max_correlation = 0.0
        for other_name, other_values in other_features.items():
            if other_name == feature_name or len(other_values) != len(feature_values):
                continue
            
            try:
                corr = np.corrcoef(feature_values, other_values)[0, 1]
                if not np.isnan(corr):
                    max_correlation = max(max_correlation, abs(float(corr)))
            except:
                pass
        
        # High correlation = redundant
        return float(max_correlation)
    
    def evaluate_degradation(
        self,
        feature_name: str,
        feature_values: Sequence[float],
        target_values: Sequence[int]
    ) -> float:
        """Evaluate if feature performance degrades over time."""
        if len(feature_values) < 40:
            return 0.0
        
        # Split into quarters
        n_quarters = 4
        quarter_size = len(feature_values) // n_quarters
        
        quarter_scores = []
        for i in range(n_quarters):
            start = i * quarter_size
            end = (i + 1) * quarter_size if i < n_quarters - 1 else len(feature_values)
            
            quarter_feature = feature_values[start:end]
            quarter_target = target_values[start:end]
            
            if len(quarter_feature) >= 5:
                importance = self.evaluate_feature_importance(feature_name, quarter_feature, quarter_target)
                quarter_scores.append(importance)
        
        if len(quarter_scores) < 2:
            return 0.0
        
        # Measure trend in importance over time
        if len(quarter_scores) >= 2:
            trend = np.polyfit(range(len(quarter_scores)), quarter_scores, 1)[0]
            # Negative trend = degradation
            degradation = max(0.0, -float(trend))
        else:
            degradation = 0.0
        
        return degradation
    
    def evaluate_oos_improvement(
        self,
        feature_name: str,
        train_features: Dict[str, Sequence[float]],
        train_targets: Sequence[int],
        test_features: Dict[str, Sequence[float]],
        test_targets: Sequence[int]
    ) -> float:
        """Evaluate if feature provides OOS improvement."""
        if feature_name not in train_features or feature_name not in test_features:
            return 0.0
        
        train_feature = train_features[feature_name]
        test_feature = test_features[feature_name]
        
        # Simple baseline: predict majority class
        baseline_train_acc = max(np.mean(train_targets), 1 - np.mean(train_targets))
        baseline_test_acc = max(np.mean(test_targets), 1 - np.mean(test_targets))
        
        # Feature-based prediction: use feature to predict
        # Simplified: if feature > median, predict class 1, else 0
        train_median = np.median(train_feature)
        train_pred = (train_feature > train_median).astype(int)
        test_pred = (test_feature > train_median).astype(int)
        
        feature_train_acc = np.mean(train_pred == train_targets)
        feature_test_acc = np.mean(test_pred == test_targets)
        
        # Improvement over baseline
        train_improvement = feature_train_acc - baseline_train_acc
        test_improvement = feature_test_acc - baseline_test_acc
        
        # Only count if it improves both train and test
        if train_improvement > 0 and test_improvement > 0:
            return float(test_improvement)
        else:
            return 0.0
    
    def evaluate_all_features(
        self,
        history_digits: Sequence[int],
        train_ratio: float = 0.7
    ) -> Dict[str, FeatureEvaluation]:
        """Comprehensive evaluation of all features."""
        digits = [int(d) for d in history_digits]
        sizes = [1 if d >= 5 else 0 for d in digits]
        
        # Extract features for entire history
        all_features = self.extract_features(digits)
        
        if not all_features:
            return {}
        
        # Split into train/test
        split_idx = int(len(digits) * train_ratio)
        train_digits = digits[:split_idx]
        test_digits = digits[split_idx:]
        train_sizes = sizes[:split_idx]
        test_sizes = sizes[split_idx:]
        
        # Extract features for train and test
        train_features = self.extract_features(train_digits)
        test_features = self.extract_features(test_digits)
        
        evaluations = {}
        
        for feature_name in all_features.keys():
            if feature_name not in train_features or feature_name not in test_features:
                continue
            
            train_values = train_features[feature_name]
            test_values = test_features[feature_name]
            
            # Evaluate each aspect
            importance = self.evaluate_feature_importance(feature_name, train_values, train_sizes)
            stability = self.evaluate_feature_stability(feature_name, train_values)
            conditional_value = self.evaluate_conditional_value(feature_name, train_values, train_sizes)
            redundancy = self.evaluate_redundancy(feature_name, train_values, train_features)
            degradation = self.evaluate_degradation(feature_name, train_values, train_sizes)
            oos_improvement = self.evaluate_oos_improvement(
                feature_name, train_features, train_sizes, test_features, test_sizes
            )
            
            # Calculate overall value score
            # Weight components: importance (0.3), stability (0.2), conditional (0.2), 
            # redundancy inverse (0.1), degradation inverse (0.1), OOS (0.1)
            redundancy_penalty = 1.0 - redundancy
            degradation_penalty = 1.0 - min(1.0, degradation)
            
            overall_value = (
                0.3 * importance +
                0.2 * stability +
                0.2 * conditional_value +
                0.1 * redundancy_penalty +
                0.1 * degradation_penalty +
                0.1 * oos_improvement
            )
            
            # Feature is useful if overall value > threshold
            is_useful = overall_value > 0.3 and oos_improvement > 0.0
            
            evaluation = FeatureEvaluation(
                feature_name=feature_name,
                importance_score=importance,
                stability_score=stability,
                conditional_value=conditional_value,
                redundancy_score=redundancy,
                degradation_score=degradation,
                oos_improvement=oos_improvement,
                overall_value=overall_value,
                is_useful=is_useful
            )
            
            evaluations[feature_name] = evaluation
            self.feature_evaluations[feature_name] = evaluation
        
        return evaluations
    
    def get_feature_weights(self) -> Dict[str, float]:
        """Get recommended weights for features based on their evaluation."""
        weights = {}
        for feature_name, evaluation in self.feature_evaluations.items():
            if evaluation.is_useful:
                weights[feature_name] = evaluation.overall_value
            else:
                weights[feature_name] = 0.0
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def get_removal_recommendations(self) -> List[str]:
        """Get list of features that should be removed or downweighted."""
        return [
            feature_name 
            for feature_name, evaluation in self.feature_evaluations.items()
            if not evaluation.is_useful
        ]
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report on feature information value."""
        useful_features = sum(1 for e in self.feature_evaluations.values() if e.is_useful)
        total_features = len(self.feature_evaluations)
        
        sorted_features = sorted(
            self.feature_evaluations.values(),
            key=lambda x: x.overall_value,
            reverse=True
        )
        
        return {
            "total_features": total_features,
            "useful_features": useful_features,
            "useless_features": total_features - useful_features,
            "top_features": [f.to_dict() for f in sorted_features[:5]],
            "bottom_features": [f.to_dict() for f in sorted_features[-5:]],
            "removal_recommendations": self.get_removal_recommendations(),
            "feature_weights": self.get_feature_weights()
        }
