import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter, deque
import math

class AdvancedPatternExtractor:
    """
    Extract advanced temporal patterns and sequence features for improved prediction.
    """
    
    def __init__(self, max_context: int = 500):
        self.max_context = max_context
        self.pattern_cache = {}
        self.frequency_matrix = np.zeros((10, 10, 10))  # 3-gram transitions
        self.temporal_decay = 0.98
        
    def extract_ngram_patterns(self, sequence: List[int], n: int = 3) -> Dict[str, float]:
        """Extract n-gram pattern frequencies and their recent strengths."""
        if len(sequence) < n:
            return {}
        
        ngrams = []
        for i in range(len(sequence) - n + 1):
            ngram = tuple(sequence[i:i+n])
            ngrams.append(ngram)
        
        # Count n-grams with recent weighting
        recent_ngrams = ngrams[-min(100, len(ngrams)):]
        recent_counter = Counter(recent_ngrams)
        
        # Calculate pattern strength
        pattern_strength = {}
        for ngram, count in recent_counter.items():
            pattern_strength[f"ngram_{ngram}"] = count / len(recent_ngrams)
        
        return pattern_strength
    
    def extract_temporal_patterns(self, sequence: List[int]) -> Dict[str, float]:
        """Extract temporal patterns like streaks, alternations, and cycles."""
        if len(sequence) < 10:
            return {}
        
        patterns = {}
        
        # Streak detection
        current_streak = 1
        max_streak = 1
        streak_type = sequence[0] >= 5  # True for Big, False for Small
        
        for i in range(1, len(sequence)):
            current_type = sequence[i] >= 5
            if current_type == streak_type:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
                streak_type = current_type
        
        patterns["max_streak"] = max_streak
        patterns["current_streak"] = current_streak
        
        # Alternation rate
        alternations = sum(1 for i in range(1, len(sequence)) 
                         if (sequence[i] >= 5) != (sequence[i-1] >= 5))
        patterns["alternation_rate"] = alternations / (len(sequence) - 1)
        
        # Gap analysis (time between same outcomes)
        big_indices = [i for i, x in enumerate(sequence) if x >= 5]
        small_indices = [i for i, x in enumerate(sequence) if x < 5]
        
        if len(big_indices) > 1:
            big_gaps = [big_indices[i+1] - big_indices[i] for i in range(len(big_indices)-1)]
            patterns["avg_big_gap"] = np.mean(big_gaps)
            patterns["big_gap_std"] = np.std(big_gaps)
        
        if len(small_indices) > 1:
            small_gaps = [small_indices[i+1] - small_indices[i] for i in range(len(small_indices)-1)]
            patterns["avg_small_gap"] = np.mean(small_gaps)
            patterns["small_gap_std"] = np.std(small_gaps)
        
        return patterns
    
    def extract_cyclical_patterns(self, sequence: List[int], max_cycle: int = 20) -> Dict[str, float]:
        """Detect cyclical patterns using autocorrelation analysis."""
        if len(sequence) < max_cycle * 2:
            return {}
        
        patterns = {}
        sequence_array = np.array(sequence)
        
        # Test different cycle lengths
        cycle_strengths = []
        for cycle_len in range(2, min(max_cycle + 1, len(sequence) // 2)):
            # Calculate correlation with shifted version
            shifted = sequence_array[cycle_len:]
            original = sequence_array[:-cycle_len]
            
            if len(shifted) > 10:
                correlation = np.corrcoef(original, shifted)[0, 1]
                if not np.isnan(correlation):
                    cycle_strengths.append((cycle_len, abs(correlation)))
        
        # Find strongest cycles
        if cycle_strengths:
            cycle_strengths.sort(key=lambda x: x[1], reverse=True)
            top_cycles = cycle_strengths[:3]
            
            for i, (cycle_len, strength) in enumerate(top_cycles):
                patterns[f"cycle_{i+1}_length"] = cycle_len
                patterns[f"cycle_{i+1}_strength"] = strength
        
        return patterns
    
    def extract_positional_patterns(self, sequence: List[int]) -> Dict[str, float]:
        """Extract position-based patterns (e.g., certain numbers at certain positions)."""
        if len(sequence) < 20:
            return {}
        
        patterns = {}
        
        # Position frequency in recent window
        recent = sequence[-min(50, len(sequence)):]
        position_freq = Counter(recent)
        
        for digit in range(10):
            patterns[f"pos_freq_{digit}"] = position_freq.get(digit, 0) / len(recent)
        
        # Last digit influence
        if len(sequence) >= 2:
            last_digit = sequence[-1]
            patterns["last_digit"] = last_digit
            
            # Track what typically follows the last digit
            followers = [sequence[i+1] for i in range(len(sequence)-1) if sequence[i] == last_digit]
            if followers:
                follower_freq = Counter(followers)
                most_likely_follower = follower_freq.most_common(1)[0][0]
                patterns["likely_follower"] = most_likely_follower
                patterns["follower_confidence"] = follower_freq[most_likely_follower] / len(followers)
        
        return patterns
    
    def extract_entropy_dynamics(self, sequence: List[int], window: int = 20) -> Dict[str, float]:
        """Extract entropy dynamics over sliding windows."""
        if len(sequence) < window * 2:
            return {}
        
        patterns = {}
        
        # Calculate entropy in sliding windows
        entropies = []
        for i in range(window, len(sequence)):
            window_seq = sequence[i-window:i]
            counts = np.bincount(window_seq, minlength=10)
            probs = counts / len(window_seq)
            entropy = 0.0
            for p in probs:
                if p > 0:
                    entropy -= p * math.log2(p)
            entropies.append(entropy)
        
        if entropies:
            patterns["entropy_mean"] = np.mean(entropies)
            patterns["entropy_std"] = np.std(entropies)
            patterns["entropy_trend"] = (entropies[-1] - entropies[0]) / len(entropies) if len(entropies) > 1 else 0
        
        return patterns
    
    def extract_volatility_patterns(self, sequence: List[int], window: int = 10) -> Dict[str, float]:
        """Extract volatility and momentum patterns."""
        if len(sequence) < window * 2:
            return {}
        
        patterns = {}
        sequence_array = np.array(sequence, dtype=float)
        
        # Rolling volatility
        volatilities = []
        for i in range(window, len(sequence)):
            window_seq = sequence_array[i-window:i]
            volatilities.append(np.std(window_seq))
        
        if volatilities:
            patterns["volatility_mean"] = np.mean(volatilities)
            patterns["volatility_current"] = volatilities[-1]
            patterns["volatility_trend"] = (volatilities[-1] - volatilities[0]) / len(volatilities) if len(volatilities) > 1 else 0
        
        # Momentum (rate of change)
        if len(sequence) >= window:
            recent_mean = np.mean(sequence_array[-window:])
            earlier_mean = np.mean(sequence_array[-(window*2):-window]) if len(sequence) >= window*2 else recent_mean
            patterns["momentum"] = recent_mean - earlier_mean
        
        return patterns
    
    def extract_color_patterns(self, sequence: List[int]) -> Dict[str, float]:
        """Extract color-based patterns (Red, Green, Violet)."""
        if len(sequence) < 10:
            return {}
        
        patterns = {}
        
        # Map digits to colors
        color_map = []
        for digit in sequence:
            if digit in [1, 3, 7, 9]:
                color_map.append(0)  # Green
            elif digit in [0, 5]:
                color_map.append(2)  # Violet
            else:
                color_map.append(1)  # Red
        
        # Color streaks
        color_streaks = []
        current_streak = 1
        for i in range(1, len(color_map)):
            if color_map[i] == color_map[i-1]:
                current_streak += 1
            else:
                color_streaks.append(current_streak)
                current_streak = 1
        color_streaks.append(current_streak)
        
        if color_streaks:
            patterns["color_streak_mean"] = np.mean(color_streaks)
            patterns["color_streak_max"] = max(color_streaks)
        
        # Color frequencies
        color_counter = Counter(color_map)
        for color_id, color_name in [(0, "green"), (1, "red"), (2, "violet")]:
            patterns[f"color_{color_name}_freq"] = color_counter.get(color_id, 0) / len(color_map)
        
        return patterns
    
    def extract_comprehensive_features(self, sequence: List[int]) -> Dict[str, float]:
        """Extract all advanced patterns into a comprehensive feature set."""
        features = {}
        
        # Extract all pattern types
        features.update(self.extract_ngram_patterns(sequence, n=2))
        features.update(self.extract_ngram_patterns(sequence, n=3))
        features.update(self.extract_temporal_patterns(sequence))
        features.update(self.extract_cyclical_patterns(sequence))
        features.update(self.extract_positional_patterns(sequence))
        features.update(self.extract_entropy_dynamics(sequence))
        features.update(self.extract_volatility_patterns(sequence))
        features.update(self.extract_color_patterns(sequence))
        
        return features
    
    def get_pattern_confidence(self, sequence: List[int], prediction: str) -> float:
        """
        Calculate confidence in prediction based on pattern consistency.
        """
        if len(sequence) < 10:
            return 0.5
        
        features = self.extract_comprehensive_features(sequence)
        
        # Base confidence from multiple pattern signals
        confidence = 0.5
        
        # Momentum alignment
        momentum = features.get("momentum", 0)
        if prediction == "Big" and momentum > 0.5:
            confidence += 0.15
        elif prediction == "Small" and momentum < -0.5:
            confidence += 0.15
        
        # Volatility adjustment
        volatility = features.get("volatility_current", 2.0)
        if volatility < 1.5:  # Low volatility - more confident
            confidence += 0.1
        elif volatility > 3.0:  # High volatility - less confident
            confidence -= 0.1
        
        # Streak continuation
        current_streak = features.get("current_streak", 1)
        if current_streak >= 3:
            # Strong streak - likely to continue or reverse
            confidence += 0.05
        
        # Cycle strength
        cycle_strength = features.get("cycle_1_strength", 0)
        if cycle_strength > 0.3:
            confidence += 0.1
        
        # Clamp confidence
        return max(0.3, min(0.95, confidence))
    
    def update_pattern_cache(self, sequence: List[int]) -> None:
        """Update internal pattern cache with new sequence data."""
        if len(sequence) < 3:
            return
        
        # Update 3-gram frequency matrix with temporal decay
        for i in range(len(sequence) - 2):
            triplet = tuple(sequence[i:i+3])
            a, b, c = triplet
            self.frequency_matrix[a, b, c] = (self.frequency_matrix[a, b, c] * self.temporal_decay + 1)
    
    def get_3gram_probability(self, context: List[int]) -> np.ndarray:
        """Get probability distribution based on 3-gram patterns."""
        if len(context) < 2:
            return np.ones(10) / 10
        
        last_two = tuple(context[-2:])
        a, b = last_two
        
        # Get frequencies for all possible next values
        frequencies = self.frequency_matrix[a, b, :]
        
        # Normalize to probabilities
        total = np.sum(frequencies)
        if total > 0:
            return frequencies / total
        else:
            return np.ones(10) / 10