import numpy as np
import math
from typing import Dict, List
import scipy.stats as stats

class EnhancedStatisticalTests:
    @staticmethod
    def chi_square_frequency(sequence: np.ndarray, num_classes: int = 10) -> Dict:
        """
        Tests whether each value appears approximately uniformly.
        Null hypothesis: Uniform distribution.
        """
        if len(sequence) < num_classes:
            return {"chi2": 0.0, "p_value": 1.0, "is_uniform": True}
            
        counts = np.bincount(sequence, minlength=num_classes)
        expected = len(sequence) / num_classes
        
        chi2_stat = np.sum((counts - expected)**2 / expected)
        p_value = 1.0 - stats.chi2.cdf(chi2_stat, df=num_classes-1)
        
        return {
            "chi2": float(chi2_stat),
            "p_value": float(p_value),
            "is_uniform": p_value > 0.05
        }

    @staticmethod
    def bit_frequency_test(bitwise_seq: np.ndarray) -> Dict:
        """
        Tests if Big/Small bits are 50/50.
        """
        if len(bitwise_seq) < 10:
            return {"p_value": 1.0, "is_uniform": True}
        ones = np.sum(bitwise_seq)
        zeros = len(bitwise_seq) - ones
        
        n = len(bitwise_seq)
        expected = n / 2.0
        chi2_stat = ((ones - expected)**2 / expected) + ((zeros - expected)**2 / expected)
        p_value = 1.0 - stats.chi2.cdf(chi2_stat, df=1)
        
        return {
            "p_value": float(p_value),
            "is_uniform": p_value > 0.05
        }

    @staticmethod
    def runs_test(bitwise_seq: np.ndarray) -> Dict:
        """
        Tests whether sequences of repeated bits happen too often or too rarely.
        """
        if len(bitwise_seq) < 2:
            return {"z_stat": 0.0, "p_value": 1.0}
            
        runs = 1
        for i in range(1, len(bitwise_seq)):
            if bitwise_seq[i] != bitwise_seq[i-1]:
                runs += 1
                
        ones = np.sum(bitwise_seq)
        zeros = len(bitwise_seq) - ones
        n = len(bitwise_seq)
        
        expected_runs = ((2.0 * ones * zeros) / n) + 1
        variance = (2.0 * ones * zeros * (2.0 * ones * zeros - n)) / ((n**2) * (n - 1)) if n > 1 else 1.0
        
        if variance == 0:
            return {"z_stat": 0.0, "p_value": 1.0}
            
        z_stat = (runs - expected_runs) / math.sqrt(variance)
        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
        
        return {
            "z_stat": float(z_stat),
            "p_value": float(p_value)
        }

    @staticmethod
    def shannon_entropy(sequence: np.ndarray, num_classes: int = 10) -> float:
        """
        Estimate Shannon entropy.
        """
        if len(sequence) == 0:
            return 0.0
            
        counts = np.bincount(sequence, minlength=num_classes)
        probs = counts / len(sequence)
        
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)
                
        return float(entropy)

    @staticmethod
    def kolmogorov_smirnov_test(sequence: np.ndarray) -> Dict:
        """
        Test if sequence follows uniform distribution using KS test.
        """
        if len(sequence) < 10:
            return {"statistic": 0.0, "p_value": 1.0}
            
        # Normalize to [0,1] range
        normalized = (sequence.astype(float) + 0.5) / 10.0  # assuming digits 0-9
        
        ks_stat, p_value = stats.kstest(normalized, 'uniform')
        
        return {
            "statistic": float(ks_stat),
            "p_value": float(p_value),
            "is_uniform": p_value > 0.05
        }

    @staticmethod
    def autocorrelation_test(sequence: np.ndarray, max_lag: int = 10) -> Dict:
        """
        Test for significant autocorrelation at multiple lags.
        """
        if len(sequence) < max_lag + 10:
            return {"max_acf": 0.0, "significant_lags": []}
            
        autocorrs = []
        significant_lags = []
        
        for lag in range(1, max_lag + 1):
            if len(sequence) <= lag:
                break
                
            x = sequence[:-lag].astype(float)
            y = sequence[lag:].astype(float)
            
            x_centered = x - x.mean()
            y_centered = y - y.mean()
            
            denominator = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
            if denominator > 0:
                acf = float(np.sum(x_centered * y_centered) / denominator)
                autocorrs.append(acf)
                
                # Test significance (approximate)
                se = 1.0 / math.sqrt(len(sequence))
                if abs(acf) > 1.96 * se:
                    significant_lags.append(lag)
        
        return {
            "max_acf": max(abs(ac) for ac in autocorrs) if autocorrs else 0.0,
            "significant_lags": significant_lags,
            "autocorrelations": autocorrs
        }

    @staticmethod
    def pattern_momentum_test(sequence: np.ndarray, window: int = 5) -> Dict:
        """
        Test for momentum patterns in recent history.
        """
        if len(sequence) < window * 2:
            return {"momentum_score": 0.0, "direction": "neutral"}
            
        recent = sequence[-window:]
        previous = sequence[-(window*2):-window]
        
        recent_big = np.mean(recent >= 5)
        previous_big = np.mean(previous >= 5)
        
        momentum = recent_big - previous_big
        direction = "big" if momentum > 0.1 else "small" if momentum < -0.1 else "neutral"
        
        return {
            "momentum_score": float(momentum),
            "direction": direction,
            "recent_big_ratio": float(recent_big),
            "previous_big_ratio": float(previous_big)
        }

    @staticmethod
    def cyclical_pattern_test(sequence: np.ndarray, cycle_lengths: List[int] = [3, 5, 7, 10]) -> Dict:
        """
        Test for cyclical patterns at specified lengths.
        """
        if len(sequence) < max(cycle_lengths) * 2:
            return {"strongest_cycle": None, "cycle_strength": 0.0}
            
        cycle_scores = {}
        
        for cycle_len in cycle_lengths:
            if len(sequence) < cycle_len * 2:
                continue
                
            # Calculate correlation between positions separated by cycle length
            correlations = []
            for i in range(len(sequence) - cycle_len):
                if sequence[i] == sequence[i + cycle_len]:
                    correlations.append(1)
                else:
                    correlations.append(0)
            
            if correlations:
                cycle_scores[cycle_len] = np.mean(correlations)
        
        if cycle_scores:
            strongest_cycle = max(cycle_scores, key=cycle_scores.get)
            return {
                "strongest_cycle": strongest_cycle,
                "cycle_strength": float(cycle_scores[strongest_cycle]),
                "all_cycles": cycle_scores
            }
        
        return {"strongest_cycle": None, "cycle_strength": 0.0}

    @staticmethod
    def comprehensive_analysis(sequence: np.ndarray) -> Dict:
        """
        Run comprehensive statistical analysis and return all metrics.
        """
        int_seq = sequence if sequence.dtype in [np.int32, np.int64] else sequence.astype(np.int32)
        bit_seq = np.where(int_seq >= 5, 1, 0)
        
        return {
            "chi_square": EnhancedStatisticalTests.chi_square_frequency(int_seq),
            "bit_frequency": EnhancedStatisticalTests.bit_frequency_test(bit_seq),
            "runs": EnhancedStatisticalTests.runs_test(bit_seq),
            "entropy": EnhancedStatisticalTests.shannon_entropy(int_seq),
            "ks_test": EnhancedStatisticalTests.kolmogorov_smirnov_test(int_seq),
            "autocorrelation": EnhancedStatisticalTests.autocorrelation_test(int_seq),
            "momentum": EnhancedStatisticalTests.pattern_momentum_test(int_seq),
            "cyclical": EnhancedStatisticalTests.cyclical_pattern_test(int_seq)
        }

# Maintain backward compatibility
StatisticalTests = EnhancedStatisticalTests
