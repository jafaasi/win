import numpy as np
import math
from typing import Dict, List
import scipy.stats as stats

class StatisticalTests:
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
