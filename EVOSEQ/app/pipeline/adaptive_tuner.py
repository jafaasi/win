import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import math

class AdaptiveHyperparameterTuner:
    """
    Adaptive hyperparameter tuning for dynamic model optimization.
    Adjusts parameters based on recent performance and regime changes.
    """
    
    def __init__(self, initial_params: Dict = None):
        self.default_params = {
            "temperature": 1.1,
            "learning_rate": 0.001,
            "ensemble_weight_decay": 0.95,
            "regime_sensitivity": 0.5,
            "volatility_threshold": 2.5,
            "momentum_threshold": 0.3,
            "confidence_base": 85.0,
            "pattern_weight": 0.3
        }
        
        self.current_params = self.default_params.copy()
        if initial_params:
            self.current_params.update(initial_params)
        
        self.performance_history = deque(maxlen=100)
        self.param_history = deque(maxlen=50)
        self.best_params = self.current_params.copy()
        self.best_performance = 0.0
        
        # Adaptive learning
        self.improvement_rate = 0.0
        self.stagnation_counter = 0
        self.exploration_phase = False
        
    def evaluate_performance(self, metrics: Dict) -> float:
        """
        Calculate composite performance score from multiple metrics.
        Higher is better.
        """
        # Weight different metrics
        weights = {
            "accuracy": 0.4,
            "calibration": 0.2,
            "stability": 0.2,
            "predictive_score": 0.1,
            "null_advantage": 0.1
        }
        
        score = 0.0
        for metric, weight in weights.items():
            value = metrics.get(metric, 0.5)
            # Normalize to 0-1 range
            if metric == "accuracy":
                normalized = value / 100.0
            elif metric == "calibration":
                normalized = value
            elif metric == "stability":
                normalized = value
            elif metric == "predictive_score":
                normalized = value
            elif metric == "null_advantage":
                normalized = min(1.0, value * 10)  # scale up advantage
            else:
                normalized = 0.5
            
            score += weight * normalized
        
        return score
    
    def update_performance(self, metrics: Dict) -> None:
        """Update performance history and trigger parameter adjustments."""
        performance = self.evaluate_performance(metrics)
        self.performance_history.append(performance)
        
        # Calculate improvement rate
        if len(self.performance_history) >= 10:
            recent_avg = np.mean(list(self.performance_history)[-10:])
            earlier_avg = np.mean(list(self.performance_history)[-20:-10]) if len(self.performance_history) >= 20 else recent_avg
            self.improvement_rate = (recent_avg - earlier_avg) / (earlier_avg + 0.001)
        
        # Check for stagnation
        if len(self.performance_history) >= 20:
            recent_variance = np.var(list(self.performance_history)[-20:])
            if recent_variance < 0.001:  # Very low variance = stagnation
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
        
        # Update best parameters
        if performance > self.best_performance:
            self.best_performance = performance
            self.best_params = self.current_params.copy()
            self.stagnation_counter = 0
        
        # Trigger adaptive tuning
        self.adapt_parameters(metrics)
    
    def adapt_parameters(self, metrics: Dict) -> None:
        """Adapt parameters based on performance and regime."""
        if len(self.performance_history) < 5:
            return
        
        recent_performance = np.mean(list(self.performance_history)[-5:])
        
        # Enter exploration phase if stagnating
        if self.stagnation_counter > 3:
            self.exploration_phase = True
            self._explore_parameters()
        elif self.exploration_phase and recent_performance > self.best_performance * 0.95:
            self.exploration_phase = False
        
        # Regime-based adaptation
        drift_level = metrics.get("drift_level", "STABLE")
        volatility = metrics.get("volatility", 2.0)
        momentum = metrics.get("momentum_score", 0.0)
        
        if drift_level in ["HIGH_VOLATILITY", "MEAN_SHIFT"]:
            # Increase temperature for uncertainty
            self.current_params["temperature"] = min(1.5, self.current_params["temperature"] * 1.05)
            # Reduce learning rate for stability
            self.current_params["learning_rate"] = max(0.0001, self.current_params["learning_rate"] * 0.95)
        elif drift_level == "STABLE" and recent_performance > 0.7:
            # Optimize for stable conditions
            self.current_params["temperature"] = max(0.9, self.current_params["temperature"] * 0.98)
            self.current_params["learning_rate"] = min(0.005, self.current_params["learning_rate"] * 1.02)
        
        # Momentum-based adaptation
        if abs(momentum) > self.current_params["momentum_threshold"]:
            # Increase pattern weight during strong momentum
            self.current_params["pattern_weight"] = min(0.5, self.current_params["pattern_weight"] * 1.1)
        else:
            # Reduce pattern weight during equilibrium
            self.current_params["pattern_weight"] = max(0.1, self.current_params["pattern_weight"] * 0.95)
        
        # Volatility-based adaptation
        if volatility > self.current_params["volatility_threshold"]:
            # Increase confidence base during high volatility (more conservative)
            self.current_params["confidence_base"] = min(95.0, self.current_params["confidence_base"] * 1.02)
        else:
            # Reduce confidence base during low volatility (more aggressive)
            self.current_params["confidence_base"] = max(80.0, self.current_params["confidence_base"] * 0.98)
        
        # Record parameter history
        self.param_history.append(self.current_params.copy())
    
    def _explore_parameters(self) -> None:
        """Explore new parameter combinations during stagnation."""
        # Random perturbation within reasonable bounds
        perturbation_scale = 0.1
        
        for param in ["temperature", "learning_rate", "ensemble_weight_decay", "pattern_weight"]:
            if param in self.current_params:
                perturbation = np.random.uniform(-perturbation_scale, perturbation_scale)
                new_value = self.current_params[param] * (1 + perturbation)
                
                # Apply bounds
                if param == "temperature":
                    new_value = max(0.8, min(2.0, new_value))
                elif param == "learning_rate":
                    new_value = max(0.0001, min(0.01, new_value))
                elif param == "ensemble_weight_decay":
                    new_value = max(0.85, min(0.99, new_value))
                elif param == "pattern_weight":
                    new_value = max(0.05, min(0.6, new_value))
                
                self.current_params[param] = new_value
    
    def get_current_params(self) -> Dict:
        """Get current adapted parameters."""
        return self.current_params.copy()
    
    def get_best_params(self) -> Dict:
        """Get best performing parameters."""
        return self.best_params.copy()
    
    def reset_to_best(self) -> None:
        """Reset to best performing parameters."""
        self.current_params = self.best_params.copy()
        self.stagnation_counter = 0
        self.exploration_phase = False
    
    def get_param_sensitivity(self) -> Dict[str, float]:
        """Calculate sensitivity of performance to parameter changes."""
        if len(self.param_history) < 10:
            return {}
        
        sensitivity = {}
        param_names = list(self.param_history[0].keys())
        
        for param in param_names:
            values = [state[param] for state in self.param_history]
            performances = list(self.performance_history)[-len(values):]
            
            if len(values) == len(performances) and len(values) > 5:
                # Calculate correlation between parameter values and performance
                correlation = np.corrcoef(values, performances)[0, 1]
                if not np.isnan(correlation):
                    sensitivity[param] = abs(correlation)
        
        return sensitivity
    
    def get_tuning_status(self) -> Dict:
        """Get current tuning status and statistics."""
        return {
            "current_performance": float(np.mean(list(self.performance_history)[-5:]) if self.performance_history else 0.0),
            "best_performance": float(self.best_performance),
            "improvement_rate": float(self.improvement_rate),
            "stagnation_counter": self.stagnation_counter,
            "exploration_phase": self.exploration_phase,
            "performance_samples": len(self.performance_history),
            "param_adjustments": len(self.param_history),
            "sensitivity": self.get_param_sensitivity()
        }
    
    def apply_params_to_models(self, predictor, transformer, mamba) -> None:
        """Apply current parameters to the models."""
        params = self.current_params
        
        # Apply temperature to models
        if hasattr(transformer, 'temperature'):
            transformer.temperature = params["temperature"]
        if hasattr(mamba, 'temperature'):
            mamba.temperature = params["temperature"]
        
        # Apply learning rate
        if hasattr(transformer, 'lr'):
            transformer.lr = params["learning_rate"]
            # Update optimizer
            for param_group in transformer.optimizer.param_groups:
                param_group['lr'] = params["learning_rate"]
        
        if hasattr(mamba, 'lr'):
            mamba.lr = params["learning_rate"]
            for param_group in mamba.optimizer.param_groups:
                param_group['lr'] = params["learning_rate"]
        
        # Apply ensemble weight decay to predictor
        if hasattr(predictor, 'decay_factor'):
            predictor.decay_factor = params["ensemble_weight_decay"]
    
    def optimize_for_regime(self, regime: str) -> Dict:
        """Get optimized parameters for specific regime."""
        regime_params = self.default_params.copy()
        
        if regime in ["STRONG_BIG_MOMENTUM", "STRONG_SMALL_MOMENTUM"]:
            regime_params["pattern_weight"] = 0.4
            regime_params["momentum_threshold"] = 0.2
            regime_params["confidence_base"] = 90.0
        elif regime == "HIGH_VOLATILITY":
            regime_params["temperature"] = 1.3
            regime_params["learning_rate"] = 0.0005
            regime_params["confidence_base"] = 88.0
        elif regime == "LOW_VOLATILITY":
            regime_params["temperature"] = 0.95
            regime_params["learning_rate"] = 0.002
            regime_params["confidence_base"] = 92.0
        elif regime == "EQUILIBRIUM":
            regime_params["pattern_weight"] = 0.2
            regime_params["temperature"] = 1.1
            regime_params["confidence_base"] = 85.0
        
        return regime_params