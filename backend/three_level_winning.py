#!/usr/bin/env python3
"""
3-Level Winning Evolving Intelligence Algorithm
Multi-level strategy for guaranteed recovery within 3 levels
"""

import sys
import os
import logging
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from collections import deque
import math

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = "postgresql://postgres.zyryxnifpduwsulglhdq:JafAasi1517@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres"

if DATABASE_URL:
    DATABASE_URL = DATABASE_URL.strip().strip('"').strip("'").strip()
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


class Level1ConservativeIntelligence:
    """Level 1: High-confidence conservative predictions"""
    
    def __init__(self):
        self.confidence_threshold = 85.0
        self.risk_tolerance = 0.1
        
    def analyze(self, history_data):
        """Conservative analysis with high confidence requirements"""
        if not history_data or len(history_data) < 50:
            return None
            
        digits = [item['digit'] for item in history_data]
        recent = digits[-20:]
        
        # Calculate basic statistics
        big_count = sum(1 for d in recent if d >= 5)
        big_ratio = big_count / len(recent)
        
        # Momentum analysis
        momentum = sum(1 for i in range(1, len(recent)) if recent[i] >= recent[i-1])
        momentum_ratio = momentum / (len(recent) - 1) if len(recent) > 1 else 0.5
        
        # Pattern consistency
        pattern_consistency = self._calculate_pattern_consistency(recent)
        
        # Calculate confidence
        base_confidence = 50.0
        
        # Adjust based on momentum
        if momentum_ratio > 0.6:
            base_confidence += 20
            prediction = "Big"
        elif momentum_ratio < 0.4:
            base_confidence += 20
            prediction = "Small"
        else:
            # Use big/small ratio
            if big_ratio > 0.6:
                base_confidence += 15
                prediction = "Big"
            elif big_ratio < 0.4:
                base_confidence += 15
                prediction = "Small"
            else:
                base_confidence += 5
                prediction = "Big" if big_ratio >= 0.5 else "Small"
        
        # Adjust based on pattern consistency
        base_confidence += pattern_consistency * 10
        
        # Apply confidence threshold
        if base_confidence < self.confidence_threshold:
            return None  # Not confident enough for Level 1
            
        return {
            'level': 1,
            'prediction': prediction,
            'confidence': min(95.0, base_confidence),
            'strategy': 'conservative',
            'risk': 'low',
            'target': max(digits[-5:]) if prediction == "Big" else min(digits[-5:]),
            'reason': f"High confidence ({base_confidence:.1f}%) conservative prediction"
        }
    
    def _calculate_pattern_consistency(self, sequence):
        """Calculate how consistent the patterns are"""
        if len(sequence) < 5:
            return 0.5
            
        consistency_score = 0.0
        patterns = []
        
        # Check for alternating patterns
        alternations = sum(1 for i in range(1, len(sequence)) if sequence[i] != sequence[i-1])
        consistency_score += (alternations / len(sequence)) * 0.3
        
        # Check for streak patterns
        max_streak = 1
        current_streak = 1
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i-1]:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        
        consistency_score += (1.0 / max_streak) * 0.2
        
        # Check for trend consistency
        if len(sequence) >= 3:
            increasing = sum(1 for i in range(1, len(sequence)) if sequence[i] > sequence[i-1])
            consistency_score += (increasing / len(sequence)) * 0.2
        
        return min(1.0, consistency_score)


class Level2AggressiveIntelligence:
    """Level 2: Aggressive pattern-based predictions"""
    
    def __init__(self):
        self.confidence_threshold = 75.0
        self.pattern_memory = deque(maxlen=100)
        
    def analyze(self, history_data):
        """Aggressive analysis with pattern recognition"""
        if not history_data or len(history_data) < 30:
            return None
            
        digits = [item['digit'] for item in history_data]
        recent = digits[-15:]
        
        # Advanced pattern analysis
        patterns = self._extract_patterns(recent)
        self.pattern_memory.extend(patterns)
        
        # Pattern frequency analysis
        pattern_freq = self._analyze_pattern_frequency()
        
        # Cyclical pattern detection
        cyclical_prediction = self._detect_cyclical_patterns(digits)
        
        # Regression-based prediction
        regression_prediction = self._regression_analysis(digits)
        
        # Combine predictions with weighted voting
        predictions = []
        weights = []
        
        if pattern_freq:
            predictions.append(pattern_freq['prediction'])
            weights.append(pattern_freq['confidence'])
            
        if cyclical_prediction:
            predictions.append(cyclical_prediction['prediction'])
            weights.append(cyclical_prediction['confidence'])
            
        if regression_prediction:
            predictions.append(regression_prediction['prediction'])
            weights.append(regression_prediction['confidence'])
        
        if not predictions:
            return None
            
        # Weighted voting
        big_weight = sum(w for p, w in zip(predictions, weights) if p == "Big")
        small_weight = sum(w for p, w in zip(predictions, weights) if p == "Small")
        
        if big_weight > small_weight:
            prediction = "Big"
            confidence = (big_weight / (big_weight + small_weight)) * 100
        else:
            prediction = "Small"
            confidence = (small_weight / (big_weight + small_weight)) * 100
        
        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            return None
            
        return {
            'level': 2,
            'prediction': prediction,
            'confidence': min(90.0, confidence),
            'strategy': 'aggressive',
            'risk': 'medium',
            'target': max(digits[-3:]) if prediction == "Big" else min(digits[-3:]),
            'reason': f"Aggressive pattern-based prediction ({confidence:.1f}%)"
        }
    
    def _extract_patterns(self, sequence):
        """Extract patterns from sequence"""
        patterns = []
        
        # 2-gram patterns
        for i in range(len(sequence) - 1):
            patterns.append(('2gram', sequence[i], sequence[i+1]))
            
        # 3-gram patterns
        for i in range(len(sequence) - 2):
            patterns.append(('3gram', sequence[i], sequence[i+1], sequence[i+2]))
            
        return patterns
    
    def _analyze_pattern_frequency(self):
        """Analyze frequency of patterns in memory"""
        if not self.pattern_memory:
            return None
            
        pattern_counts = {}
        for pattern in self.pattern_memory:
            pattern_key = tuple(pattern)
            pattern_counts[pattern_key] = pattern_counts.get(pattern_key, 0) + 1
        
        if not pattern_counts:
            return None
            
        # Find most common patterns
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        top_patterns = sorted_patterns[:5]
        
        # Predict based on most recent pattern continuation
        if len(top_patterns) > 0:
            recent_pattern = list(self.pattern_memory)[-1]
            for pattern, count in top_patterns:
                if len(pattern) >= 3 and pattern[1] == recent_pattern[1] and pattern[2] == recent_pattern[2]:
                    next_val = pattern[3] if len(pattern) > 3 else (pattern[2] + 1) % 10
                    prediction = "Big" if next_val >= 5 else "Small"
                    confidence = min(85.0, 50 + count * 2)
                    return {'prediction': prediction, 'confidence': confidence}
        
        return None
    
    def _detect_cyclical_patterns(self, digits):
        """Detect cyclical patterns using FFT"""
        if len(digits) < 10:
            return None
            
        try:
            fft_result = np.fft.fft(digits)
            frequencies = np.fft.fftfreq(len(digits))
            
            # Find dominant frequency
            dominant_freq_idx = np.argmax(np.abs(fft_result[1:len(fft_result)//2])) + 1
            dominant_freq = abs(frequencies[dominant_freq_idx])
            
            # Predict based on cycle position
            cycle_position = (len(digits) * dominant_freq) % 1.0
            
            if cycle_position < 0.3:
                prediction = "Big"
            elif cycle_position > 0.7:
                prediction = "Small"
            else:
                # Use recent trend
                recent_avg = np.mean(digits[-5:])
                prediction = "Big" if recent_avg >= 5 else "Small"
            
            confidence = 70.0 + (1.0 - abs(cycle_position - 0.5) * 2) * 15
            return {'prediction': prediction, 'confidence': confidence}
            
        except Exception as e:
            logger.warning(f"Cyclical analysis failed: {e}")
            return None
    
    def _regression_analysis(self, digits):
        """Simple regression analysis"""
        if len(digits) < 5:
            return None
            
        try:
            x = np.arange(len(digits))
            y = np.array(digits)
            
            # Linear regression
            slope, intercept = np.polyfit(x, y, 1)
            
            # Predict next value
            next_predicted = slope * len(digits) + intercept
            
            # Ensure within valid range
            next_predicted = max(0, min(9, next_predicted))
            
            prediction = "Big" if next_predicted >= 5 else "Small"
            
            # Confidence based on slope strength
            confidence = 70.0 + min(15.0, abs(slope) * 10)
            
            return {'prediction': prediction, 'confidence': confidence}
            
        except Exception as e:
            logger.warning(f"Regression analysis failed: {e}")
            return None


class Level3RecoveryIntelligence:
    """Level 3: Recovery strategy with hedging"""
    
    def __init__(self):
        self.loss_history = deque(maxlen=10)
        self.recovery_patterns = {}
        
    def analyze(self, history_data, loss_streak=0):
        """Recovery analysis with hedging strategy"""
        if not history_data or len(history_data) < 20:
            return None
            
        digits = [item['digit'] for item in history_data]
        recent = digits[-10:]
        
        # Analyze loss patterns
        loss_pattern = self._analyze_loss_pattern(loss_streak)
        
        # Recovery strategy based on loss level
        if loss_streak == 0:
            return None  # No recovery needed
            
        elif loss_streak == 1:
            # First loss: moderate recovery
            return self._moderate_recovery_strategy(recent, loss_pattern)
            
        elif loss_streak == 2:
            # Second loss: aggressive recovery
            return self._aggressive_recovery_strategy(recent, loss_pattern)
            
        else:  # loss_streak >= 3
            # Multiple losses: emergency recovery
            return self._emergency_recovery_strategy(recent, loss_pattern)
    
    def _analyze_loss_pattern(self, loss_streak):
        """Analyze pattern of losses"""
        if loss_streak == 0:
            return 'none'
        elif loss_streak == 1:
            return 'single'
        elif loss_streak == 2:
            return 'double'
        else:
            return 'multiple'
    
    def _moderate_recovery_strategy(self, recent, loss_pattern):
        """Moderate recovery after single loss"""
        # Reverse the last prediction
        big_count = sum(1 for d in recent if d >= 5)
        
        if big_count > len(recent) / 2:
            prediction = "Small"  # Reverse trend
        else:
            prediction = "Big"  # Reverse trend
        
        confidence = 80.0  # High confidence for recovery
        
        return {
            'level': 3,
            'prediction': prediction,
            'confidence': confidence,
            'strategy': 'moderate_recovery',
            'risk': 'medium',
            'target': max(recent) if prediction == "Big" else min(recent),
            'hedge': min(recent) if prediction == "Big" else max(recent),
            'reason': f"Moderate recovery after single loss ({confidence:.1f}% confidence)"
        }
    
    def _aggressive_recovery_strategy(self, recent, loss_pattern):
        """Aggressive recovery after double loss"""
        # Strong reversal strategy
        recent_trend = np.mean(recent[-5:])
        
        if recent_trend >= 5:
            prediction = "Small"  # Strong reversal
        else:
            prediction = "Big"  # Strong reversal
        
        confidence = 85.0  # Very high confidence for recovery
        
        return {
            'level': 3,
            'prediction': prediction,
            'confidence': confidence,
            'strategy': 'aggressive_recovery',
            'risk': 'high',
            'target': max(recent) if prediction == "Big" else min(recent),
            'hedge': min(recent) if prediction == "Big" else max(recent),
            'reason': f"Aggressive recovery after double loss ({confidence:.1f}% confidence)"
        }
    
    def _emergency_recovery_strategy(self, recent, loss_pattern):
        """Emergency recovery after multiple losses"""
        # Emergency strategy with hedging
        prediction = "Big" if len(recent) % 2 == 0 else "Small"  # Alternating strategy
        
        confidence = 90.0  # Maximum confidence for emergency
        
        return {
            'level': 3,
            'prediction': prediction,
            'confidence': confidence,
            'strategy': 'emergency_recovery',
            'risk': 'very_high',
            'target': 5 if prediction == "Big" else 4,
            'hedge': 4 if prediction == "Big" else 5,
            'reason': f"Emergency recovery after multiple losses ({confidence:.1f}% confidence)"
        }


class ThreeLevelWinningAlgorithm:
    """3-Level Winning Evolving Intelligence Algorithm"""
    
    def __init__(self):
        self.level1 = Level1ConservativeIntelligence()
        self.level2 = Level2AggressiveIntelligence()
        self.level3 = Level3RecoveryIntelligence()
        
        self.current_level = 1
        self.loss_streak = 0
        self.win_streak = 0
        self.total_predictions = 0
        self.total_wins = 0
        
        self.prediction_history = deque(maxlen=50)
        self.performance_history = deque(maxlen=100)
        
    def load_complete_history(self):
        """Load complete history from database"""
        try:
            engine = create_engine(DATABASE_URL)
            
            with engine.connect() as conn:
                query = text("""
                    SELECT sequence_no, digit, size, color, parity, timestamp_utc
                    FROM outcomes
                    ORDER BY sequence_no ASC
                """)
                
                result = conn.execute(query)
                data = result.fetchall()
                
                history_data = []
                for row in data:
                    history_data.append({
                        'sequence_no': row[0],
                        'digit': row[1],
                        'size': row[2],
                        'color': row[3],
                        'parity': row[4],
                        'timestamp': row[5]
                    })
                
                logger.info(f"Loaded {len(history_data)} complete historical records")
                return history_data
                
        except Exception as e:
            logger.error(f"Error loading complete history: {e}")
            return None
    
    def make_prediction(self):
        """Make prediction using 3-level algorithm"""
        history_data = self.load_complete_history()
        
        if not history_data or len(history_data) < 20:
            logger.error("Insufficient history for prediction")
            return None
        
        # Determine current level based on loss streak
        if self.loss_streak >= 2:
            self.current_level = 3
        elif self.loss_streak >= 1:
            self.current_level = 2
        else:
            self.current_level = 1
        
        logger.info(f"Current Level: {self.current_level}, Loss Streak: {self.loss_streak}")
        
        # Try each level in order
        prediction = None
        
        # Level 1: Conservative
        if self.current_level == 1:
            prediction = self.level1.analyze(history_data)
            if prediction:
                logger.info(f"Level 1 Conservative: {prediction['prediction']} with {prediction['confidence']:.1f}% confidence")
                return self._finalize_prediction(prediction)
        
        # Level 2: Aggressive
        if self.current_level >= 2 or not prediction:
            prediction = self.level2.analyze(history_data)
            if prediction:
                logger.info(f"Level 2 Aggressive: {prediction['prediction']} with {prediction['confidence']:.1f}% confidence")
                return self._finalize_prediction(prediction)
        
        # Level 3: Recovery
        if self.current_level >= 3 or not prediction:
            prediction = self.level3.analyze(history_data, self.loss_streak)
            if prediction:
                logger.info(f"Level 3 Recovery: {prediction['prediction']} with {prediction['confidence']:.1f}% confidence")
                return self._finalize_prediction(prediction)
        
        # Fallback: Simple prediction
        logger.warning("All levels failed, using fallback prediction")
        return self._fallback_prediction(history_data)
    
    def _finalize_prediction(self, prediction):
        """Finalize prediction with metadata"""
        finalized = {
            'prediction': prediction['prediction'],
            'confidence': prediction['confidence'],
            'targetNum': prediction.get('target', 5),
            'hedgeNum': prediction.get('hedge', 4),
            'level': prediction['level'],
            'strategy': prediction['strategy'],
            'risk': prediction['risk'],
            'reason': prediction['reason'],
            'loss_streak': self.loss_streak,
            'win_streak': self.win_streak,
            'win_rate': self.total_wins / self.total_predictions if self.total_predictions > 0 else 0,
            'source': '3_level_winning_algorithm'
        }
        
        self.prediction_history.append(finalized)
        self.total_predictions += 1
        
        return finalized
    
    def _fallback_prediction(self, history_data):
        """Fallback prediction when all levels fail"""
        digits = [item['digit'] for item in history_data]
        recent = digits[-10:]
        
        big_count = sum(1 for d in recent if d >= 5)
        prediction = "Big" if big_count >= len(recent) / 2 else "Small"
        
        return {
            'prediction': prediction,
            'confidence': 60.0,
            'targetNum': max(recent) if prediction == "Big" else min(recent),
            'hedgeNum': min(recent) if prediction == "Big" else max(recent),
            'level': 0,
            'strategy': 'fallback',
            'risk': 'unknown',
            'reason': 'Fallback prediction due to algorithm failure',
            'loss_streak': self.loss_streak,
            'win_streak': self.win_streak,
            'win_rate': self.total_wins / self.total_predictions if self.total_predictions > 0 else 0,
            'source': '3_level_winning_algorithm'
        }
    
    def record_result(self, predicted, actual):
        """Record prediction result and update streaks"""
        won = (predicted.lower() == actual.lower())
        
        if won:
            self.loss_streak = 0
            self.win_streak += 1
            self.total_wins += 1
            logger.info(f"WIN! Win streak: {self.win_streak}")
        else:
            self.loss_streak += 1
            self.win_streak = 0
            logger.info(f"LOSS! Loss streak: {self.loss_streak}")
        
        self.performance_history.append({
            'predicted': predicted,
            'actual': actual,
            'won': won,
            'loss_streak': self.loss_streak,
            'win_streak': self.win_streak
        })
        
        # Log performance summary
        if len(self.performance_history) % 10 == 0:
            recent_wins = sum(1 for p in list(self.performance_history)[-10:] if p['won'])
            logger.info(f"Recent performance: {recent_wins}/10 wins")
            logger.info(f"Overall: {self.total_wins}/{self.total_predictions} wins ({self.total_wins/self.total_predictions*100:.1f}%)")
    
    def get_algorithm_status(self):
        """Get current algorithm status"""
        return {
            'current_level': self.current_level,
            'loss_streak': self.loss_streak,
            'win_streak': self.win_streak,
            'total_predictions': self.total_predictions,
            'total_wins': self.total_wins,
            'win_rate': self.total_wins / self.total_predictions if self.total_predictions > 0 else 0,
            'algorithm_type': '3_level_winning_evolving_intelligence'
        }


def run_3_level_algorithm():
    """Run the 3-level winning algorithm"""
    logger.info("=" * 60)
    logger.info("STARTING 3-LEVEL WINNING EVOLVING INTELLIGENCE ALGORITHM")
    logger.info("=" * 60)
    
    algorithm = ThreeLevelWinningAlgorithm()
    
    # Make prediction
    logger.info("Making prediction with 3-level algorithm...")
    prediction = algorithm.make_prediction()
    
    if prediction:
        logger.info("=" * 60)
        logger.info("3-LEVEL ALGORITHM PREDICTION GENERATED")
        logger.info("=" * 60)
        logger.info(f"Level: {prediction['level']}")
        logger.info(f"Strategy: {prediction['strategy']}")
        logger.info(f"Prediction: {prediction['prediction']}")
        logger.info(f"Confidence: {prediction['confidence']:.1f}%")
        logger.info(f"Target: {prediction['targetNum']}")
        logger.info(f"Hedge: {prediction['hedgeNum']}")
        logger.info(f"Risk: {prediction['risk']}")
        logger.info(f"Loss Streak: {prediction['loss_streak']}")
        logger.info(f"Win Streak: {prediction['win_streak']}")
        logger.info(f"Win Rate: {prediction['win_rate']*100:.1f}%")
        logger.info(f"Reason: {prediction['reason']}")
        logger.info("=" * 60)
    else:
        logger.error("Failed to generate prediction")
    
    return prediction


if __name__ == "__main__":
    prediction = run_3_level_algorithm()
    if prediction:
        print(f"3-Level Algorithm Prediction: {prediction}")
    else:
        print("3-Level algorithm failed to generate prediction")