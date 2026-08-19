#!/usr/bin/env python3
"""
Extraordinary Intelligence Engine
Uses complete history database with advanced AI for superior predictions
"""

import sys
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from collections import defaultdict, deque
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


class AdvancedFeatureEngineer:
    """Extract extraordinary features from complete history"""
    
    def __init__(self):
        self.pattern_cache = {}
        self.temporal_features = {}
        
    def extract_comprehensive_features(self, history_data):
        """Extract 100+ advanced features from complete history"""
        features = {}
        
        if not history_data or len(history_data) < 100:
            return features
            
        digits = [item['digit'] for item in history_data]
        sizes = [item['size'] for item in history_data]
        colors = [item['color'] for item in history_data]
        
        # Basic statistics
        features['mean_digit'] = np.mean(digits)
        features['std_digit'] = np.std(digits)
        features['mean_size'] = np.mean(sizes)
        features['std_size'] = np.std(sizes)
        
        # Advanced temporal features
        for window in [10, 20, 50, 100, 200, 500]:
            if len(digits) >= window:
                window_data = digits[-window:]
                features[f'mean_{window}'] = np.mean(window_data)
                features[f'std_{window}'] = np.std(window_data)
                features[f'min_{window}'] = np.min(window_data)
                features[f'max_{window}'] = np.max(window_data)
                features[f'range_{window}'] = np.max(window_data) - np.min(window_data)
                
                # Trend analysis
                if len(window_data) >= 3:
                    features[f'trend_{window}'] = np.polyfit(range(len(window_data)), window_data, 1)[0]
                    
        # Pattern frequency analysis
        for length in [2, 3, 4, 5]:
            patterns = self._extract_ngrams(digits, length)
            features.update(self._pattern_frequency_features(patterns, length))
            
        # Cyclical pattern detection
        features.update(self._detect_cyclical_patterns(digits))
        
        # Momentum indicators
        features.update(self._calculate_momentum_indicators(digits))
        
        # Volatility features
        features.update(self._calculate_volatility_features(digits))
        
        # Entropy and complexity
        features.update(self._calculate_entropy_features(digits))
        
        # Autocorrelation features
        features.update(self._calculate_autocorrelation_features(digits))
        
        # Distribution features
        features.update(self._calculate_distribution_features(digits))
        
        # Sequence analysis
        features.update(self._analyze_sequences(digits, sizes, colors))
        
        # Rare pattern detection
        features.update(self._detect_rare_patterns(digits))
        
        return features
    
    def _extract_ngrams(self, sequence, n):
        """Extract n-grams from sequence"""
        return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]
    
    def _pattern_frequency_features(self, patterns, length):
        """Calculate pattern frequency features"""
        features = {}
        if not patterns:
            return features
            
        pattern_counts = defaultdict(int)
        for pattern in patterns:
            pattern_counts[pattern] += 1
            
        total = len(patterns)
        features[f'unique_patterns_{length}'] = len(pattern_counts)
        features[f'pattern_diversity_{length}'] = len(pattern_counts) / total if total > 0 else 0
        
        # Most common pattern frequency
        if pattern_counts:
            most_common = max(pattern_counts.values())
            features[f'most_common_freq_{length}'] = most_common / total
            
        return features
    
    def _detect_cyclical_patterns(self, sequence):
        """Detect cyclical patterns using FFT"""
        features = {}
        if len(sequence) < 10:
            return features
            
        try:
            fft_result = np.fft.fft(sequence)
            frequencies = np.fft.fftfreq(len(sequence))
            
            # Dominant frequency
            dominant_freq_idx = np.argmax(np.abs(fft_result[1:len(fft_result)//2])) + 1
            features['dominant_frequency'] = abs(frequencies[dominant_freq_idx])
            
            # Spectral entropy
            power_spectrum = np.abs(fft_result) ** 2
            power_spectrum = power_spectrum / np.sum(power_spectrum)
            spectral_entropy = -np.sum(power_spectrum * np.log2(power_spectrum + 1e-10))
            features['spectral_entropy'] = spectral_entropy
            
        except Exception as e:
            logger.warning(f"FFT analysis failed: {e}")
            
        return features
    
    def _calculate_momentum_indicators(self, sequence):
        """Calculate momentum indicators"""
        features = {}
        if len(sequence) < 5:
            return features
            
        # RSI-like indicator
        gains = []
        losses = []
        for i in range(1, len(sequence)):
            change = sequence[i] - sequence[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
                
        if gains and losses:
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
            
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                features['rsi'] = 100 - (100 / (1 + rs))
            else:
                features['rsi'] = 100
                
        # Moving averages
        for window in [5, 10, 20]:
            if len(sequence) >= window:
                ma = np.mean(sequence[-window:])
                features[f'ma_{window}'] = ma
                features[f'price_above_ma_{window}'] = 1 if sequence[-1] > ma else 0
                
        return features
    
    def _calculate_volatility_features(self, sequence):
        """Calculate volatility features"""
        features = {}
        if len(sequence) < 2:
            return features
            
        returns = np.diff(sequence)
        
        features['volatility_5'] = np.std(returns[-5:]) if len(returns) >= 5 else 0
        features['volatility_10'] = np.std(returns[-10:]) if len(returns) >= 10 else 0
        features['volatility_20'] = np.std(returns[-20:]) if len(returns) >= 20 else 0
        
        # Parkinson volatility estimator
        if len(sequence) >= 2:
            high = max(sequence[-20:]) if len(sequence) >= 20 else max(sequence)
            low = min(sequence[-20:]) if len(sequence) >= 20 else min(sequence)
            features['parkinson_vol'] = np.sqrt((np.log(high/low)**2) / (4 * np.log(2)))
            
        return features
    
    def _calculate_entropy_features(self, sequence):
        """Calculate entropy and complexity features"""
        features = {}
        
        # Shannon entropy
        value_counts = defaultdict(int)
        for val in sequence:
            value_counts[val] += 1
            
        total = len(sequence)
        entropy = 0
        for count in value_counts.values():
            probability = count / total
            entropy -= probability * np.log2(probability + 1e-10)
            
        features['shannon_entropy'] = entropy
        features['max_entropy'] = np.log2(len(value_counts)) if value_counts else 0
        features['entropy_ratio'] = entropy / features['max_entropy'] if features['max_entropy'] > 0 else 0
        
        # Lempel-Ziv complexity (simplified)
        features['lz_complexity'] = self._calculate_lz_complexity(sequence)
        
        return features
    
    def _calculate_lz_complexity(self, sequence):
        """Calculate Lempel-Ziv complexity"""
        n = len(sequence)
        complexity = 1
        i = 0
        
        while i < n:
            j = i + 1
            while j <= n:
                subsequence = sequence[i:j]
                if subsequence in sequence[:i]:
                    j += 1
                else:
                    complexity += 1
                    break
            i = j
            
        return complexity / n if n > 0 else 0
    
    def _calculate_autocorrelation_features(self, sequence):
        """Calculate autocorrelation features"""
        features = {}
        if len(sequence) < 10:
            return features
            
        for lag in [1, 2, 3, 5, 10]:
            if len(sequence) > lag:
                autocorr = np.corrcoef(sequence[:-lag], sequence[lag:])[0, 1]
                features[f'autocorr_{lag}'] = autocorr if not np.isnan(autocorr) else 0
                
        return features
    
    def _calculate_distribution_features(self, sequence):
        """Calculate distribution features"""
        features = {}
        
        if len(sequence) < 10:
            return features
            
        # Skewness and kurtosis
        features['skewness'] = self._calculate_skewness(sequence)
        features['kurtosis'] = self._calculate_kurtosis(sequence)
        
        # Percentile-based features
        features['p25'] = np.percentile(sequence, 25)
        features['p50'] = np.percentile(sequence, 50)
        features['p75'] = np.percentile(sequence, 75)
        features['iqr'] = features['p75'] - features['p25']
        
        return features
    
    def _calculate_skewness(self, data):
        """Calculate skewness"""
        if len(data) < 3:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        n = len(data)
        skew = (n / ((n-1) * (n-2))) * np.sum(((data - mean) / std) ** 3)
        return skew
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis"""
        if len(data) < 4:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        n = len(data)
        kurt = (n * (n+1) / ((n-1) * (n-2) * (n-3))) * np.sum(((data - mean) / std) ** 4) - (3 * (n-1)**2) / ((n-2) * (n-3))
        return kurt
    
    def _analyze_sequences(self, digits, sizes, colors):
        """Analyze sequences and patterns"""
        features = {}
        
        # Consecutive same digits
        max_consecutive = 1
        current_consecutive = 1
        for i in range(1, len(digits)):
            if digits[i] == digits[i-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        features['max_consecutive'] = max_consecutive
        
        # Alternating patterns
        alternations = sum(1 for i in range(1, len(digits)) if digits[i] != digits[i-1])
        features['alternation_rate'] = alternations / len(digits) if len(digits) > 1 else 0
        
        # Size patterns
        big_streaks = self._find_streaks(sizes, 1)
        small_streaks = self._find_streaks(sizes, 0)
        
        features['max_big_streak'] = max(big_streaks) if big_streaks else 0
        features['max_small_streak'] = max(small_streaks) if small_streaks else 0
        features['avg_big_streak'] = np.mean(big_streaks) if big_streaks else 0
        features['avg_small_streak'] = np.mean(small_streaks) if small_streaks else 0
        
        return features
    
    def _find_streaks(self, sequence, value):
        """Find streaks of a specific value"""
        streaks = []
        current_streak = 0
        
        for item in sequence:
            if item == value:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
                
        if current_streak > 0:
            streaks.append(current_streak)
            
        return streaks
    
    def _detect_rare_patterns(self, sequence):
        """Detect rare and unusual patterns"""
        features = {}
        
        if len(sequence) < 20:
            return features
            
        # Detect increasing sequences
        increasing_sequences = 0
        decreasing_sequences = 0
        
        for i in range(len(sequence) - 3):
            if sequence[i] < sequence[i+1] < sequence[i+2] < sequence[i+3]:
                increasing_sequences += 1
            elif sequence[i] > sequence[i+1] > sequence[i+2] > sequence[i+3]:
                decreasing_sequences += 1
                
        features['increasing_sequences'] = increasing_sequences
        features['decreasing_sequences'] = decreasing_sequences
        
        # Detect palindromic patterns
        palindromes = 0
        for i in range(len(sequence) - 4):
            if sequence[i:i+3] == sequence[i+2:i+5][::-1]:
                palindromes += 1
                
        features['palindromic_patterns'] = palindromes
        
        return features


class ExtraordinaryDeepLearning(nn.Module):
    """Extraordinary deep learning model for pattern recognition"""
    
    def __init__(self, input_size, hidden_size=256, num_layers=4, dropout=0.3):
        super(ExtraordinaryDeepLearning, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Multi-layer LSTM with attention
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True, dropout=dropout)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout)
        
        # Additional deep layers
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 64)
        self.fc4 = nn.Linear(64, 10)  # Output for digits 0-9
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size * 2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # LSTM layers
        lstm_out1, _ = self.lstm1(x)
        lstm_out2, _ = self.lstm2(lstm_out1)
        
        # Attention mechanism
        attn_out, _ = self.attention(lstm_out2, lstm_out2, lstm_out2)
        
        # Take the last time step
        final_hidden = attn_out[:, -1, :]
        
        # Deep layers with batch normalization
        out = self.fc1(final_hidden)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc4(out)
        
        return self.softmax(out)


class ExtraordinaryIntelligence:
    """Extraordinary intelligence using complete history database"""
    
    def __init__(self):
        self.feature_engineer = AdvancedFeatureEngineer()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history_cache = None
        self.features_cache = None
        self.is_initialized = False
        
    def load_complete_history(self):
        """Load complete history from database"""
        try:
            engine = create_engine(DATABASE_URL)
            
            with engine.connect() as conn:
                # Fetch ALL historical data (not just recent)
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
                self.history_cache = history_data
                return history_data
                
        except Exception as e:
            logger.error(f"Error loading complete history: {e}")
            return None
    
    def extract_extraordinary_features(self):
        """Extract extraordinary features from complete history"""
        if not self.history_cache:
            self.load_complete_history()
            
        if not self.history_cache:
            logger.error("No history data available")
            return None
            
        logger.info("Extracting extraordinary features from complete history...")
        features = self.feature_engineer.extract_comprehensive_features(self.history_cache)
        logger.info(f"Extracted {len(features)} extraordinary features")
        
        self.features_cache = features
        return features
    
    def initialize_model(self):
        """Initialize the extraordinary deep learning model"""
        if self.features_cache is None:
            self.extract_extraordinary_features()
            
        # Calculate input size based on features
        input_size = len(self.features_cache) if self.features_cache else 100
        
        self.model = ExtraordinaryDeepLearning(
            input_size=input_size,
            hidden_size=256,
            num_layers=4,
            dropout=0.3
        ).to(self.device)
        
        logger.info(f"Initialized extraordinary model with {input_size} input features")
        self.is_initialized = True
        
    def train_on_complete_history(self, epochs=50, batch_size=32):
        """Train model on complete history"""
        if not self.is_initialized:
            self.initialize_model()
            
        if not self.history_cache or len(self.history_cache) < 100:
            logger.error("Insufficient history data for training")
            return False
            
        logger.info(f"Training on complete history ({len(self.history_cache)} records)...")
        
        # Prepare training data
        X, y = self._prepare_training_data()
        
        if X is None or y is None:
            logger.error("Failed to prepare training data")
            return False
            
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for i in range(0, len(X), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X.unsqueeze(1))  # Add sequence dimension
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            epoch_loss = total_loss / (len(X) // batch_size)
            epoch_acc = 100 * correct / total
            scheduler.step(epoch_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
        logger.info("Training completed successfully")
        return True
    
    def _prepare_training_data(self):
        """Prepare training data from complete history"""
        try:
            if not self.history_cache or len(self.history_cache) < 100:
                return None, None
                
            # Use feature engineering to create input data
            sequence_length = 50
            X = []
            y = []
            
            digits = [item['digit'] for item in self.history_cache]
            
            for i in range(len(digits) - sequence_length):
                # Extract features for the sequence
                sequence_data = self.history_cache[i:i+sequence_length]
                features = self.feature_engineer.extract_comprehensive_features(sequence_data)
                
                # Convert features to array
                feature_array = np.array(list(features.values()))
                X.append(feature_array)
                
                # Target is the next digit
                y.append(digits[i + sequence_length])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None
    
    def predict_next(self):
        """Make extraordinary prediction using complete history"""
        if not self.is_initialized:
            self.initialize_model()
            
        if not self.history_cache or len(self.history_cache) < 50:
            logger.error("Insufficient history for prediction")
            return None
            
        # Extract features from recent history
        recent_data = self.history_cache[-50:]
        features = self.feature_engineer.extract_comprehensive_features(recent_data)
        
        if not features:
            logger.error("Failed to extract features for prediction")
            return None
            
        # Convert to tensor
        feature_array = np.array(list(features.values()))
        X_tensor = torch.FloatTensor(feature_array).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = outputs.cpu().numpy()[0]
            
        # Get prediction
        predicted_digit = np.argmax(probabilities)
        prediction = "Big" if predicted_digit >= 5 else "Small"
        confidence = probabilities[predicted_digit] * 100
        
        # Get hedge (second best)
        sorted_indices = np.argsort(probabilities)[::-1]
        hedge_digit = sorted_indices[1] if len(sorted_indices) > 1 else 0
        
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'targetNum': predicted_digit,
            'hedgeNum': hedge_digit,
            'probabilities': probabilities.tolist(),
            'source': 'extraordinary_intelligence',
            'features_used': len(features),
            'history_records': len(self.history_cache)
        }
        
        logger.info(f"Extraordinary prediction: {prediction} with {confidence:.1f}% confidence")
        return result


def run_extraordinary_intelligence():
    """Run the extraordinary intelligence system"""
    logger.info("=" * 60)
    logger.info("STARTING EXTRAORDINARY INTELLIGENCE SYSTEM")
    logger.info("=" * 60)
    
    # Initialize extraordinary intelligence
    ei = ExtraordinaryIntelligence()
    
    # Load complete history
    logger.info("Loading complete history database...")
    history = ei.load_complete_history()
    
    if not history or len(history) < 100:
        logger.error("Insufficient historical data for extraordinary intelligence")
        return None
    
    # Extract extraordinary features
    logger.info("Extracting extraordinary features...")
    features = ei.extract_extraordinary_features()
    
    if not features:
        logger.error("Failed to extract extraordinary features")
        return None
    
    # Initialize model
    logger.info("Initializing extraordinary deep learning model...")
    ei.initialize_model()
    
    # Train on complete history
    logger.info("Training extraordinary model on complete history...")
    training_success = ei.train_on_complete_history(epochs=50, batch_size=32)
    
    if not training_success:
        logger.error("Model training failed")
        return None
    
    # Make prediction
    logger.info("Making extraordinary prediction...")
    prediction = ei.predict_next()
    
    if prediction:
        logger.info("=" * 60)
        logger.info("EXTRAORDINARY PREDICTION GENERATED")
        logger.info("=" * 60)
        logger.info(f"Prediction: {prediction['prediction']}")
        logger.info(f"Confidence: {prediction['confidence']:.1f}%")
        logger.info(f"Target: {prediction['targetNum']}")
        logger.info(f"Hedge: {prediction['hedgeNum']}")
        logger.info(f"Features Used: {prediction['features_used']}")
        logger.info(f"History Records: {prediction['history_records']}")
        logger.info("=" * 60)
    
    return prediction


if __name__ == "__main__":
    prediction = run_extraordinary_intelligence()
    if prediction:
        print(f"Extraordinary Prediction: {prediction}")
    else:
        print("Extraordinary intelligence failed to generate prediction")