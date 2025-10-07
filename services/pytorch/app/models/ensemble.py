import numpy as np
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta
import logging

from .prophet_model import ProphetPredictor
from .lstm_model import LSTMPredictor

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Ensemble model combining Prophet and LSTM predictions"""
    
    def __init__(
        self,
        prophet_weight: float = 0.6,
        lstm_weight: float = 0.4,
        sequence_length: int = 24
    ):
        self.prophet_weight = prophet_weight
        self.lstm_weight = lstm_weight
        
        # Initialize individual models
        self.prophet_model = ProphetPredictor()
        self.lstm_model = LSTMPredictor(sequence_length=sequence_length)
        
        self.is_trained = False
        self.last_train_time = None
        self.model_performances = {
            'prophet': {'mae': None, 'weight': prophet_weight},
            'lstm': {'mae': None, 'weight': lstm_weight}
        }
    
    def train(
        self,
        timestamps: List[datetime],
        values: List[float],
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Train both models and calculate optimal weights"""
        try:
            # Split data for validation
            split_idx = int(len(values) * (1 - validation_split))
            
            train_timestamps = timestamps[:split_idx]
            train_values = values[:split_idx]
            val_timestamps = timestamps[split_idx:]
            val_values = values[split_idx:]
            
            logger.info(f"Training ensemble with {len(train_values)} samples, validating with {len(val_values)}")
            
            # Train Prophet model
            logger.info("Training Prophet model...")
            prophet_metrics = self.prophet_model.train(train_timestamps, train_values)
            
            # Train LSTM model
            logger.info("Training LSTM model...")
            lstm_metrics = self.lstm_model.train(train_timestamps, train_values)
            
            # Validate both models
            prophet_val_mae = self._validate_prophet(val_timestamps, val_values)
            lstm_val_mae = self._validate_lstm(val_values)
            
            # Update model performances
            self.model_performances['prophet']['mae'] = prophet_val_mae
            self.model_performances['lstm']['mae'] = lstm_val_mae
            
            # Calculate dynamic weights based on validation performance
            if prophet_val_mae > 0 and lstm_val_mae > 0:
                prophet_inv_error = 1.0 / prophet_val_mae
                lstm_inv_error = 1.0 / lstm_val_mae
                total_inv_error = prophet_inv_error + lstm_inv_error
                
                self.prophet_weight = prophet_inv_error / total_inv_error
                self.lstm_weight = lstm_inv_error / total_inv_error
                
                self.model_performances['prophet']['weight'] = self.prophet_weight
                self.model_performances['lstm']['weight'] = self.lstm_weight
            
            self.is_trained = True
            self.last_train_time = datetime.utcnow()
            
            logger.info(f"Ensemble trained - Prophet weight: {self.prophet_weight:.3f}, LSTM weight: {self.lstm_weight:.3f}")
            
            return {
                'prophet_metrics': prophet_metrics,
                'lstm_metrics': lstm_metrics,
                'prophet_val_mae': float(prophet_val_mae),
                'lstm_val_mae': float(lstm_val_mae),
                'prophet_weight': float(self.prophet_weight),
                'lstm_weight': float(self.lstm_weight),
                'training_samples': len(train_values),
                'validation_samples': len(val_values)
            }
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            raise
    
    def _validate_prophet(
        self,
        val_timestamps: List[datetime],
        val_values: List[float]
    ) -> float:
        """Validate Prophet model"""
        try:
            # Get predictions for validation period
            forecast_horizon = len(val_values)
            _, predictions, _, _ = self.prophet_model.predict(forecast_horizon)
            
            # Calculate MAE
            mae = np.mean(np.abs(np.array(val_values) - np.array(predictions[:len(val_values)])))
            return float(mae)
            
        except Exception as e:
            logger.warning(f"Prophet validation failed: {e}")
            return float('inf')
    
    def _validate_lstm(self, val_values: List[float]) -> float:
        """Validate LSTM model"""
        try:
            # Get predictions for validation period
            forecast_horizon = len(val_values)
            predictions, _, _ = self.lstm_model.predict(val_values, forecast_horizon)
            
            # Calculate MAE
            mae = np.mean(np.abs(np.array(val_values) - np.array(predictions[:len(val_values)])))
            return float(mae)
            
        except Exception as e:
            logger.warning(f"LSTM validation failed: {e}")
            return float('inf')
    
    def predict(
        self,
        timestamps: List[datetime],
        recent_values: List[float],
        forecast_horizon: int
    ) -> Tuple[List[datetime], List[float], List[float], List[float]]:
        """
        Generate ensemble predictions
        
        Returns:
            timestamps, predictions, lower_bounds, upper_bounds
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call train() first.")
        
        try:
            # Get Prophet predictions
            prophet_timestamps, prophet_preds, prophet_lower, prophet_upper = \
                self.prophet_model.predict(forecast_horizon)
            
            # Get LSTM predictions
            lstm_preds, lstm_lower, lstm_upper = \
                self.lstm_model.predict(recent_values, forecast_horizon)
            
            # Combine predictions using weighted average
            ensemble_preds = (
                np.array(prophet_preds) * self.prophet_weight +
                np.array(lstm_preds) * self.lstm_weight
            )
            
            ensemble_lower = (
                np.array(prophet_lower) * self.prophet_weight +
                np.array(lstm_lower) * self.lstm_weight
            )
            
            ensemble_upper = (
                np.array(prophet_upper) * self.prophet_weight +
                np.array(lstm_upper) * self.lstm_weight
            )
            
            # Ensure non-negative values
            ensemble_preds = np.maximum(ensemble_preds, 0)
            ensemble_lower = np.maximum(ensemble_lower, 0)
            ensemble_upper = np.maximum(ensemble_upper, 0)
            
            logger.info(f"Generated ensemble predictions for {forecast_horizon} periods")
            
            return (
                prophet_timestamps,
                ensemble_preds.tolist(),
                ensemble_lower.tolist(),
                ensemble_upper.tolist()
            )
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            raise
    
    def detect_anomalies(
        self,
        timestamps: List[datetime],
        values: List[float],
        threshold: float = 2.5
    ) -> List[Dict[str, Any]]:
        """Detect anomalies using both models"""
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call train() first.")
        
        try:
            # Get anomalies from Prophet
            prophet_anomalies = self.prophet_model.detect_anomalies(
                timestamps, values, threshold
            )
            
            # Get anomaly indices from LSTM
            lstm_anomaly_indices = self.lstm_model.detect_anomalies(values, threshold)
            
            # Combine anomalies
            anomaly_dict = {}
            
            # Add Prophet anomalies
            for anomaly in prophet_anomalies:
                ts = anomaly['timestamp']
                anomaly_dict[ts] = {
                    'timestamp': ts,
                    'value': anomaly['value'],
                    'expected_value': anomaly['expected_value'],
                    'severity': anomaly['severity'],
                    'confidence': anomaly['confidence'],
                    'detected_by': ['prophet']
                }
            
            # Add LSTM anomalies
            for idx in lstm_anomaly_indices:
                if idx < len(timestamps):
                    ts = timestamps[idx]
                    if ts in anomaly_dict:
                        # Both models detected this anomaly - increase confidence
                        anomaly_dict[ts]['detected_by'].append('lstm')
                        anomaly_dict[ts]['confidence'] = min(
                            anomaly_dict[ts]['confidence'] * 1.3, 1.0
                        )
                        if anomaly_dict[ts]['severity'] == 'medium':
                            anomaly_dict[ts]['severity'] = 'high'
                    else:
                        anomaly_dict[ts] = {
                            'timestamp': ts,
                            'value': float(values[idx]),
                            'expected_value': float(values[idx]),  # Could be improved
                            'severity': 'medium',
                            'confidence': 0.7,
                            'detected_by': ['lstm']
                        }
            
            # Convert to list and sort by timestamp
            anomalies = sorted(anomaly_dict.values(), key=lambda x: x['timestamp'])
            
            logger.info(f"Detected {len(anomalies)} anomalies using ensemble")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Ensemble anomaly detection failed: {e}")
            raise
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for individual models"""
        return self.model_performances
    
    def needs_retraining(self, max_age_hours: int = 24) -> bool:
        """Check if ensemble needs retraining"""
        if not self.is_trained or not self.last_train_time:
            return True
        
        age = datetime.utcnow() - self.last_train_time
        return age > timedelta(hours=max_age_hours)
    
    def update_weights(self, prophet_weight: float, lstm_weight: float):
        """Manually update ensemble weights"""
        if not (0 <= prophet_weight <= 1 and 0 <= lstm_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
        
        total = prophet_weight + lstm_weight
        if total == 0:
            raise ValueError("At least one weight must be non-zero")
        
        # Normalize weights
        self.prophet_weight = prophet_weight / total
        self.lstm_weight = lstm_weight / total
        
        self.model_performances['prophet']['weight'] = self.prophet_weight
        self.model_performances['lstm']['weight'] = self.lstm_weight
        
        logger.info(f"Updated ensemble weights - Prophet: {self.prophet_weight:.3f}, LSTM: {self.lstm_weight:.3f}")