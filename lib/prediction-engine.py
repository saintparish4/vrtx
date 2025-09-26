#!/usr/bin/env python3
"""
Predictive Infrastructure Auto-Scaling Engine
Core ML models for traffic prediction and failure detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Time series forecasting
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Data storage
import redis
import psycopg2
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrafficPredictor:
    """Advanced traffic prediction using ensemble of models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series data for training"""
        # Feature engineering
        data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
        data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        
        # Rolling statistics
        data['rolling_mean_24h'] = data['value'].rolling(window=24).mean()
        data['rolling_std_24h'] = data['value'].rolling(window=24).std()
        data['rolling_mean_7d'] = data['value'].rolling(window=168).mean()
        
        # Lag features
        for lag in [1, 2, 3, 6, 12, 24, 48]:
            data[f'lag_{lag}'] = data['value'].shift(lag)
            
        # Remove NaN values
        data = data.dropna()
        
        feature_columns = ['hour', 'day_of_week', 'is_weekend', 'rolling_mean_24h', 
                          'rolling_std_24h', 'rolling_mean_7d'] + \
                         [f'lag_{lag}' for lag in [1, 2, 3, 6, 12, 24, 48]]
        
        X = data[feature_columns].values
        y = data['value'].values
        
        return X, y
    
    def train_prophet_model(self, data: pd.DataFrame, resource_id: str) -> Prophet:
        """Train Facebook Prophet model"""
        logger.info(f"Training Prophet model for resource {resource_id}")
        
        # Prepare data for Prophet
        prophet_data = data[['timestamp', 'value']].copy()
        prophet_data.columns = ['ds', 'y']
        prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
        
        # Add custom seasonalities
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            interval_width=0.8
        )
        
        # Add custom regressors for business hours
        prophet_data['business_hours'] = pd.to_datetime(prophet_data['ds']).dt.hour.between(9, 17).astype(int)
        model.add_regressor('business_hours')
        
        model.fit(prophet_data)
        self.models[f'prophet_{resource_id}'] = model
        
        return model
    
    def train_lstm_model(self, data: pd.DataFrame, resource_id: str) -> tf.keras.Model:
        """Train LSTM model for complex patterns"""
        logger.info(f"Training LSTM model for resource {resource_id}")
        
        # Prepare sequences
        sequence_length = 24  # 24 hours of data
        X, y = self.prepare_sequences(data['value'].values, sequence_length)
        
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        self.scalers[f'lstm_{resource_id}'] = scaler
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train model
        model.fit(X_scaled, y_scaled, epochs=50, batch_size=32, verbose=0)
        
        self.models[f'lstm_{resource_id}'] = model
        return model
    
    def prepare_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def train_ensemble_model(self, data: pd.DataFrame, resource_id: str):
        """Train ensemble of multiple models"""
        logger.info(f"Training ensemble model for resource {resource_id}")
        
        X, y = self.prepare_data(data)
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        self.models[f'rf_{resource_id}'] = rf_model
        
        # Scale features for other models
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[f'ensemble_{resource_id}'] = scaler
        
    def predict_traffic(self, resource_id: str, hours_ahead: int = 24) -> List[Dict]:
        """Generate traffic predictions"""
        predictions = []
        
        # Get recent data
        recent_data = self.get_recent_metrics(resource_id, hours=168)  # 7 days
        
        if len(recent_data) < 24:
            logger.warning(f"Insufficient data for resource {resource_id}")
            return predictions
        
        # Prophet predictions
        if f'prophet_{resource_id}' in self.models:
            prophet_preds = self.predict_with_prophet(resource_id, hours_ahead)
            predictions.extend(prophet_preds)
        
        # LSTM predictions
        if f'lstm_{resource_id}' in self.models:
            lstm_preds = self.predict_with_lstm(resource_id, hours_ahead)
            predictions.extend(lstm_preds)
        
        # Ensemble predictions
        ensemble_preds = self.predict_with_ensemble(resource_id, hours_ahead)
        predictions.extend(ensemble_preds)
        
        return self.combine_predictions(predictions)
    
    def predict_with_prophet(self, resource_id: str, hours_ahead: int) -> List[Dict]:
        """Generate Prophet predictions"""
        model = self.models[f'prophet_{resource_id}']
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=hours_ahead, freq='H')
        future['business_hours'] = pd.to_datetime(future['ds']).dt.hour.between(9, 17).astype(int)
        
        forecast = model.predict(future)
        
        predictions = []
        for i in range(-hours_ahead, 0):
            predictions.append({
                'timestamp': forecast.iloc[i]['ds'].timestamp(),
                'predicted_load': max(0, forecast.iloc[i]['yhat']),
                'confidence': 1 - (forecast.iloc[i]['yhat_upper'] - forecast.iloc[i]['yhat_lower']) / forecast.iloc[i]['yhat'],
                'model': 'prophet'
            })
        
        return predictions
    
    def detect_anomalies(self, data: pd.DataFrame) -> List[Dict]:
        """Detect anomalies using Isolation Forest"""
        logger.info("Detecting anomalies in traffic patterns")
        
        X, _ = self.prepare_data(data)
        
        # Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(X)
        
        anomaly_results = []
        for i, is_anomaly in enumerate(anomalies):
            if is_anomaly == -1:  # Anomaly detected
                anomaly_results.append({
                    'timestamp': data.iloc[i]['timestamp'],
                    'value': data.iloc[i]['value'],
                    'anomaly_score': iso_forest.score_samples([X[i]])[0],
                    'severity': 'high' if iso_forest.score_samples([X[i]])[0] < -0.5 else 'medium'
                })
        
        return anomaly_results
    
    def get_recent_metrics(self, resource_id: str, hours: int = 24) -> pd.DataFrame:
        """Fetch recent metrics from database"""
        # This would connect to your actual metrics database
        # For now, returning sample data structure
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Sample data generation for demonstration
        timestamps = pd.date_range(start=start_time, end=end_time, freq='H')
        values = np.random.normal(100, 20, len(timestamps)) + \
                np.sin(np.arange(len(timestamps)) * 2 * np.pi / 24) * 30  # Daily pattern
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'value': np.maximum(0, values),
            'resource_id': resource_id
        })
    
    def combine_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """Combine predictions from multiple models"""
        if not predictions:
            return []
        
        # Group by timestamp
        grouped = {}
        for pred in predictions:
            ts = pred['timestamp']
            if ts not in grouped:
                grouped[ts] = []
            grouped[ts].append(pred)
        
        # Ensemble predictions
        combined = []
        for timestamp, preds in grouped.items():
            avg_load = np.mean([p['predicted_load'] for p in preds])
            avg_confidence = np.mean([p['confidence'] for p in preds])
            
            # Calculate spike probability
            spike_threshold = avg_load * 1.5
            spike_prob = min(1.0, max(0.0, (avg_load - spike_threshold) / spike_threshold))
            
            combined.append({
                'timestamp': timestamp,
                'predicted_load': avg_load,
                'confidence': avg_confidence,
                'spike_probability': spike_prob
            })
        
        return sorted(combined, key=lambda x: x['timestamp'])

class FailurePredictor:
    """Predict infrastructure failures before they occur"""
    
    def __init__(self):
        self.models = {}
        self.thresholds = {
            'cpu_exhaustion': 0.85,
            'memory_leak': 0.90,
            'disk_full': 0.95,
            'network_congestion': 0.80
        }
    
    def analyze_resource_health(self, resource_id: str, metrics: pd.DataFrame) -> Dict:
        """Analyze resource health and predict failures"""
        health_score = 1.0
        failure_predictions = []
        
        # CPU analysis
        if 'cpu_usage' in metrics.columns:
            cpu_trend = self.analyze_trend(metrics['cpu_usage'])
            if cpu_trend['slope'] > 0.01:  # Increasing CPU usage
                failure_prob = min(1.0, cpu_trend['current_value'] / self.thresholds['cpu_exhaustion'])
                if failure_prob > 0.7:
                    failure_predictions.append({
                        'resource_id': resource_id,
                        'failure_probability': failure_prob,
                        'failure_type': 'cpu_exhaustion',
                        'predicted_failure_time': self.estimate_failure_time(cpu_trend, self.thresholds['cpu_exhaustion']),
                        'recommended_action': 'Scale up instances or optimize CPU-intensive processes'
                    })
        
        # Memory analysis
        if 'memory_usage' in metrics.columns:
            memory_trend = self.analyze_trend(metrics['memory_usage'])
            if memory_trend['slope'] > 0.005:  # Memory leak detection
                failure_prob = min(1.0, memory_trend['current_value'] / self.thresholds['memory_leak'])
                if failure_prob > 0.6:
                    failure_predictions.append({
                        'resource_id': resource_id,
                        'failure_probability': failure_prob,
                        'failure_type': 'memory_leak',
                        'predicted_failure_time': self.estimate_failure_time(memory_trend, self.thresholds['memory_leak']),
                        'recommended_action': 'Restart services or investigate memory leaks'
                    })
        
        return {
            'resource_id': resource_id,
            'health_score': health_score,
            'failure_predictions': failure_predictions,
            'analysis_timestamp': datetime.now().timestamp()
        }
    
    def analyze_trend(self, series: pd.Series) -> Dict:
        """Analyze trend in time series data"""
        x = np.arange(len(series))
        coeffs = np.polyfit(x, series.values, 1)
        slope = coeffs[0]
        
        return {
            'slope': slope,
            'current_value': series.iloc[-1],
            'trend': 'increasing' if slope > 0 else 'decreasing'
        }
    
    def estimate_failure_time(self, trend: Dict, threshold: float) -> Optional[float]:
        """Estimate when failure might occur based on trend"""
        if trend['slope'] <= 0:
            return None
        
        current_value = trend['current_value']
        if current_value >= threshold:
            return datetime.now().timestamp()
        
        hours_to_failure = (threshold - current_value) / trend['slope']
        failure_time = datetime.now() + timedelta(hours=hours_to_failure)
        
        return failure_time.timestamp()

class CostOptimizer:
    """Optimize infrastructure costs while maintaining performance"""
    
    def __init__(self):
        self.cost_models = {}
        self.provider_costs = {
            'aws': {'t3.micro': 0.0104, 't3.small': 0.0208, 't3.medium': 0.0416, 't3.large': 0.0832},
            'gcp': {'e2-micro': 0.0063, 'e2-small': 0.0126, 'e2-medium': 0.0252, 'e2-standard-2': 0.0504},
            'azure': {'B1S': 0.0052, 'B1MS': 0.0104, 'B2S': 0.0208, 'B2MS': 0.0416}
        }
    
    def optimize_resource_allocation(self, predictions: List[Dict], current_config: Dict) -> Dict:
        """Optimize resource allocation based on predictions"""
        recommendations = []
        total_current_cost = 0
        total_optimized_cost = 0
        
        for prediction in predictions:
            predicted_load = prediction['predicted_load']
            confidence = prediction['confidence']
            
            # Calculate required capacity
            safety_margin = 1.2 if confidence > 0.8 else 1.5
            required_capacity = predicted_load * safety_margin
            
            # Find optimal instance type and count
            optimal_config = self.find_optimal_instances(
                required_capacity, 
                current_config['provider'], 
                current_config['instance_type']
            )
            
            current_cost = current_config['current_instances'] * \
                          self.provider_costs[current_config['provider']][current_config['instance_type']]
            optimized_cost = optimal_config['instance_count'] * \
                           self.provider_costs[current_config['provider']][optimal_config['instance_type']]
            
            total_current_cost += current_cost
            total_optimized_cost += optimized_cost
            
            if abs(optimized_cost - current_cost) / current_cost > 0.1:  # 10% cost difference
                recommendations.append({
                    'timestamp': prediction['timestamp'],
                    'current_instances': current_config['current_instances'],
                    'optimal_instances': optimal_config['instance_count'],
                    'current_instance_type': current_config['instance_type'],
                    'optimal_instance_type': optimal_config['instance_type'],
                    'cost_savings': current_cost - optimized_cost,
                    'reason': f"Load prediction: {predicted_load:.1f}, Confidence: {confidence:.2f}"
                })
        
        savings_percentage = ((total_current_cost - total_optimized_cost) / total_current_cost) * 100 if total_current_cost > 0 else 0
        
        return {
            'current_cost_per_hour': total_current_cost,
            'optimized_cost_per_hour': total_optimized_cost,
            'savings_percentage': savings_percentage,
            'recommendations': recommendations
        }
    
    def find_optimal_instances(self, required_capacity: float, provider: str, current_type: str) -> Dict:
        """Find optimal instance configuration"""
        instance_capacities = {
            't3.micro': 10, 't3.small': 20, 't3.medium': 40, 't3.large': 80,
            'e2-micro': 8, 'e2-small': 16, 'e2-medium': 32, 'e2-standard-2': 64,
            'B1S': 5, 'B1MS': 10, 'B2S': 20, 'B2MS': 40
        }
        
        provider_instances = {k: v for k, v in instance_capacities.items() 
                            if k in self.provider_costs[provider]}
        
        best_config = None
        best_cost = float('inf')
        
        for instance_type, capacity in provider_instances.items():
            instance_count = max(1, int(np.ceil(required_capacity / capacity)))
            cost = instance_count * self.provider_costs[provider][instance_type]
            
            if cost < best_cost and instance_count * capacity >= required_capacity:
                best_cost = cost
                best_config = {
                    'instance_type': instance_type,
                    'instance_count': instance_count,
                    'total_capacity': instance_count * capacity,
                    'cost_per_hour': cost
                }
        
        return best_config or {
            'instance_type': current_type,
            'instance_count': max(1, int(np.ceil(required_capacity / instance_capacities.get(current_type, 20)))),
            'total_capacity': required_capacity,
            'cost_per_hour': 0
        }

if __name__ == "__main__":
    # Example usage
    config = {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'db_connection': 'postgresql://user:pass@localhost/metrics'
    }
    
    # Initialize predictors
    traffic_predictor = TrafficPredictor(config)
    failure_predictor = FailurePredictor()
    cost_optimizer = CostOptimizer()
    
    # Example prediction workflow
    resource_id = "web-server-cluster-1"
    
    # Get sample data and train models
    sample_data = traffic_predictor.get_recent_metrics(resource_id, hours=168)
    traffic_predictor.train_prophet_model(sample_data, resource_id)
    traffic_predictor.train_ensemble_model(sample_data, resource_id)
    
    # Generate predictions
    predictions = traffic_predictor.predict_traffic(resource_id, hours_ahead=24)
    
    # Analyze failures
    health_analysis = failure_predictor.analyze_resource_health(resource_id, sample_data)
    
    # Optimize costs
    current_config = {
        'provider': 'aws',
        'instance_type': 't3.medium',
        'current_instances': 3
    }
    cost_optimization = cost_optimizer.optimize_resource_allocation(predictions, current_config)
    
    print("Traffic Predictions:", json.dumps(predictions[:5], indent=2))
    print("\nHealth Analysis:", json.dumps(health_analysis, indent=2))
    print("\nCost Optimization:", json.dumps(cost_optimization, indent=2))
