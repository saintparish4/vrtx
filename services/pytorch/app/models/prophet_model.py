from prophet import Prophet
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from datetime import datetime, timedelta
import logging
logger = logging.getLogger(__name__)

class ProphetPredictor:
    """Time series forecasting with Prophet"""

    def __init__(self):
        self.model = None
        self.is_trained = False
        self.last_train_time = None

        def prepare_data(self, timestamps: List[datetime], values: List[float]) -> pd.DataFrame:
            """Convert data to Prophet format (ds, y)"""
            df = pd.DataFrame({
                'ds': timestamps,
                'y': values 
            })
            df = df.sort_values('ds').reset_index(drop=True)

            # Remove duplicates and handle missing values
            df = df.drop_duplicates(subset=['ds'])
            df['y'] = df['y'].interpolate(method='linear')

            return df 

    def train(
        self,
        timestamps: List[datetime],
        values: List[float],
        seasonality_mode: str = 'multiplicative',
        changepoint_prior_scale: float = 0.05
    ) -> Dict[str, float]:
        """Train Prophet model on historical data"""
        try:
            df = self.prepare_data(timestamps, values)
            
            if len(df) < 10:
                raise ValueError("Need at least 10 data points to train Prophet model")
            
            # Initialize Prophet with custom parameters
            self.model = Prophet(
                seasonality_mode=seasonality_mode,
                changepoint_prior_scale=changepoint_prior_scale,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                interval_width=0.95
            )
            
            # Fit the model
            self.model.fit(df)
            self.is_trained = True
            self.last_train_time = datetime.utcnow()
            
            # Calculate training metrics
            forecast = self.model.predict(df)
            mae = np.mean(np.abs(df['y'] - forecast['yhat']))
            mape = np.mean(np.abs((df['y'] - forecast['yhat']) / df['y'])) * 100
            
            logger.info(f"Prophet model trained - MAE: {mae:.2f}, MAPE: {mape:.2f}%")
            
            return {
                'mae': float(mae),
                'mape': float(mape),
                'training_samples': len(df)
            }
            
        except Exception as e:
            logger.error(f"Prophet training failed: {e}")
            raise
    
    def predict(
        self,
        forecast_horizon: int,
        frequency: str = 'H'
    ) -> Tuple[List[datetime], List[float], List[float], List[float]]:
        """
        Generate predictions
        
        Returns:
            timestamps, predictions, lower_bounds, upper_bounds
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(
                periods=forecast_horizon,
                freq=frequency
            )
            
            # Generate forecast
            forecast = self.model.predict(future)
            
            # Extract predictions (only future values)
            future_forecast = forecast.tail(forecast_horizon)
            
            timestamps = future_forecast['ds'].tolist()
            predictions = future_forecast['yhat'].tolist()
            lower_bounds = future_forecast['yhat_lower'].tolist()
            upper_bounds = future_forecast['yhat_upper'].tolist()
            
            # Ensure non-negative predictions for resource metrics
            predictions = [max(0, p) for p in predictions]
            lower_bounds = [max(0, lb) for lb in lower_bounds]
            upper_bounds = [max(0, ub) for ub in upper_bounds]
            
            return timestamps, predictions, lower_bounds, upper_bounds
            
        except Exception as e:
            logger.error(f"Prophet prediction failed: {e}")
            raise
    
    def detect_anomalies(
        self,
        timestamps: List[datetime],
        values: List[float],
        threshold: float = 2.5
    ) -> List[Dict[str, Any]]:
        """Detect anomalies using Prophet's prediction intervals"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            df = self.prepare_data(timestamps, values)
            forecast = self.model.predict(df)
            
            anomalies = []
            
            for i, row in df.iterrows():
                pred_row = forecast.iloc[i]
                actual = row['y']
                predicted = pred_row['yhat']
                lower = pred_row['yhat_lower']
                upper = pred_row['yhat_upper']
                
                # Calculate deviation in standard deviations
                std = (upper - lower) / 4  # Approximate std from 95% interval
                deviation = abs(actual - predicted) / std if std > 0 else 0
                
                if deviation > threshold:
                    severity = 'high' if deviation > threshold * 1.5 else 'medium'
                    
                    anomalies.append({
                        'timestamp': row['ds'],
                        'value': float(actual),
                        'expected_value': float(predicted),
                        'severity': severity,
                        'confidence': float(min(deviation / (threshold * 2), 1.0))
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            raise
    
    def get_trend_components(self) -> pd.DataFrame:
        """Extract trend components from trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(self.model.history)
    
    def needs_retraining(self, max_age_hours: int = 24) -> bool:
        """Check if model needs retraining based on age"""
        if not self.is_trained or not self.last_train_time:
            return True
        
        age = datetime.utcnow() - self.last_train_time
        return age > timedelta(hours=max_age_hours)
    