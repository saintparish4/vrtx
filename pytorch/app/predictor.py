import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MetricsPredictor:
    """
    Prophet-based time-series forecasting for CPU and memory Metrics
    Handles seasonality patterns and trend detection.
    """

    def __init__(self):
        self.cpu_model = None
        self.memory_model = None

    def predict(self, historical_data: list) -> dict:
        """
        Generate predictions for next time window (1 hour ahead)

        Args:
            historical_data (list): List of dicts with 'timestamp', 'cpu_usage', 'memory_usage'

        Returns:
            dict with predicted_cpu, predicted_memory, and confidence
        """
        if len(historical_data) < 24:
            raise ValueError("Insufficient data: at least 24 data points required")

        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        # Train models if not already trained
        cpu_pred = self._predict_metric(df, "cpu_usage", "cpu")
        memory_pred = self._predict_metric(df, "memory_usage", "memory")

        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(df)

        return {"cpu": cpu_pred, "memory": memory_pred, "confidence": confidence}

    def train(self, historical_data: list):
        """
        Explicitly train models with historical data
        Useful for batch retraining
        """
        df = pd.DataFrame(historical_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        self.cpu_model = self._train_model(
            df, "cpu_usage"
        )  # Fixed typo 'slef' to 'self'
        self.memory_model = self._train_model(df, "memory_usage")

        logger.info(f"Models trained on {len(df)} data points")

    def _predict_metric(
        self, df: pd.DataFrame, metric_col: str, metric_name: str
    ) -> float:
        """Train and predict for a single metric"""

        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        prophet_df = pd.DataFrame({"ds": df["timestamp"], "y": df[metric_col]})

        # Configure Prophet for hourly data with daily seasonality
        model = Prophet(
            daily_seasonality="auto",
            weekly_seasonality="auto",
            yearly_seasonality="auto",
            changepoint_prior_scale=0.05,  # Less sensitive to changes
            seasonality_prior_scale=10.0,  # Strong seasonality
            interval_width=0.8,
        )

        # Train model
        model.fit(prophet_df)

        # Predict 1 hour ahead
        future = model.make_future_dataframe(periods=1, freq="H")
        forecast = model.predict(future)

        # Get the last prediction (1 hour ahead)
        predicted_value = forecast["yhat"].iloc[-1]

        # Ensure non-negative
        predicted_value = max(0, predicted_value)

        logger.info(f"{metric_name} prediction: {predicted_value:.2f}")

        return float(predicted_value)

    def _train_model(self, df: pd.DataFrame, metric_col: str) -> Prophet:
        """Train and return a Prophet model"""
        prophet_df = pd.DataFrame({"ds": df["timestamp"], "y": df[metric_col]})

        model = Prophet(
            daily_seasonality="auto",
            weekly_seasonality="auto",
            yearly_seasonality="auto",
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
        )

        model.fit(prophet_df)
        return model

    def _calculate_confidence(self, df: pd.DataFrame) -> float:
        """
        Calculate  confidence score based on data quality Metrics

        Factors:
        - Data recency (more recent = higher confidence)
        - Data completeness (fewer gaps = higher confidence)
        - Data variance (Stable patterns = higher confidence)
        """

        # Recency score (data from last 24h gets higher score)
        now = datetime.utcnow()
        latest_time = df["timestamp"].max()
        hours_old = (now - latest_time).total_seconds() / 3600
        recency_score = max(0, 1 - (hours_old / 24))  # Decay over 24h

        # Completeness score (check for gaps)
        expected_points = len(df)
        actual_points = df["cpu_usage"].notna().sum()
        completeness_score = actual_points / expected_points

        # Variance score (lower variance = more predictable)
        cpu_cv = df["cpu_usage"].std() / (df["cpu_usage"].mean() + 1e-6)
        variance_score = max(0, 1 - min(1, cpu_cv))  # Cap at 1

        # Weighted average
        confidence = (
            0.4 * recency_score + 0.3 * completeness_score + 0.3 * variance_score
        )

        return round(confidence, 2)
