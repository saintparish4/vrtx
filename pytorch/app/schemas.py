from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class MetricData(BaseModel):
    """Incoming metric data from Kubernetes cluster"""

    namespace: str = Field(..., description="Kubernetes namespace")
    deployment: str = Field(..., description="Deployment name")
    timestamp: datetime = Field(..., description="Metric timestamp")
    cpu_usage: float = Field(..., ge=0, description="CPU usage in millicores")
    memory_usage: float = Field(..., ge=0, description="Memory usage in bytes")
    replica_count: Optional[int] = Field(None, ge=0, description="Number of replicas")

    class Config:
        json_schema_extra = {
            "example": {
                "namespace": "default",
                "deployment": "my-app",
                "timestamp": "2025-10-07T12:00:00Z",
                "cpu_usage": 250.5,
                "memory_usage": 524288000,
                "replica_count": 3,
            }
        }


class PredictionResponse(BaseModel):
    """ML prediction output"""

    predicted_cpu: float = Field(..., description="Predicted CPU usage in millicores")
    predicted_memory: float = Field(..., description="Predicted memory usage in bytes")
    confidence: float = Field(
        ..., ge=0, le=1, description="Prediction confidence score (0-1)"
    )
    timestamp: datetime = Field(..., description="Prediction generated timestamp")
    forecast_horizon: str = Field(
        default="1h", description="How far ahead the prediction is"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_cpu": 320.8,
                "predicted_memory": 687194112,
                "confidence": 0.85,
                "timestamp": "2025-10-07T12:00:00Z",
                "forecast_horizon": "1h",
            }
        }


class TrainingRequest(BaseModel):
    """Request to train/retrain the model"""

    namespace: str = Field(..., description="Kubernetes namespace")
    deployment: str = Field(..., description="Deployment name")
    lookback_hours: int = Field(
        168, ge=24, le=720, description="Hours of historical data to use"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "namespace": "default",
                "deployment": "my-app",
                "lookback_hours": 168,
            }
        }
