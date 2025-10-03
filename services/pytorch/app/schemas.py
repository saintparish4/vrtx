from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class ResourceType(str, Enum):
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"

class MetricPoint(BaseModel):
    timestamp: datetime
    value: float
    resource_id: str
    resource_type: ResourceType

class PredictionRequest(BaseModel):
    resource_id: str
    resource_type: ResourceType
    historical_data: List[MetricPoint]
    forecast_horizon: int = Field(default=24, ge=1, le=168) # 1 hour to 7 days
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99)

class PredictionPoint(BaseModel):
    timestamp: datetime
    predicted_value: float
    lower_bound: float
    upper_bound: float
    confidence: float

class PredictionResponse(BaseModel):
    resource_id: str
    resource_type: ResourceType
    predictions: List[PredictionPoint]
    model_used: str
    accuracy_score: Optional[float] = None
    generated_at: datetime = Field(default_factory=datetime.now)

class AnomalyDetectionRequest(BaseModel):
    resource_id: str
    resource_type: ResourceType
    current_metrics: List[MetricPoint]
    threshold: float = Field(default=2.5, ge=1.0, le=5.0)

class Anomaly(BaseModel):
    timestamp: datetime
    value: float
    expected_value: float
    severity: str # low, medium, high
    confidence: float

class AnomalyDetectionResponse(BaseModel):
    resource_id: str
    anomalies: List[Anomaly]
    is_anomalous: bool
    risk_score: float

class ModelTrainRequest(BaseModel):
    resource_id: str
    resource_type: ResourceType
    training_data: List[MetricPoint]
    model_type: str = Field(default="ensemble") # prophet, lstm, ensemble

class ModelTrainResponse(BaseModel):
    resource_id: str
    model_type: str
    training_accuracy: float
    validation_accuracy: float
    training_completed_at: datetime = Field(default_factory=datetime.now)

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: Dict[str, bool]
    uptime_seconds: float 

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)