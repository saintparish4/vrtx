from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from datetime import datetime
import logging

from db import Database
from predictor import MetricsPredictor
from schemas import PredictionResponse, MetricData, TrainingRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, cleanup on shutdown"""
    app.state.db = Database()
    app.state.predictor = MetricsPredictor()
    logger.info("ML Prediction Service started")
    yield
    logger.info("ML Prediction Service shutting down")


app = FastAPI(
    title="K8s ML Prediction Service",
    description="Prophet-based time-series forecasting for Kubernetes resource metrics",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "ml-prediction",
    }


@app.post("/api/metrics/ingest")
async def ingest_metrics(data: MetricData):
    """
    Ingest real-time metrics from Kubernetes cluster
    This endpoint receives CPU/memory data to build historical dataset
    """
    try:
        app.state.db.insert_metric(
            namespace=data.namespace,
            deployment=data.deployment,
            timestamp=data.timestamp,
            cpu_usage=data.cpu_usage,
            memory_usage=data.memory_usage,
            replica_count=data.replica_count,
        )
        return {"status": "success", "message": "Metric ingested"}
    except Exception as e:
        logger.error(f"Failed to ingest metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions", response_model=PredictionResponse)
async def get_prediction(
    namespace: str = Query(..., description="Kubernetes namespace"),
    deployment: str = Query(..., description="Deployment name"),
):
    """
    Get ML-based prediction for future resource usage
    Returns predicted CPU/memory for next time window
    """
    try:
        # Fetch historical data
        historical_data = app.state.db.get_metrics(
            namespace, deployment, limit=168
        )  # Last week

        if len(historical_data) < 24:  # Need at least 24 hours of data
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data: {len(historical_data)} records found, need at least 24",
            )

        # Generate prediction
        prediction = app.state.predictor.predict(historical_data)

        return PredictionResponse(
            predicted_cpu=prediction["cpu"],
            predicted_memory=prediction["memory"],
            confidence=prediction["confidence"],
            timestamp=datetime.utcnow(),
            forecast_horizon="1h",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train")
async def train_model(request: TrainingRequest):
    """
    Manually trigger model training with historical data
    Useful for retraining with updated parameters
    """
    try:
        historical_data = app.state.db.get_metrics(
            request.namespace, request.deployment, limit=request.lookback_hours
        )

        if len(historical_data) < 24:
            raise HTTPException(
                status_code=400, detail="Insufficient data for training"
            )

        app.state.predictor.train(historical_data)

        return {
            "status": "success",
            "message": f"Model trained on {len(historical_data)} data points",
            "deployment": request.deployment,
        }

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/history")
async def get_historical_metrics(
    namespace: str = Query(...),
    deployment: str = Query(...),
    limit: int = Query(100, ge=1, le=1000),
):
    """
    Retrieve historical metrics for a deployment
    Useful for debugging and visualization
    """
    try:
        data = app.state.db.get_metrics(namespace, deployment, limit)
        return {
            "namespace": namespace,
            "deployment": deployment,
            "count": len(data),
            "metrics": data,
        }
    except Exception as e:
        logger.error(f"Failed to fetch history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
