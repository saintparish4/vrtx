┌──────────────┐         ┌─────────────────┐
│  Kubernetes  │────────▶│ Metrics         │
│   Cluster    │ hourly  │ Collector       │
└──────────────┘         │ (CronJob)       │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  ML Service     │
                         │  (FastAPI)      │
                         │                 │
                         │  - Prophet      │
                         │  - SQLite       │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Cluster API    │
                         │  (Go)           │
                         └─────────────────┘

## ML Prediction Service
Prophet-based time-series forecasting service for Kubernetes resource prediction. Analyzes historical CPU/memory patterns to enable proactive autoscaling.
## Features

**Prophet ML Model:** Facebook's time-series forecasting library
**Hourly Predictions:** Forecast CPU/memory usage 1 hour ahead
**Confidence Scoring:** Data quality-based confidence metrics
**Historical Storage:** SQLite database for metrics (upgradable to PostgreSQL)
**Auto-Collection:** CronJob to automatically gather K8s metrics
**FastAPI Backend:** High-performance async API