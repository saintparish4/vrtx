┌─────────────┐      ┌──────────────────┐      ┌─────────────┐
│   Next.js   │─────▶│  Cluster API     │─────▶│ Kubernetes  │
│  Frontend   │      │  (This Service)  │      │   Cluster   │
└─────────────┘      └──────────────────┘      └─────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  Python ML API   │
                     │    (Prophet)     │
                     └──────────────────┘

## Cluster Management API

A **Go-based Kubernetes cluster management service** that automates pod scaling operations using ML predictions from a Python Prophet service.

### Features

- **Cluster Information:**  
  Query namespaces, deployments, pods, and resource usage.

- **Manual Scaling:**  
  Directly scale deployments via API.

- **Predictive Scaling:**  
  Proactively scale resources based on Prophet ML forecasts.

- **Scaling History:**  
  Track all scaling events and decisions.

- **Health Monitoring:**  
  Built-in health checks and metrics endpoints.