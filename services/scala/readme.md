# Scaler Service

A high-performance Rust-based scaling service for the Predictive Autoscaler platform. Handles AWS EC2 and Kubernetes scaling operations with safety constraints and cost optimization.

## Features

- **Multi-Cloud Support**: AWS EC2 Auto Scaling Groups and Kubernetes Deployments/StatefulSets
- **Intelligent Scaling Decisions**: ML-driven predictions with safety constraints
- **Cost Estimation**: Real-time cost calculations for scaling operations
- **Safety Constraints**: Min/max instances, cooldown periods, rate limiting
- **Event Tracking**: Complete audit trail of all scaling operations
- **High Performance**: Built with Rust and Actix-web for maximum throughput

## Architecture

```
┌─────────────────┐
│   API Gateway   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│ Decision Engine │◄────►│  PostgreSQL  │
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│ Cloud Providers │◄────►│    Redis     │
│  (AWS + K8s)    │      └──────────────┘
└─────────────────┘
```

## API Endpoints

### Health Check
```
GET /health
```

### Make Scaling Decision
```
POST /scale/decide
Content-Type: application/json

{
  "resource_id": "uuid",
  "use_prediction": true
}
```

**Response:**
```json
{
  "id": "uuid",
  "resource_id": "uuid",
  "decision_type": "scaleup|scaledown|noaction",
  "current_instances": 3,
  "target_instances": 5,
  "reason": "Predicted CPU: 85%, Memory: 70%...",
  "estimated_cost": 0.50,
  "confidence": 0.92,
  "created_at": "2025-10-01T12:00:00Z",
  "executed": false
}
```

### Execute Scaling Decision
```
POST /scale/execute
Content-Type: application/json

{
  "decision_id": "uuid"
}
```

**Response:**
```json
{
  "success": true,
  "event": {
    "id": "uuid",
    "resource_id": "uuid",
    "decision_id": "uuid",
    "action": "ScaleUp",
    "previous_instances": 3,
    "new_instances": 5,
    "success": true,
    "execution_time_ms": 1234
  }
}
```

### Get Scaling History
```
GET /scale/history?resource_id=uuid&limit=50&offset=0
```

### Estimate Cost
```
POST /cost/estimate
Content-Type: application/json

{
  "resource_id": "uuid",
  "instance_count": 5
}
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Server bind address | `0.0.0.0` |
| `PORT` | Server port | `8082` |
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `REDIS_URL` | Redis connection string | Required |
| `AWS_REGION` | AWS region | `us-east-1` |
| `RUST_LOG` | Log level | `info` |

### Resource Configuration

Resources are stored in PostgreSQL with the following format:

```sql
CREATE TABLE resources (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    resource_type VARCHAR(50) NOT NULL, -- 'ec2' or 'kubernetes'
    cloud_provider VARCHAR(50) NOT NULL, -- 'aws' or 'kubernetes'
    min_instances INTEGER NOT NULL,
    max_instances INTEGER NOT NULL,
    current_instances INTEGER NOT NULL,
    target_cpu DOUBLE PRECISION NOT NULL,
    target_memory DOUBLE PRECISION NOT NULL,
    cooldown_period INTEGER NOT NULL, -- seconds
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## Cloud Provider Setup

### AWS

**Resource Name Format**: Auto Scaling Group name
```
Example: "my-app-asg"
```

**Required IAM Permissions**:
```json
{
  "