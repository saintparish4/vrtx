# Predictive Infrastructure Auto-Scaler

An AI-powered infrastructure auto-scaling system that predicts traffic spikes before they happen and automatically scales cloud resources accordingly. Features advanced cost optimization algorithms and failure prediction to prevent over/under-provisioning that bleeds money from streaming services and marketplaces.

## 🚀 Features

### Core Capabilities
- **🤖 AI-Powered Predictions**: Multiple ML models (Prophet, LSTM, Ensemble) for accurate traffic forecasting
- **⚡ Real-time Auto-Scaling**: Instant scaling decisions based on predicted load and failure risks
- **💰 Cost Optimization**: Advanced algorithms to minimize infrastructure costs while maintaining performance
- **🛡️ Failure Prediction**: Proactive detection of potential system failures before they occur
- **📊 Real-time Dashboard**: Beautiful, responsive monitoring interface with live metrics
- **☁️ Multi-Cloud Support**: AWS, GCP, Azure, and Kubernetes integrations

### Advanced Features
- **Anomaly Detection**: Identify unusual traffic patterns and system behavior
- **Spike Protection**: Automatic buffer allocation for predicted traffic spikes
- **Confidence-based Scaling**: Scale more conservatively when predictions have lower confidence
- **Historical Analysis**: Learn from past scaling decisions to improve future predictions
- **Cost Impact Analysis**: Show estimated cost changes before executing scaling decisions

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Next.js UI    │    │  Python ML API  │    │  Cloud Providers │
│   Dashboard     │◄──►│  Prediction     │◄──►│  AWS/GCP/Azure  │
│                 │    │  Engine         │    │  Kubernetes     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────┐    ┌─┴─────────────────┐    ┌─────────────────┐
         │   PostgreSQL    │    │     Redis         │    │   Monitoring    │
         │   Database      │    │     Cache         │    │   (Prometheus   │
         │                 │    │                   │    │   + Grafana)    │
         └─────────────────┘    └───────────────────┘    └─────────────────┘
```

## 🛠️ Technology Stack

### Frontend
- **Next.js 14** with TypeScript and App Router
- **Tailwind CSS** for styling
- **Recharts** for data visualization
- **Lucide React** for icons

### Backend
- **Node.js/TypeScript** API layer
- **Python** ML prediction engine
- **FastAPI** for ML model serving
- **PostgreSQL** for data persistence
- **Redis** for caching and real-time data

### Machine Learning
- **Facebook Prophet** for time series forecasting
- **TensorFlow/Keras** for LSTM neural networks
- **Scikit-learn** for ensemble methods and anomaly detection
- **Pandas/NumPy** for data processing

### Infrastructure
- **Docker** containerization
- **Docker Compose** for local development
- **NGINX** load balancer
- **Prometheus** metrics collection
- **Grafana** advanced monitoring

### Cloud Integrations
- **AWS SDK** (EC2, Auto Scaling, CloudWatch)
- **Google Cloud SDK** (Compute Engine, Monitoring)
- **Azure SDK** (Virtual Machines, Monitor)
- **Kubernetes API** for container orchestration

## 🚦 Getting Started

### Prerequisites
- Node.js 18+ and npm/yarn
- Python 3.11+
- Docker and Docker Compose
- Cloud provider credentials (optional for demo)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd predictive-autoscaler
   ```

2. **Install dependencies**
   ```bash
   npm install
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start with Docker Compose**
   ```bash
   docker-compose up -d
   ```

5. **Access the dashboard**
   - Main Dashboard: http://localhost:3000
   - Python API: http://localhost:8000
   - Grafana: http://localhost:3001 (admin/admin)
   - Prometheus: http://localhost:9090

### Manual Setup

1. **Start the database**
   ```bash
   docker-compose up -d db redis
   ```

2. **Run database migrations**
   ```bash
   psql -h localhost -U postgres -d autoscaler -f init.sql
   ```

3. **Start the Python API**
   ```bash
   cd lib
   python prediction-engine.py
   # Or with FastAPI
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

4. **Start the Next.js app**
   ```bash
   npm run dev
   ```

## 📊 Usage

### Dashboard Overview
The main dashboard provides:
- **Real-time metrics** for all monitored resources
- **Traffic predictions** for the next 24 hours
- **Cost optimization** recommendations
- **Failure predictions** and health scores
- **Scaling history** and decision logs

### API Endpoints

#### Predictions
```bash
# Get traffic predictions
POST /api/predictions
{
  "resource_id": "web-server-cluster-1",
  "hours_ahead": 24,
  "include_failures": true,
  "include_cost_optimization": true
}

# Get cached predictions
GET /api/predictions?resource_id=web-server-cluster-1
```

#### Scaling
```bash
# Auto-scale based on predictions
POST /api/scaling
{
  "resource_id": "web-server-cluster-1",
  "action": "auto"
}

# Manual scaling
POST /api/scaling
{
  "resource_id": "web-server-cluster-1",
  "action": "scale_up",
  "target_instances": 5
}

# Get scaling history
GET /api/scaling?resource_id=web-server-cluster-1&hours=24
```

#### Resources
```bash
# List all resources
GET /api/resources

# Create new resource
POST /api/resources
{
  "id": "new-cluster",
  "type": "ec2",
  "provider": "aws",
  "min_instances": 1,
  "max_instances": 10,
  "current_instances": 2,
  "target_cpu": 70,
  "target_memory": 80,
  "cost_per_hour": 0.0416
}
```

### Python ML API
```bash
# Direct prediction API
POST http://localhost:8000/predict/web-server-cluster-1
{
  "hours_ahead": 24,
  "include_failures": true,
  "include_cost_optimization": true
}

# Anomaly detection
GET http://localhost:8000/anomalies/web-server-cluster-1?hours=24
```

## 🔧 Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/autoscaler

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Cloud Providers
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

GCP_PROJECT_ID=your-project-id
GCP_ZONE=us-central1-a
GCP_KEY_FILE=path/to/service-account.json

AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret
AZURE_TENANT_ID=your-tenant-id

# Kubernetes
KUBECONFIG=path/to/kubeconfig
```

### Model Configuration
Models can be configured in the database `system_config` table:

```sql
UPDATE system_config 
SET value = '{
  "enabled": ["prophet", "lstm", "ensemble"],
  "default": "ensemble",
  "retrain_interval_hours": 24,
  "confidence_threshold": 0.7
}'
WHERE key = 'prediction_models';
```

### Scaling Policy
```sql
UPDATE system_config 
SET value = '{
  "cooldown_minutes": 5,
  "scale_up_threshold": 0.7,
  "scale_down_threshold": 0.3,
  "max_scale_factor": 2.0,
  "safety_margin": 1.2
}'
WHERE key = 'scaling_policy';
```

## 📈 Monitoring

### Metrics Collected
- **System Metrics**: CPU, Memory, Disk, Network usage
- **Application Metrics**: Request rate, response time, error rate
- **Cost Metrics**: Hourly costs, optimization savings
- **Prediction Metrics**: Model accuracy, confidence scores
- **Scaling Metrics**: Scaling frequency, success rate

### Alerts
The system automatically generates alerts for:
- High failure prediction probability (>70%)
- Scaling failures
- Cost optimization opportunities (>10% savings)
- Anomaly detection
- Model accuracy degradation

### Grafana Dashboards
Pre-configured dashboards include:
- Infrastructure Overview
- Prediction Accuracy
- Cost Analysis
- Scaling Performance
- System Health

## 🧪 Testing

### Unit Tests
```bash
npm test
python -m pytest tests/
```

### Integration Tests
```bash
npm run test:integration
python -m pytest tests/integration/
```

### Load Testing
```bash
# Generate sample traffic data
python scripts/generate_test_data.py

# Run prediction accuracy tests
python scripts/test_predictions.py
```

## 🚀 Deployment

### Production Deployment
1. **Configure environment variables** for production
2. **Set up cloud provider credentials**
3. **Deploy with Docker Compose**:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

### Kubernetes Deployment
```bash
kubectl apply -f k8s/
```

### Cloud-specific Deployments
- **AWS**: Use ECS or EKS with provided CloudFormation templates
- **GCP**: Deploy to GKE with provided deployment configs
- **Azure**: Use AKS with provided ARM templates

## 🔒 Security

- **API Authentication**: JWT-based authentication for API endpoints
- **Database Security**: Encrypted connections and user permissions
- **Cloud Credentials**: Secure credential management with IAM roles
- **Network Security**: VPC isolation and security groups
- **Monitoring**: Security event logging and alerting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

- **Documentation**: Check the `/docs` folder for detailed guides
- **Issues**: Report bugs and feature requests via GitHub Issues
- **Community**: Join our Discord server for discussions
- **Email**: support@predictive-autoscaler.com

## 🎯 Roadmap

### v2.0 Planned Features
- [ ] Multi-region scaling coordination
- [ ] Advanced cost models (spot instances, reserved capacity)
- [ ] Custom metric integration
- [ ] Mobile dashboard app
- [ ] Advanced ML models (Transformers, Graph Neural Networks)
- [ ] Integration with service mesh (Istio, Linkerd)
- [ ] Chaos engineering integration
- [ ] Automated performance testing

### v3.0 Vision
- [ ] Self-healing infrastructure
- [ ] Predictive security scaling
- [ ] Carbon footprint optimization
- [ ] Quantum-resistant algorithms
- [ ] Edge computing support

---

Built with ❤️ for the cloud-native community. Save money, prevent outages, scale smarter.
