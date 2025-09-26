-- Database initialization script for Predictive Auto-Scaler

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS autoscaler;
CREATE SCHEMA IF NOT EXISTS metrics;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Set search path
SET search_path TO autoscaler, public;

-- Resources table
CREATE TABLE IF NOT EXISTS resources (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL CHECK (type IN ('ec2', 'gce', 'azure_vm', 'kubernetes')),
    provider VARCHAR(20) NOT NULL CHECK (provider IN ('aws', 'gcp', 'azure', 'k8s')),
    min_instances INTEGER NOT NULL DEFAULT 1,
    max_instances INTEGER NOT NULL DEFAULT 10,
    current_instances INTEGER NOT NULL DEFAULT 1,
    target_cpu DECIMAL(5,2) NOT NULL DEFAULT 70.0,
    target_memory DECIMAL(5,2) NOT NULL DEFAULT 80.0,
    cost_per_hour DECIMAL(10,4) NOT NULL DEFAULT 0.0,
    region VARCHAR(50),
    availability_zone VARCHAR(50),
    instance_type VARCHAR(50),
    tags JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT check_instances CHECK (min_instances <= current_instances AND current_instances <= max_instances)
);

-- Metrics table for storing historical data
CREATE TABLE IF NOT EXISTS metrics.resource_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource_id VARCHAR(255) REFERENCES resources(id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,4) NOT NULL,
    unit VARCHAR(20),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    tags JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Scaling actions log
CREATE TABLE IF NOT EXISTS scaling_actions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource_id VARCHAR(255) REFERENCES resources(id) ON DELETE CASCADE,
    action VARCHAR(20) NOT NULL CHECK (action IN ('scale_up', 'scale_down', 'maintain')),
    previous_instances INTEGER NOT NULL,
    target_instances INTEGER NOT NULL,
    actual_instances INTEGER,
    reason TEXT,
    confidence DECIMAL(5,4),
    cost_impact DECIMAL(10,4),
    execution_status VARCHAR(20) DEFAULT 'pending' CHECK (execution_status IN ('pending', 'success', 'failed', 'timeout')),
    execution_time_ms INTEGER,
    error_message TEXT,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Traffic predictions
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource_id VARCHAR(255) REFERENCES resources(id) ON DELETE CASCADE,
    prediction_type VARCHAR(50) NOT NULL CHECK (prediction_type IN ('traffic', 'failure', 'cost')),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    prediction_horizon_hours INTEGER NOT NULL,
    prediction_data JSONB NOT NULL,
    confidence_score DECIMAL(5,4),
    accuracy_score DECIMAL(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    valid_until TIMESTAMP WITH TIME ZONE NOT NULL
);

-- Failure predictions
CREATE TABLE IF NOT EXISTS failure_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource_id VARCHAR(255) REFERENCES resources(id) ON DELETE CASCADE,
    failure_type VARCHAR(100) NOT NULL,
    failure_probability DECIMAL(5,4) NOT NULL,
    predicted_failure_time TIMESTAMP WITH TIME ZONE,
    contributing_factors JSONB,
    recommended_action TEXT,
    severity VARCHAR(20) DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'resolved', 'false_positive')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- Cost optimization recommendations
CREATE TABLE IF NOT EXISTS cost_optimizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource_id VARCHAR(255) REFERENCES resources(id) ON DELETE CASCADE,
    optimization_type VARCHAR(50) NOT NULL,
    current_cost_per_hour DECIMAL(10,4) NOT NULL,
    optimized_cost_per_hour DECIMAL(10,4) NOT NULL,
    savings_percentage DECIMAL(5,2) NOT NULL,
    recommendations JSONB NOT NULL,
    implementation_status VARCHAR(20) DEFAULT 'pending' CHECK (implementation_status IN ('pending', 'implemented', 'rejected')),
    estimated_monthly_savings DECIMAL(12,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    implemented_at TIMESTAMP WITH TIME ZONE
);

-- Monitoring alerts
CREATE TABLE IF NOT EXISTS monitoring.alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource_id VARCHAR(255) REFERENCES resources(id) ON DELETE CASCADE,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('info', 'warning', 'critical')),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    alert_data JSONB,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'acknowledged', 'resolved')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    acknowledged_by VARCHAR(255),
    resolved_by VARCHAR(255)
);

-- System configuration
CREATE TABLE IF NOT EXISTS system_config (
    key VARCHAR(100) PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    category VARCHAR(50),
    updated_by VARCHAR(255),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_metrics_resource_timestamp ON metrics.resource_metrics(resource_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics.resource_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics.resource_metrics(metric_name);

CREATE INDEX IF NOT EXISTS idx_scaling_actions_resource ON scaling_actions(resource_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_scaling_actions_timestamp ON scaling_actions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_scaling_actions_status ON scaling_actions(execution_status);

CREATE INDEX IF NOT EXISTS idx_predictions_resource ON predictions(resource_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_valid ON predictions(valid_until);
CREATE INDEX IF NOT EXISTS idx_predictions_type ON predictions(prediction_type);

CREATE INDEX IF NOT EXISTS idx_failure_predictions_resource ON failure_predictions(resource_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_failure_predictions_probability ON failure_predictions(failure_probability DESC);
CREATE INDEX IF NOT EXISTS idx_failure_predictions_status ON failure_predictions(status);

CREATE INDEX IF NOT EXISTS idx_alerts_resource ON monitoring.alerts(resource_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON monitoring.alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON monitoring.alerts(severity);

-- Create functions and triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_resources_updated_at BEFORE UPDATE ON resources
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to clean old metrics
CREATE OR REPLACE FUNCTION clean_old_metrics(days_to_keep INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM metrics.resource_metrics 
    WHERE timestamp < NOW() - INTERVAL '1 day' * days_to_keep;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get resource health score
CREATE OR REPLACE FUNCTION get_resource_health_score(resource_id_param VARCHAR)
RETURNS DECIMAL AS $$
DECLARE
    health_score DECIMAL DEFAULT 1.0;
    critical_failures INTEGER;
    recent_scaling_failures INTEGER;
BEGIN
    -- Check for critical failure predictions
    SELECT COUNT(*) INTO critical_failures
    FROM failure_predictions 
    WHERE resource_id = resource_id_param 
    AND failure_probability > 0.7 
    AND status = 'active';
    
    -- Check for recent scaling failures
    SELECT COUNT(*) INTO recent_scaling_failures
    FROM scaling_actions 
    WHERE resource_id = resource_id_param 
    AND execution_status = 'failed'
    AND timestamp > NOW() - INTERVAL '24 hours';
    
    -- Calculate health score
    health_score = health_score - (critical_failures * 0.3) - (recent_scaling_failures * 0.2);
    health_score = GREATEST(0.0, health_score);
    
    RETURN health_score;
END;
$$ LANGUAGE plpgsql;

-- Insert sample data
INSERT INTO resources (id, name, type, provider, min_instances, max_instances, current_instances, target_cpu, target_memory, cost_per_hour, region, instance_type) VALUES
('web-server-cluster-1', 'Web Server Cluster', 'ec2', 'aws', 2, 20, 5, 70.0, 80.0, 0.0416, 'us-east-1', 't3.medium'),
('api-gateway-cluster', 'API Gateway Cluster', 'kubernetes', 'k8s', 3, 15, 6, 65.0, 75.0, 0.02, 'us-central1', 'pod'),
('database-cluster', 'Database Cluster', 'gce', 'gcp', 2, 8, 3, 80.0, 85.0, 0.0504, 'us-central1-a', 'e2-standard-2')
ON CONFLICT (id) DO NOTHING;

-- Insert system configuration
INSERT INTO system_config (key, value, description, category) VALUES
('prediction_models', '{"enabled": ["prophet", "lstm", "ensemble"], "default": "ensemble", "retrain_interval_hours": 24}', 'ML model configuration', 'ai'),
('scaling_policy', '{"cooldown_minutes": 5, "scale_up_threshold": 0.7, "scale_down_threshold": 0.3, "max_scale_factor": 2.0}', 'Auto-scaling policy settings', 'scaling'),
('cost_optimization', '{"enabled": true, "aggressive_mode": false, "min_savings_threshold": 10.0, "max_cost_increase": 5.0}', 'Cost optimization settings', 'cost'),
('monitoring', '{"metrics_retention_days": 90, "alert_cooldown_minutes": 15, "health_check_interval_seconds": 30}', 'Monitoring and alerting configuration', 'monitoring')
ON CONFLICT (key) DO NOTHING;

-- Grant permissions
GRANT USAGE ON SCHEMA autoscaler TO postgres;
GRANT USAGE ON SCHEMA metrics TO postgres;
GRANT USAGE ON SCHEMA monitoring TO postgres;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA autoscaler TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA metrics TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO postgres;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA autoscaler TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA metrics TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO postgres;
