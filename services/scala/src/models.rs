use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    pub id: Uuid,
    pub name: String,
    pub resource_type: String, // ec2 or kubernetes
    pub cloud_provider: String, // aws or gcp or azure or k8s
    pub min_instances: i32,
    pub max_instances: i32,
    pub current_instances: i32,
    pub target_cpu: f64,
    pub target_memory: f64,
    pub cooldown_period: i32, // seconds
    pub created_at: DateTime<Utc>, 
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConstraints {
    pub min_instances: i32,
    pub max_instances: i32,
    pub max_scale_up_step: i32,
    pub max_scale_down_step: i32,
    pub cooldown_seconds: i32,
    pub min_time_between_scales: i32, 
}

impl Default for SafetyConstraints {
    fn default() -> Self {
        Self {
            min_instances: 1,
            max_instances: 10,
            max_scale_up_step: 3,
            max_scale_down_step: 2,
            cooldown_seconds: 300,
            min_time_between_scales: 60, 
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionData {
    pub resource_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub predicted_cpu: f64,
    pub predicted_memory: f64,
    pub predicted_requests: f64,
    pub confidence: f64, 
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingDecision {
    pub id: Uuid,
    pub resource_id: Uuid,
    pub decision_type: DecisionType,
    pub current_instances: i32,
    pub target_instances: i32,
    pub reason: String,
    pub estimated_cost: f64,
    pub confidence: f64,
    pub created_at: DateTime<Utc>,
    pub executed: bool, 
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DecisionType {
    ScaleUp,
    ScaleDown,
    NoAction, 
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    pub hourly_cost: f64,
    pub daily_cost: f64,
    pub monthly_cost: f64,
    pub instance_type: String,
    pub instance_count: i32, 
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingEvent {
    pub id: Uuid,
    pub resource_id: Uuid,
    pub decision_id: Uuid,
    pub action: String,
    pub previous_instances: i32,
    pub new_instances: i32,
    pub success: bool,
    pub error_message: Option<String>,
    pub execution_time_ms: i64,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ScaleRequest {
    pub resource_id: Uuid,
    pub use_prediction: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ScaleExecuteRequest {
    pub decision_id: Uuid,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HistoryQuery {
    pub resource_id: Option<Uuid>,
    pub limit: Option<i32>,
    pub offset: Option<i32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CostQuery {
    pub resource_id: Uuid,
    pub instance_count: i32,
}