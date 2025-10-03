use anyhow::Result;
use async_trait::async_trait;

pub mod aws;
pub mod kubernetes;

#[async_trait]
pub trait CloudProvider: Send + Sync {
    async fn scale_to(&self, resource_id: &str, target_instances: i32) -> Result<()>;
    async fn get_current_instances(&self, resource_id: &str) -> Result<i32>;
    async fn validate_resource(&self, resource_id: &str) -> Result<bool>;
}

#[derive(Debug)]
pub enum CloudError {
    ResourceNotFound(String),
    ScalingFailed(String),
    InvalidConfiguration(String),
    PermissionDenied(String),
}

impl std::fmt::Display for CloudError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CloudError::ResourceNotFound(msg) => write!(f, "Resource not found: {}", msg),
            CloudError::ScalingFailed(msg) => write!(f, "Scaling failed: {}", msg),
            CloudError::InvalidConfiguration(msg) => write!(f, "Invalid configuration: {}", msg),
            CloudError::PermissionDenied(msg) => write!(f, "Permission denied: {}", msg),
        }
    }
}

impl std::error::Error for CloudError {}