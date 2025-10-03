use super::{CloudError, CloudProvider};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use aws_sdk_autoscaling::Client as AutoScalingClient;
use aws_sdk_ec2::Client as EC2Client;
use log::{info, warn};

pub struct AwsProvider {
    ec2_client: EC2Client,
    autoscaling_client: AutoScalingClient, 
}

impl AwsProvider {
    pub async fn new() -> Result<Self> {
        let config = aws_config::from_env().load().await;
        let ec2_client = EC2Client::new(&config);
        let autoscaling_client = AutoScalingClient::new(&config);

        Ok(Self {
            ec2_client,
            autoscaling_client, 
        })
    }

    async fn scale_auto_scaling_group(&self, asg_name: &str, desired_capacity: i32) -> Result<()> {
        info!(
            "Scaling Auto Scaling Group: '{}' to {} instances",
            asg_name, desired_capacity 
        );

        self.autoscaling_client
            .set_desired_capacity()
            .auto_scaling_group_name(asg_name)
            .desired_capacity(desired_capacity)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to scale ASG: {}", e))?;
            
        info!(
            "Successfully scaled ASG: '{}' to {} instances",
            asg_name, desired_capacity 
        );

        Ok(()) 
    }

    async fn get_asg_capacity(&self, asg_name: &str) -> Result<i32> {
        let result = self
            .autoscaling_client
            .describe_auto_scaling_groups()
            .auto_scaling_group_names(asg_name)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to get ASG capacity: {}", e))?;

        let asg = result 
            .auto_scaling_groups()
            .first()
            .ok_or_else(|| CloudError::ResourceNotFound(format!("ASG '{}' not found", asg_name)))?;

        Ok(asg.desired_capacity().unwrap_or(0)) 
    }

    async fn validate_asg(&self, asg_name: &str) -> Result<bool> {
        let result = self 
            .autoscaling_client
            .describe_auto_scaling_groups()
            .auto_scaling_group_names(asg_name)
            .send()
            .await;

        match result {
            Ok(response) => Ok(!response.auto_scaling_groups().is_empty()),
            Err(e) => {
                warn!("Failed to validate ASG: '{}' : {}", asg_name, e);
                Ok(false) 
            }
        }
    }

    pub async fn terminate_instances(&self, instance_ids: Vec<String>) -> Result<()> {
        if instance_ids.is_empty() {
            return Ok(());  
        }

        info!("Terminating {} EC2 instances", instance_ids.len());

        self.ec2_client
            .terminate_instances()
            .set_instance_ids(Some(instance_ids))
            .send()
            .await
            .map_err(|e| anyhow!("Failed to terminate instances: {}", e))?;

        info!("Successfully initiated termination for instances");

        Ok(()) 
    }

    pub async fn launch_instances(
        &self,
        image_id: &str,
        instance_type: &str,
        count: i32,
        subnet_id: Option<&str>,
        security_group_ids: Vec<String>, 
    ) -> Result<Vec<String>> {
        info!(
            "Launching {} instances of type '{}' with AMI '{}'",
            count, instance_type, image_id 
        );

        let mut request = self 
            .ec2_client
            .run_instances()
            .image_id(image_id)
            .instance_type(instance_type.parse().unwrap())
            .min_count(count)
            .max_count(count);

        if let Some(subnet) = subnet_id {
            request = request.subnet_id(subnet); 
        }

        for sg in security_group_ids {
            request = request.security_group_ids(sg); 
        }

        let result = request 
            .send()
            .await
            .map_err(|e| anyhow!("Failed to launch instances: {}", e))?;

        let instance_ids: Vec<String> = result 
            .instances()
            .iter()
            .map(|i| i.instance_id().unwrap_or("").to_string())
            .collect();

        info!("Successfully launched {} instances", instance_ids.len());

        Ok(instance_ids) 
    }

    pub async fn get_instance_status(&self, instance_ids: Vec<String>) -> Result<Vec<String>> {
        let result = self
            .ec2_client
            .describe_instance_status()
            .set_instance_ids(Some(instance_ids))
            .send()
            .await
            .map_err(|e| anyhow!("Failed to get instance status: {}", e))?;

        let statuses: Vec<String> = result
            .instance_statuses()
            .iter()
            .filter_map(|s| s.instance_state().and_then(|state| state.name().map(|n| n.as_str().to_string())))
            .collect();

        Ok(statuses)
    }
}

#[async_trait]
impl CloudProvider for AwsProvider {
    async fn scale_to(&self, resource_id: &str, target_instances: i32) -> Result<()> {
        // Assume resource_id is the Auto Scaling Group name
        self.scale_auto_scaling_group(resource_id, target_instances)
            .await
    }

    async fn get_current_instances(&self, resource_id: &str) -> Result<i32> {
        self.get_asg_capacity(resource_id).await
    }

    async fn validate_resource(&self, resource_id: &str) -> Result<bool> {
        self.validate_asg(resource_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Ignore by default as it requires AWS credentials
    async fn test_aws_provider_creation() {
        let provider = AwsProvider::new().await;
        assert!(provider.is_ok());
    }
}