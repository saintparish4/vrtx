use super::{CloudError, CloudProvider};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use k8s_openapi::api::apps::v1::{Deployment, StatefulSet};
use k8s_openapi::api::autoscaling::v2::HorizontalPodAutoscaler;
use kube::{
    api::{Api, Patch, PatchParams},
    Client,
};
use log::{info, warn};
use serde_json::json;

pub struct KubernetesProvider {
    client: Client,
}

impl KubernetesProvider {
    pub async fn new() -> Result<Self> {
        let client = Client::try_default()
            .await
            .map_err(|e| anyhow!("Failed to create Kubernetes client: {}", e))?;

        Ok(Self { client })
    }

    async fn scale_deployment(
        &self,
        namespace: &str,
        name: &str,
        replicas: i32,
    ) -> Result<()> {
        info!(
            "Scaling Deployment '{}/{}' to {} replicas",
            namespace, name, replicas
        );

        let deployments: Api<Deployment> = Api::namespaced(self.client.clone(), namespace);

        let patch = json!({
            "spec": {
                "replicas": replicas
            }
        });

        deployments
            .patch(
                name,
                &PatchParams::default(),
                &Patch::Strategic(patch),
            )
            .await
            .map_err(|e| anyhow!("Failed to scale deployment: {}", e))?;

        info!(
            "Successfully scaled Deployment '{}/{}' to {} replicas",
            namespace, name, replicas
        );

        Ok(())
    }

    async fn scale_statefulset(
        &self,
        namespace: &str,
        name: &str,
        replicas: i32,
    ) -> Result<()> {
        info!(
            "Scaling StatefulSet '{}/{}' to {} replicas",
            namespace, name, replicas
        );

        let statefulsets: Api<StatefulSet> = Api::namespaced(self.client.clone(), namespace);

        let patch = json!({
            "spec": {
                "replicas": replicas
            }
        });

        statefulsets
            .patch(
                name,
                &PatchParams::default(),
                &Patch::Strategic(patch),
            )
            .await
            .map_err(|e| anyhow!("Failed to scale statefulset: {}", e))?;

        info!(
            "Successfully scaled StatefulSet '{}/{}' to {} replicas",
            namespace, name, replicas
        );

        Ok(())
    }

    async fn get_deployment_replicas(&self, namespace: &str, name: &str) -> Result<i32> {
        let deployments: Api<Deployment> = Api::namespaced(self.client.clone(), namespace);

        let deployment = deployments
            .get(name)
            .await
            .map_err(|e| anyhow!("Failed to get deployment: {}", e))?;

        Ok(deployment
            .spec
            .and_then(|spec| spec.replicas)
            .unwrap_or(0))
    }

    async fn get_statefulset_replicas(&self, namespace: &str, name: &str) -> Result<i32> {
        let statefulsets: Api<StatefulSet> = Api::namespaced(self.client.clone(), namespace);

        let statefulset = statefulsets
            .get(name)
            .await
            .map_err(|e| anyhow!("Failed to get statefulset: {}", e))?;

        Ok(statefulset
            .spec
            .and_then(|spec| spec.replicas)
            .unwrap_or(0))
    }

    async fn validate_deployment(&self, namespace: &str, name: &str) -> Result<bool> {
        let deployments: Api<Deployment> = Api::namespaced(self.client.clone(), namespace);

        match deployments.get(name).await {
            Ok(_) => Ok(true),
            Err(e) => {
                warn!("Failed to validate deployment '{}/{}': {}", namespace, name, e);
                Ok(false)
            }
        }
    }

    async fn validate_statefulset(&self, namespace: &str, name: &str) -> Result<bool> {
        let statefulsets: Api<StatefulSet> = Api::namespaced(self.client.clone(), namespace);

        match statefulsets.get(name).await {
            Ok(_) => Ok(true),
            Err(e) => {
                warn!("Failed to validate statefulset '{}/{}': {}", namespace, name, e);
                Ok(false)
            }
        }
    }

    pub async fn update_hpa(
        &self,
        namespace: &str,
        name: &str,
        min_replicas: i32,
        max_replicas: i32,
    ) -> Result<()> {
        info!(
            "Updating HPA '{}/{}' to min: {}, max: {}",
            namespace, name, min_replicas, max_replicas
        );

        let hpas: Api<HorizontalPodAutoscaler> = Api::namespaced(self.client.clone(), namespace);

        let patch = json!({
            "spec": {
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas
            }
        });

        hpas.patch(
            name,
            &PatchParams::default(),
            &Patch::Strategic(patch),
        )
        .await
        .map_err(|e| anyhow!("Failed to update HPA: {}", e))?;

        info!("Successfully updated HPA '{}/{}'", namespace, name);

        Ok(())
    }

    pub async fn get_hpa_status(&self, namespace: &str, name: &str) -> Result<(i32, i32, i32)> {
        let hpas: Api<HorizontalPodAutoscaler> = Api::namespaced(self.client.clone(), namespace);

        let hpa = hpas
            .get(name)
            .await
            .map_err(|e| anyhow!("Failed to get HPA: {}", e))?;

        let current = hpa
            .status
            .as_ref()
            .and_then(|s| s.current_replicas)
            .unwrap_or(0);

        let min = hpa
            .spec
            .as_ref()
            .and_then(|s| s.min_replicas)
            .unwrap_or(1);

        let max = hpa
            .spec
            .as_ref()
            .map(|s| s.max_replicas)
            .unwrap_or(10);

        Ok((current, min, max))
    }

    // Parse resource_id format: "deployment:namespace/name" or "statefulset:namespace/name"
    fn parse_resource_id(&self, resource_id: &str) -> Result<(String, String, String)> {
        let parts: Vec<&str> = resource_id.split(':').collect();
        if parts.len() != 2 {
            return Err(anyhow!(
                "Invalid resource_id format. Expected 'type:namespace/name'"
            ));
        }

        let resource_type = parts[0];
        let ns_name: Vec<&str> = parts[1].split('/').collect();
        
        if ns_name.len() != 2 {
            return Err(anyhow!(
                "Invalid resource_id format. Expected 'type:namespace/name'"
            ));
        }

        Ok((
            resource_type.to_string(),
            ns_name[0].to_string(),
            ns_name[1].to_string(),
        ))
    }
}

#[async_trait]
impl CloudProvider for KubernetesProvider {
    async fn scale_to(&self, resource_id: &str, target_instances: i32) -> Result<()> {
        let (resource_type, namespace, name) = self.parse_resource_id(resource_id)?;

        match resource_type.as_str() {
            "deployment" => self.scale_deployment(&namespace, &name, target_instances).await,
            "statefulset" => self.scale_statefulset(&namespace, &name, target_instances).await,
            _ => Err(anyhow!("Unsupported resource type: {}", resource_type)),
        }
    }

    async fn get_current_instances(&self, resource_id: &str) -> Result<i32> {
        let (resource_type, namespace, name) = self.parse_resource_id(resource_id)?;

        match resource_type.as_str() {
            "deployment" => self.get_deployment_replicas(&namespace, &name).await,
            "statefulset" => self.get_statefulset_replicas(&namespace, &name).await,
            _ => Err(anyhow!("Unsupported resource type: {}", resource_type)),
        }
    }

    async fn validate_resource(&self, resource_id: &str) -> Result<bool> {
        let (resource_type, namespace, name) = self.parse_resource_id(resource_id)?;

        match resource_type.as_str() {
            "deployment" => self.validate_deployment(&namespace, &name).await,
            "statefulset" => self.validate_statefulset(&namespace, &name).await,
            _ => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Ignore by default as it requires k8s cluster
    async fn test_kubernetes_provider_creation() {
        let provider = KubernetesProvider::new().await;
        assert!(provider.is_ok());
    }

    #[test]
    fn test_parse_resource_id() {
        let provider = KubernetesProvider {
            client: Client::try_default().await.unwrap(),
        };

        let (rtype, ns, name) = provider
            .parse_resource_id("deployment:default/my-app")
            .unwrap();
        
        assert_eq!(rtype, "deployment");
        assert_eq!(ns, "default");
        assert_eq!(name, "my-app");
    }
}