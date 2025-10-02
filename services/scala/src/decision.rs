use crate::models::*;
use anyhow::{anyhow, Result};
use chrono::Utc;
use uuid::Uuid;

pub struct ScalingDecisionEngine;

impl ScalingDecisionEngine {
    pub fn new() -> Self {
        Self
    }

    pub fn make_decision(
        &self,
        resource: &ResourceConfig,
        prediction: Option<&PredictionData>,
        constraints: &SafetyConstraints,
    ) -> Result<ScalingDecision> {
        let current = resource.current_instances;
        
        // Determine target instances based on prediction or current metrics
        let target = if let Some(pred) = prediction {
            self.calculate_required_instances(resource, pred)?
        } else {
            // Without prediction, maintain current if within bounds
            current
        };

        // Apply safety constraints
        let safe_target = self.apply_constraints(current, target, constraints);
        
        // Determine decision type
        let decision_type = if safe_target > current {
            DecisionType::ScaleUp
        } else if safe_target < current {
            DecisionType::ScaleDown
        } else {
            DecisionType::NoAction
        };

        // Generate reason
        let reason = self.generate_reason(
            &decision_type,
            current,
            safe_target,
            prediction,
            resource,
        );

        // Estimate cost
        let estimated_cost = self.estimate_cost(&resource.resource_type, safe_target);

        Ok(ScalingDecision {
            id: Uuid::new_v4(),
            resource_id: resource.id,
            decision_type,
            current_instances: current,
            target_instances: safe_target,
            reason,
            estimated_cost,
            confidence: prediction.map(|p| p.confidence).unwrap_or(0.5),
            created_at: Utc::now(),
            executed: false,
        })
    }

    fn calculate_required_instances(
        &self,
        resource: &ResourceConfig,
        prediction: &PredictionData,
    ) -> Result<i32> {
        // Calculate instances needed based on CPU utilization
        let cpu_based = if prediction.predicted_cpu > 0.0 {
            ((prediction.predicted_cpu / resource.target_cpu) * resource.current_instances as f64)
                .ceil() as i32
        } else {
            resource.current_instances
        };

        // Calculate instances needed based on memory utilization
        let memory_based = if prediction.predicted_memory > 0.0 {
            ((prediction.predicted_memory / resource.target_memory)
                * resource.current_instances as f64)
                .ceil() as i32
        } else {
            resource.current_instances
        };

        // Take the maximum to ensure we have enough capacity
        let required = cpu_based.max(memory_based);

        // Add buffer for safety (10% overhead)
        let with_buffer = (required as f64 * 1.1).ceil() as i32;

        Ok(with_buffer)
    }

    fn apply_constraints(
        &self,
        current: i32,
        target: i32,
        constraints: &SafetyConstraints,
    ) -> i32 {
        let mut safe_target = target;

        // Enforce min/max bounds
        safe_target = safe_target.max(constraints.min_instances);
        safe_target = safe_target.min(constraints.max_instances);

        // Enforce max step size
        if safe_target > current {
            let max_up = current + constraints.max_scale_up_step;
            safe_target = safe_target.min(max_up);
        } else if safe_target < current {
            let max_down = current - constraints.max_scale_down_step;
            safe_target = safe_target.max(max_down);
        }

        safe_target
    }

    fn generate_reason(
        &self,
        decision_type: &DecisionType,
        current: i32,
        target: i32,
        prediction: Option<&PredictionData>,
        resource: &ResourceConfig,
    ) -> String {
        match decision_type {
            DecisionType::ScaleUp => {
                if let Some(pred) = prediction {
                    format!(
                        "Scaling up from {} to {} instances. Predicted CPU: {:.2}%, Memory: {:.2}%. Target CPU: {:.2}%, Memory: {:.2}%",
                        current, target, pred.predicted_cpu, pred.predicted_memory,
                        resource.target_cpu, resource.target_memory
                    )
                } else {
                    format!("Scaling up from {} to {} instances based on current metrics", current, target)
                }
            }
            DecisionType::ScaleDown => {
                if let Some(pred) = prediction {
                    format!(
                        "Scaling down from {} to {} instances. Predicted CPU: {:.2}%, Memory: {:.2}% below target thresholds",
                        current, target, pred.predicted_cpu, pred.predicted_memory
                    )
                } else {
                    format!("Scaling down from {} to {} instances to optimize costs", current, target)
                }
            }
            DecisionType::NoAction => {
                format!("No scaling required. Current {} instances optimal", current)
            }
        }
    }

    fn estimate_cost(&self, resource_type: &str, instance_count: i32) -> f64 {
        // Basic cost estimation (hourly rate per instance)
        let hourly_rate = match resource_type {
            "ec2" => 0.10, // $0.10/hour per t3.medium instance (example)
            "kubernetes" => 0.05, // $0.05/hour per pod (example)
            _ => 0.08,
        };

        hourly_rate * instance_count as f64
    }

    pub fn validate_constraints(&self, constraints: &SafetyConstraints) -> Result<()> {
        if constraints.min_instances < 0 {
            return Err(anyhow!("min_instances cannot be negative"));
        }

        if constraints.max_instances < constraints.min_instances {
            return Err(anyhow!(
                "max_instances must be greater than or equal to min_instances"
            ));
        }

        if constraints.max_scale_up_step < 1 {
            return Err(anyhow!("max_scale_up_step must be at least 1"));
        }

        if constraints.max_scale_down_step < 1 {
            return Err(anyhow!("max_scale_down_step must be at least 1"));
        }

        Ok(())
    }

    pub fn calculate_cost_estimate(
        &self,
        resource_type: &str,
        instance_count: i32,
    ) -> CostEstimate {
        let hourly = self.estimate_cost(resource_type, instance_count);
        
        CostEstimate {
            hourly_cost: hourly,
            daily_cost: hourly * 24.0,
            monthly_cost: hourly * 24.0 * 30.0,
            instance_type: resource_type.to_string(),
            instance_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_apply_constraints() {
        let engine = ScalingDecisionEngine::new();
        let constraints = SafetyConstraints {
            min_instances: 2,
            max_instances: 10,
            max_scale_up_step: 3,
            max_scale_down_step: 2,
            cooldown_seconds: 300,
            min_time_between_scales: 60,
        };

        // Test max scale up
        assert_eq!(engine.apply_constraints(5, 10, &constraints), 8);

        // Test max scale down
        assert_eq!(engine.apply_constraints(5, 1, &constraints), 3);

        // Test min constraint
        assert_eq!(engine.apply_constraints(3, 1, &constraints), 2);

        // Test max constraint
        assert_eq!(engine.apply_constraints(8, 15, &constraints), 10);
    }

    #[test]
    fn test_validate_constraints() {
        let engine = ScalingDecisionEngine::new();
        
        let valid = SafetyConstraints::default();
        assert!(engine.validate_constraints(&valid).is_ok());

        let invalid = SafetyConstraints {
            min_instances: 10,
            max_instances: 5,
            ..Default::default()
        };
        assert!(engine.validate_constraints(&invalid).is_err());
    }
}