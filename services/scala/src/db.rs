use crate::models::*;
use anyhow::Result;
use chrono::Utc;
use redis::AsyncCommands;
use sqlx::{PgPool, Row};
use uuid::Uuid;

pub struct Database {
    pg_pool: PgPool,
    redis_client: redis::Client,
}

impl Database {
    pub async fn new(database_url: &str, redis_url: &str) -> Result<Self> {
        let pg_pool = PgPool::connect(database_url).await?;
        let redis_client = redis::Client::open(redis_url)?;

        Ok(Self {
            pg_pool,
            redis_client, 
        })
    }

    pub async fn get_resource(&self, resource_id: Uuid) -> Result<ResourceConfig> {
        let resource = sqlx::query_as!(
            ResourceConfig,
            r#"
            SELECT id, name, resource_type, cloud_provider, min_instances, max_instances, current_instances, target_cpu, target_memory, cooldown_period, created_at
            FROM resources
            WHERE id = $1
            "#,
            resource_id
        )
        .fetch_one(&self.pg_pool)
        .await?;

        Ok(resource)
    }

    pub async fn get_latest_prediction(&self, resource_id: Uuid) -> Result<Option<PredictionData>> {
        let prediction = sqlx::query(
            r#"
            SELECT resource_id, timestamp, predicted_cpu, predicted_memory,
                   predicted_requests, confidence
            FROM predictions
            WHERE resource_id = $1
            ORDER BY timestamp DESC
            LIMIT 1
            "#
        )
        .bind(resource_id)
        .fetch_optional(&self.pg_pool)
        .await?;

        Ok(prediction.map(|row| PredictionData {
            resource_id: row.get("resource_id"),
            timestamp: row.get("timestamp"),
            predicted_cpu: row.get("predicted_cpu"),
            predicted_memory: row.get("predicted_memory"),
            predicted_requests: row.get("predicted_requests"),
            confidence: row.get("confidence"),
        }))
    }

    pub async fn save_decision(&self, decision: &ScalingDecision) -> Result<()> {
        sqlx::query!(
            r#"
            INSERT INTO scaling_decisions 
            (id, resource_id, decision_type, current_instances, target_instances,
             reason, estimated_cost, confidence, created_at, executed)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            "#,
            decision.id,
            decision.resource_id,
            serde_json::to_string(&decision.decision_type)?,
            decision.current_instances,
            decision.target_instances,
            decision.reason,
            decision.estimated_cost,
            decision.confidence,
            decision.created_at,
            decision.executed
        )
        .execute(&self.pg_pool)
        .await?;

        Ok(())
    }

    pub async fn get_decision(&self, decision_id: Uuid) -> Result<ScalingDecision> {
        let row = sqlx::query(
            r#"
            SELECT id, resource_id, decision_type, current_instances, target_instances,
                   reason, estimated_cost, confidence, created_at, executed
            FROM scaling_decisions
            WHERE id = $1
            "#
        )
        .bind(decision_id)
        .fetch_one(&self.pg_pool)
        .await?;

        let decision_type: String = row.get("decision_type");
        
        Ok(ScalingDecision {
            id: row.get("id"),
            resource_id: row.get("resource_id"),
            decision_type: serde_json::from_str(&decision_type)?,
            current_instances: row.get("current_instances"),
            target_instances: row.get("target_instances"),
            reason: row.get("reason"),
            estimated_cost: row.get("estimated_cost"),
            confidence: row.get("confidence"),
            created_at: row.get("created_at"),
            executed: row.get("executed"),
        })
    }

    pub async fn mark_decision_executed(&self, decision_id: Uuid) -> Result<()> {
        sqlx::query!(
            "UPDATE scaling_decisions SET executed = true WHERE id = $1",
            decision_id
        )
        .execute(&self.pg_pool)
        .await?;

        Ok(())
    }

    pub async fn save_scaling_event(&self, event: &ScalingEvent) -> Result<()> {
        sqlx::query!(
            r#"
            INSERT INTO scaling_events
            (id, resource_id, decision_id, action, previous_instances, new_instances,
             success, error_message, execution_time_ms, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            "#,
            event.id,
            event.resource_id,
            event.decision_id,
            event.action,
            event.previous_instances,
            event.new_instances,
            event.success,
            event.error_message,
            event.execution_time_ms,
            event.created_at
        )
        .execute(&self.pg_pool)
        .await?;

        Ok(())
    }

    pub async fn get_scaling_history(
        &self,
        resource_id: Option<Uuid>,
        limit: i32,
        offset: i32,
    ) -> Result<Vec<ScalingEvent>> {
        let events = if let Some(rid) = resource_id {
            sqlx::query(
                r#"
                SELECT id, resource_id, decision_id, action, previous_instances,
                       new_instances, success, error_message, execution_time_ms, created_at
                FROM scaling_events
                WHERE resource_id = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
                "#
            )
            .bind(rid)
            .bind(limit)
            .bind(offset)
            .fetch_all(&self.pg_pool)
            .await?
        } else {
            sqlx::query(
                r#"
                SELECT id, resource_id, decision_id, action, previous_instances,
                       new_instances, success, error_message, execution_time_ms, created_at
                FROM scaling_events
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
                "#
            )
            .bind(limit)
            .bind(offset)
            .fetch_all(&self.pg_pool)
            .await?
        };

        Ok(events
            .into_iter()
            .map(|row| ScalingEvent {
                id: row.get("id"),
                resource_id: row.get("resource_id"),
                decision_id: row.get("decision_id"),
                action: row.get("action"),
                previous_instances: row.get("previous_instances"),
                new_instances: row.get("new_instances"),
                success: row.get("success"),
                error_message: row.get("error_message"),
                execution_time_ms: row.get("execution_time_ms"),
                created_at: row.get("created_at"),
            })
            .collect())
    }

    pub async fn update_resource_instances(&self, resource_id: Uuid, instances: i32) -> Result<()> {
        sqlx::query!(
            "UPDATE resources SET current_instances = $1 WHERE id = $2",
            instances,
            resource_id
        )
        .execute(&self.pg_pool)
        .await?;

        Ok(())
    }

    pub async fn check_cooldown(&self, resource_id: Uuid, cooldown_seconds: i32) -> Result<bool> {
        let mut conn = self.redis_client.get_async_connection().await?;
        let key = format!("cooldown:{}", resource_id);
        
        let exists: bool = conn.exists(&key).await?;
        
        if !exists {
            let _: () = conn.set_ex(&key, "1", cooldown_seconds as u64).await?;
            Ok(true) // Not in cooldown, can scale
        } else {
            Ok(false) // In cooldown, cannot scale
        }
    }

    pub async fn get_last_scaling_time(&self, resource_id: Uuid) -> Result<Option<DateTime<Utc>>> {
        let row = sqlx::query(
            r#"
            SELECT created_at
            FROM scaling_events
            WHERE resource_id = $1 AND success = true
            ORDER BY created_at DESC
            LIMIT 1
            "#
        )
        .bind(resource_id)
        .fetch_optional(&self.pg_pool)
        .await?;

        Ok(row.map(|r| r.get("created_at")))
    }
}