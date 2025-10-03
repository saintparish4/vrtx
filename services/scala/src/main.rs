mod cloud;
mod db;
mod decision;
mod models;

use actix_web::{
    middleware::Logger,
    web::{self, Data},
    App, HttpResponse, HttpServer, Result as ActixResult,
};
use anyhow::Result;
use chrono::Utc;
use cloud::{aws::AwsProvider, kubernetes::KubernetesProvider, CloudProvider};
use db::Database;
use decision::ScalingDecisionEngine;
use log::info;
use models::*;
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

struct AppState {
    db: Arc<Database>,
    decision_engine: Arc<ScalingDecisionEngine>,
    aws_provider: Arc<AwsProvider>,
    k8s_provider: Arc<KubernetesProvider>,
}

#[actix_web::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();
    env_logger::init();

    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5432/predictive_scaler".to_string());
    let redis_url = std::env::var("REDIS_URL")
        .unwrap_or_else(|_| "redis://localhost:6379".to_string());
    let host = std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = std::env::var("PORT").unwrap_or_else(|_| "8082".to_string());

    info!("Connecting to database...");
    let db = Arc::new(Database::new(&database_url, &redis_url).await?);

    info!("Initializing cloud providers...");
    let aws_provider = Arc::new(AwsProvider::new().await?);
    let k8s_provider = Arc::new(KubernetesProvider::new().await?);

    let decision_engine = Arc::new(ScalingDecisionEngine::new());

    let app_state = Data::new(AppState {
        db,
        decision_engine,
        aws_provider,
        k8s_provider,
    });

    info!("Starting server on {}:{}", host, port);

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .wrap(Logger::default())
            .route("/health", web::get().to(health_check))
            .route("/scale/decide", web::post().to(make_scaling_decision))
            .route("/scale/execute", web::post().to(execute_scaling))
            .route("/scale/history", web::get().to(get_scaling_history))
            .route("/cost/estimate", web::post().to(estimate_cost))
    })
    .bind(format!("{}:{}", host, port))?
    .run()
    .await?;

    Ok(())
}

async fn health_check() -> ActixResult<HttpResponse> {
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "status": "healthy",
        "service": "scaler",
        "timestamp": Utc::now()
    })))
}

async fn make_scaling_decision(
    state: Data<AppState>,
    req: web::Json<ScaleRequest>,
) -> ActixResult<HttpResponse> {
    let resource = state
        .db
        .get_resource(req.resource_id)
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

    // Check cooldown
    let can_scale = state
        .db
        .check_cooldown(req.resource_id, resource.cooldown_period)
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

    if !can_scale {
        return Ok(HttpResponse::TooManyRequests().json(serde_json::json!({
            "error": "Resource is in cooldown period",
            "resource_id": req.resource_id
        })));
    }

    // Get prediction if requested
    let prediction = if req.use_prediction {
        state
            .db
            .get_latest_prediction(req.resource_id)
            .await
            .map_err(|e| actix_web::error::ErrorInternalServerError(e))?
    } else {
        None
    };

    // Create constraints from resource config
    let constraints = SafetyConstraints {
        min_instances: resource.min_instances,
        max_instances: resource.max_instances,
        max_scale_up_step: 3,
        max_scale_down_step: 2,
        cooldown_seconds: resource.cooldown_period,
        min_time_between_scales: 60,
    };

    // Make decision
    let decision = state
        .decision_engine
        .make_decision(&resource, prediction.as_ref(), &constraints)
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

    // Save decision
    state
        .db
        .save_decision(&decision)
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

    Ok(HttpResponse::Ok().json(decision))
}

async fn execute_scaling(
    state: Data<AppState>,
    req: web::Json<ScaleExecuteRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();

    // Get the decision
    let decision = state
        .db
        .get_decision(req.decision_id)
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

    if decision.executed {
        return Ok(HttpResponse::BadRequest().json(serde_json::json!({
            "error": "Decision already executed"
        })));
    }

    if decision.decision_type == DecisionType::NoAction {
        return Ok(HttpResponse::Ok().json(serde_json::json!({
            "message": "No scaling action required"
        })));
    }

    // Get resource details
    let resource = state
        .db
        .get_resource(decision.resource_id)
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

    // Execute scaling based on cloud provider
    let result = match resource.cloud_provider.as_str() {
        "aws" => {
            state
                .aws_provider
                .scale_to(&resource.name, decision.target_instances)
                .await
        }
        "kubernetes" => {
            state
                .k8s_provider
                .scale_to(&resource.name, decision.target_instances)
                .await
        }
        _ => Err(anyhow::anyhow!("Unsupported cloud provider")),
    };

    let execution_time = start_time.elapsed().as_millis() as i64;

    // Create scaling event
    let event = ScalingEvent {
        id: Uuid::new_v4(),
        resource_id: decision.resource_id,
        decision_id: decision.id,
        action: format!("{:?}", decision.decision_type),
        previous_instances: decision.current_instances,
        new_instances: decision.target_instances,
        success: result.is_ok(),
        error_message: result.as_ref().err().map(|e| e.to_string()),
        execution_time_ms: execution_time,
        created_at: Utc::now(),
    };

    // Save event
    state
        .db
        .save_scaling_event(&event)
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

    if result.is_ok() {
        // Mark decision as executed
        state
            .db
            .mark_decision_executed(decision.id)
            .await
            .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

        // Update resource current instances
        state
            .db
            .update_resource_instances(decision.resource_id, decision.target_instances)
            .await
            .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

        Ok(HttpResponse::Ok().json(serde_json::json!({
            "success": true,
            "event": event,
            "execution_time_ms": execution_time
        })))
    } else {
        Ok(HttpResponse::InternalServerError().json(serde_json::json!({
            "success": false,
            "error": result.err().unwrap().to_string(),
            "event": event
        })))
    }
}

async fn get_scaling_history(
    state: Data<AppState>,
    query: web::Query<HistoryQuery>,
) -> ActixResult<HttpResponse> {
    let limit = query.limit.unwrap_or(50).min(100);
    let offset = query.offset.unwrap_or(0);

    let events = state
        .db
        .get_scaling_history(query.resource_id, limit, offset)
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "events": events,
        "count": events.len(),
        "limit": limit,
        "offset": offset
    })))
}

async fn estimate_cost(
    state: Data<AppState>,
    req: web::Json<CostQuery>,
) -> ActixResult<HttpResponse> {
    let resource = state
        .db
        .get_resource(req.resource_id)
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

    let estimate = state
        .decision_engine
        .calculate_cost_estimate(&resource.resource_type, req.instance_count);

    Ok(HttpResponse::Ok().json(estimate))
}