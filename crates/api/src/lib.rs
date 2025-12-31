//! REST API layer for neuro-compiler web applications
//!
//! This crate provides a web API that exposes neuro-compiler functionality
//! for integration with web-based user interfaces and applications.

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::CorsLayer;

/// API state containing shared resources
#[derive(Clone)]
pub struct ApiState {
    // Add any shared state here (database connections, caches, etc.)
}

/// API response types
#[derive(Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message.into()),
        }
    }
}

/// Target information response
#[derive(Serialize)]
pub struct TargetInfo {
    pub name: String,
    pub vendor: String,
    pub family: String,
    pub description: Option<String>,
}

/// Compilation request
#[derive(Deserialize)]
pub struct CompileRequest {
    pub input: serde_json::Value, // NIR graph as JSON
    pub target: String,
    pub pipeline: Option<String>,
}

/// Compilation response
#[derive(Serialize)]
pub struct CompileResponse {
    pub artifact_path: String,
    pub target: String,
    pub populations: usize,
    pub connections: usize,
    pub probes: usize,
}

/// Simulation request
#[derive(Deserialize)]
pub struct SimulateRequest {
    pub input: serde_json::Value, // NIR graph as JSON
    pub simulator: String,
    pub out_dir: Option<String>,
}

/// Simulation response
#[derive(Serialize)]
pub struct SimulateResponse {
    pub output_directory: String,
    pub simulator: String,
    pub message: String,
}

/// Error type for API operations
#[derive(thiserror::Error, Debug)]
pub enum ApiError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("compilation failed: {0}")]
    CompilationFailed(String),
    #[error("simulation failed: {0}")]
    SimulationFailed(String),
    #[error("target not found: {0}")]
    TargetNotFound(String),
    #[error("internal error: {0}")]
    Internal(String),
}

impl From<ApiError> for (StatusCode, Json<ApiResponse<()>>) {
    fn from(err: ApiError) -> Self {
        let status = match err {
            ApiError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::TargetNotFound(_) => StatusCode::NOT_FOUND,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };

        (status, Json(ApiResponse::error(err.to_string())))
    }
}

/// Health check endpoint
async fn health_check() -> Json<ApiResponse<String>> {
    Json(ApiResponse::success("Neuro-compiler API is running".to_string()))
}

/// List available targets
async fn list_targets() -> Result<Json<ApiResponse<Vec<TargetInfo>>>, (StatusCode, Json<ApiResponse<()>>)> {
    let targets = nc_hal::builtin_targets();

    let target_info: Vec<TargetInfo> = targets
        .iter()
        .filter_map(|&name| {
            // Try to load the manifest for detailed info
            match nc_hal::parse_target_manifest_path(&std::path::PathBuf::from(format!("targets/{}.toml", name))) {
                Ok(manifest) => Some(TargetInfo {
                    name: manifest.name,
                    vendor: manifest.vendor,
                    family: manifest.family,
                    description: manifest.notes,
                }),
                Err(_) => Some(TargetInfo {
                    name: name.to_string(),
                    vendor: "Unknown".to_string(),
                    family: "Unknown".to_string(),
                    description: None,
                }),
            }
        })
        .collect();

    Ok(Json(ApiResponse::success(target_info)))
}

/// Compile a neural network model
async fn compile_model(
    Json(request): Json<CompileRequest>,
) -> Result<Json<ApiResponse<CompileResponse>>, (StatusCode, Json<ApiResponse<()>>)> {
    // Parse the NIR graph from the request
    let graph: nc_nir::Graph = serde_json::from_value(request.input)
        .map_err(|e| ApiError::InvalidRequest(format!("invalid NIR graph: {}", e)))?;

    // Validate the graph
    graph.validate()
        .map_err(|e| ApiError::InvalidRequest(format!("graph validation failed: {}", e)))?;

    // Ensure versioning
    let mut graph = graph;
    graph.ensure_version_tag();

    // Load target manifest
    let manifest_path = std::path::PathBuf::from(format!("targets/{}.toml", request.target));
    let manifest = nc_hal::parse_target_manifest_path(&manifest_path)
        .map_err(|_| ApiError::TargetNotFound(request.target.clone()))?;

    nc_hal::validate_manifest(&manifest)
        .map_err(|e| ApiError::InvalidRequest(format!("invalid manifest: {}", e)))?;

    // Perform compilation based on target
    let artifact_path = compile_to_target(&graph, &manifest, &request.target)
        .await?;

    let response = CompileResponse {
        artifact_path,
        target: request.target,
        populations: graph.populations.len(),
        connections: graph.connections.len(),
        probes: graph.probes.len(),
    };

    Ok(Json(ApiResponse::success(response)))
}

/// Run simulation
async fn simulate_model(
    Json(request): Json<SimulateRequest>,
) -> Result<Json<ApiResponse<SimulateResponse>>, (StatusCode, Json<ApiResponse<()>>)> {
    // Parse the NIR graph from the request
    let graph: nc_nir::Graph = serde_json::from_value(request.input)
        .map_err(|e| ApiError::InvalidRequest(format!("invalid NIR graph: {}", e)))?;

    // Validate the graph
    graph.validate()
        .map_err(|e| ApiError::InvalidRequest(format!("graph validation failed: {}", e)))?;

    // Ensure versioning
    let mut graph = graph;
    graph.ensure_version_tag();

    // Perform simulation based on simulator
    let out_dir = request.out_dir.unwrap_or_else(|| format!("target/sim-{}-web", request.simulator));
    let message = simulate_with_backend(&graph, &request.simulator, &out_dir)
        .await?;

    let response = SimulateResponse {
        output_directory: out_dir,
        simulator: request.simulator,
        message,
    };

    Ok(Json(ApiResponse::success(response)))
}

/// Compile to specific target (internal function)
async fn compile_to_target(
    graph: &nc_nir::Graph,
    manifest: &nc_hal::TargetManifest,
    target: &str,
) -> Result<String, ApiError> {
    match target {
        "loihi2" => {
            #[cfg(feature = "backend-loihi")]
            {
                nc_backend_loihi::compile(graph, manifest)
                    .map_err(|e| ApiError::CompilationFailed(e.to_string()))
            }
            #[cfg(not(feature = "backend-loihi"))]
            {
                Err(ApiError::Internal("backend-loihi feature not enabled".to_string()))
            }
        }
        "truenorth" => {
            #[cfg(feature = "backend-truenorth")]
            {
                nc_backend_truenorth::compile(graph, manifest)
                    .map_err(|e| ApiError::CompilationFailed(e.to_string()))
            }
            #[cfg(not(feature = "backend-truenorth"))]
            {
                Err(ApiError::Internal("backend-truenorth feature not enabled".to_string()))
            }
        }
        "akida" => {
            #[cfg(feature = "backend-akida")]
            {
                nc_backend_akida::compile(graph, manifest)
                    .map_err(|e| ApiError::CompilationFailed(e.to_string()))
            }
            #[cfg(not(feature = "backend-akida"))]
            {
                Err(ApiError::Internal("backend-akida feature not enabled".to_string()))
            }
        }
        "spinnaker2" => {
            #[cfg(feature = "backend-spinnaker")]
            {
                nc_backend_spinnaker::compile(graph, manifest)
                    .map_err(|e| ApiError::CompilationFailed(e.to_string()))
            }
            #[cfg(not(feature = "backend-spinnaker"))]
            {
                Err(ApiError::Internal("backend-spinnaker feature not enabled".to_string()))
            }
        }
        "neurogrid" => {
            #[cfg(feature = "backend-neurogrid")]
            {
                nc_backend_neurogrid::compile(graph, manifest)
                    .map_err(|e| ApiError::CompilationFailed(e.to_string()))
            }
            #[cfg(not(feature = "backend-neurogrid"))]
            {
                Err(ApiError::Internal("backend-neurogrid feature not enabled".to_string()))
            }
        }
        "dynaps" => {
            #[cfg(feature = "backend-dynaps")]
            {
                nc_backend_dynaps::compile(graph, manifest)
                    .map_err(|e| ApiError::CompilationFailed(e.to_string()))
            }
            #[cfg(not(feature = "backend-dynaps"))]
            {
                Err(ApiError::Internal("backend-dynaps feature not enabled".to_string()))
            }
        }
        "memxbar" => {
            #[cfg(feature = "backend-memxbar")]
            {
                nc_backend_memxbar::compile(graph, manifest)
                    .map_err(|e| ApiError::CompilationFailed(e.to_string()))
            }
            #[cfg(not(feature = "backend-memxbar"))]
            {
                Err(ApiError::Internal("backend-memxbar feature not enabled".to_string()))
            }
        }
        "custom_asic" => {
            #[cfg(feature = "backend-custom-asic")]
            {
                nc_backend_custom_asic::compile(graph, manifest)
                    .map_err(|e| ApiError::CompilationFailed(e.to_string()))
            }
            #[cfg(not(feature = "backend-custom-asic"))]
            {
                Err(ApiError::Internal("backend-custom-asic feature not enabled".to_string()))
            }
        }
        "riscv64gcv_linux" | "riscv32imac_bare" | "riscv64gc_ctrl" => {
            #[cfg(feature = "backend-riscv")]
            {
                nc_backend_riscv::compile(graph, manifest)
                    .map_err(|e| ApiError::CompilationFailed(e.to_string()))
            }
            #[cfg(not(feature = "backend-riscv"))]
            {
                Err(ApiError::Internal("backend-riscv feature not enabled".to_string()))
            }
        }
        "speck" => {
            #[cfg(feature = "backend-speck")]
            {
                nc_backend_speck::compile(graph, manifest)
                    .map_err(|e| ApiError::CompilationFailed(e.to_string()))
            }
            #[cfg(not(feature = "backend-speck"))]
            {
                Err(ApiError::Internal("backend-speck feature not enabled".to_string()))
            }
        }
        "xylo" => {
            #[cfg(feature = "backend-xylo")]
            {
                nc_backend_xylo::compile(graph, manifest)
                    .map_err(|e| ApiError::CompilationFailed(e.to_string()))
            }
            #[cfg(not(feature = "backend-xylo"))]
            {
                Err(ApiError::Internal("backend-xylo feature not enabled".to_string()))
            }
        }
        "brainscales2" => {
            #[cfg(feature = "backend-brainscales")]
            {
                nc_backend_brainscales::compile(graph, manifest)
                    .map_err(|e| ApiError::CompilationFailed(e.to_string()))
            }
            #[cfg(not(feature = "backend-brainscales"))]
            {
                Err(ApiError::Internal("backend-brainscales feature not enabled".to_string()))
            }
        }
        _ => Err(ApiError::TargetNotFound(target.to_string())),
    }
}

/// Simulate with specific backend (internal function)
async fn simulate_with_backend(
    graph: &nc_nir::Graph,
    simulator: &str,
    out_dir: &str,
) -> Result<String, ApiError> {
    match simulator {
        "neuron" => {
            #[cfg(feature = "sim-neuron")]
            {
                nc_sim_neuron::emit_artifacts(graph, &std::path::PathBuf::from(out_dir))
                    .map_err(|e| ApiError::SimulationFailed(e.to_string()))?;
                Ok(format!("Neuron artifacts written to {}", out_dir))
            }
            #[cfg(not(feature = "sim-neuron"))]
            {
                Err(ApiError::Internal("sim-neuron feature not enabled".to_string()))
            }
        }
        "coreneuron" => {
            #[cfg(feature = "sim-coreneuron")]
            {
                nc_sim_coreneuron::emit_artifacts(graph, &std::path::PathBuf::from(out_dir))
                    .map_err(|e| ApiError::SimulationFailed(e.to_string()))?;
                Ok(format!("CoreNeuron artifacts written to {}", out_dir))
            }
            #[cfg(not(feature = "sim-coreneuron"))]
            {
                Err(ApiError::Internal("sim-coreneuron feature not enabled".to_string()))
            }
        }
        "arbor" => {
            #[cfg(feature = "sim-arbor")]
            {
                nc_sim_arbor::emit_artifacts(graph, &std::path::PathBuf::from(out_dir))
                    .map_err(|e| ApiError::SimulationFailed(e.to_string()))?;
                Ok(format!("Arbor artifacts written to {}", out_dir))
            }
            #[cfg(not(feature = "sim-arbor"))]
            {
                Err(ApiError::Internal("sim-arbor feature not enabled".to_string()))
            }
        }
        "hw" => {
            #[cfg(feature = "sim-hw-specific")]
            {
                nc_sim_hw_specific::emit_artifacts(graph, &std::path::PathBuf::from(out_dir))
                    .map_err(|e| ApiError::SimulationFailed(e.to_string()))?;
                Ok(format!("Hardware-specific artifacts written to {}", out_dir))
            }
            #[cfg(not(feature = "sim-hw-specific"))]
            {
                Err(ApiError::Internal("sim-hw-specific feature not enabled".to_string()))
            }
        }
        _ => Err(ApiError::InvalidRequest(format!("unsupported simulator: {}", simulator))),
    }
}

/// Create the API router
pub fn create_router() -> Router {
    let cors = CorsLayer::permissive();

    Router::new()
        .route("/health", get(health_check))
        .route("/targets", get(list_targets))
        .route("/compile", post(compile_model))
        .route("/simulate", post(simulate_model))
        .layer(cors)
}

/// Start the API server
pub async fn start_server(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let app = create_router();

    let addr = format!("0.0.0.0:{}", port);
    println!("🚀 Neuro-compiler API server starting on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_health_check() {
        let app = create_router();

        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_list_targets() {
        let app = create_router();

        let response = app
            .oneshot(Request::builder().uri("/targets").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}