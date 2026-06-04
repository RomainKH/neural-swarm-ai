use axum::{routing::get, Router};
use neural_swarm_ai::transport::server::swarm_handler;
use neural_swarm_ai::Orchestrator;
use std::sync::Arc;
use tokio::net::TcpListener;

#[tokio::main]
async fn main() {
    println!("🧠 Starting NeuralSwarmAI Master Node...");

    // Initialize an orchestrator for a 32-layer model, wrapped in Arc for axum
    let shared_secret = "my_super_secret_token".to_string();
    let orchestrator = Arc::new(Orchestrator::new(32, shared_secret.clone()));
    let state = neural_swarm_ai::transport::server::ServerState {
        orchestrator,
        shared_secret,
    };
    println!("✅ Orchestrator initialized for a 32-layer model.");

    // Setup the axum router
    let app = Router::new()
        .route("/swarm", get(swarm_handler))
        .with_state(state);

    let addr = "127.0.0.1:3000";
    let listener = TcpListener::bind(addr).await.unwrap();

    println!("🚀 Server is running!");
    println!("🔌 Waiting for workers to connect at ws://{}/swarm", addr);

    axum::serve(listener, app).await.unwrap();
}
