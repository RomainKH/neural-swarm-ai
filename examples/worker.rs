use neural_swarm_ai::compute::{ComputeMonitor, NodeProfile};
use neural_swarm_ai::transport::client::connect_to_cluster;

#[tokio::main]
async fn main() {
    println!("🤖 Starting NeuralSwarmAI Worker Node...");

    // Auto-detect hardware profile (CPU, RAM, architecture, device type)
    let profile = NodeProfile::detect();

    println!("✅ Initialized for device: {}", profile.hostname);
    println!(
        "📊 Hardware Profile: {:?} with {} cores, {} MB RAM",
        profile.device_type, profile.cpu_cores, profile.ram_total_mb
    );

    // Get initial status and start the compute monitor in the background
    let (mut monitor, _status_rx) = ComputeMonitor::new(Default::default());
    let initial_status = monitor.sample_once();
    tokio::spawn(monitor.run());

    println!("⏳ Connecting to Orchestrator...");

    let url = "ws://127.0.0.1:3000/swarm";
    if let Err(e) = connect_to_cluster(url, profile, initial_status).await {
        eprintln!("❌ Failed to connect: {}", e);
    }
}
