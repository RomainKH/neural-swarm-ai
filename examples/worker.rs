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
    let shared_secret = "my_super_secret_token";
    match connect_to_cluster(url, shared_secret, profile, initial_status).await {
        Ok(cluster_key) => {
            println!("✅ Joined swarm! Cluster key obtained.");
            // In a real app, initialize Executor here
            // let executor = Executor::new(hostname, cluster_key);

            // For the example, we just keep the connection alive
            println!("⏳ Waiting for tasks (press Ctrl+C to exit)...");
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
            }
        }
        Err(e) => eprintln!("❌ Failed to connect: {}", e),
    }
}
