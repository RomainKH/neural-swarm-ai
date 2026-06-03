use neural_swarm_ai::{Orchestrator, SwarmMessage};

#[tokio::main]
async fn main() {
    println!("🧠 Starting NeuralSwarmAI Master Node...");

    // Initialize an orchestrator for a 32-layer model
    let orchestrator = Orchestrator::new(32);

    println!("✅ Orchestrator initialized for a 32-layer model.");

    // Simulate nodes joining the swarm with different compute capabilities
    let nodes = vec![("macbook-pro-m3", 100), ("raspberry-pi-5", 25)];

    for (device_id, power) in nodes {
        println!(
            "\n⏳ Node join request from {} (power: {})...",
            device_id, power
        );

        match orchestrator.handle_join(device_id.to_string(), power) {
            Ok(SwarmMessage::JoinResponse {
                assigned_layers,
                total_layers,
            }) => {
                println!("🎉 Node {} joined successfully!", device_id);
                println!(
                    "   -> Assigned {} out of {} layers",
                    assigned_layers.len(),
                    total_layers
                );
                println!("   -> Layers: {:?}", assigned_layers);
            }
            Ok(_) => println!("❌ Unexpected response from orchestrator"),
            Err(e) => println!("❌ Error joining swarm: {}", e),
        }
    }

    println!(
        "\n📊 Cluster status: {} active workers",
        orchestrator.worker_count()
    );
    println!("Master node example completed.");
}
