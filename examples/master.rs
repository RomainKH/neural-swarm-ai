use neural_swarm_ai::compute::{DeviceType, NodeProfile, NodeStatus};
use neural_swarm_ai::{Orchestrator, SwarmMessage};

#[tokio::main]
async fn main() {
    println!("🧠 Starting NeuralSwarmAI Master Node...");

    // Initialize an orchestrator for a 32-layer model
    let orchestrator = Orchestrator::new(32);
    println!("✅ Orchestrator initialized for a 32-layer model.");

    // SIMULATION: In a real environment, the Master doesn't hardcode these!
    // Real workers auto-detect their hardware using `NodeProfile::detect()`
    // and send it to the Master via a `NodeAnnounce` WebSocket message.
    // We mock them here to demonstrate how the Orchestrator distributes
    // layers differently based on device capabilities.
    let simulated_nodes = vec![
        ("gpu-server", DeviceType::Server, 32, 65536u64, 0.05f32),
        ("macbook-pro", DeviceType::Laptop, 10, 16384, 0.20),
        ("raspberry-pi", DeviceType::Embedded, 4, 4096, 0.10),
    ];

    for (name, device_type, cores, ram, cpu_usage) in simulated_nodes {
        let profile = NodeProfile::custom(device_type, cores, ram, name.into());
        let status = NodeStatus {
            cpu_usage,
            ram_used_mb: 0,
            ram_available_mb: ram - 1024,
            thermal: neural_swarm_ai::ThermalState::Nominal,
            measured_at: None,
        };

        println!("\n⏳ {} ({:?}) joining...", name, device_type);

        match orchestrator.handle_announce(name.to_string(), profile, status) {
            Ok(SwarmMessage::JoinResponse {
                assigned_layers,
                total_layers,
            }) => {
                println!("🎉 {} joined successfully!", name);
                println!(
                    "   → Assigned {} / {} layers: {:?}",
                    assigned_layers.len(),
                    total_layers,
                    assigned_layers
                );
            }
            Ok(_) => println!("❌ Unexpected response"),
            Err(e) => println!("❌ Error: {}", e),
        }
    }

    println!(
        "\n📊 Cluster: {} active nodes",
        orchestrator.active_node_count()
    );

    // Simulate the MacBook getting busy (user starts coding)
    println!("\n⚡ Simulating MacBook getting busy (CPU → 80%)...");
    let busy_status = NodeStatus {
        cpu_usage: 0.80,
        ram_used_mb: 12000,
        ram_available_mb: 4384,
        thermal: neural_swarm_ai::ThermalState::Warm,
        measured_at: None,
    };

    match orchestrator.handle_status_update("macbook-pro", busy_status) {
        Ok(Some(SwarmMessage::RebalanceCommand { new_layers })) => {
            println!(
                "🔄 MacBook rebalanced → now has {} layers",
                new_layers.len()
            );
        }
        Ok(None) => println!("✅ No rebalance needed"),
        Ok(_) => println!("❌ Unexpected response"),
        Err(e) => println!("❌ Error: {}", e),
    }

    println!("\nMaster node example completed.");
}
