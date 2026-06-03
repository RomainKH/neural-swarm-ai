use neural_swarm_ai::Orchestrator;
use neural_swarm_ai::SwarmMessage;

#[tokio::main]
async fn main() {
    println!("🧠 Starting NeuralSwarmAI Master Node...");
    
    // Initialize an orchestrator for a 32-layer model
    let orchestrator = Orchestrator::new(32); 
    
    println!("✅ Orchestrator initialized for a 32-layer model.");

    // Simulate a node joining the swarm
    let device_id = "macbook-pro-m3".to_string();
    let compute_power = 100; // Mock score
    
    println!("⏳ Simulating node join request from {}...", device_id);
    let response = orchestrator.handle_join(device_id.clone(), compute_power);
    
    match response {
        SwarmMessage::JoinResponse { assigned_layers, total_layers } => {
            println!("🎉 Node {} joined successfully!", device_id);
            println!("   -> Assigned layers: {:?}", assigned_layers);
            println!("   -> Total model layers: {}", total_layers);
        },
        _ => println!("❌ Unexpected response from orchestrator"),
    }
    
    println!("Master node example completed.");
}
