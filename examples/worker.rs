use neural_swarm_ai::compute::{ComputeMonitor, NodeProfile};
use neural_swarm_ai::Executor;

#[tokio::main]
async fn main() {
    println!("🤖 Starting NeuralSwarmAI Worker Node...");

    // Auto-detect hardware profile (CPU, RAM, architecture, device type)
    let profile = NodeProfile::detect();
    let executor = Executor::new(profile.hostname.clone());

    println!("✅ Executor initialized for device: {}", executor.device_id);
    println!(
        "📊 Hardware Profile: {:?} with {} cores, {} MB RAM",
        profile.device_type, profile.cpu_cores, profile.ram_total_mb
    );

    // Start the compute monitor in the background to track real-time CPU/RAM usage
    let (monitor, _status_rx) = ComputeMonitor::new(Default::default());
    tokio::spawn(monitor.run());

    println!("⏳ Waiting for tasks from Orchestrator...");

    // In a real scenario, you would:
    // 1. Connect to the Master via WebSocket
    // 2. Initialize your inference backend:
    //
    //    With llama.cpp (requires `features = ["llama"]`):
    //      let mut backend = LlamaBackend::new(&mut llama_ctx);
    //
    //    With a custom backend:
    //      let mut backend = MyCustomBackend::new(...);
    //
    // 3. Process tasks:
    //      let result = executor.run_task(&mut backend, task_message)?;
    //      // Send result back to Master or forward to next node

    println!("Worker node example completed.");
}
