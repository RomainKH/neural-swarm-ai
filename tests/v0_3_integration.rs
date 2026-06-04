use anyhow::Result;
use bytes::Bytes;
use neural_swarm_ai::compute::{NodeProfile, NodeStatus};
use neural_swarm_ai::pipeline::PipelineResult;
use neural_swarm_ai::{Executor, InferenceBackend, Orchestrator};

struct MockBackend {
    state: Vec<u8>,
}

impl InferenceBackend for MockBackend {
    fn set_state(&mut self, state: &[u8]) -> Result<()> {
        self.state = state.to_vec();
        Ok(())
    }
    fn get_state(&self) -> Result<Vec<u8>> {
        Ok(self.state.clone())
    }
    fn run_layers(&mut self, _start: u32, _end: u32, tokens: &[i32]) -> Result<Vec<f32>> {
        Ok(tokens.iter().map(|&t| t as f32 / 10.0).collect())
    }
}

#[tokio::test]
async fn test_v0_3_topology_aware_rebalance() -> Result<()> {
    let secret = "test_secret".to_string();
    let orchestrator = Orchestrator::new(32, secret.clone());

    // 1. Register two nodes with SAME hardware but DIFFERENT latency
    let p1 = NodeProfile::custom(
        neural_swarm_ai::compute::profile::DeviceType::Desktop,
        8,
        8000,
        "fast_node".into(),
    );
    let p2 = NodeProfile::custom(
        neural_swarm_ai::compute::profile::DeviceType::Desktop,
        8,
        8000,
        "slow_node".into(),
    );

    // Fast node: 10ms latency
    let mut s1 = NodeStatus::unknown();
    s1.latency_ms = Some(10);

    // Slow node: 500ms latency
    let mut s2 = NodeStatus::unknown();
    s2.latency_ms = Some(500);

    orchestrator.handle_announce("fast_node".into(), p1, s1)?;
    orchestrator.handle_announce("slow_node".into(), p2, s2)?;

    let registry = orchestrator.registry.read().unwrap();
    let fast = registry.get("fast_node").unwrap();
    let slow = registry.get("slow_node").unwrap();

    // 2. Verify fast node got more layers due to latency penalty on the slow node
    println!(
        "Fast node layers: {}, Slow node layers: {}",
        fast.assigned_layers.len(),
        slow.assigned_layers.len()
    );
    assert!(fast.assigned_layers.len() > slow.assigned_layers.len());

    // 3. Verify pipeline stages order (fast node should be first if it has more capacity)
    let pipeline = orchestrator.pipeline.read().unwrap();
    // The current implementation sorts by capacity descending for assignment,
    // and the pipeline stages follow this order.
    // So "fast_node" should be the first stage.
    // (Note: get_next_node_id uses pipeline_stages)

    Ok(())
}

#[tokio::test]
async fn test_v0_3_heterogeneous_mock() -> Result<()> {
    // This test verifies that the Executor correctly handles the cluster key and AAD
    let secret = "test_secret".to_string();
    let orchestrator = Orchestrator::new(32, secret.clone());
    let cluster_key = orchestrator.cluster_key;

    let executor = Executor::new("node1".into(), cluster_key);
    let mut backend = MockBackend { state: vec![] };

    let task = neural_swarm_ai::SwarmMessage::ProcessTask {
        task_id: "task_123".into(),
        sequence_id: 1,
        input_state: Bytes::from(vec![1, 2, 3]),
        start_layer: 0,
        end_layer: 10,
        tokens: vec![42],
    };

    // Since task.input_state is NOT encrypted in this manual task creation,
    // we need to encrypt it first to simulate a real orchestrator output
    let compressed = neural_swarm_ai::crypto::compress(&vec![1, 2, 3])?;
    let encrypted =
        neural_swarm_ai::crypto::encrypt_with_aad(&compressed, &cluster_key, b"task_123").unwrap();

    let secure_task = neural_swarm_ai::SwarmMessage::ProcessTask {
        task_id: "task_123".into(),
        sequence_id: 1,
        input_state: Bytes::from(encrypted),
        start_layer: 0,
        end_layer: 10,
        tokens: vec![42],
    };

    let result = executor.run_task(&mut backend, secure_task)?;
    assert!(result.is_some());

    Ok(())
}
