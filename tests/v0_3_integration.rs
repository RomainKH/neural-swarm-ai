use anyhow::Result;
use bytes::Bytes;
use libp2p::PeerId;
use neural_swarm_ai::compute::{NodeProfile, NodeStatus};
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
    // Node IDs must be valid PeerIds in v0.4
    let id1 = PeerId::random();
    let id2 = PeerId::random();

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

    orchestrator.handle_announce(id1.to_string(), p1, s1)?;
    orchestrator.handle_announce(id2.to_string(), p2, s2)?;

    let registry = orchestrator.registry.read().unwrap();
    let n1 = registry.get(&id1.to_string()).unwrap();
    let n2 = registry.get(&id2.to_string()).unwrap();

    // 2. Verify fast node got more layers due to latency penalty on the slow node
    println!(
        "Node 1 ({:?}) layers: {}, Node 2 ({:?}) layers: {}",
        id1,
        n1.assigned_layers.len(),
        id2,
        n2.assigned_layers.len()
    );

    // One of them should have 10ms and the other 500ms.
    // The one with 10ms (id1) should have more layers than the one with 500ms (id2).
    assert!(n1.assigned_layers.len() > n2.assigned_layers.len());

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

    // Since task.input_state is NOT encrypted in this manual task creation,
    // we need to encrypt it first to simulate a real orchestrator output
    let compressed = neural_swarm_ai::crypto::compress(&[1, 2, 3])?;
    let encrypted =
        neural_swarm_ai::crypto::encrypt_with_aad(&compressed, &cluster_key, b"task_123").unwrap();

    let secure_task = neural_swarm_ai::SwarmMessage::ProcessTask {
        task_id: "task_123".into(),
        sequence_id: 1,
        input_state: Bytes::from(encrypted),
        start_layer: 0,
        end_layer: 10,
        tokens: vec![42],
        route: None,
    };

    let result = executor.run_task(&mut backend, secure_task)?;
    assert!(result.is_some());

    Ok(())
}
