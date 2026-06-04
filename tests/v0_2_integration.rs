use anyhow::Result;
use bytes::Bytes;
use libp2p::PeerId;
use neural_swarm_ai::compute::{NodeProfile, NodeStatus};
use neural_swarm_ai::pipeline::PipelineResult;
use neural_swarm_ai::{Executor, InferenceBackend, Orchestrator, SwarmMessage};

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
        // Just return some dummy logits based on tokens
        Ok(tokens.iter().map(|&t| t as f32 / 10.0).collect())
    }
}

#[tokio::test]
async fn test_v0_2_full_pipeline_flow() -> Result<()> {
    let secret = "test_secret".to_string();
    let orchestrator = Orchestrator::new(32, secret.clone());

    // 1. Register two nodes with valid PeerIds
    let id1 = PeerId::random();
    let id2 = PeerId::random();

    let p1 = NodeProfile::custom(
        neural_swarm_ai::compute::profile::DeviceType::Desktop,
        8,
        8000,
        "node1".into(),
    );
    let p2 = NodeProfile::custom(
        neural_swarm_ai::compute::profile::DeviceType::Desktop,
        8,
        8000,
        "node2".into(),
    );

    orchestrator.handle_announce(id1.to_string(), p1, NodeStatus::unknown())?;
    orchestrator.handle_announce(id2.to_string(), p2, NodeStatus::unknown())?;

    // 2. Start a sequence
    let initial_data = vec![1, 2, 3, 4];
    let (node1_id, task1) = orchestrator
        .start_sequence(
            1,
            "task1".into(),
            vec![42],
            Bytes::from(initial_data.clone()),
        )?
        .unwrap();

    assert_eq!(node1_id, id1);

    // 3. Node 1 processes task
    let mut backend1 = MockBackend { state: vec![] };
    let executor1 = Executor::new(id1.to_string(), orchestrator.cluster_key);

    let result1 = executor1.run_task(&mut backend1, task1)?.unwrap();

    // 4. Master handles result and sends to Node 2
    let pipeline_res = orchestrator.handle_task_result(&result1)?.unwrap();

    if let PipelineResult::NextStage(node2_id, task2) = pipeline_res {
        assert_eq!(node2_id, id2);

        // 5. Node 2 processes task
        let mut backend2 = MockBackend { state: vec![] };
        let executor2 = Executor::new(id2.to_string(), orchestrator.cluster_key);
        let result2 = executor2.run_task(&mut backend2, task2)?.unwrap();

        // 6. Master handles final result
        let final_res = orchestrator.handle_task_result(&result2)?.unwrap();

        if let PipelineResult::Finished(logits) = final_res {
            assert_eq!(logits, vec![4.2]);
            // Verify state was preserved through encryption/compression
            assert_eq!(backend2.state, initial_data);
            println!("✅ Full pipeline flow with encryption and compression verified!");
        } else {
            panic!("Expected Finished");
        }
    } else {
        panic!("Expected NextStage");
    }

    Ok(())
}
