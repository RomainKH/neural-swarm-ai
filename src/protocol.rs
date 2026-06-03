use bytes::Bytes;
use serde::{Deserialize, Serialize};

/// Messages exchanged between nodes in the NeuralSwarmAI cluster.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SwarmMessage {
    /// Handshake: Node requests to join the cluster.
    JoinRequest {
        device_id: String,
        compute_power: u32,
    },
    /// Handshake: Master responds with assigned layers.
    JoinResponse {
        assigned_layers: Vec<u32>,
        total_layers: u32,
    },

    /// Inference: Master sends a task to a node.
    ProcessTask {
        task_id: String,
        /// Serialized KV Cache state.
        input_state: Bytes,
        start_layer: u32,
        end_layer: u32,
        tokens: Vec<i32>,
    },
    /// Inference: Node sends result back to Master.
    TaskResult {
        task_id: String,
        /// Serialized KV Cache state after computation.
        output_state: Bytes,
        /// Output probabilities for the next token.
        logits: Vec<f32>,
    },

    /// Health check.
    Heartbeat,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: serialize then deserialize via bincode and assert equality.
    fn roundtrip(msg: &SwarmMessage) {
        let encoded = bincode::serialize(msg).expect("Failed to serialize");
        let decoded: SwarmMessage = bincode::deserialize(&encoded).expect("Failed to deserialize");
        assert_eq!(msg, &decoded);
    }

    #[test]
    fn test_roundtrip_join_request() {
        roundtrip(&SwarmMessage::JoinRequest {
            device_id: "raspberry-pi-5".into(),
            compute_power: 42,
        });
    }

    #[test]
    fn test_roundtrip_join_response() {
        roundtrip(&SwarmMessage::JoinResponse {
            assigned_layers: vec![0, 1, 2, 3, 4, 5],
            total_layers: 32,
        });
    }

    #[test]
    fn test_roundtrip_process_task() {
        roundtrip(&SwarmMessage::ProcessTask {
            task_id: "task-001".into(),
            input_state: Bytes::from(vec![0xDE, 0xAD, 0xBE, 0xEF]),
            start_layer: 0,
            end_layer: 16,
            tokens: vec![1, 2, 3, 4, 5],
        });
    }

    #[test]
    fn test_roundtrip_task_result() {
        roundtrip(&SwarmMessage::TaskResult {
            task_id: "task-001".into(),
            output_state: Bytes::from(vec![0xCA, 0xFE]),
            logits: vec![0.1, 0.5, 0.3, 0.1],
        });
    }

    #[test]
    fn test_roundtrip_heartbeat() {
        roundtrip(&SwarmMessage::Heartbeat);
    }

    #[test]
    fn test_empty_state_roundtrip() {
        roundtrip(&SwarmMessage::ProcessTask {
            task_id: "empty".into(),
            input_state: Bytes::new(),
            start_layer: 0,
            end_layer: 0,
            tokens: vec![],
        });
    }
}
