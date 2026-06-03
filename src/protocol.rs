use crate::compute::profile::NodeProfile;
use crate::compute::status::NodeStatus;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

/// Messages exchanged between nodes in the NeuralSwarmAI cluster.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SwarmMessage {
    // ── Handshake ───────────────────────────────────────────────
    /// Node announces itself to the master with its hardware profile.
    NodeAnnounce {
        device_id: String,
        profile: NodeProfile,
        initial_status: NodeStatus,
    },
    /// Master responds with assigned layers and cluster info.
    JoinResponse {
        assigned_layers: Vec<u32>,
        total_layers: u32,
    },

    // ── Inference ───────────────────────────────────────────────
    /// Master sends a computation task to a node.
    ProcessTask {
        task_id: String,
        /// Serialized KV Cache state.
        input_state: Bytes,
        start_layer: u32,
        end_layer: u32,
        tokens: Vec<i32>,
    },
    /// Node sends computation result back to Master.
    TaskResult {
        task_id: String,
        /// Serialized KV Cache state after computation.
        output_state: Bytes,
        /// Output probabilities for the next token.
        logits: Vec<f32>,
    },

    // ── Dynamic Compute ────────────────────────────────────────
    /// Worker reports updated resource availability.
    /// Only sent when the change exceeds the monitor threshold.
    StatusUpdate { status: NodeStatus },

    /// Master orders a layer redistribution.
    RebalanceCommand {
        /// New layer assignment for the receiving node.
        new_layers: Vec<u32>,
    },

    /// Node acknowledges rebalance completion.
    RebalanceAck { device_id: String },

    // ── Lifecycle ──────────────────────────────────────────────
    /// Node requests to leave gracefully (finish tasks, then disconnect).
    DrainRequest { reason: String },

    /// Health check ping/pong.
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
    fn test_roundtrip_node_announce() {
        roundtrip(&SwarmMessage::NodeAnnounce {
            device_id: "macbook-pro".into(),
            profile: NodeProfile::custom(
                crate::compute::profile::DeviceType::Laptop,
                10,
                16384,
                "macbook-pro".into(),
            ),
            initial_status: NodeStatus::unknown(),
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
    fn test_roundtrip_status_update() {
        roundtrip(&SwarmMessage::StatusUpdate {
            status: NodeStatus::unknown(),
        });
    }

    #[test]
    fn test_roundtrip_rebalance_command() {
        roundtrip(&SwarmMessage::RebalanceCommand {
            new_layers: vec![10, 11, 12, 13],
        });
    }

    #[test]
    fn test_roundtrip_drain_request() {
        roundtrip(&SwarmMessage::DrainRequest {
            reason: "Battery low".into(),
        });
    }

    #[test]
    fn test_roundtrip_heartbeat() {
        roundtrip(&SwarmMessage::Heartbeat);
    }
}
