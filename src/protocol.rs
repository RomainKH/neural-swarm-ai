use bytes::Bytes;
use serde::{Deserialize, Serialize};

/// Messages exchanged between nodes in the NeuralSwarmAI cluster.
#[derive(Serialize, Deserialize, Debug, Clone)]
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
