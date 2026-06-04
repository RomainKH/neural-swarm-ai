use crate::protocol::SwarmMessage;
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A ticket defining the route for an inference sequence.
///
/// This allows the orchestrator to be "Stateless". It generates the ticket
/// and the data travels with it from node to node without master intervention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteTicket {
    pub task_id: String,
    pub sequence_id: u64,
    /// Ordered list of PeerIds in the pipeline.
    pub nodes: Vec<PeerId>,
    /// Index of the node currently processing the task.
    pub current_index: usize,
}

impl RouteTicket {
    pub fn next_peer(&self) -> Option<PeerId> {
        self.nodes.get(self.current_index + 1).cloned()
    }
}

/// Tracks an active inference sequence (Master side for monitoring).
#[derive(Debug, Clone)]
pub struct SequenceState {
    pub sequence_id: u64,
    pub task_id: String,
    pub tokens: Vec<i32>,
    pub current_node_index: usize,
}

/// The outcome of processing a task result in the pipeline.
pub enum PipelineResult {
    /// The pipeline continues; send the provided task to the specified node.
    NextStage(String, SwarmMessage),
    /// The pipeline has finished; these are the final logits.
    Finished(Vec<f32>),
}

/// Manages the routing of tokens through the active nodes in the swarm.
#[derive(Default)]
pub struct InferencePipeline {
    /// Maps a task_id to its ongoing sequence state
    active_sequences: HashMap<String, SequenceState>,
    /// Sorted list of node IDs (now mapped to libp2p PeerIds in v0.4)
    pipeline_stages: Vec<String>,
}

impl InferencePipeline {
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates the pipeline stages based on the latest sorted registry nodes.
    pub fn update_stages(&mut self, sorted_node_ids: Vec<String>) {
        self.pipeline_stages = sorted_node_ids;
    }

    /// Starts a new sequence. Returns the ProcessTask for the first node if available.
    pub fn start_sequence(
        &mut self,
        sequence_id: u64,
        task_id: String,
        tokens: Vec<i32>,
        initial_state: bytes::Bytes,
        first_node_layers: (u32, u32),
    ) -> Option<(String, SwarmMessage)> {
        if self.pipeline_stages.is_empty() {
            return None;
        }

        self.active_sequences.insert(
            task_id.clone(),
            SequenceState {
                sequence_id,
                task_id: task_id.clone(),
                tokens: tokens.clone(),
                current_node_index: 0,
            },
        );

        let first_node_id = self.pipeline_stages[0].clone();
        let task = SwarmMessage::ProcessTask {
            task_id,
            sequence_id,
            input_state: initial_state,
            start_layer: first_node_layers.0,
            end_layer: first_node_layers.1,
            tokens,
        };

        Some((first_node_id, task))
    }

    /// Returns the ID of the next node in the pipeline for a given task, if any.
    pub fn get_next_node_id(&self, task_id: &str) -> Option<String> {
        if let Some(state) = self.active_sequences.get(task_id) {
            let next_index = state.current_node_index + 1;
            if next_index < self.pipeline_stages.len() {
                return Some(self.pipeline_stages[next_index].clone());
            }
        }
        None
    }

    /// Handles a result from a node. If there is a next stage, returns NextStage.
    /// Otherwise returns Finished with the final logits.
    pub fn handle_task_result(
        &mut self,
        result: &SwarmMessage,
        next_node_layers: (u32, u32),
    ) -> Option<PipelineResult> {
        if let SwarmMessage::TaskResult {
            task_id,
            sequence_id,
            output_state,
            logits,
        } = result
        {
            if let Some(state) = self.active_sequences.get_mut(task_id) {
                // Ensure we are processing the correct sequence (safety check)
                if state.sequence_id != *sequence_id {
                    return None;
                }

                state.current_node_index += 1;

                if state.current_node_index < self.pipeline_stages.len() {
                    let next_node_id = self.pipeline_stages[state.current_node_index].clone();
                    let next_task = SwarmMessage::ProcessTask {
                        task_id: task_id.clone(),
                        sequence_id: *sequence_id,
                        input_state: output_state.clone(),
                        start_layer: next_node_layers.0,
                        end_layer: next_node_layers.1,
                        tokens: state.tokens.clone(),
                    };
                    return Some(PipelineResult::NextStage(next_node_id, next_task));
                } else {
                    // Pipeline finished for this task
                    let final_logits = logits.clone();
                    self.active_sequences.remove(task_id);
                    return Some(PipelineResult::Finished(final_logits));
                }
            }
        }
        None
    }
}
