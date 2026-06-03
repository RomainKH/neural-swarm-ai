use crate::protocol::SwarmMessage;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Represents a compute node in the swarm.
pub struct WorkerNode {
    pub id: String,
    pub compute_power: u32,
    pub assigned_layers: Vec<u32>,
}

/// Orchestrates the distributed inference across the swarm.
pub struct Orchestrator {
    pub workers: Arc<RwLock<HashMap<String, WorkerNode>>>,
    pub total_model_layers: u32,
}

impl Orchestrator {
    /// Creates a new orchestrator for a model with a given number of layers.
    pub fn new(total_layers: u32) -> Self {
        Self {
            workers: Arc::new(RwLock::new(HashMap::new())),
            total_model_layers: total_layers,
        }
    }

    /// Handles a new node joining the swarm and assigns computation layers.
    pub fn handle_join(&self, device_id: String, power: u32) -> SwarmMessage {
        let mut workers = self
            .workers
            .write()
            .expect("Failed to lock workers for writing");

        // Dynamic layer assignment logic based on compute power.
        // For the PoC, we assign all layers to the node.
        let node = WorkerNode {
            id: device_id.clone(),
            compute_power: power,
            assigned_layers: (0..self.total_model_layers).collect(),
        };

        workers.insert(device_id, node);

        SwarmMessage::JoinResponse {
            assigned_layers: (0..self.total_model_layers).collect(),
            total_layers: self.total_model_layers,
        }
    }

    /// Removes a node from the swarm.
    pub fn handle_leave(&self, device_id: &str) {
        let mut workers = self
            .workers
            .write()
            .expect("Failed to lock workers for writing");
        workers.remove(device_id);
    }
}
