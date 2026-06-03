use crate::protocol::SwarmMessage;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Represents a compute node in the swarm.
#[derive(Debug, Clone)]
pub struct WorkerNode {
    pub id: String,
    pub compute_power: u32,
    pub assigned_layers: Vec<u32>,
}

/// Orchestrates the distributed inference across the swarm.
///
/// The `Orchestrator` manages worker nodes, dynamically assigns model layers
/// based on each node's compute power, and rebalances on join/leave events.
///
/// # Layer Assignment Algorithm
///
/// Layers are distributed proportionally to each node's `compute_power`:
/// - A node with 2× the compute power gets 2× the layers.
/// - All layers are guaranteed to be assigned (remainder goes to the most powerful node).
///
/// # Example
///
/// ```rust
/// use neural_swarm_ai::Orchestrator;
///
/// let orchestrator = Orchestrator::new(32);
///
/// // A powerful GPU node joins
/// let resp = orchestrator.handle_join("gpu-node".into(), 200).unwrap();
///
/// // A lighter CPU node joins — layers are rebalanced automatically
/// let resp = orchestrator.handle_join("cpu-node".into(), 50).unwrap();
/// ```
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

    /// Handles a new node joining the swarm.
    ///
    /// Registers the node and **rebalances all layer assignments** across the
    /// swarm proportionally to each node's compute power.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal worker lock is poisoned.
    pub fn handle_join(&self, device_id: String, power: u32) -> Result<SwarmMessage> {
        let mut workers = self
            .workers
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire worker lock: {}", e))?;

        // Register the new node (will be assigned layers during rebalance)
        let node = WorkerNode {
            id: device_id.clone(),
            compute_power: power,
            assigned_layers: vec![],
        };
        workers.insert(device_id.clone(), node);

        // Rebalance all layer assignments across the swarm
        Self::rebalance_layers(&mut workers, self.total_model_layers);

        // Return the assignment for the newly joined node
        let assigned = workers
            .get(&device_id)
            .context("Node disappeared after registration")?;

        Ok(SwarmMessage::JoinResponse {
            assigned_layers: assigned.assigned_layers.clone(),
            total_layers: self.total_model_layers,
        })
    }

    /// Removes a node from the swarm and rebalances remaining layer assignments.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal worker lock is poisoned.
    pub fn handle_leave(&self, device_id: &str) -> Result<()> {
        let mut workers = self
            .workers
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire worker lock: {}", e))?;

        workers.remove(device_id);

        // Rebalance remaining nodes so all layers stay covered
        if !workers.is_empty() {
            Self::rebalance_layers(&mut workers, self.total_model_layers);
        }

        Ok(())
    }

    /// Returns the number of currently connected workers.
    pub fn worker_count(&self) -> usize {
        self.workers.read().map(|w| w.len()).unwrap_or(0)
    }

    /// Distributes layers proportionally to compute power.
    ///
    /// Algorithm:
    /// 1. Sort workers by compute power (descending) for deterministic assignment.
    /// 2. Calculate each node's share: `node_power / total_power * total_layers`.
    /// 3. Assign remaining layers (from integer rounding) to the most powerful node.
    fn rebalance_layers(workers: &mut HashMap<String, WorkerNode>, total_layers: u32) {
        if workers.is_empty() {
            return;
        }

        let total_power: u32 = workers.values().map(|w| w.compute_power).sum();
        if total_power == 0 {
            return;
        }

        // Sort by compute power descending for deterministic assignment
        let mut sorted_ids: Vec<String> = workers.keys().cloned().collect();
        sorted_ids.sort_by(|a, b| {
            let pa = workers.get(a).map(|w| w.compute_power).unwrap_or(0);
            let pb = workers.get(b).map(|w| w.compute_power).unwrap_or(0);
            pb.cmp(&pa).then(a.cmp(b))
        });

        let mut current_layer: u32 = 0;
        let mut assignments: Vec<(String, Vec<u32>)> = Vec::new();

        for (i, id) in sorted_ids.iter().enumerate() {
            let power = workers.get(id).map(|w| w.compute_power).unwrap_or(0);
            let is_last = i == sorted_ids.len() - 1;

            // Calculate this node's share of layers
            let layer_count = if is_last {
                // Last node gets all remaining layers (handles rounding)
                total_layers.saturating_sub(current_layer)
            } else {
                ((power as f64 / total_power as f64) * total_layers as f64).round() as u32
            };

            let end_layer = (current_layer + layer_count).min(total_layers);
            let layers: Vec<u32> = (current_layer..end_layer).collect();

            assignments.push((id.clone(), layers));
            current_layer = end_layer;
        }

        // Apply assignments
        for (id, layers) in assignments {
            if let Some(worker) = workers.get_mut(&id) {
                worker.assigned_layers = layers;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_node_gets_all_layers() {
        let orch = Orchestrator::new(32);
        let resp = orch.handle_join("node-1".into(), 100).unwrap();

        if let SwarmMessage::JoinResponse {
            assigned_layers,
            total_layers,
        } = resp
        {
            assert_eq!(total_layers, 32);
            assert_eq!(assigned_layers.len(), 32);
            assert_eq!(assigned_layers, (0..32).collect::<Vec<u32>>());
        } else {
            panic!("Expected JoinResponse");
        }
    }

    #[test]
    fn test_equal_power_splits_evenly() {
        let orch = Orchestrator::new(32);
        orch.handle_join("node-1".into(), 100).unwrap();
        let resp = orch.handle_join("node-2".into(), 100).unwrap();

        if let SwarmMessage::JoinResponse {
            assigned_layers,
            total_layers,
        } = resp
        {
            assert_eq!(total_layers, 32);
            assert_eq!(assigned_layers.len(), 16);
        } else {
            panic!("Expected JoinResponse");
        }
    }

    #[test]
    fn test_proportional_distribution() {
        let orch = Orchestrator::new(30);
        // Node with 2x power should get ~2x layers
        orch.handle_join("strong".into(), 200).unwrap();
        orch.handle_join("weak".into(), 100).unwrap();

        let workers = orch.workers.read().unwrap();
        let strong = workers.get("strong").unwrap();
        let weak = workers.get("weak").unwrap();

        // strong (200/300 * 30 = 20), weak (100/300 * 30 = 10)
        assert_eq!(strong.assigned_layers.len(), 20);
        assert_eq!(weak.assigned_layers.len(), 10);

        // All layers must be covered with no gaps
        let mut all_layers: Vec<u32> = strong
            .assigned_layers
            .iter()
            .chain(weak.assigned_layers.iter())
            .copied()
            .collect();
        all_layers.sort();
        assert_eq!(all_layers, (0..30).collect::<Vec<u32>>());
    }

    #[test]
    fn test_leave_rebalances() {
        let orch = Orchestrator::new(32);
        orch.handle_join("node-1".into(), 100).unwrap();
        orch.handle_join("node-2".into(), 100).unwrap();

        // node-2 leaves → node-1 should get all 32 layers back
        orch.handle_leave("node-2").unwrap();

        let workers = orch.workers.read().unwrap();
        let node1 = workers.get("node-1").unwrap();
        assert_eq!(node1.assigned_layers.len(), 32);
    }

    #[test]
    fn test_worker_count() {
        let orch = Orchestrator::new(32);
        assert_eq!(orch.worker_count(), 0);

        orch.handle_join("node-1".into(), 100).unwrap();
        assert_eq!(orch.worker_count(), 1);

        orch.handle_join("node-2".into(), 50).unwrap();
        assert_eq!(orch.worker_count(), 2);

        orch.handle_leave("node-1").unwrap();
        assert_eq!(orch.worker_count(), 1);
    }
}
