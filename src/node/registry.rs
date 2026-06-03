use crate::compute::{EffectiveCapacity, NodeProfile, NodeStatus, ResourceReservation};
use crate::node::state::NodeState;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::time::Instant;

/// A complete record for a node in the swarm.
///
/// Contains everything the orchestrator needs to make decisions:
/// hardware profile, current status, computed capacity, assigned layers,
/// and lifecycle state.
#[derive(Debug, Clone)]
pub struct NodeEntry {
    /// Unique device identifier.
    pub id: String,

    /// Static hardware profile (detected once at join).
    pub profile: NodeProfile,

    /// Most recent dynamic resource status.
    pub status: NodeStatus,

    /// Current lifecycle state.
    pub state: NodeState,

    /// Computed effective capacity (derived from profile + status + reservation).
    pub capacity: EffectiveCapacity,

    /// Model layers currently assigned to this node.
    pub assigned_layers: Vec<u32>,

    /// Resource reservation (safety margin) for this node.
    pub reservation: ResourceReservation,

    /// Timestamp of the last heartbeat received.
    pub last_heartbeat: Instant,
}

/// Thread-safe registry of all nodes in the swarm.
///
/// Replaces the raw `HashMap<String, WorkerNode>` with a proper
/// data structure that tracks the full lifecycle of each node.
pub struct NodeRegistry {
    nodes: HashMap<String, NodeEntry>,
}

impl NodeRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    /// Registers a new node with its hardware profile.
    ///
    /// The node starts in `Joining` state and gets a default reservation
    /// based on its detected device type.
    pub fn register(&mut self, id: String, profile: NodeProfile) -> Result<&NodeEntry> {
        let reservation = ResourceReservation::for_device(&profile.device_type);
        let status = NodeStatus::unknown();
        let capacity = EffectiveCapacity::zero();

        let entry = NodeEntry {
            id: id.clone(),
            profile,
            status,
            state: NodeState::Joining,
            capacity,
            assigned_layers: vec![],
            reservation,
            last_heartbeat: Instant::now(),
        };

        self.nodes.insert(id.clone(), entry);
        self.nodes
            .get(&id)
            .context("Node disappeared after registration")
    }

    /// Marks a node as ready (handshake complete) and computes initial capacity.
    pub fn mark_ready(&mut self, id: &str, initial_status: NodeStatus) -> Result<()> {
        let entry = self
            .nodes
            .get_mut(id)
            .context(format!("Node not found: {}", id))?;

        entry.state = entry.state.transition_to(NodeState::Ready)?;
        entry.status = initial_status;
        entry.capacity =
            EffectiveCapacity::compute(&entry.profile, &entry.status, &entry.reservation);
        Ok(())
    }

    /// Updates a node's status and recomputes its effective capacity.
    /// Returns true if the capacity changed significantly (> threshold).
    pub fn update_status(&mut self, id: &str, status: NodeStatus) -> Result<bool> {
        let entry = self
            .nodes
            .get_mut(id)
            .context(format!("Node not found: {}", id))?;

        let old_composite = entry.capacity.composite;

        entry.status = status;
        entry.last_heartbeat = Instant::now();
        entry.capacity =
            EffectiveCapacity::compute(&entry.profile, &entry.status, &entry.reservation);

        // Check if the node should transition to Degraded or recover
        if entry.state == NodeState::Active && entry.capacity.composite < 0.1 {
            entry.state = entry
                .state
                .transition_to(NodeState::Degraded)
                .unwrap_or(entry.state);
        } else if entry.state == NodeState::Degraded && entry.capacity.composite >= 0.15 {
            entry.state = entry
                .state
                .transition_to(NodeState::Active)
                .unwrap_or(entry.state);
        }

        let delta = (entry.capacity.composite - old_composite).abs();
        Ok(delta > 0.15)
    }

    /// Records a heartbeat from a node.
    pub fn heartbeat(&mut self, id: &str) {
        if let Some(entry) = self.nodes.get_mut(id) {
            entry.last_heartbeat = Instant::now();
        }
    }

    /// Removes a node from the registry.
    pub fn remove(&mut self, id: &str) -> Option<NodeEntry> {
        self.nodes.remove(id)
    }

    /// Returns a reference to a node entry.
    pub fn get(&self, id: &str) -> Option<&NodeEntry> {
        self.nodes.get(id)
    }

    /// Returns a mutable reference to a node entry.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut NodeEntry> {
        self.nodes.get_mut(id)
    }

    /// Returns the number of registered nodes (all states).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns an iterator over all active nodes (can accept tasks).
    pub fn active_nodes(&self) -> impl Iterator<Item = &NodeEntry> {
        self.nodes.values().filter(|n| n.state.can_accept_tasks())
    }

    /// Returns an iterator over all alive nodes (not Left).
    pub fn alive_nodes(&self) -> impl Iterator<Item = &NodeEntry> {
        self.nodes.values().filter(|n| n.state.is_alive())
    }

    /// Returns all nodes as a slice, sorted by composite capacity descending.
    /// Used by the rebalancer for layer distribution.
    pub fn sorted_by_capacity(&self) -> Vec<&NodeEntry> {
        let mut entries: Vec<&NodeEntry> = self.active_nodes().collect();
        entries.sort_by(|a, b| {
            b.capacity
                .composite
                .partial_cmp(&a.capacity.composite)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.id.cmp(&b.id))
        });
        entries
    }

    /// Updates the assigned layers for a node.
    pub fn assign_layers(&mut self, id: &str, layers: Vec<u32>) -> Result<()> {
        let entry = self
            .nodes
            .get_mut(id)
            .context(format!("Node not found: {}", id))?;

        entry.assigned_layers = layers;

        // Transition to Active if currently Ready
        if entry.state == NodeState::Ready {
            entry.state = entry
                .state
                .transition_to(NodeState::Active)
                .unwrap_or(entry.state);
        }
        Ok(())
    }

    /// Returns a mutable iterator over all nodes (for batch operations like rebalance).
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&String, &mut NodeEntry)> {
        self.nodes.iter_mut()
    }
}

impl Default for NodeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::profile::DeviceType;
    use crate::compute::status::ThermalState;

    fn make_profile(name: &str) -> NodeProfile {
        NodeProfile::custom(DeviceType::Desktop, 8, 16384, name.into())
    }

    fn make_status(cpu: f32, ram_avail: u64) -> NodeStatus {
        NodeStatus {
            cpu_usage: cpu,
            ram_used_mb: 0,
            ram_available_mb: ram_avail,
            thermal: ThermalState::Nominal,
            measured_at: None,
        }
    }

    #[test]
    fn test_register_and_ready() {
        let mut reg = NodeRegistry::new();
        reg.register("node-1".into(), make_profile("node-1"))
            .unwrap();

        assert_eq!(reg.len(), 1);
        assert_eq!(reg.get("node-1").unwrap().state, NodeState::Joining);

        reg.mark_ready("node-1", make_status(0.2, 12000)).unwrap();
        assert_eq!(reg.get("node-1").unwrap().state, NodeState::Ready);
        assert!(reg.get("node-1").unwrap().capacity.composite > 0.0);
    }

    #[test]
    fn test_update_status_triggers_degraded() {
        let mut reg = NodeRegistry::new();
        reg.register("node-1".into(), make_profile("node-1"))
            .unwrap();
        reg.mark_ready("node-1", make_status(0.2, 12000)).unwrap();
        reg.assign_layers("node-1", vec![0, 1, 2, 3]).unwrap();

        assert_eq!(reg.get("node-1").unwrap().state, NodeState::Active);

        // Node becomes overloaded → should transition to Degraded
        reg.update_status("node-1", make_status(0.95, 500)).unwrap();
        assert_eq!(reg.get("node-1").unwrap().state, NodeState::Degraded);
    }

    #[test]
    fn test_update_status_recovers_from_degraded() {
        let mut reg = NodeRegistry::new();
        reg.register("node-1".into(), make_profile("node-1"))
            .unwrap();
        reg.mark_ready("node-1", make_status(0.2, 12000)).unwrap();
        reg.assign_layers("node-1", vec![0, 1, 2, 3]).unwrap();

        // Degrade
        reg.update_status("node-1", make_status(0.95, 500)).unwrap();
        assert_eq!(reg.get("node-1").unwrap().state, NodeState::Degraded);

        // Recover
        reg.update_status("node-1", make_status(0.2, 12000))
            .unwrap();
        assert_eq!(reg.get("node-1").unwrap().state, NodeState::Active);
    }

    #[test]
    fn test_sorted_by_capacity() {
        let mut reg = NodeRegistry::new();

        // Register two nodes with different capabilities
        reg.register("weak".into(), make_profile("weak")).unwrap();
        reg.mark_ready("weak", make_status(0.7, 4000)).unwrap();
        reg.assign_layers("weak", vec![0]).unwrap();

        reg.register("strong".into(), make_profile("strong"))
            .unwrap();
        reg.mark_ready("strong", make_status(0.1, 14000)).unwrap();
        reg.assign_layers("strong", vec![1]).unwrap();

        let sorted = reg.sorted_by_capacity();
        assert_eq!(sorted.len(), 2);
        assert_eq!(sorted[0].id, "strong");
        assert_eq!(sorted[1].id, "weak");
    }

    #[test]
    fn test_remove_node() {
        let mut reg = NodeRegistry::new();
        reg.register("node-1".into(), make_profile("node-1"))
            .unwrap();
        assert_eq!(reg.len(), 1);

        reg.remove("node-1");
        assert_eq!(reg.len(), 0);
        assert!(reg.get("node-1").is_none());
    }

    #[test]
    fn test_active_nodes_filter() {
        let mut reg = NodeRegistry::new();

        reg.register("active".into(), make_profile("active"))
            .unwrap();
        reg.mark_ready("active", make_status(0.2, 12000)).unwrap();

        reg.register("joining".into(), make_profile("joining"))
            .unwrap();
        // Don't mark ready → stays in Joining state

        // Only the Ready node should show as active
        let active: Vec<_> = reg.active_nodes().collect();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, "active");
    }
}
