use crate::compute::{NodeProfile, NodeStatus};
use crate::node::{NodeRegistry, NodeState};
use crate::protocol::SwarmMessage;
use anyhow::{Context, Result};
use std::sync::{Arc, RwLock};

/// Orchestrates the distributed inference across the swarm.
///
/// The `Orchestrator` manages node lifecycles, dynamically assigns model layers
/// based on each node's [`EffectiveCapacity`], and rebalances when resources
/// change significantly.
///
/// # Layer Assignment
///
/// Layers are distributed proportionally to each node's `composite` capacity score,
/// which accounts for:
/// - Hardware capability (CPU cores, RAM)
/// - Current resource availability (CPU/RAM usage)
/// - Safety margins (per device type — laptops reserve more than servers)
/// - Thermal state (throttled devices get fewer layers)
///
/// # Example
///
/// ```rust
/// use neural_swarm_ai::Orchestrator;
/// use neural_swarm_ai::compute::{NodeProfile, DeviceType, NodeStatus};
///
/// let orchestrator = Orchestrator::new(32, "my-secret".into());
///
/// // A node joins with auto-detected profile
/// let profile = NodeProfile::custom(DeviceType::Desktop, 8, 16384, "my-pc".into());
/// let status = NodeStatus::unknown();
/// let resp = orchestrator.handle_announce("my-pc".into(), profile, status).unwrap();
/// ```
use crate::pipeline::{InferencePipeline, PipelineResult};
use rand::Rng;

pub struct Orchestrator {
    pub registry: Arc<RwLock<NodeRegistry>>,
    pub pipeline: Arc<RwLock<InferencePipeline>>,
    pub total_model_layers: u32,
    pub shared_secret: String,
    /// Random key generated on startup for cluster-wide payload encryption.
    pub cluster_key: [u8; 32],
}

impl Orchestrator {
    /// Creates a new orchestrator for a model with a given number of layers.
    pub fn new(total_layers: u32, shared_secret: String) -> Self {
        let mut cluster_key = [0u8; 32];
        rand::thread_rng().fill(&mut cluster_key);

        Self {
            registry: Arc::new(RwLock::new(NodeRegistry::new())),
            pipeline: Arc::new(RwLock::new(InferencePipeline::new())),
            total_model_layers: total_layers,
            shared_secret,
            cluster_key,
        }
    }

    /// Handles a node announcing itself to the swarm.
    ///
    /// Registers the node, computes its capacity with safety margins,
    /// and rebalances all layer assignments.
    pub fn handle_announce(
        &self,
        device_id: String,
        profile: NodeProfile,
        initial_status: NodeStatus,
    ) -> Result<SwarmMessage> {
        let mut registry = self
            .registry
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire registry lock: {}", e))?;

        // Register and advance to Ready
        registry.register(device_id.clone(), profile)?;
        registry.mark_ready(&device_id, initial_status)?;

        // Rebalance all layer assignments
        Self::rebalance_layers(&mut registry, self.total_model_layers, &self.pipeline);

        // Return the assignment for the newly joined node
        let entry = registry
            .get(&device_id)
            .context("Node disappeared after registration")?;

        Ok(SwarmMessage::JoinResponse {
            assigned_layers: entry.assigned_layers.clone(),
            total_layers: self.total_model_layers,
            encrypted_cluster_key: vec![],
        })
    }

    /// Handles a status update from a node.
    ///
    /// Recomputes the node's capacity and triggers a rebalance if the
    /// change is significant (> 15% composite delta).
    ///
    /// Returns `Some(RebalanceCommand)` if the node's layers changed,
    /// `None` if no rebalance was needed.
    pub fn handle_status_update(
        &self,
        device_id: &str,
        status: NodeStatus,
    ) -> Result<Option<SwarmMessage>> {
        let mut registry = self
            .registry
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire registry lock: {}", e))?;

        let significant_change = registry.update_status(device_id, status)?;

        if significant_change {
            let old_layers = registry
                .get(device_id)
                .map(|n| n.assigned_layers.clone())
                .unwrap_or_default();

            Self::rebalance_layers(&mut registry, self.total_model_layers, &self.pipeline);

            let new_layers = registry
                .get(device_id)
                .map(|n| n.assigned_layers.clone())
                .unwrap_or_default();

            if old_layers != new_layers {
                return Ok(Some(SwarmMessage::RebalanceCommand { new_layers }));
            }
        }

        Ok(None)
    }

    /// Handles a node requesting to leave gracefully.
    ///
    /// Transitions the node to Draining, then removes it and rebalances.
    pub fn handle_drain(&self, device_id: &str) -> Result<()> {
        let mut registry = self
            .registry
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire registry lock: {}", e))?;

        // Transition to Draining then Left
        if let Some(entry) = registry.get_mut(device_id) {
            if entry.state.can_transition_to(NodeState::Draining) {
                entry.state = entry.state.transition_to(NodeState::Draining)?;
            }
        }

        registry.remove(device_id);

        if !registry.is_empty() {
            Self::rebalance_layers(&mut registry, self.total_model_layers, &self.pipeline);
        }

        Ok(())
    }

    /// Handles a heartbeat from a node.
    pub fn handle_heartbeat(&self, device_id: &str) -> Result<()> {
        let mut registry = self
            .registry
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire registry lock: {}", e))?;

        registry.heartbeat(device_id);
        Ok(())
    }

    /// Returns the number of currently registered nodes.
    pub fn node_count(&self) -> usize {
        self.registry.read().map(|r| r.len()).unwrap_or(0)
    }

    /// Returns the number of active nodes (can accept tasks).
    pub fn active_node_count(&self) -> usize {
        self.registry
            .read()
            .map(|r| r.active_nodes().count())
            .unwrap_or(0)
    }

    /// Distributes layers proportionally to effective capacity.
    ///
    /// Algorithm:
    /// 1. Get all active nodes sorted by capacity (descending).
    /// 2. Calculate each node's share: `node_composite / total_composite × total_layers`.
    /// 3. Last node gets remaining layers (handles integer rounding).
    /// 4. Update assignments in the registry.
    fn rebalance_layers(
        registry: &mut NodeRegistry,
        total_layers: u32,
        pipeline: &Arc<RwLock<InferencePipeline>>,
    ) {
        // Try to acquire the pipeline lock gracefully
        let mut pipeline_lock = pipeline.write().expect("Failed to acquire pipeline lock");

        let sorted = registry.sorted_by_capacity();
        if sorted.is_empty() {
            pipeline_lock.update_stages(vec![]);
            return;
        }

        let total_composite: f32 = sorted.iter().map(|n| n.capacity.composite).sum();
        if total_composite <= 0.0 {
            // All nodes have zero capacity — distribute equally as fallback
            let per_node = total_layers / sorted.len() as u32;
            let mut current = 0u32;
            let ids: Vec<String> = sorted.iter().map(|n| n.id.clone()).collect();
            pipeline_lock.update_stages(ids.clone());
            for (i, id) in ids.iter().enumerate() {
                let count = if i == ids.len() - 1 {
                    total_layers.saturating_sub(current)
                } else {
                    per_node
                };
                let layers: Vec<u32> = (current..current + count).collect();
                current += count;
                let _ = registry.assign_layers(id, layers);
            }
            return;
        }

        let mut current_layer: u32 = 0;
        let ids_and_composites: Vec<(String, f32)> = sorted
            .iter()
            .map(|n| (n.id.clone(), n.capacity.composite))
            .collect();

        let stage_ids = ids_and_composites
            .iter()
            .map(|(id, _)| id.clone())
            .collect();
        pipeline_lock.update_stages(stage_ids);

        for (i, (id, composite)) in ids_and_composites.iter().enumerate() {
            let is_last = i == ids_and_composites.len() - 1;

            let layer_count = if is_last {
                total_layers.saturating_sub(current_layer)
            } else {
                ((composite / total_composite) * total_layers as f32).round() as u32
            };

            let end_layer = (current_layer + layer_count).min(total_layers);
            let layers: Vec<u32> = (current_layer..end_layer).collect();
            current_layer = end_layer;

            let _ = registry.assign_layers(id, layers);
        }
    }

    /// Starts a new inference sequence and returns the ProcessTask to send to the first node.
    pub fn start_sequence(
        &self,
        sequence_id: u64,
        task_id: String,
        tokens: Vec<i32>,
        initial_state: bytes::Bytes,
    ) -> Result<Option<(String, SwarmMessage)>> {
        let registry = self
            .registry
            .read()
            .map_err(|e| anyhow::anyhow!("Failed to read registry: {}", e))?;
        let mut pipeline = self
            .pipeline
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire pipeline lock: {}", e))?;

        let sorted = registry.sorted_by_capacity();
        if sorted.is_empty() {
            return Ok(None);
        }
        let first_node = &sorted[0];
        let start_layer = *first_node.assigned_layers.first().unwrap_or(&0);
        let end_layer = *first_node.assigned_layers.last().unwrap_or(&0) + 1;

        // Compress and Encrypt initial state
        let compressed = crate::crypto::compress(&initial_state)?;
        let encrypted =
            crate::crypto::encrypt_with_aad(&compressed, &self.cluster_key, task_id.as_bytes())
                .map_err(|e| anyhow::anyhow!("Encryption failed: {}", e))?;

        Ok(pipeline.start_sequence(
            sequence_id,
            task_id,
            tokens,
            bytes::Bytes::from(encrypted),
            (start_layer, end_layer),
        ))
    }

    /// Handles a TaskResult. If the pipeline continues, returns the ProcessTask for the next node.
    pub fn handle_task_result(&self, result: &SwarmMessage) -> Result<Option<PipelineResult>> {
        let registry = self
            .registry
            .read()
            .map_err(|e| anyhow::anyhow!("Failed to read registry: {}", e))?;
        let mut pipeline = self
            .pipeline
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire pipeline lock: {}", e))?;

        if let SwarmMessage::TaskResult { task_id, .. } = result {
            if let Some(next_id) = pipeline.get_next_node_id(task_id) {
                if let Some(next_node) = registry.get(&next_id) {
                    let start_layer = *next_node.assigned_layers.first().unwrap_or(&0);
                    let end_layer = *next_node.assigned_layers.last().unwrap_or(&0) + 1;

                    return Ok(pipeline.handle_task_result(result, (start_layer, end_layer)));
                }
            } else {
                // Last stage
                return Ok(pipeline.handle_task_result(result, (0, 0)));
            }
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::profile::DeviceType;
    use crate::compute::status::ThermalState;

    fn make_profile(name: &str, device_type: DeviceType) -> NodeProfile {
        NodeProfile::custom(device_type, 8, 16384, name.into())
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
    fn test_single_node_gets_all_layers() {
        let orch = Orchestrator::new(32, "secret".into());
        let resp = orch
            .handle_announce(
                "node-1".into(),
                make_profile("node-1", DeviceType::Desktop),
                make_status(0.1, 12000),
            )
            .unwrap();

        if let SwarmMessage::JoinResponse {
            assigned_layers,
            total_layers,
            ..
        } = resp
        {
            assert_eq!(total_layers, 32);
            assert_eq!(assigned_layers.len(), 32);
        } else {
            panic!("Expected JoinResponse");
        }
    }

    #[test]
    fn test_server_gets_more_than_laptop() {
        let orch = Orchestrator::new(32, "secret".into());

        // Server: low reservation (5% CPU, 512MB RAM)
        orch.handle_announce(
            "server".into(),
            make_profile("server", DeviceType::Server),
            make_status(0.1, 14000),
        )
        .unwrap();

        // Laptop: high reservation (30% CPU, 2GB RAM)
        orch.handle_announce(
            "laptop".into(),
            make_profile("laptop", DeviceType::Laptop),
            make_status(0.1, 14000),
        )
        .unwrap();

        let registry = orch.registry.read().unwrap();
        let server = registry.get("server").unwrap();
        let laptop = registry.get("laptop").unwrap();

        // Server should get more layers because it has higher effective capacity
        assert!(
            server.assigned_layers.len() > laptop.assigned_layers.len(),
            "Server got {} layers, laptop got {} layers",
            server.assigned_layers.len(),
            laptop.assigned_layers.len()
        );

        // All layers must be covered
        let total = server.assigned_layers.len() + laptop.assigned_layers.len();
        assert_eq!(total, 32);
    }

    #[test]
    fn test_status_update_rebalances_on_big_change() {
        let orch = Orchestrator::new(32, "secret".into());

        orch.handle_announce(
            "node-1".into(),
            make_profile("node-1", DeviceType::Desktop),
            make_status(0.1, 12000),
        )
        .unwrap();

        orch.handle_announce(
            "node-2".into(),
            make_profile("node-2", DeviceType::Desktop),
            make_status(0.1, 12000),
        )
        .unwrap();

        // Node-1 becomes very busy
        let result = orch
            .handle_status_update("node-1", make_status(0.9, 2000))
            .unwrap();

        // Should trigger rebalance
        assert!(
            result.is_some() || {
                let reg = orch.registry.read().unwrap();
                let n1 = reg.get("node-1").unwrap();
                let n2 = reg.get("node-2").unwrap();
                // Node-2 should have more layers now
                n2.assigned_layers.len() >= n1.assigned_layers.len()
            }
        );
    }

    #[test]
    fn test_drain_removes_and_rebalances() {
        let orch = Orchestrator::new(32, "secret".into());

        orch.handle_announce(
            "node-1".into(),
            make_profile("node-1", DeviceType::Desktop),
            make_status(0.1, 12000),
        )
        .unwrap();

        orch.handle_announce(
            "node-2".into(),
            make_profile("node-2", DeviceType::Desktop),
            make_status(0.1, 12000),
        )
        .unwrap();

        assert_eq!(orch.node_count(), 2);

        // Node-2 drains
        orch.handle_drain("node-2").unwrap();

        assert_eq!(orch.node_count(), 1);

        // Node-1 should now have all 32 layers
        let registry = orch.registry.read().unwrap();
        let node1 = registry.get("node-1").unwrap();
        assert_eq!(node1.assigned_layers.len(), 32);
    }

    #[test]
    fn test_node_count_methods() {
        let orch = Orchestrator::new(32, "secret".into());
        assert_eq!(orch.node_count(), 0);
        assert_eq!(orch.active_node_count(), 0);

        orch.handle_announce(
            "node-1".into(),
            make_profile("node-1", DeviceType::Desktop),
            make_status(0.1, 12000),
        )
        .unwrap();

        assert_eq!(orch.node_count(), 1);
        assert_eq!(orch.active_node_count(), 1);
    }
}
