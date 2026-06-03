use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

/// Lifecycle state of a node in the swarm.
///
/// Each node follows a strict state machine with validated transitions.
/// The orchestrator uses this to determine what actions are valid for a node.
///
/// ```text
///                 ┌──────────┐
///                 │ Joining  │
///                 └────┬─────┘
///                      │ handshake complete
///                      ▼
///                 ┌──────────┐
///          ┌──────│  Ready   │
///          │      └────┬─────┘
///          │           │ layers assigned
///          │           ▼
///          │      ┌──────────┐
///          │      │  Active  │◄────────┐
///          │      └──┬────┬──┘         │
///          │         │    │             │
///          │    (ok)  │    │ degraded   │ recovered
///          │         │    ▼             │
///          │         │  ┌──────────┐   │
///          │         │  │ Degraded │───┘
///          │         │  └────┬─────┘
///          │         │       │ timeout / critical
///          │         │       ▼
///          │         │  ┌──────────┐
///          └─────────┴─►│ Draining │
///                       └────┬─────┘
///                            │ all tasks complete
///                            ▼
///                       ┌──────────┐
///                       │   Left   │
///                       └──────────┘
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeState {
    /// Handshake in progress, node is being registered.
    Joining,

    /// Registered and ready to receive layer assignments.
    Ready,

    /// Actively computing with assigned layers.
    Active,

    /// Resources are low but node is still functional.
    /// Keeps current tasks, but receives fewer new ones.
    Degraded,

    /// Gracefully shutting down. Finishes current tasks,
    /// doesn't accept new ones. Layers will be redistributed.
    Draining,

    /// Disconnected from the swarm.
    Left,
}

impl NodeState {
    /// Attempts to transition to a new state.
    /// Returns an error if the transition is invalid.
    pub fn transition_to(&self, target: NodeState) -> Result<NodeState> {
        if self.can_transition_to(target) {
            Ok(target)
        } else {
            bail!("Invalid state transition: {:?} → {:?}", self, target)
        }
    }

    /// Checks if a transition to the target state is valid.
    pub fn can_transition_to(&self, target: NodeState) -> bool {
        matches!(
            (self, target),
            // Normal flow
            (NodeState::Joining, NodeState::Ready)
                | (NodeState::Ready, NodeState::Active)
                | (NodeState::Active, NodeState::Degraded)
                | (NodeState::Degraded, NodeState::Active) // recovery
                | (NodeState::Active, NodeState::Draining)
                | (NodeState::Degraded, NodeState::Draining)
                | (NodeState::Draining, NodeState::Left)
                // Skip states (e.g., node leaves before getting layers)
                | (NodeState::Ready, NodeState::Draining)
                | (NodeState::Joining, NodeState::Left) // connection failed
        )
    }

    /// Returns true if the node can accept new tasks.
    pub fn can_accept_tasks(&self) -> bool {
        matches!(self, NodeState::Active | NodeState::Ready)
    }

    /// Returns true if the node has an active presence in the swarm.
    pub fn is_alive(&self) -> bool {
        !matches!(self, NodeState::Left)
    }
}

impl std::fmt::Display for NodeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeState::Joining => write!(f, "Joining"),
            NodeState::Ready => write!(f, "Ready"),
            NodeState::Active => write!(f, "Active"),
            NodeState::Degraded => write!(f, "Degraded"),
            NodeState::Draining => write!(f, "Draining"),
            NodeState::Left => write!(f, "Left"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_normal_flow() {
        let state = NodeState::Joining;
        let state = state.transition_to(NodeState::Ready).unwrap();
        let state = state.transition_to(NodeState::Active).unwrap();
        let state = state.transition_to(NodeState::Draining).unwrap();
        let state = state.transition_to(NodeState::Left).unwrap();
        assert_eq!(state, NodeState::Left);
    }

    #[test]
    fn test_valid_degradation_and_recovery() {
        let state = NodeState::Active;
        let state = state.transition_to(NodeState::Degraded).unwrap();
        let state = state.transition_to(NodeState::Active).unwrap();
        assert_eq!(state, NodeState::Active);
    }

    #[test]
    fn test_valid_degraded_to_draining() {
        let state = NodeState::Degraded;
        let state = state.transition_to(NodeState::Draining).unwrap();
        assert_eq!(state, NodeState::Draining);
    }

    #[test]
    fn test_invalid_left_to_active() {
        let state = NodeState::Left;
        assert!(state.transition_to(NodeState::Active).is_err());
    }

    #[test]
    fn test_invalid_joining_to_active() {
        let state = NodeState::Joining;
        assert!(state.transition_to(NodeState::Active).is_err());
    }

    #[test]
    fn test_invalid_draining_to_active() {
        let state = NodeState::Draining;
        assert!(state.transition_to(NodeState::Active).is_err());
    }

    #[test]
    fn test_can_accept_tasks() {
        assert!(NodeState::Active.can_accept_tasks());
        assert!(NodeState::Ready.can_accept_tasks());
        assert!(!NodeState::Degraded.can_accept_tasks());
        assert!(!NodeState::Draining.can_accept_tasks());
        assert!(!NodeState::Left.can_accept_tasks());
    }

    #[test]
    fn test_is_alive() {
        assert!(NodeState::Active.is_alive());
        assert!(NodeState::Degraded.is_alive());
        assert!(NodeState::Draining.is_alive());
        assert!(!NodeState::Left.is_alive());
    }

    #[test]
    fn test_early_disconnect() {
        // Connection failed during joining
        let state = NodeState::Joining;
        let state = state.transition_to(NodeState::Left).unwrap();
        assert_eq!(state, NodeState::Left);
    }
}
