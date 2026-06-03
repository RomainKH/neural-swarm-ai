use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Dynamic resource availability snapshot for a node.
///
/// This is measured periodically by the [`ComputeMonitor`](super::ComputeMonitor)
/// and reported to the master when significant changes occur.
///
/// Unlike [`NodeProfile`](super::NodeProfile) which is static, `NodeStatus`
/// changes continuously as the device's workload evolves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStatus {
    /// Current CPU usage across all cores (0.0 = idle, 1.0 = fully loaded).
    pub cpu_usage: f32,

    /// RAM currently in use by all processes (in MB).
    pub ram_used_mb: u64,

    /// RAM available for new allocations (in MB).
    pub ram_available_mb: u64,

    /// Current thermal state of the device.
    pub thermal: ThermalState,

    /// Monotonic timestamp of when this status was measured.
    /// Not serialized over the network (each node has its own clock).
    #[serde(skip)]
    pub measured_at: Option<Instant>,
}

impl NodeStatus {
    /// Creates an initial "unknown" status.
    pub fn unknown() -> Self {
        Self {
            cpu_usage: 0.0,
            ram_used_mb: 0,
            ram_available_mb: 0,
            thermal: ThermalState::Nominal,
            measured_at: None,
        }
    }

    /// Returns the relative change between this status and another.
    /// Used to determine if a status update should be reported.
    pub fn delta(&self, other: &NodeStatus) -> f32 {
        let cpu_delta = (self.cpu_usage - other.cpu_usage).abs();
        let ram_total = (self.ram_used_mb + self.ram_available_mb).max(1) as f32;
        let ram_delta =
            (self.ram_available_mb as f32 - other.ram_available_mb as f32).abs() / ram_total;

        // Weighted average: CPU matters more for inference workloads
        cpu_delta * 0.7 + ram_delta * 0.3
    }
}

impl PartialEq for NodeStatus {
    fn eq(&self, other: &Self) -> bool {
        self.cpu_usage == other.cpu_usage
            && self.ram_used_mb == other.ram_used_mb
            && self.ram_available_mb == other.ram_available_mb
            && self.thermal == other.thermal
    }
}

/// Thermal state of the device.
///
/// Affects the [`EffectiveCapacity`](super::EffectiveCapacity) computation:
/// throttled devices get a lower score to avoid overheating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermalState {
    /// Normal operating temperature.
    Nominal,
    /// Running warm, not yet throttling.
    Warm,
    /// CPU/GPU is being throttled due to heat.
    Throttling,
    /// Critical temperature — should drain and disconnect.
    Critical,
}

impl ThermalState {
    /// Returns the penalty factor applied to capacity scoring.
    /// 1.0 = no penalty, 0.0 = fully penalized.
    pub fn penalty_factor(&self) -> f32 {
        match self {
            ThermalState::Nominal => 1.0,
            ThermalState::Warm => 0.85,
            ThermalState::Throttling => 0.4,
            ThermalState::Critical => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_penalties() {
        assert_eq!(ThermalState::Nominal.penalty_factor(), 1.0);
        assert_eq!(ThermalState::Critical.penalty_factor(), 0.0);
        assert!(ThermalState::Warm.penalty_factor() > ThermalState::Throttling.penalty_factor());
    }

    #[test]
    fn test_status_delta_no_change() {
        let s = NodeStatus {
            cpu_usage: 0.5,
            ram_used_mb: 4000,
            ram_available_mb: 4000,
            thermal: ThermalState::Nominal,
            measured_at: None,
        };
        assert_eq!(s.delta(&s), 0.0);
    }

    #[test]
    fn test_status_delta_significant_cpu_change() {
        let s1 = NodeStatus {
            cpu_usage: 0.2,
            ram_used_mb: 4000,
            ram_available_mb: 4000,
            thermal: ThermalState::Nominal,
            measured_at: None,
        };
        let s2 = NodeStatus {
            cpu_usage: 0.8,
            ram_used_mb: 4000,
            ram_available_mb: 4000,
            thermal: ThermalState::Nominal,
            measured_at: None,
        };
        let delta = s1.delta(&s2);
        // CPU went from 0.2 to 0.8 = 0.6 delta * 0.7 weight = 0.42
        assert!(delta > 0.4);
    }
}
