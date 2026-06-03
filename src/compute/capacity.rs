use super::profile::{DeviceType, NodeProfile};
use super::status::NodeStatus;
use serde::{Deserialize, Serialize};

/// Defines how much of a device's resources are **reserved for the user**
/// and off-limits to the swarm.
///
/// This is the safety margin that prevents a connected Mac from lagging
/// when the user is coding while contributing compute to the swarm.
///
/// # How it works
///
/// ```text
/// Total CPU: 10 cores (from NodeProfile)
/// Reserved:   2 cores (20% for Desktop)
/// Max for swarm: 8 cores
///
/// If user is using 3 cores right now:
///   Effective for swarm = max(0, 8 - 3) = 5 cores → score = 5/10 = 0.50
///
/// If user is idle:
///   Effective for swarm = max(0, 8 - 0) = 8 cores → score = 8/10 = 0.80
///
/// The score NEVER exceeds 0.80 for a Desktop, even if fully idle.
/// This guarantees the user always has headroom.
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ResourceReservation {
    /// Fraction of total CPU reserved for the user (0.0 → 1.0).
    pub cpu_reserved: f32,

    /// Minimum RAM (in MB) always reserved for the OS and user applications.
    pub ram_reserved_mb: u64,
}

impl ResourceReservation {
    /// Returns the default reservation for a given device type.
    ///
    /// These defaults are intentionally conservative: it's better to
    /// under-utilize a node than to make the user's machine lag.
    pub fn for_device(device_type: &DeviceType) -> Self {
        match device_type {
            DeviceType::Server => Self {
                cpu_reserved: 0.05,
                ram_reserved_mb: 512,
            },
            DeviceType::Desktop => Self {
                cpu_reserved: 0.20,
                ram_reserved_mb: 2048,
            },
            DeviceType::Laptop => Self {
                cpu_reserved: 0.30,
                ram_reserved_mb: 2048,
            },
            DeviceType::Mobile => Self {
                cpu_reserved: 0.40,
                ram_reserved_mb: 1024,
            },
            DeviceType::Embedded => Self {
                cpu_reserved: 0.10,
                ram_reserved_mb: 256,
            },
        }
    }

    /// Creates a custom reservation (for advanced users who know their setup).
    pub fn custom(cpu_reserved: f32, ram_reserved_mb: u64) -> Self {
        Self {
            cpu_reserved: cpu_reserved.clamp(0.0, 0.95),
            ram_reserved_mb,
        }
    }
}

/// The computed effective capacity of a node, used by the orchestrator
/// for layer assignment decisions.
///
/// This is the **single source of truth** for scheduling. The orchestrator
/// never looks at raw Profile or Status — only at EffectiveCapacity.
///
/// Formula:
/// ```text
/// cpu_score = max(0, (1 - cpu_reserved) - cpu_usage) / 1.0
/// ram_score = max(0, ram_available - ram_reserved) / ram_total
/// composite = (0.6 × cpu_score + 0.4 × ram_score) × thermal_penalty
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EffectiveCapacity {
    /// Available CPU capacity after reservation and current usage (0.0 → 1.0).
    pub cpu_score: f32,

    /// Available RAM for inference after reservation (in MB).
    pub ram_available_mb: u64,

    /// Final composite score used for proportional layer distribution (0.0 → 1.0).
    pub composite: f32,
}

impl EffectiveCapacity {
    /// Computes the effective capacity from a node's profile, current status,
    /// and resource reservation.
    pub fn compute(
        profile: &NodeProfile,
        status: &NodeStatus,
        reservation: &ResourceReservation,
    ) -> Self {
        // CPU: what's the usable fraction after reservation and current usage?
        // ceiling = 1.0 - reserved (e.g., 0.80 for Desktop)
        // available = ceiling - current_usage
        let cpu_ceiling = 1.0 - reservation.cpu_reserved;
        let cpu_score = (cpu_ceiling - status.cpu_usage).max(0.0);

        // RAM: how much is available for inference after reservation?
        let ram_after_reservation = status
            .ram_available_mb
            .saturating_sub(reservation.ram_reserved_mb);
        let ram_total = profile.ram_total_mb.max(1);
        let ram_score = ram_after_reservation as f32 / ram_total as f32;

        // Composite: weighted CPU + RAM with thermal penalty
        let thermal_penalty = status.thermal.penalty_factor();
        let raw_composite = (0.6 * cpu_score + 0.4 * ram_score).clamp(0.0, 1.0);
        let composite = raw_composite * thermal_penalty;

        Self {
            cpu_score,
            ram_available_mb: ram_after_reservation,
            composite,
        }
    }

    /// Returns a zero capacity (for nodes that are offline or critical).
    pub fn zero() -> Self {
        Self {
            cpu_score: 0.0,
            ram_available_mb: 0,
            composite: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::status::ThermalState;

    fn make_profile(cpu_cores: u32, ram_total_mb: u64) -> NodeProfile {
        NodeProfile::custom(DeviceType::Desktop, cpu_cores, ram_total_mb, "test".into())
    }

    fn make_status(cpu_usage: f32, ram_available_mb: u64) -> NodeStatus {
        NodeStatus {
            cpu_usage,
            ram_used_mb: 0,
            ram_available_mb,
            thermal: ThermalState::Nominal,
            measured_at: None,
        }
    }

    #[test]
    fn test_reservation_defaults() {
        let server = ResourceReservation::for_device(&DeviceType::Server);
        let laptop = ResourceReservation::for_device(&DeviceType::Laptop);

        // Server should reserve less than laptop
        assert!(server.cpu_reserved < laptop.cpu_reserved);
        assert!(server.ram_reserved_mb < laptop.ram_reserved_mb);
    }

    #[test]
    fn test_custom_reservation_clamped() {
        let r = ResourceReservation::custom(1.5, 1024);
        assert_eq!(r.cpu_reserved, 0.95); // Clamped to max 95%
    }

    #[test]
    fn test_idle_desktop_capacity() {
        let profile = make_profile(10, 16384);
        let status = make_status(0.0, 12000); // Idle, 12GB free
        let reservation = ResourceReservation::for_device(&DeviceType::Desktop);

        let cap = EffectiveCapacity::compute(&profile, &status, &reservation);

        // CPU: ceiling = 0.80, usage = 0.0, score = 0.80
        assert!((cap.cpu_score - 0.80).abs() < 0.01);
        // RAM: 12000 - 2048 = 9952 available
        assert_eq!(cap.ram_available_mb, 9952);
        // Composite should be high but capped by reservation
        assert!(cap.composite > 0.5);
        assert!(cap.composite <= 1.0);
    }

    #[test]
    fn test_busy_desktop_capacity() {
        let profile = make_profile(10, 16384);
        let status = make_status(0.6, 4000); // 60% CPU, 4GB free
        let reservation = ResourceReservation::for_device(&DeviceType::Desktop);

        let cap = EffectiveCapacity::compute(&profile, &status, &reservation);

        // CPU: ceiling = 0.80, usage = 0.60, score = 0.20
        assert!((cap.cpu_score - 0.20).abs() < 0.01);
        // RAM: 4000 - 2048 = 1952 available
        assert_eq!(cap.ram_available_mb, 1952);
        // Composite should be low
        assert!(cap.composite < 0.3);
    }

    #[test]
    fn test_overloaded_desktop_floors_at_zero() {
        let profile = make_profile(10, 16384);
        let status = make_status(0.95, 1000); // 95% CPU, only 1GB free
        let reservation = ResourceReservation::for_device(&DeviceType::Desktop);

        let cap = EffectiveCapacity::compute(&profile, &status, &reservation);

        // CPU: ceiling = 0.80, usage = 0.95 → score = 0.0 (clamped)
        assert_eq!(cap.cpu_score, 0.0);
        // RAM: 1000 - 2048 → 0 (saturating sub)
        assert_eq!(cap.ram_available_mb, 0);
        // Composite should be 0
        assert_eq!(cap.composite, 0.0);
    }

    #[test]
    fn test_thermal_throttling_penalty() {
        let profile = make_profile(10, 16384);
        let status_nominal = make_status(0.2, 12000);
        let mut status_throttled = status_nominal.clone();
        status_throttled.thermal = ThermalState::Throttling;
        let reservation = ResourceReservation::for_device(&DeviceType::Desktop);

        let cap_nominal = EffectiveCapacity::compute(&profile, &status_nominal, &reservation);
        let cap_throttled = EffectiveCapacity::compute(&profile, &status_throttled, &reservation);

        // Throttled should be significantly lower
        assert!(cap_throttled.composite < cap_nominal.composite * 0.5);
    }

    #[test]
    fn test_server_has_more_capacity_than_laptop() {
        let profile = make_profile(10, 16384);
        let status = make_status(0.3, 10000); // Same load

        let server_cap = EffectiveCapacity::compute(
            &profile,
            &status,
            &ResourceReservation::for_device(&DeviceType::Server),
        );
        let laptop_cap = EffectiveCapacity::compute(
            &profile,
            &status,
            &ResourceReservation::for_device(&DeviceType::Laptop),
        );

        // Same hardware, same load, but server reserves less → higher score
        assert!(server_cap.composite > laptop_cap.composite);
    }

    #[test]
    fn test_zero_capacity() {
        let cap = EffectiveCapacity::zero();
        assert_eq!(cap.composite, 0.0);
        assert_eq!(cap.cpu_score, 0.0);
        assert_eq!(cap.ram_available_mb, 0);
    }
}
