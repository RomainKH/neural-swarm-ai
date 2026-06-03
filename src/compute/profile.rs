use serde::{Deserialize, Serialize};
use sysinfo::System;

/// Static hardware profile for a node, detected once at startup.
///
/// This represents the **ceiling** of what a device can contribute to the swarm.
/// The actual available resources depend on the dynamic [`NodeStatus`](super::NodeStatus).
///
/// # Detection
///
/// ```rust,ignore
/// let profile = NodeProfile::detect();
/// println!("Detected: {} cores, {} MB RAM, type: {:?}",
///     profile.cpu_cores, profile.ram_total_mb, profile.device_type);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NodeProfile {
    /// CPU architecture.
    pub arch: Architecture,

    /// Detected device type (affects resource reservation margins).
    pub device_type: DeviceType,

    /// Total number of logical CPU cores.
    pub cpu_cores: u32,

    /// Total physical RAM in megabytes.
    pub ram_total_mb: u64,

    /// GPU information, if a dedicated/integrated GPU is detected.
    pub gpu: Option<GpuProfile>,

    /// Human-readable hostname for display in dashboards.
    pub hostname: String,
}

impl NodeProfile {
    /// Auto-detects the hardware profile of the current machine.
    pub fn detect() -> Self {
        let sys = System::new_all();

        let cpu_cores = sys.cpus().len() as u32;
        let ram_total_mb = sys.total_memory() / (1024 * 1024);
        let hostname = System::host_name().unwrap_or_else(|| "unknown".to_string());
        let arch = Architecture::detect();
        let device_type = DeviceType::detect(cpu_cores, ram_total_mb);

        Self {
            arch,
            device_type,
            cpu_cores,
            ram_total_mb,
            gpu: None, // GPU detection requires platform-specific APIs
            hostname,
        }
    }

    /// Creates a profile with explicit values (for testing or manual configuration).
    pub fn custom(
        device_type: DeviceType,
        cpu_cores: u32,
        ram_total_mb: u64,
        hostname: String,
    ) -> Self {
        Self {
            arch: Architecture::detect(),
            device_type,
            cpu_cores,
            ram_total_mb,
            gpu: None,
            hostname,
        }
    }
}

/// CPU architecture of the node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Architecture {
    X86_64,
    Aarch64,
    Other,
}

impl Architecture {
    /// Detects the current architecture at compile time.
    pub fn detect() -> Self {
        if cfg!(target_arch = "x86_64") {
            Architecture::X86_64
        } else if cfg!(target_arch = "aarch64") {
            Architecture::Aarch64
        } else {
            Architecture::Other
        }
    }
}

/// Classification of the device, determines default resource reservation margins.
///
/// The key insight: a **Laptop** user is likely doing other work (coding, browsing)
/// while contributing to the swarm. A **Server** is fully dedicated. The margins
/// reflect this reality.
///
/// | Type     | CPU reserved | RAM reserved | Rationale                    |
/// |----------|-------------|-------------|------------------------------|
/// | Server   | 5%          | 512 MB      | Dedicated to swarm           |
/// | Desktop  | 20%         | 2 GB        | Mixed use, user is present   |
/// | Laptop   | 30%         | 2 GB        | Battery + mixed use          |
/// | Mobile   | 40%         | 1 GB        | UX is priority               |
/// | Embedded | 10%         | 256 MB      | Limited but usually dedicated|
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    /// Dedicated server or headless machine.
    Server,
    /// Desktop workstation (iMac, gaming PC, etc.).
    Desktop,
    /// Laptop / portable computer (MacBook, ThinkPad, etc.).
    Laptop,
    /// Smartphone or tablet.
    Mobile,
    /// Single-board computer (Raspberry Pi, Jetson Nano, etc.).
    Embedded,
}

impl DeviceType {
    /// Heuristic detection based on hardware characteristics.
    ///
    /// Rules:
    /// - Very low RAM (< 2 GB) → Embedded
    /// - Low RAM (< 4 GB) and few cores → Mobile
    /// - Moderate specs → Desktop (default for desktops/laptops without battery detection)
    /// - High core count + high RAM → Server
    ///
    /// This is intentionally conservative: Desktop is the safe default because
    /// it reserves a reasonable margin without being too aggressive.
    ///
    /// Users can override this via `NodeProfile::custom()`.
    pub fn detect(cpu_cores: u32, ram_total_mb: u64) -> Self {
        match (cpu_cores, ram_total_mb) {
            (_, ram) if ram < 2048 => DeviceType::Embedded,
            (cores, ram) if cores <= 4 && ram < 4096 => DeviceType::Mobile,
            (cores, ram) if cores >= 16 && ram >= 32768 => DeviceType::Server,
            _ => DeviceType::Desktop,
        }
    }
}

/// GPU information for compute-capable devices.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuProfile {
    /// GPU model name (e.g., "Apple M3 Pro", "NVIDIA RTX 4090").
    pub name: String,

    /// Video RAM in megabytes.
    pub vram_mb: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_embedded() {
        assert_eq!(DeviceType::detect(4, 1024), DeviceType::Embedded);
    }

    #[test]
    fn test_detect_mobile() {
        assert_eq!(DeviceType::detect(4, 3072), DeviceType::Mobile);
    }

    #[test]
    fn test_detect_desktop() {
        assert_eq!(DeviceType::detect(8, 16384), DeviceType::Desktop);
    }

    #[test]
    fn test_detect_server() {
        assert_eq!(DeviceType::detect(32, 65536), DeviceType::Server);
    }

    #[test]
    fn test_profile_detect_runs() {
        // Integration test: just verify detection doesn't panic
        let profile = NodeProfile::detect();
        assert!(profile.cpu_cores > 0);
        assert!(profile.ram_total_mb > 0);
        assert!(!profile.hostname.is_empty());
    }

    #[test]
    fn test_profile_custom() {
        let profile = NodeProfile::custom(DeviceType::Laptop, 10, 16384, "my-mac".into());
        assert_eq!(profile.device_type, DeviceType::Laptop);
        assert_eq!(profile.cpu_cores, 10);
        assert_eq!(profile.ram_total_mb, 16384);
    }

    #[test]
    fn test_profile_serde_roundtrip() {
        let profile = NodeProfile::custom(DeviceType::Desktop, 8, 32768, "test-pc".into());
        let encoded = bincode::serialize(&profile).unwrap();
        let decoded: NodeProfile = bincode::deserialize(&encoded).unwrap();
        assert_eq!(profile, decoded);
    }
}
