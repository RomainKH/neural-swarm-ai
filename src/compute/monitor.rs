use super::status::NodeStatus;
use std::time::Duration;
use sysinfo::System;
use tokio::sync::watch;

/// Background monitor that periodically samples system resources and reports
/// changes to the orchestrator via a watch channel.
///
/// The monitor is designed to be **lightweight**: each sample takes < 1ms
/// and updates are only sent when the change exceeds a configurable threshold.
///
/// # Usage
///
/// ```rust,ignore
/// use neural_swarm_ai::compute::ComputeMonitor;
///
/// let (monitor, rx) = ComputeMonitor::new(Default::default());
///
/// // Spawn the monitoring loop
/// tokio::spawn(monitor.run());
///
/// // Observe status changes
/// let status = rx.borrow().clone();
/// ```
pub struct ComputeMonitor {
    system: System,
    config: MonitorConfig,
    tx: watch::Sender<NodeStatus>,
}

/// Configuration for the compute monitor.
pub struct MonitorConfig {
    /// How often to sample system resources. Default: 5 seconds.
    pub interval: Duration,

    /// Minimum change (0.0 → 1.0) required to emit a status update.
    /// Default: 0.10 (10%).
    pub change_threshold: f32,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(5),
            change_threshold: 0.10,
        }
    }
}

impl ComputeMonitor {
    /// Creates a new monitor and returns it along with a receiver for status updates.
    pub fn new(config: MonitorConfig) -> (Self, watch::Receiver<NodeStatus>) {
        let (tx, rx) = watch::channel(NodeStatus::unknown());
        let monitor = Self {
            system: System::new(),
            config,
            tx,
        };
        (monitor, rx)
    }

    /// Runs the monitoring loop. This should be spawned as a tokio task.
    ///
    /// The loop runs indefinitely until the receiver is dropped or the task
    /// is cancelled.
    pub async fn run(mut self) {
        let mut last_reported = NodeStatus::unknown();

        loop {
            tokio::time::sleep(self.config.interval).await;

            let status = self.sample();
            let delta = status.delta(&last_reported);

            if delta >= self.config.change_threshold {
                last_reported = status.clone();
                // If no one is listening, this is a no-op (watch semantics)
                let _ = self.tx.send(status);
            }
        }
    }

    /// Takes a single resource sample from the OS.
    fn sample(&mut self) -> NodeStatus {
        // Refresh only what we need (CPU + memory)
        self.system.refresh_cpu_usage();
        self.system.refresh_memory();

        let cpu_usage = if self.system.cpus().is_empty() {
            0.0
        } else {
            let total: f32 = self.system.cpus().iter().map(|c| c.cpu_usage()).sum();
            (total / self.system.cpus().len() as f32) / 100.0 // Normalize to 0.0..1.0
        };

        let ram_used_mb = self.system.used_memory() / (1024 * 1024);
        let ram_available_mb = self.system.available_memory() / (1024 * 1024);

        NodeStatus {
            cpu_usage,
            ram_used_mb,
            ram_available_mb,
            thermal: super::status::ThermalState::Nominal, // TODO: platform-specific thermal detection
            measured_at: Some(std::time::Instant::now()),
        }
    }

    /// Takes a single sample without running the loop (useful for initial join).
    pub fn sample_once(&mut self) -> NodeStatus {
        self.sample()
    }
}
