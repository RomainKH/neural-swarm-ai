pub mod backend;
pub mod compute;
pub mod executor;
pub mod node;
pub mod orchestrator;
pub mod protocol;

#[cfg(feature = "llama")]
pub mod llama_backend;

#[cfg(any(feature = "server", feature = "client"))]
pub mod transport;

pub use backend::InferenceBackend;
pub use compute::{
    DeviceType, EffectiveCapacity, NodeProfile, NodeStatus, ResourceReservation, ThermalState,
};
pub use executor::Executor;
pub use node::{NodeEntry, NodeRegistry, NodeState};
pub use orchestrator::Orchestrator;
pub use protocol::SwarmMessage;

#[cfg(feature = "llama")]
pub use llama_backend::LlamaBackend;
