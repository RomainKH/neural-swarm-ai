use crate::backend::InferenceBackend;
use crate::protocol::SwarmMessage;
use anyhow::Result;
use bytes::Bytes;

/// Executes computation tasks on the local model using any [`InferenceBackend`].
///
/// The `Executor` is the core compute unit on each node in the swarm. It
/// receives tasks from the [`Orchestrator`](crate::Orchestrator), runs
/// inference on the assigned layers, and returns results.
///
/// # Example
///
/// ```rust,ignore
/// use neural_swarm_ai::Executor;
///
/// let executor = Executor::new("node-1".into());
///
/// // With any backend implementing InferenceBackend:
/// let result = executor.run_task(&mut my_backend, task)?;
/// ```
pub struct Executor {
    pub device_id: String,
}

impl Executor {
    /// Creates a new executor for the current device.
    pub fn new(id: String) -> Self {
        Self { device_id: id }
    }

    /// Processes a computation task using the provided inference backend.
    ///
    /// This is a **safe** function — all unsafe operations are encapsulated
    /// within the backend implementation.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend fails to set/get state or run inference.
    pub fn run_task(
        &self,
        backend: &mut dyn InferenceBackend,
        task: SwarmMessage,
    ) -> Result<Option<SwarmMessage>> {
        if let SwarmMessage::ProcessTask {
            task_id,
            input_state,
            start_layer,
            end_layer,
            tokens,
        } = task
        {
            // 1. Inject received state (KV Cache) from previous node
            backend.set_state(&input_state)?;

            // 2. Run inference on assigned layers
            let logits = backend.run_layers(start_layer, end_layer, &tokens)?;

            // 3. Extract updated state for forwarding to next node
            let output_state = backend.get_state()?;

            return Ok(Some(SwarmMessage::TaskResult {
                task_id,
                output_state: Bytes::from(output_state),
                logits,
            }));
        }
        Ok(None)
    }
}
