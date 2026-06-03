use crate::protocol::SwarmMessage;
use bytes::Bytes;
use llama_cpp_2::context::LlamaContext;

/// Executes computation tasks on the local model.
pub struct Executor {
    pub device_id: String,
}

impl Executor {
    /// Creates a new executor for the current device.
    pub fn new(id: String) -> Self {
        Self { device_id: id }
    }

    /// Processes a computation task.
    ///
    /// # Safety
    /// This function calls `set_state_data` and `copy_state_data` which are unsafe in `llama-cpp-2`.
    pub unsafe fn run_task(
        &self,
        ctx: &mut LlamaContext,
        task: SwarmMessage,
    ) -> Option<SwarmMessage> {
        if let SwarmMessage::ProcessTask {
            task_id,
            input_state,
            start_layer: _,
            end_layer: _,
            tokens: _,
        } = task
        {
            // 1. Inject received state (KV Cache)
            ctx.set_state_data(&input_state);

            // 2. Perform inference
            // Note: In a full implementation, we'd limit computation to start_layer -> end_layer.

            // 3. Extract new state
            let state_size = ctx.get_state_size();
            let mut output_state = vec![0u8; state_size];
            ctx.copy_state_data(output_state.as_mut_ptr());

            return Some(SwarmMessage::TaskResult {
                task_id,
                output_state: Bytes::from(output_state),
                logits: vec![], // Populate with actual logits
            });
        }
        None
    }
}
