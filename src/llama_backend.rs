use crate::backend::InferenceBackend;
use anyhow::Result;
use llama_cpp_2::context::LlamaContext;

/// Inference backend powered by [llama.cpp](https://github.com/ggerganov/llama.cpp)
/// via the `llama-cpp-2` Rust bindings.
///
/// This backend is only available when the `llama` feature is enabled:
///
/// ```toml
/// [dependencies]
/// neural-swarm-ai = { version = "0.1", features = ["llama"] }
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use neural_swarm_ai::{Executor, LlamaBackend};
///
/// // Assuming you have a LlamaContext from llama-cpp-2:
/// let mut backend = LlamaBackend::new(&mut llama_ctx);
/// let result = executor.run_task(&mut backend, task)?;
/// ```
pub struct LlamaBackend<'a> {
    ctx: &'a mut LlamaContext,
}

impl<'a> LlamaBackend<'a> {
    /// Wraps an existing `LlamaContext` into a NeuralSwarmAI-compatible backend.
    pub fn new(ctx: &'a mut LlamaContext) -> Self {
        Self { ctx }
    }
}

impl InferenceBackend for LlamaBackend<'_> {
    fn set_state(&mut self, state: &[u8]) -> Result<()> {
        // SAFETY: `set_state_data` is unsafe in llama-cpp-2 because it
        // performs raw pointer operations on the KV cache. We trust the
        // caller to provide valid serialized state from `get_state`.
        unsafe { self.ctx.set_state_data(state) };
        Ok(())
    }

    fn get_state(&self) -> Result<Vec<u8>> {
        let state_size = self.ctx.get_state_size();
        let mut buf = vec![0u8; state_size];
        // SAFETY: `copy_state_data` writes into the provided buffer.
        // We allocate exactly `state_size` bytes as required.
        unsafe { self.ctx.copy_state_data(buf.as_mut_ptr()) };
        Ok(buf)
    }

    fn run_layers(
        &mut self,
        _start_layer: u32,
        _end_layer: u32,
        _tokens: &[i32],
    ) -> Result<Vec<f32>> {
        // TODO: Implement layer-specific inference.
        // In a full implementation, this would:
        // 1. Decode the tokens through the specified layer range
        // 2. Return the output logits from the final layer
        //
        // llama.cpp doesn't natively support layer-range execution,
        // so this would require either:
        // - A custom fork with layer slicing support
        // - Running the full model and extracting intermediate activations
        Ok(vec![])
    }
}
