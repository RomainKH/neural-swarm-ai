use anyhow::Result;

/// Trait that any inference backend must implement to work with NeuralSwarmAI.
///
/// This abstraction allows the library to remain agnostic of the underlying ML
/// runtime (llama.cpp, candle, custom engine, etc.). The orchestration and
/// networking layers don't need to know *how* inference is performed — only that
/// it conforms to this contract.
///
/// # Implementing a custom backend
///
/// ```rust,ignore
/// use neural_swarm_ai::InferenceBackend;
/// use anyhow::Result;
///
/// struct MyBackend { /* ... */ }
///
/// impl InferenceBackend for MyBackend {
///     fn set_state(&mut self, state: &[u8]) -> Result<()> {
///         // Inject serialized KV Cache into your engine
///         Ok(())
///     }
///
///     fn get_state(&self) -> Result<Vec<u8>> {
///         // Extract current KV Cache state
///         Ok(vec![])
///     }
///
///     fn run_layers(&mut self, start_layer: u32, end_layer: u32, tokens: &[i32], sequence_id: usize) -> Result<Vec<f32>> {
///         // Run inference on layers [start_layer, end_layer) with the given tokens
///         Ok(vec![])
///     }
/// }
/// ```
pub trait InferenceBackend: Send + Sync {
    /// Injects a serialized KV Cache state into the inference context.
    ///
    /// This is called before `run_layers` to restore state received from a
    /// previous node in the pipeline.
    fn set_state(&mut self, state: &[u8]) -> Result<()>;

    /// Returns the serialized KV Cache state from the inference context.
    ///
    /// This is called after `run_layers` to extract the updated state that
    /// will be forwarded to the next node in the pipeline.
    fn get_state(&self) -> Result<Vec<u8>>;

    /// Runs inference on the specified layer range with the given input tokens.
    ///
    /// - `start_layer`: First layer to process (inclusive).
    /// - `end_layer`: Last layer to process (exclusive).
    /// - `tokens`: Input token IDs.
    ///
    /// Returns the output logits (probability distribution over vocabulary).
    fn run_layers(&mut self, start_layer: u32, end_layer: u32, tokens: &[i32], sequence_id: usize) -> Result<Vec<f32>>;
}
