use crate::backend::InferenceBackend;
use crate::protocol::SwarmMessage;
use anyhow::Result;
use bytes::Bytes;
use tokio::sync::mpsc;

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
/// let executor = Executor::new("node-1".into(), [0u8; 32]);
///
/// // Start the executor loop with a backend and channels
/// // executor.run_loop(backend, task_rx, result_tx).await;
/// ```
pub struct Executor {
    pub device_id: String,
    pub cluster_key: [u8; 32],
}

impl Executor {
    /// Creates a new executor for the current device.
    pub fn new(id: String, key: [u8; 32]) -> Self {
        Self {
            device_id: id,
            cluster_key: key,
        }
    }

    /// Starts the executor loop in a separate task.
    ///
    /// This allows the node to receive the next task from the network
    /// (e.g., KV Cache for layer N+1) while the GPU is still busy
    /// computing the current task (layer N).
    pub async fn run_loop(
        &self,
        mut backend: Box<dyn InferenceBackend>,
        mut task_rx: mpsc::Receiver<SwarmMessage>,
        result_tx: mpsc::Sender<SwarmMessage>,
    ) -> Result<()> {
        while let Some(task) = task_rx.recv().await {
            if let Some(result) = self.run_task(backend.as_mut(), task)? {
                result_tx.send(result).await?;
            }
        }
        Ok(())
    }

    /// Processes a computation task using the provided inference backend.
    pub fn run_task(
        &self,
        backend: &mut dyn InferenceBackend,
        task: SwarmMessage,
    ) -> Result<Option<SwarmMessage>> {
        if let SwarmMessage::ProcessTask {
            task_id,
            sequence_id,
            input_state,
            start_layer,
            end_layer,
            tokens,
            ..
        } = task
        {
            // 1. Decrypt and Decompress received state
            // Use task_id as AAD to prevent replay/cross-task attacks
            let decrypted = crate::crypto::decrypt_with_aad(
                &input_state,
                &self.cluster_key,
                task_id.as_bytes(),
            )
            .map_err(|e| anyhow::anyhow!("Decryption failed: {}", e))?;
            let decompressed = crate::crypto::decompress(&decrypted)?;

            // 2. Inject state into backend
            backend.set_state(&decompressed)?;

            // 3. Run inference on assigned layers
            let logits = backend.run_layers(start_layer, end_layer, &tokens, sequence_id as usize)?;

            // 4. Extract updated state
            let output_raw = backend.get_state()?;

            // 5. Compress and Encrypt for forwarding
            let compressed = crate::crypto::compress(&output_raw)?;
            let encrypted =
                crate::crypto::encrypt_with_aad(&compressed, &self.cluster_key, task_id.as_bytes())
                    .map_err(|e| anyhow::anyhow!("Encryption failed: {}", e))?;

            return Ok(Some(SwarmMessage::TaskResult {
                task_id,
                output_state: Bytes::from(encrypted),
                logits,
                sequence_id,
            }));
        }
        Ok(None)
    }
}
