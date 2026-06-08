use crate::quantized_llama::ModelWeights;
use crate::InferenceBackend;
use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};

/// Inference backend using the HuggingFace Candle framework (100% Rust).
///
/// Support for Heterogeneous Compute: uses GPU (Metal/CUDA/WGPU) when available.
pub struct CandleBackend {
    pub primary_device: Device,
    pub model: Option<ModelWeights>,
    pub dtype: DType,
    pub intermediate_state: Option<Tensor>,
    // TODO: Add proper distributed KV cache
}

impl CandleBackend {
    pub fn new(device: Device, dtype: DType) -> Self {
        Self {
            primary_device: device,
            model: None,
            dtype,
            intermediate_state: None,
        }
    }

    /// Loads a Llama model from a GGUF file.
    pub fn load_model(&mut self, filename: &std::path::Path) -> Result<()> {
        let mut file = std::fs::File::open(filename)?;
        let gguf = gguf_file::Content::read(&mut file)?;
        let model = ModelWeights::from_gguf(gguf, &mut file, &self.primary_device)?;
        self.model = Some(model);
        Ok(())
    }
}

// SAFETY: Safe to share across threads in the Tokio executor.
unsafe impl Send for CandleBackend {}
unsafe impl Sync for CandleBackend {}

use candle_core::{IndexOp, Module};

impl InferenceBackend for CandleBackend {
    fn set_state(&mut self, state: &[u8]) -> Result<()> {
        if !state.is_empty() {
            let (d1, d2, d3, data): (usize, usize, usize, Vec<f32>) = bincode::deserialize(state)?;
            let tensor = Tensor::from_vec(data, (d1, d2, d3), &self.primary_device)?;
            self.intermediate_state = Some(tensor);
        }
        Ok(())
    }

    fn get_state(&self) -> Result<Vec<u8>> {
        if let Some(state) = &self.intermediate_state {
            let (d1, d2, d3) = state.dims3()?;
            let data: Vec<f32> = state.flatten_all()?.to_vec1()?;
            let buffer = bincode::serialize(&(d1, d2, d3, data))?;
            return Ok(buffer);
        }
        Ok(vec![])
    }

    fn run_layers(&mut self, start_layer: u32, end_layer: u32, tokens: &[i32]) -> Result<Vec<f32>> {
        println!(
            "🔥 [Candle] Running layers {} to {} with {} tokens on primary device ({:?})",
            start_layer,
            end_layer,
            tokens.len(),
            self.primary_device
        );

        if let Some(ref mut model) = self.model {
            // Track sequence length manually or get from input
            let mut seq_len = tokens.len();

            let mut layer_in = if start_layer == 0 {
                let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
                let input =
                    Tensor::new(tokens_u32.as_slice(), &self.primary_device)?.unsqueeze(0)?;
                model.tok_embeddings.forward(&input)?
            } else {
                // Deserialize intermediate tensor state here!
                // For now, we return dummy tensor.
                // In a complete implementation, `set_state` sets this.
                let dummy = Tensor::zeros((1, seq_len, 4096), self.dtype, &self.primary_device)?;
                dummy
            };

            let index_pos = Default::default(); // TODO: track index_pos properly

            let mask = if seq_len == 1 {
                None
            } else {
                Some(model.mask(seq_len, &self.primary_device)?)
            };

            // Limit end_layer to max layers
            let actual_end_layer = std::cmp::min(end_layer as usize, model.layers.len());

            for layer_idx in (start_layer as usize)..actual_end_layer {
                let layer = &mut model.layers[layer_idx];

                let x = &layer_in;
                let residual = x;
                let x = layer.attention_norm.forward(x)?;

                // MQA / SDPA attention
                let attn = layer.forward_attn(&x, mask.as_ref(), index_pos)?;
                let x = (attn + residual)?;

                // MLP
                let _enter = layer.span_mlp.enter();
                let residual = &x;
                let x = layer.ffn_norm.forward(&x)?;
                let x = layer.mlp_or_moe.forward(&x)?;
                let x = (x + residual)?;
                layer_in = x;
            }

            // Save intermediate state (to be serialized in get_state)
            // self.intermediate_state = Some(layer_in.clone());

            if end_layer as usize >= model.layers.len() {
                let x = model.norm.forward(&layer_in)?;
                let x = x.i((.., seq_len - 1, ..))?;
                let logits_tensor = model.output.forward(&x)?;

                // Flatten logits back to vec
                let logits: Vec<f32> = logits_tensor.flatten_all()?.to_vec1()?;
                return Ok(logits);
            } else {
                // If not final layer, return empty logits (next node will compute)
                return Ok(vec![]);
            }
        }

        Ok(vec![])
    }
}
