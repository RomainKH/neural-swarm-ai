use crate::quantized_llama::ModelWeights;
use crate::InferenceBackend;
use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};

/// Inference backend using the HuggingFace Candle framework (100% Rust).
///
/// Support for Heterogeneous Compute: uses GPU (Metal/CUDA/WGPU) when available.
/// Where a node reads its layer slice from.
#[derive(Clone)]
pub enum ModelSource {
    /// A local GGUF file.
    Path(std::path::PathBuf),
    /// A remote GGUF fetched over HTTP range requests — participate without a local copy.
    #[cfg(feature = "http")]
    Url(String),
}

pub struct CandleBackend {
    pub primary_device: Device,
    pub model: Option<ModelWeights>,
    pub dtype: DType,
    pub intermediate_state: Option<Tensor>,
    /// Where this node loads its layer slice from (local file or remote URL).
    pub source: Option<ModelSource>,
    /// The absolute layer range currently resident in memory, if any.
    pub loaded_range: Option<(u32, u32)>,
    // TODO: Add proper distributed KV cache
}

impl CandleBackend {
    pub fn new(device: Device, dtype: DType) -> Self {
        Self {
            primary_device: device,
            model: None,
            dtype,
            intermediate_state: None,
            source: None,
            loaded_range: None,
        }
    }

    /// Loads the FULL model from a GGUF file (all layers). Use for a single-node
    /// setup or the coordinator; distributed nodes prefer lazy partial loading.
    pub fn load_model(&mut self, filename: &std::path::Path) -> Result<()> {
        let mut file = std::fs::File::open(filename)?;
        let gguf = gguf_file::Content::read(&mut file)?;
        let model = ModelWeights::from_gguf(gguf, &mut file, &self.primary_device)?;
        let total = model.total_layers as u32;
        self.model = Some(model);
        self.source = Some(ModelSource::Path(filename.to_path_buf()));
        self.loaded_range = Some((0, total));
        Ok(())
    }

    /// Registers a local GGUF file WITHOUT loading any layers yet. The actual slice
    /// is loaded lazily by `ensure_layers` based on the range each task requests — so
    /// a node only ever holds the layers it was assigned, never the whole model.
    pub fn set_model_source(&mut self, filename: &std::path::Path) {
        self.source = Some(ModelSource::Path(filename.to_path_buf()));
        self.model = None;
        self.loaded_range = None;
    }

    /// Registers a REMOTE GGUF (URL). Layer slices are streamed over HTTP range
    /// requests on demand — the node participates without ever storing the model.
    #[cfg(feature = "http")]
    pub fn set_model_url(&mut self, url: &str) {
        self.source = Some(ModelSource::Url(url.to_string()));
        self.model = None;
        self.loaded_range = None;
    }

    /// Ensures the layer slice `[start, end)` is the one resident in memory,
    /// (re)loading just that slice from the GGUF source if the assignment changed.
    /// No-op if the model was already fully/correctly loaded for this range.
    pub fn load_layer_range(&mut self, start: u32, end: u32) -> Result<()> {
        if self.loaded_range == Some((start, end)) && self.model.is_some() {
            return Ok(());
        }
        let source = self
            .source
            .clone()
            .ok_or_else(|| anyhow::anyhow!("CandleBackend: no model source set"))?;
        let model = match source {
            ModelSource::Path(path) => {
                let mut file = std::fs::File::open(&path)?;
                let gguf = gguf_file::Content::read(&mut file)?;
                ModelWeights::from_gguf_partial(
                    gguf,
                    &mut file,
                    &self.primary_device,
                    start as usize,
                    end as usize,
                )?
            }
            #[cfg(feature = "http")]
            ModelSource::Url(url) => {
                let mut reader = crate::remote::HttpRangeReader::new(&url)
                    .map_err(|e| anyhow::anyhow!("remote model open: {e}"))?;
                let gguf = gguf_file::Content::read(&mut reader)?;
                ModelWeights::from_gguf_partial(
                    gguf,
                    &mut reader,
                    &self.primary_device,
                    start as usize,
                    end as usize,
                )?
            }
        };
        println!(
            "📦 [Candle] Chargé couches {}..{} ({} couche(s) en mémoire sur {} au total)",
            start,
            end,
            model.layers.len(),
            model.total_layers
        );
        self.model = Some(model);
        self.loaded_range = Some((start, end));
        self.intermediate_state = None;
        Ok(())
    }
}

// SAFETY: Safe to share across threads in the Tokio executor.
unsafe impl Send for CandleBackend {}
unsafe impl Sync for CandleBackend {}

use candle_core::{IndexOp, Module};

#[derive(serde::Serialize, serde::Deserialize)]
enum KVCacheState {
    F32(usize, usize, usize, Vec<f32>),
    Q8_0(usize, usize, usize, f32, Vec<i8>),
}

impl InferenceBackend for CandleBackend {
    fn ensure_layers(&mut self, start_layer: u32, end_layer: u32) -> Result<()> {
        self.load_layer_range(start_layer, end_layer)
    }

    fn set_state(&mut self, state: &[u8]) -> Result<()> {
        if !state.is_empty() {
            let result: std::result::Result<KVCacheState, _> = bincode::deserialize(state);
            let (d1, d2, d3, data) = match result {
                Ok(KVCacheState::Q8_0(d1, d2, d3, scale, q_data)) => {
                    let f_data: Vec<f32> = q_data.into_iter().map(|v| (v as f32) * scale).collect();
                    (d1, d2, d3, f_data)
                }
                Ok(KVCacheState::F32(d1, d2, d3, f_data)) => (d1, d2, d3, f_data),
                Err(_) => {
                    let (d1, d2, d3, f_data): (usize, usize, usize, Vec<f32>) =
                        bincode::deserialize(state)?;
                    (d1, d2, d3, f_data)
                }
            };
            let tensor = Tensor::from_vec(data, (d1, d2, d3), &self.primary_device)?;
            self.intermediate_state = Some(tensor);
        }
        Ok(())
    }

    fn get_state(&self) -> Result<Vec<u8>> {
        if let Some(state) = &self.intermediate_state {
            let (d1, d2, d3) = state.dims3()?;
            let data: Vec<f32> = state.flatten_all()?.to_vec1()?;

            // Serialize the hidden state losslessly (F32). These are inter-layer
            // ACTIVATIONS transiting between pipeline stages: int8 (Q8_0) quantization
            // with a single global scale corrupts them badly (outliers dominate the
            // scale), which garbles generation once a model is split across nodes.
            // zstd still compresses the F32 payload before it hits the wire.
            let state_enum = KVCacheState::F32(d1, d2, d3, data);
            let buffer = bincode::serialize(&state_enum)?;
            return Ok(buffer);
        }
        Ok(vec![])
    }

    fn run_layers(
        &mut self,
        start_layer: u32,
        end_layer: u32,
        tokens: &[i32],
        sequence_id: usize,
    ) -> Result<Vec<f32>> {
        println!(
            "🔥 [Candle] Running layers {} to {} with {} tokens on primary device ({:?})",
            start_layer,
            end_layer,
            tokens.len(),
            self.primary_device
        );

        if let Some(ref mut model) = self.model {
            let seq_len = tokens.len();
            let offset = model.layer_offset;
            let total = model.total_layers;

            // Entry node embeds the tokens; every other node ingests the hidden
            // state forwarded by the previous stage.
            let mut layer_in = if start_layer == 0 {
                let tok = model.tok_embeddings.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("entry node (start_layer 0) is missing token embeddings")
                })?;
                let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
                let input =
                    Tensor::new(tokens_u32.as_slice(), &self.primary_device)?.unsqueeze(0)?;
                tok.forward(&input)?
            } else if let Some(state) = self.intermediate_state.take() {
                state
            } else {
                return Err(anyhow::anyhow!(
                    "non-entry node received no upstream hidden state"
                ));
            };

            let index_pos = sequence_id;

            let mask = if seq_len == 1 {
                None
            } else {
                Some(model.mask(seq_len, &self.primary_device)?)
            };

            // Only iterate layers this node actually holds in memory ([offset, offset+len)).
            let local_end = (end_layer as usize).min(offset + model.layers.len());
            for layer_idx in (start_layer as usize).max(offset)..local_end {
                let layer = &mut model.layers[layer_idx - offset];

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

            // Save intermediate state (serialized in get_state, forwarded to the next stage)
            self.intermediate_state = Some(layer_in.clone());

            // The exit node (owns the last layer) applies the final norm + output head.
            if end_layer as usize >= total {
                let norm = model
                    .norm
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("exit node is missing the final norm"))?;
                let output = model
                    .output
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("exit node is missing the output head"))?;
                let x = norm.forward(&layer_in)?;
                let x = x.i((.., seq_len - 1, ..))?;
                let logits_tensor = output.forward(&x)?;
                let logits: Vec<f32> = logits_tensor.flatten_all()?.to_vec1()?;
                return Ok(logits);
            } else {
                // Not the final stage — the next node continues from this state.
                return Ok(vec![]);
            }
        }

        Ok(vec![])
    }
}
