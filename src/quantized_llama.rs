use std::collections::HashMap;

use candle_core::quantized::QTensor;
use candle_core::quantized::{ggml_file, gguf_file};
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};
use candle_transformers::quantized_nn::RmsNorm;

pub const MAX_SEQ_LEN: usize = 4096;

// QMatMul wrapper adding some tracing.
#[derive(Debug, Clone)]
pub struct QMatMul {
    inner: candle_core::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    pub fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
pub struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        self.feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

#[derive(Debug, Clone)]
pub enum MlpOrMoe {
    Mlp(Mlp),
    MoE {
        n_expert_used: usize,
        feed_forward_gate_inp: QMatMul,
        experts: Vec<Mlp>,
    },
}

impl Module for MlpOrMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::MoE {
                feed_forward_gate_inp,
                experts,
                n_expert_used,
            } => {
                let (b_size, seq_len, hidden_dim) = xs.dims3()?;
                let xs = xs.reshape(((), hidden_dim))?;
                let router_logits = feed_forward_gate_inp.forward(&xs)?;
                let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

                // In order to extract topk, we extract the data from the tensor and manipulate it
                // directly. Maybe we will want to use some custom ops instead at some point.
                let routing_weights = routing_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;

                // routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
                // top_x contains the row indexes to evaluate for each expert.
                let mut top_x = vec![vec![]; experts.len()];
                let mut selected_rws = vec![vec![]; experts.len()];
                for (row_idx, rw) in routing_weights.iter().enumerate() {
                    let mut dst = (0..rw.len() as u32).collect::<Vec<u32>>();
                    dst.sort_by(|&i, &j| rw[j as usize].total_cmp(&rw[i as usize]));
                    let mut sum_routing_weights = 0f32;
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        sum_routing_weights += routing_weight;
                        top_x[expert_idx].push(row_idx as u32);
                    }
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        selected_rws[expert_idx].push(routing_weight / sum_routing_weights)
                    }
                }

                // routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
                // expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

                let mut ys = xs.zeros_like()?;
                for (expert_idx, expert_layer) in experts.iter().enumerate() {
                    let top_x = &top_x[expert_idx];
                    if top_x.is_empty() {
                        continue;
                    }
                    let top_x = Tensor::new(top_x.as_slice(), xs.device())?;
                    let selected_rws =
                        Tensor::new(selected_rws[expert_idx].as_slice(), xs.device())?
                            .reshape(((), 1))?;
                    // Index the correct hidden states and compute the expert hidden state for
                    // the current expert. We need to make sure to multiply the output hidden
                    // states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                    let current_state = xs.index_select(&top_x, 0)?.reshape(((), hidden_dim))?;
                    // current_hidden_states = expert_layer(current_state, routing_weights[top_x_list, idx_list, None])
                    let current_hidden_states = expert_layer.forward(&current_state)?;
                    let current_hidden_states =
                        current_hidden_states.broadcast_mul(&selected_rws)?;
                    ys = ys.index_add(&top_x, &current_hidden_states, 0)?;
                }

                let ys = ys.reshape((b_size, seq_len, hidden_dim))?;
                Ok(ys)
            }
            Self::Mlp(mlp) => mlp.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub attention_wq: QMatMul,
    pub attention_wk: QMatMul,
    pub attention_wv: QMatMul,
    pub attention_wo: QMatMul,
    // Additive q/k/v projection biases (Qwen2 has them; Llama/Mistral don't).
    pub attention_bq: Option<Tensor>,
    pub attention_bk: Option<Tensor>,
    pub attention_bv: Option<Tensor>,
    pub attention_norm: RmsNorm,
    pub mlp_or_moe: MlpOrMoe,
    pub ffn_norm: RmsNorm,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub cos: Tensor,
    pub sin: Tensor,
    pub neg_inf: Tensor,
    pub kv_cache: Option<(Tensor, Tensor)>,
    pub span_attn: tracing::Span,
    pub span_rot: tracing::Span,
    pub span_mlp: tracing::Span,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

impl LayerWeights {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        // The call to contiguous below is only necessary when processing the prompt.
        // When the seq_len is 1 in the inference loop, this is a no-op.
        candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
    }

    pub fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;
        // Qwen2-style additive biases on the projections (no-op for Llama/Mistral).
        let q = match &self.attention_bq { Some(b) => q.broadcast_add(b)?, None => q };
        let k = match &self.attention_bk { Some(b) => k.broadcast_add(b)?, None => k };
        let v = match &self.attention_bv { Some(b) => v.broadcast_add(b)?, None => v };

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            // This call to contiguous ensures that the fast kernel can be called below. It's
            // actually a no-op except when processing the initial prompt so has no significant
            // impact on performance.
            .contiguous()?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let y = if q.device().is_metal() && seq_len == 1 {
            // SDPA will do MQA for us
            candle_nn::ops::sdpa(&q, &k, &v, 1. / (self.head_dim as f32).sqrt(), 1.)?
        } else {
            // Support for MQA, useful for 70B models and mistral.
            let k = candle_transformers::utils::repeat_kv(k, self.n_head / self.n_kv_head)?;
            let v = candle_transformers::utils::repeat_kv(v, self.n_head / self.n_kv_head)?;

            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = match mask {
                None => att,
                Some(mask) => {
                    let mask = mask.broadcast_as(att.shape())?;
                    masked_fill(&att, &mask, &self.neg_inf)?
                }
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            // Convert to contiguous as matmul doesn't support strided vs for now.
            att.matmul(&v.contiguous()?)?
        };

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        let y = self.attention_wo.forward(&y)?;
        Ok(y)
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    /// Token embeddings — only loaded on the pipeline ENTRY node (owns layer 0).
    pub tok_embeddings: Option<Embedding>,
    /// Only the transformer layers assigned to THIS node (a contiguous slice).
    pub layers: Vec<LayerWeights>,
    /// Final norm — only on the EXIT node (owns the last layer).
    pub norm: Option<RmsNorm>,
    /// Output (un-embedding) head — only on the EXIT node.
    pub output: Option<QMatMul>,
    /// Absolute index of `layers[0]` in the full model (0 for the entry node).
    pub layer_offset: usize,
    /// Total number of layers in the full model (so the node knows the global shape).
    pub total_layers: usize,
    pub masks: HashMap<usize, Tensor>,
    pub span: tracing::Span,
    pub span_output: tracing::Span,
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

impl ModelWeights {
    pub fn from_ggml(mut ct: ggml_file::Content, gqa: usize) -> Result<Self> {
        let head_dim = (ct.hparams.n_embd / ct.hparams.n_head) as usize;
        let (cos, sin) = precomput_freqs_cis(head_dim, 10000., &ct.device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &ct.device)?;
        let tok_embeddings = ct.remove("tok_embeddings.weight")?;
        let tok_embeddings = tok_embeddings.dequantize(&ct.device)?;
        let norm = RmsNorm::from_qtensor(ct.remove("norm.weight")?, 1e-5)?;
        let output = ct.remove("output.weight")?;
        let mut layers = Vec::with_capacity(ct.hparams.n_layer as usize);
        for layer_idx in 0..ct.hparams.n_layer {
            let prefix = format!("layers.{layer_idx}");
            let attention_wq = ct.remove(&format!("{prefix}.attention.wq.weight"))?;
            let attention_wk = ct.remove(&format!("{prefix}.attention.wk.weight"))?;
            let attention_wv = ct.remove(&format!("{prefix}.attention.wv.weight"))?;
            let attention_wo = ct.remove(&format!("{prefix}.attention.wo.weight"))?;
            let mlp_or_moe = {
                let feed_forward_w1 = ct.remove(&format!("{prefix}.feed_forward.w1.weight"))?;
                let feed_forward_w2 = ct.remove(&format!("{prefix}.feed_forward.w2.weight"))?;
                let feed_forward_w3 = ct.remove(&format!("{prefix}.feed_forward.w3.weight"))?;
                MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                })
            };
            let attention_norm = ct.remove(&format!("{prefix}.attention_norm.weight"))?;
            let ffn_norm = ct.remove(&format!("{prefix}.ffn_norm.weight"))?;
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_bq: None,
                attention_bk: None,
                attention_bv: None,
                attention_norm: RmsNorm::from_qtensor(attention_norm, 1e-5)?,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, 1e-5)?,
                n_head: ct.hparams.n_head as usize,
                n_kv_head: ct.hparams.n_head as usize / gqa,
                head_dim: (ct.hparams.n_embd / ct.hparams.n_head) as usize,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                kv_cache: None,
                span_attn,
                span_rot,
                span_mlp,
            })
        }
        let n_layer = ct.hparams.n_layer as usize;
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            tok_embeddings: Some(Embedding::new(tok_embeddings, ct.hparams.n_embd as usize)),
            layers,
            norm: Some(norm),
            output: Some(QMatMul::from_qtensor(output)?),
            layer_offset: 0,
            total_layers: n_layer,
            masks: HashMap::new(),
            span,
            span_output,
        })
    }

    /// Loads the full model (all layers + embeddings + output head).
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        // Metadata keys are namespaced by architecture (llama.*, qwen2.*, …).
        let arch = ct
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "llama".to_string());
        let block_count = ct
            .metadata
            .get(&format!("{arch}.block_count"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(0) as usize;
        Self::from_gguf_partial(ct, reader, device, 0, block_count)
    }

    /// Loads ONLY the layers in `[start_layer, end_layer)` into memory, plus the
    /// token embeddings if this node owns layer 0, and the final norm + output head
    /// if it owns the last layer. This is what lets a weak machine hold just a slice
    /// of a model far too big to fit whole.
    pub fn from_gguf_partial<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<Self> {
        // GGUF namespaces hyperparameters by architecture: llama.*, qwen2.*, etc.
        // Detect it so the same loader handles Llama/Mistral/SmolLM (llama) and Qwen2.
        let arch = ct
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "llama".to_string());
        let md_get = |suffix: &str| {
            let key = format!("{arch}.{suffix}");
            match ct.metadata.get(&key) {
                None => candle_core::bail!("cannot find {key} in metadata"),
                Some(v) => Ok(v),
            }
        };

        // Parameter extraction from metadata.
        let n_expert = md_get("expert_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let n_expert_used = md_get("expert_used_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let head_count = md_get("attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("block_count")?.to_u32()? as usize;
        let embedding_length = md_get("embedding_length")?.to_u32()? as usize;
        // Qwen2 omits rope.dimension_count; it equals head_dim = embedding/head_count.
        let rope_dim = md_get("rope.dimension_count")
            .and_then(|v| v.to_u32())
            .map(|v| v as usize)
            .unwrap_or(embedding_length / head_count);
        // Strangely this value is generally 1e-6 in GGUF file but used to be 1e-5 by default.
        let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")?.to_f32()? as f64;

        // Qwen2.5 uses a RoPE base of 1e6; Llama-3.2 uses 5e5; older Llama 1e4.
        let rope_freq_base = md_get("rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);
        let (cos, sin) = precomput_freqs_cis(rope_dim, rope_freq_base, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        // Clamp the requested slice to the real model bounds.
        let start = start_layer.min(block_count);
        let end = end_layer.min(block_count).max(start);
        let is_entry = start == 0;
        let is_exit = end >= block_count;

        // Token embeddings: only the entry node needs them to embed input tokens.
        let tok_embeddings = if is_entry {
            let q = ct.tensor(reader, "token_embd.weight", device)?;
            Some(Embedding::new(q.dequantize(device)?, embedding_length))
        } else {
            None
        };

        // Final norm + output head: only the exit node produces logits.
        let (norm, output) = if is_exit {
            let norm = RmsNorm::from_qtensor(
                ct.tensor(reader, "output_norm.weight", device)?,
                rms_norm_eps,
            )?;
            // Some models tie the output head to the input embeddings.
            let output_q = match ct.tensor(reader, "output.weight", device) {
                Ok(tensor) => tensor,
                Err(_) => ct.tensor(reader, "token_embd.weight", device)?,
            };
            (Some(norm), Some(QMatMul::from_qtensor(output_q)?))
        } else {
            (None, None)
        };

        let mut layers = Vec::with_capacity(end - start);
        for layer_idx in start..end {
            let prefix = format!("blk.{layer_idx}");
            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;
            // Optional q/k/v biases (present in Qwen2, absent in Llama). Stored as F32.
            let mut load_bias = |name: &str| -> Option<Tensor> {
                ct.tensor(reader, &format!("{prefix}.{name}"), device)
                    .ok()
                    .and_then(|q| q.dequantize(device).ok())
            };
            let attention_bq = load_bias("attn_q.bias");
            let attention_bk = load_bias("attn_k.bias");
            let attention_bv = load_bias("attn_v.bias");
            let mlp_or_moe = if n_expert <= 1 {
                let feed_forward_w1 =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
                let feed_forward_w2 =
                    ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
                let feed_forward_w3 =
                    ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
                MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                })
            } else {
                let feed_forward_gate_inp =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate_inp.weight"), device)?;
                let mut experts = Vec::with_capacity(n_expert);
                for i in 0..n_expert {
                    let feed_forward_w1 =
                        ct.tensor(reader, &format!("{prefix}.ffn_gate.{i}.weight"), device)?;
                    let feed_forward_w2 =
                        ct.tensor(reader, &format!("{prefix}.ffn_down.{i}.weight"), device)?;
                    let feed_forward_w3 =
                        ct.tensor(reader, &format!("{prefix}.ffn_up.{i}.weight"), device)?;
                    experts.push(Mlp {
                        feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                        feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                        feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                    })
                }
                MlpOrMoe::MoE {
                    n_expert_used,
                    feed_forward_gate_inp: QMatMul::from_qtensor(feed_forward_gate_inp)?,
                    experts,
                }
            };
            let attention_norm =
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_bq,
                attention_bk,
                attention_bv,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: embedding_length / head_count,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                kv_cache: None,
                span_attn,
                span_rot,
                span_mlp,
            })
        }
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            tok_embeddings,
            layers,
            norm,
            output,
            layer_offset: start,
            total_layers: block_count,
            masks: HashMap::new(),
            span,
            span_output,
        })
    }

    pub fn mask(&mut self, t: usize, device: &Device) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), device)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }

    /// Entry I/O (coordinator role): embed token ids into the initial hidden state
    /// `[1, seq, d]`. Used by a master that delegates the transformer layers to the swarm
    /// but keeps the cheap embedding/un-embedding itself. Requires tok_embeddings loaded
    /// (e.g. `from_gguf_partial(.., 0, 0)`).
    pub fn embed_tokens(&self, tokens: &[i32], device: &Device) -> Result<Tensor> {
        let emb = self
            .tok_embeddings
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("embed_tokens: tok_embeddings not loaded".into()))?;
        let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        let input = Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?;
        emb.forward(&input)
    }

    /// Exit I/O (coordinator role): final RMSNorm + output head on the LAST position of the
    /// hidden state `[1, seq, d]` → logits over the vocabulary. Requires norm + output
    /// loaded (e.g. `from_gguf_partial(.., n_layers, n_layers)`).
    pub fn project_logits(&self, hidden: &Tensor) -> Result<Vec<f32>> {
        let norm = self
            .norm
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("project_logits: final norm not loaded".into()))?;
        let output = self
            .output
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("project_logits: output head not loaded".into()))?;
        let (_b, seq_len, _d) = hidden.dims3()?;
        let x = norm.forward(hidden)?;
        let x = x.i((.., seq_len - 1, ..))?;
        let logits = output.forward(&x)?;
        logits.flatten_all()?.to_vec1()
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, x.device())?)
        };
        let _enter = self.span.enter();
        let tok_embeddings = self
            .tok_embeddings
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("forward() needs tok_embeddings (entry node)".into()))?;
        let mut layer_in = tok_embeddings.forward(x)?;
        for layer in self.layers.iter_mut() {
            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(&x, mask.as_ref(), index_pos)?;
            let x = (attn + residual)?;

            // MLP
            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp_or_moe.forward(&x)?;
            let x = (x + residual)?;
            layer_in = x
        }
        let norm = self
            .norm
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("forward() needs norm (exit node)".into()))?;
        let output = self
            .output
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("forward() needs output (exit node)".into()))?;
        let x = norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        let _enter = self.span_output.enter();
        output.forward(&x)
    }
}
