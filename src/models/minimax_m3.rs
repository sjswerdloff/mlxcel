// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! MiniMax-M3 model with Multi-head Sparse Attention (MSA)
//!
//! Reference implementation following:
//! - arXiv:2606.13392 (MSA paper)
//! - huggingface/transformers modeling_minimax_m3_vl.py (authoritative)
//!
//! Key architecture facts verified against official Transformers code:
//! - Index Branch: index_q_proj (H_kv heads), index_k_proj (1 shared head)
//! - Q/K norm is per-head on head_dim (not on full projection)
//! - RoPE applied AFTER norm+transpose
//! - Index Branch also gets RoPE
//! - Custom SwiGLU activation: clamp + sigmoid * alpha (NOT standard SiLU)
//! - routed_scaling_factor applied after MoE expert computation
//! - Block max-pool scoring with causal masking
//! - Local block always included (score set to inf)

use crate::models::switch_layers::{SwitchLinear, gather_sort};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{GemmaRMSNorm, KVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;
use tracing::{debug, trace, warn};

/// Load a UnifiedLinear. Auto-detects quantization mode from weight shapes.
fn load_linear(weights: &WeightMap, prefix: &str, g: i32, b: i32) -> Result<UnifiedLinear, String> {
    UnifiedLinear::from_weights(weights, prefix, g, b)
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub text_config: TextConfig,
    #[serde(default)]
    pub quantization: Option<M3Quantization>,
}

/// Quantization descriptor for MiniMax-M3 checkpoints.
///
/// Accepts two on-disk schemas so the same struct deserializes both
/// canonical MiniMax-M3 quantized exports:
///
/// * **mlx-community schema** — produced by `mlx_lm.convert`
///   (e.g. `mlx-community/MiniMax-M3-*-mlx`):
///
///   ```json
///   "quantization": { "group_size": 32, "bits": 8 }
///   ```
///
/// * **HF `quantization_config` schema** — used by MiniMax's own
///   MXFP8 upload (`MiniMaxAI/MiniMax-M3-MXFP8`) and by NVIDIA-style
///   MXFP4 / NVFP4 exports:
///
///   ```json
///   "quantization_config": {
///     "quant_method": "mxfp8",
///     "activation_scheme": "dynamic",
///     "weight_block_size": [1, 32],
///     "ignored_layers": ["lm_head", "model.embed_tokens", ...]
///   }
///   ```
///
/// `group_size` / `bits` are canonical for the loader — they drive
/// `UnifiedLinear::from_weights`'s mode auto-detect (bits == 8 →
/// `mxfp8`, group_size == 16 → `nvfp4`, else → `mxfp4`, unless
/// `.biases` are present, which forces `affine`). When only the HF
/// schema is present [`M3Quantization::group_size_effective`] derives
/// `group_size` from `weight_block_size[1]` and
/// [`M3Quantization::bits_effective`] derives `bits` from
/// `quant_method`.
///
/// `ignored_layers` is descriptive only. The loader's `.scales`-key
/// auto-detect already falls back to unquantized `Linear` for layers
/// whose safetensors export lacks a scales tensor, which is exactly
/// the set the checkpoint author enumerates here (lm_head, embed
/// tokens, vision heads, MoE gates).
#[derive(Debug, Clone, Deserialize)]
pub struct M3Quantization {
    #[serde(default)]
    pub group_size: Option<i32>,
    #[serde(default)]
    pub bits: Option<i32>,

    // HF `quantization_config` fields.
    #[serde(default)]
    pub quant_method: Option<String>,
    #[serde(default)]
    pub weight_block_size: Option<Vec<i32>>,
    #[serde(default)]
    pub activation_scheme: Option<String>,
    #[serde(default)]
    pub ignored_layers: Option<Vec<String>>,
}

impl M3Quantization {
    /// Group size to feed `UnifiedLinear::from_weights`.
    ///
    /// Prefers the explicit `group_size` field (mlx-community schema).
    /// Falls back to `weight_block_size[1]` for HF exports — the outer
    /// `[1, N]` layout gives one scale per `N`-element run along the
    /// last axis, and `N` is what the loader wants. Returns `None`
    /// when neither is present.
    pub fn group_size_effective(&self) -> Option<i32> {
        if let Some(gs) = self.group_size {
            return Some(gs);
        }
        self.weight_block_size
            .as_ref()
            .and_then(|v| v.get(1).copied())
    }

    /// Bit width to feed `UnifiedLinear::from_weights`.
    ///
    /// Prefers the explicit `bits` field. Falls back to `quant_method`
    /// for HF exports: `mxfp8` → 8, `mxfp4` / `nvfp4` → 4. Returns
    /// `None` when neither is present.
    pub fn bits_effective(&self) -> Option<i32> {
        if let Some(b) = self.bits {
            return Some(b);
        }
        match self.quant_method.as_deref() {
            Some("mxfp8") => Some(8),
            Some("mxfp4") | Some("nvfp4") => Some(4),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rotary_dim: usize,
    pub partial_rotary_factor: f32,
    pub use_qk_norm: bool,
    pub tie_word_embeddings: bool,
    pub dense_intermediate_size: usize,
    pub shared_intermediate_size: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub n_shared_experts: usize,
    pub scoring_func: String,
    pub use_routing_bias: bool,
    #[serde(default, deserialize_with = "deserialize_bool_vec")]
    pub moe_layer_freq: Vec<bool>,
    pub qk_norm_type: String,
    pub swiglu_alpha: f32,
    pub swiglu_limit: f32,
    pub routed_scaling_factor: f32,
    pub sparse_attention_config: SparseAttentionConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SparseAttentionConfig {
    pub use_sparse_attention: bool,
    pub sparse_index_dim: usize,
    pub sparse_num_index_heads: usize,
    pub sparse_topk_blocks: usize,
    pub sparse_block_size: usize,
    #[serde(default = "default_score_type")]
    pub sparse_score_type: String,
    #[serde(default)]
    pub sparse_init_block: usize,
    #[serde(default = "default_local_block")]
    pub sparse_local_block: usize,
    #[serde(default, deserialize_with = "deserialize_bool_vec")]
    pub sparse_disable_index_value: Vec<bool>,
    #[serde(default, deserialize_with = "deserialize_bool_vec")]
    pub sparse_attention_freq: Vec<bool>,
}

fn deserialize_bool_vec<'de, D>(deserializer: D) -> Result<Vec<bool>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v: Vec<serde_json::Value> = Vec::deserialize(deserializer)?;
    v.iter()
        .map(|val| {
            val.as_bool()
                .or_else(|| val.as_i64().map(|i| i != 0))
                .ok_or_else(|| serde::de::Error::custom("expected bool or int"))
        })
        .collect()
}

fn default_score_type() -> String {
    "max".to_string()
}
fn default_local_block() -> usize {
    1
}

impl ModelArgs {
    pub fn group_size(&self) -> i32 {
        let gs = self
            .quantization
            .as_ref()
            .and_then(|q| q.group_size_effective())
            .unwrap_or(64);
        eprintln!(
            "[M3 ModelArgs] group_size={} (quantization={:?})",
            gs, self.quantization
        );
        gs
    }
    pub fn bits(&self) -> i32 {
        self.quantization
            .as_ref()
            .and_then(|q| q.bits_effective())
            .unwrap_or(4)
    }
    pub fn gate_bits(&self) -> i32 {
        8
    }
    pub fn use_msa_for_layer(&self, layer_idx: usize) -> bool {
        let cfg = &self.text_config.sparse_attention_config;
        if !cfg.use_sparse_attention {
            return false;
        }
        if layer_idx < cfg.sparse_attention_freq.len() {
            cfg.sparse_attention_freq[layer_idx]
        } else {
            false
        }
    }
}

// ============================================================================
// Sparse Attention (MSA)
// ============================================================================

pub struct SparseAttention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub q_norm: Option<GemmaRMSNorm>,
    pub k_norm: Option<GemmaRMSNorm>,

    pub index_q_proj: Option<UnifiedLinear>,
    pub index_k_proj: Option<UnifiedLinear>,
    pub index_q_norm: Option<GemmaRMSNorm>,
    pub index_k_norm: Option<GemmaRMSNorm>,

    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32,
    pub rope_base: f32,
    pub block_size: i32,
    pub top_k: i32,
    pub index_dim: i32,
    pub sparse_local_block: i32,

    // Set at construction by DecoderLayer::from_weights. Used by tracing
    // events so we can attribute attention dispatch + shapes to a layer.
    pub layer_idx: usize,
}

impl SparseAttention {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        trace!(
            layer = self.layer_idx,
            b = b,
            l = l,
            cache_offset = cache.offset,
            mask_present = mask.is_some(),
            "attn.forward entry"
        );

        let q_raw = self.q_proj.forward(x);
        let k_raw = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let q = mlxcel_core::reshape(&q_raw, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k_raw, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // Per-head Gemma-style Q/K norm (scales by 1+weight, not weight alone).
        // Order matches HF reference: reshape → norm → transpose → RoPE.
        let q = if let Some(ref n) = self.q_norm {
            n.forward(&q)
        } else {
            q
        };
        let k = if let Some(ref n) = self.k_norm {
            n.forward(&k)
        } else {
            k
        };

        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        // Partial RoPE: fast_rope rotates the first `rope_dims` of head_dim,
        // leaves the rest unchanged. M3 config: head_dim=128, rotary_dim=64.
        let offset = cache.offset;
        let q = mlxcel_core::fast_rope(&q, self.rope_dims, false, self.rope_base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.rope_dims, false, self.rope_base, 1.0, offset);

        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Asymmetric q/k for chunked-prefill / cached case. cache_offset is
        // the absolute position where the current chunk's queries START
        // (captured BEFORE update_and_fetch advanced cache.offset). kv_len
        // is the total cached length AFTER the current chunk is written.
        let cache_offset_at_chunk_start = offset;
        let kv_len = offset + l;

        // Ceil-div: the trailing partial block is a real block. Floor-div would silently
        // drop the tail tokens (l - floor(l/bs)*bs of them), so any prompt whose length
        // isn't a multiple of block_size would crash on the downstream reshape from
        // `[..., l, d]` to `[..., num_blocks, block_size, d]`. Matches HF reference
        // modeling_minimax_m3_vl.py:572 (`num_key_blocks = -(-k_len // block_size)`).
        let num_query_blocks = (l + self.block_size - 1) / self.block_size;
        let num_key_blocks = (kv_len + self.block_size - 1) / self.block_size;
        let padded_q_len = num_query_blocks * self.block_size;
        let padded_k_len = num_key_blocks * self.block_size;
        let pad_q_amt = padded_q_len - l;
        let pad_k_amt = padded_k_len - kv_len;

        let is_msa_eligible_layer = self.index_q_proj.is_some();

        // MSA-eligible layers MUST keep m3_idx_k in lockstep with main K
        // *regardless* of which attention dispatch this forward picks. The
        // indexer K cache is a per-token artifact (field comment on
        // `KVCache::m3_idx_k` in cache.rs documents this contract): each MSA
        // forward selects top-k blocks over the FULL cached prefix, so the
        // cache must contain idx_k for every position in main K — including
        // positions written by chunks that dispatched dense (e.g. early
        // chunks where `num_key_blocks <= top_k` saturates the selector).
        //
        // Pre-cycle-79.5 the update only happened in the MSA branch, which
        // silently broke lockstep on any session whose first chunk dispatched
        // dense and later chunks reached MSA — including the common pattern
        // of `prefill_chunk_size=512` where chunk 1's kv_len is small enough
        // to dense-saturate.
        //
        // Defense-in-depth: a PRE-update lockstep mismatch (`m3_idx_offset
        // != offset`) means the prior prefix's idx_k is missing. After the
        // cycle-79 proper fix (clone_handle/install_detached round-trip),
        // adoption restores both halves together so this should never fire
        // in healthy flows. If it does fire, we cannot retroactively
        // reconstruct the missing prefix's idx_k without re-projecting from
        // tokens we no longer have, so the only safe move is to dense-
        // fallback for THIS forward AND skip the update so subsequent
        // forwards in this session continue to dense-fallback (any partial
        // update would produce a gappy idx_k that scores wrong blocks).
        let pre_update_indexer_offset = cache.m3_idx_offset();
        let lockstep_healthy = pre_update_indexer_offset == offset;
        let cached_idx_k = if is_msa_eligible_layer && lockstep_healthy {
            // Always compute idx_k projection and update the cache. Cost is
            // one Linear(hidden, num_kv_heads*index_dim) + norm + transpose
            // + RoPE per MSA-eligible layer per forward — negligible vs the
            // main attention math.
            //
            // Order: RESHAPE → NORM → TRANSPOSE → RoPE matches the HF
            // reference (MiniMaxM3VLIndexer). The post-RoPE values are what
            // gets cached because RoPE is position-dependent and each
            // position was rotated at its absolute position when written.
            let idx_k_raw = self.index_k_proj.as_ref().unwrap().forward(x);
            let idx_k = mlxcel_core::reshape(&idx_k_raw, &[b, l, 1, self.index_dim]);
            let idx_k = if let Some(ref n) = self.index_k_norm {
                n.forward(&idx_k)
            } else {
                idx_k
            };
            let idx_k = mlxcel_core::transpose_axes(&idx_k, &[0, 2, 1, 3]);
            let idx_k =
                mlxcel_core::fast_rope(&idx_k, self.rope_dims, false, self.rope_base, 1.0, offset);
            Some(cache.m3_idx_k_update_and_fetch(&idx_k))
        } else {
            if is_msa_eligible_layer && !lockstep_healthy {
                warn!(
                    layer = self.layer_idx,
                    b = b,
                    l = l,
                    main_offset = offset,
                    indexer_offset = pre_update_indexer_offset,
                    reason = "idx_k_cache_desync_pre_update",
                    "m3_idx_k lockstep broken before update — likely adoption \
                     regression (cycle-79 proper fix should prevent this). \
                     Falling back to dense for the rest of this session; \
                     reset the session to recover sparse dispatch."
                );
            }
            None
        };

        // Dispatch decision — shape-based, with the broken-lockstep fall-
        // through baked into `cached_idx_k.is_none()`:
        //   * non-MSA-eligible layer: dense (dense layers 0-2 in M3)
        //   * MSA layer with `l <= block_size`: dense (single-block decode)
        //   * MSA layer with `num_key_blocks <= top_k`: dense (saturation —
        //     sparse-selecting all blocks reduces to dense)
        //   * broken lockstep (cached_idx_k is None): dense
        //   * else: MSA
        if !is_msa_eligible_layer
            || l <= self.block_size
            || num_key_blocks <= self.top_k
            || cached_idx_k.is_none()
        {
            let reason = if !is_msa_eligible_layer {
                "no_index_proj"
            } else if l <= self.block_size {
                "l_le_block_size"
            } else if num_key_blocks <= self.top_k {
                "num_key_blocks_le_top_k"
            } else {
                "idx_k_cache_desync"
            };
            debug!(
                layer = self.layer_idx,
                b = b,
                l = l,
                kv_len = kv_len,
                block_size = self.block_size,
                num_key_blocks = num_key_blocks,
                top_k = self.top_k,
                has_index_proj = is_msa_eligible_layer,
                main_offset = offset,
                indexer_offset = cache.m3_idx_offset(),
                branch = "dense",
                reason = reason,
                "attn.dispatch"
            );
            return self.dense_attention(&q, &cache_k, &cache_v, mask);
        }

        debug!(
            layer = self.layer_idx,
            b = b,
            l = l,
            kv_len = kv_len,
            cache_offset = cache_offset_at_chunk_start,
            num_query_blocks = num_query_blocks,
            num_key_blocks = num_key_blocks,
            top_k = self.top_k,
            block_size = self.block_size,
            branch = "msa",
            "attn.dispatch"
        );

        // MSA path. cached_idx_k is Some by construction (passed the dispatch
        // check above which guards `cached_idx_k.is_none()`). The cached
        // tensor has shape `[b, 1, kv_len, index_dim]` — the full prefix
        // including the current chunk's freshly-RoPE'd idx_k that was
        // appended during the unconditional pre-dispatch update above.
        let idx_k = cached_idx_k.unwrap();

        // Compute idx_q for the current chunk. idx_q is per-forward (no
        // caching) because the selector only operates on the current query
        // positions.
        let idx_q_raw = self.index_q_proj.as_ref().unwrap().forward(x);
        let idx_q = mlxcel_core::reshape(&idx_q_raw, &[b, l, self.num_kv_heads, self.index_dim]);
        let idx_q = if let Some(ref n) = self.index_q_norm {
            n.forward(&idx_q)
        } else {
            idx_q
        };
        let idx_q = mlxcel_core::transpose_axes(&idx_q, &[0, 2, 1, 3]);
        let idx_q =
            mlxcel_core::fast_rope(&idx_q, self.rope_dims, false, self.rope_base, 1.0, offset);

        // Pad idx_q (to padded_q_len) and idx_k (to padded_k_len) with -inf
        // so the upcoming block reshape and max-pool can never let padded
        // positions win. Real positions in the partial last block dominate
        // the per-block max (any finite value beats -inf), so block scores
        // stay real-valued. -inf appears only at padded positions that are
        // summarized away by max_axis; it never reaches a matmul.
        let idx_dtype = mlxcel_core::array_dtype(&idx_q);
        let idx_q = if pad_q_amt > 0 {
            let pad_q = mlxcel_core::full_f32(
                &[b, self.num_kv_heads, pad_q_amt, self.index_dim],
                f32::NEG_INFINITY,
                idx_dtype,
            );
            mlxcel_core::concatenate(&idx_q, &pad_q, 2)
        } else {
            idx_q
        };
        let idx_k = if pad_k_amt > 0 {
            let pad_k = mlxcel_core::full_f32(
                &[b, 1, pad_k_amt, self.index_dim],
                f32::NEG_INFINITY,
                idx_dtype,
            );
            mlxcel_core::concatenate(&idx_k, &pad_k, 2)
        } else {
            idx_k
        };

        // Block max-pool scoring.
        // K side uses num_key_blocks (full cached). Q side uses
        // num_query_blocks (current chunk only).
        let k_blocked = mlxcel_core::reshape(
            &idx_k,
            &[b, 1, num_key_blocks, self.block_size, self.index_dim],
        );
        let k_pool = mlxcel_core::max_axis(&k_blocked, 3, false);

        let q_blocked = mlxcel_core::reshape(
            &idx_q,
            &[
                b,
                self.num_kv_heads,
                num_query_blocks,
                self.block_size,
                self.index_dim,
            ],
        );
        let q_pool = mlxcel_core::max_axis(&q_blocked, 3, false);

        let scale_idx = 1.0 / (self.index_dim as f32).sqrt();
        let k_pool_t = mlxcel_core::transpose_axes(&k_pool, &[0, 1, 3, 2]);
        let block_scores = mlxcel_core::matmul(&q_pool, &k_pool_t);
        let block_scores = mlxcel_core::multiply_scalar(&block_scores, scale_idx);
        // block_scores shape: [b, num_kv_heads, num_query_blocks, num_key_blocks]

        let block_scores = self.apply_causal_block_mask_asymmetric(
            &block_scores,
            num_query_blocks,
            num_key_blocks,
            cache_offset_at_chunk_start,
        );

        // Local block always included (set to inf) - verified from Transformers code
        let block_scores = self.ensure_local_block_score_asymmetric(
            &block_scores,
            num_query_blocks,
            num_key_blocks,
            cache_offset_at_chunk_start,
            l,
        );

        // Top-K selection: produces ABSOLUTE key block indices in
        // [0, num_key_blocks). selected shape: [b, num_kv_heads,
        // num_query_blocks, top_k].
        let neg_scores = mlxcel_core::negative(&block_scores);
        let k_minus_1 = self.top_k - 1;
        let partitioned = mlxcel_core::argpartition(&neg_scores, k_minus_1, -1);
        let selected = mlxcel_core::slice(
            &partitioned,
            &[0, 0, 0, 0],
            &[b, self.num_kv_heads, num_query_blocks, self.top_k],
        );

        self.sparse_sdpa(
            &q,
            &cache_k,
            &cache_v,
            &selected,
            b,
            l,
            kv_len,
            cache_offset_at_chunk_start,
        )
    }

    #[allow(dead_code)] // Symmetric form retained for regression; #81 wires asymmetric.
    fn apply_causal_block_mask(&self, scores: &MlxArray, num_blocks: i32) -> UniquePtr<MlxArray> {
        let key_pos = mlxcel_core::arange_f32(0.0, num_blocks as f32, 1.0);
        let key_pos = mlxcel_core::reshape(&key_pos, &[1, 1, 1, num_blocks]);
        let query_pos = mlxcel_core::arange_f32(0.0, num_blocks as f32, 1.0);
        let query_pos = mlxcel_core::reshape(&query_pos, &[1, 1, num_blocks, 1]);
        let causal = mlxcel_core::greater_equal(&query_pos, &key_pos);
        let neg_inf =
            mlxcel_core::full_f32(&[1], f32::NEG_INFINITY, mlxcel_core::array_dtype(scores));
        let s = mlxcel_core::array_shape(scores);
        let neg_inf = mlxcel_core::broadcast_to(&neg_inf, &[s[0], s[1], s[2], s[3]]);
        mlxcel_core::where_cond(&causal, scores, &neg_inf)
    }

    /// Force the local block (and any additional `sparse_local_block` blocks
    /// preceding it) into the Top-K selection by setting their scores to
    /// +infinity before Top-K.
    ///
    /// Per the MSA paper (arXiv:2606.13392, eq 7 commentary): "We always
    /// include the local block containing position i" — and the explicit
    /// rationale paragraph: "This fixed allocation reserves one block slot
    /// and leaves the remaining slots to be chosen by the Index Branch,
    /// preventing degenerate selections that omit the query's immediate
    /// neighbourhood." Without it, sparse attention can drop the local
    /// context block entirely → degenerate / repetitive output.
    ///
    /// Reference Transformers code (modeling_minimax_m3_vl.py:587-591):
    /// ```python
    /// if self.local_blocks > 0:
    ///     local = torch.arange(self.local_blocks)
    ///     local_idx = (q_block[..., None] - local.view(1, 1, -1)).clamp(min=0)
    ///     block_scores.scatter_(-1, local_idx, float("inf"))
    /// ```
    ///
    /// In mlxcel's coordinate system block_scores has shape
    /// `[B, H_idx, num_q_blocks, num_k_blocks]` (Q is max-pooled per block
    /// before scoring, so we work at the block level rather than per token).
    /// The guarantee is therefore applied per Q block: for each q in
    /// `[0, num_blocks)`, the set
    /// `{q, q-1, ..., max(0, q - sparse_local_block + 1)}` of key blocks
    /// is forced to +inf so Top-K picks them.
    ///
    /// Expressed as a broadcast where-mask (no scatter needed):
    ///   diff      = query_pos - key_pos   shape [1, 1, num_blocks, num_blocks]
    ///   is_local  = (diff >= 0) AND (diff < sparse_local_block)
    ///   scores    = where(is_local, +inf, scores)
    #[allow(dead_code)] // Symmetric form retained for regression; #81 wires asymmetric.
    fn ensure_local_block_score(&self, scores: &MlxArray, num_blocks: i32) -> UniquePtr<MlxArray> {
        if self.sparse_local_block <= 0 {
            return mlxcel_core::copy(scores);
        }

        // query_pos[q, k] = q (broadcast along key axis)
        let query_pos = mlxcel_core::arange_f32(0.0, num_blocks as f32, 1.0);
        let query_pos = mlxcel_core::reshape(&query_pos, &[1, 1, num_blocks, 1]);

        // key_pos[q, k] = k (broadcast along query axis)
        let key_pos = mlxcel_core::arange_f32(0.0, num_blocks as f32, 1.0);
        let key_pos = mlxcel_core::reshape(&key_pos, &[1, 1, 1, num_blocks]);

        // is_local[q, k] = (q >= k) AND (k > q - sparse_local_block)
        //                 i.e. k in [max(0, q - sparse_local_block + 1), q]
        let causal = mlxcel_core::greater_equal(&query_pos, &key_pos);
        let window = mlxcel_core::full_f32(
            &[1],
            self.sparse_local_block as f32,
            mlxcel_core::dtype::FLOAT32,
        );
        let key_plus_window = mlxcel_core::add(&key_pos, &window);
        let within_window = mlxcel_core::less(&query_pos, &key_plus_window);
        let is_local = mlxcel_core::logical_and(&causal, &within_window);

        let dtype = mlxcel_core::array_dtype(scores);
        let pos_inf = mlxcel_core::full_f32(&[1], f32::INFINITY, dtype);
        let s = mlxcel_core::array_shape(scores);
        let pos_inf = mlxcel_core::broadcast_to(&pos_inf, &[s[0], s[1], s[2], s[3]]);

        mlxcel_core::where_cond(&is_local, &pos_inf, scores)
    }

    /// Asymmetric variant of `apply_causal_block_mask` for chunked-prefill /
    /// cached case. Query blocks are chunk-relative ([0, num_query_blocks));
    /// key blocks are absolute ([0, num_key_blocks)).
    ///
    /// For each query block i, the maximum attendable absolute key block is
    /// the block containing the query block's LAST absolute position:
    ///
    ///   max_attendable_kblock[i] = (cache_offset + (i+1)*block_size - 1) / block_size
    ///                              (integer division, floor)
    ///
    /// This is over-permissive when a query block straddles a key block
    /// boundary (cache_offset is not a multiple of block_size): we include
    /// the boundary key block even though some of its positions may be
    /// future from the query's first position's perspective. That's
    /// correct: the asymmetric mask in `sparse_sdpa` later masks per-
    /// position with the `p_k > p_q ⇒ invalid` rule, which prunes any
    /// within-block future positions. Over-permission at top-k selection
    /// time means a top-k slot may go to a partially-attendable block; the
    /// per-position mask catches the residual.
    fn apply_causal_block_mask_asymmetric(
        &self,
        scores: &MlxArray,
        num_query_blocks: i32,
        num_key_blocks: i32,
        cache_offset: i32,
    ) -> UniquePtr<MlxArray> {
        // CPU-side precompute the per-query-block max attendable absolute
        // key block. Cheap (num_query_blocks ≤ ~64 in practice).
        let max_kblocks: Vec<f32> = (0..num_query_blocks)
            .map(|i| {
                let last_abs = cache_offset + (i + 1) * self.block_size - 1;
                (last_abs / self.block_size) as f32
            })
            .collect();
        let max_kblocks_arr =
            mlxcel_core::from_slice_f32(&max_kblocks, &[1, 1, num_query_blocks, 1]);
        let key_pos = mlxcel_core::arange_f32(0.0, num_key_blocks as f32, 1.0);
        let key_pos = mlxcel_core::reshape(&key_pos, &[1, 1, 1, num_key_blocks]);
        let causal = mlxcel_core::less_equal(&key_pos, &max_kblocks_arr);
        let neg_inf =
            mlxcel_core::full_f32(&[1], f32::NEG_INFINITY, mlxcel_core::array_dtype(scores));
        let s = mlxcel_core::array_shape(scores);
        let neg_inf = mlxcel_core::broadcast_to(&neg_inf, &[s[0], s[1], s[2], s[3]]);
        mlxcel_core::where_cond(&causal, scores, &neg_inf)
    }

    /// Asymmetric variant of `ensure_local_block_score`. For each query
    /// block i, force the absolute key blocks
    /// `[max_attendable_kblock[i] - sparse_local_block + 1, max_attendable_kblock[i]]`
    /// (clipped to ≥ 0) to +inf so top-k always includes the local
    /// neighbourhood.
    ///
    /// In the cache_offset=0 case this reduces to the symmetric form (the
    /// local block IS the query's block IS the diagonal of the score
    /// matrix). With cache_offset > 0, the local block sits in the cached
    /// prefix's last absolute block(s), which is the correct
    /// neighbourhood for the current chunk's queries.
    ///
    /// `q_len` anchors the guarantee at the chunk's REAL last query
    /// position. A partial final query block (q_len not a multiple of
    /// block_size, reachable whenever a dense prompt-cache adoption resumes
    /// prefill at an arbitrary token offset — e.g. adopted_len 160 with
    /// block_size 128) would otherwise compute its anchor from the padded
    /// block end: with `sparse_local_block = 1` (the M3 default) the +inf
    /// then lands entirely on a key block one past the newest real block,
    /// forcing NOTHING — exactly the degenerate local-context omission the
    /// paper's fixed allocation exists to prevent, on the positions that
    /// produce the first token after adoption.
    fn ensure_local_block_score_asymmetric(
        &self,
        scores: &MlxArray,
        num_query_blocks: i32,
        num_key_blocks: i32,
        cache_offset: i32,
        q_len: i32,
    ) -> UniquePtr<MlxArray> {
        if self.sparse_local_block <= 0 {
            return mlxcel_core::copy(scores);
        }

        // For each query block i: the absolute key block containing the
        // query block's last REAL position. Full query blocks reduce to the
        // asymmetric causal-mask formula; the final partial block clamps to
        // the chunk's true last position instead of its padded block end.
        let last_real_abs = cache_offset + q_len - 1;
        let max_kblocks: Vec<f32> = (0..num_query_blocks)
            .map(|i| {
                let last_abs = (cache_offset + (i + 1) * self.block_size - 1).min(last_real_abs);
                (last_abs / self.block_size) as f32
            })
            .collect();
        let max_kblocks_arr =
            mlxcel_core::from_slice_f32(&max_kblocks, &[1, 1, num_query_blocks, 1]);

        // key_pos[k] = k (absolute)
        let key_pos = mlxcel_core::arange_f32(0.0, num_key_blocks as f32, 1.0);
        let key_pos = mlxcel_core::reshape(&key_pos, &[1, 1, 1, num_key_blocks]);

        // is_local[q, k] = (k ≤ max_kblock[q]) AND (k > max_kblock[q] - sparse_local_block)
        let causal = mlxcel_core::less_equal(&key_pos, &max_kblocks_arr);
        let window = mlxcel_core::full_f32(
            &[1],
            self.sparse_local_block as f32,
            mlxcel_core::dtype::FLOAT32,
        );
        let lower_bound = mlxcel_core::subtract(&max_kblocks_arr, &window);
        let within_window = mlxcel_core::greater(&key_pos, &lower_bound);
        let is_local = mlxcel_core::logical_and(&causal, &within_window);

        let dtype = mlxcel_core::array_dtype(scores);
        let pos_inf = mlxcel_core::full_f32(&[1], f32::INFINITY, dtype);
        let s = mlxcel_core::array_shape(scores);
        let pos_inf = mlxcel_core::broadcast_to(&pos_inf, &[s[0], s[1], s[2], s[3]]);

        mlxcel_core::where_cond(&is_local, &pos_inf, scores)
    }

    fn sparse_sdpa(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        v: &MlxArray,
        selected: &MlxArray,
        b: i32,
        q_len: i32,
        kv_len: i32,
        cache_offset: i32,
    ) -> UniquePtr<MlxArray> {
        // Asymmetric q/k blocking for chunked-prefill / cached case.
        // q_len: the current chunk's query length.
        // kv_len: the full cached K/V length (cache_offset + q_len).
        // cache_offset: where the current chunk starts in absolute coords.
        //
        // num_query_blocks = ceil(q_len / block_size) — used to block Q.
        // num_key_blocks   = ceil(kv_len / block_size) — used to block K/V and
        //                    bound `selected`'s value range.
        //
        // Divisibility padding. The block reshapes below require q's seq axis
        // to equal num_query_blocks * block_size and k/v's seq axis to equal
        // num_key_blocks * block_size. Pad value is zero — neutral in matmul —
        // and the additive mask below sets padded-K scores to -inf before
        // softmax so they contribute nothing.
        let num_query_blocks = (q_len + self.block_size - 1) / self.block_size;
        let num_key_blocks = (kv_len + self.block_size - 1) / self.block_size;
        let padded_q_len = num_query_blocks * self.block_size;
        let padded_k_len = num_key_blocks * self.block_size;
        let pad_q_amt = padded_q_len - q_len;
        let pad_k_amt = padded_k_len - kv_len;
        debug!(
            layer = self.layer_idx,
            b = b,
            q_len = q_len,
            kv_len = kv_len,
            cache_offset = cache_offset,
            num_query_blocks = num_query_blocks,
            num_key_blocks = num_key_blocks,
            padded_q_len = padded_q_len,
            padded_k_len = padded_k_len,
            top_k = self.top_k,
            selected_shape = ?mlxcel_core::array_shape(selected),
            "sparse_sdpa.entry"
        );

        // Materialize padded q if pad_q > 0; otherwise borrow input ref as-is.
        // Same for k, v (with their own potentially different pad amount).
        // Storage variables live for the full function scope so the &MlxArray
        // borrows below remain valid.
        let q_padded_storage;
        let q: &MlxArray = if pad_q_amt > 0 {
            let q_dtype = mlxcel_core::array_dtype(q);
            let pad_q =
                mlxcel_core::full_f32(&[b, self.num_heads, pad_q_amt, self.head_dim], 0.0, q_dtype);
            q_padded_storage = mlxcel_core::concatenate(q, &pad_q, 2);
            q_padded_storage.as_ref().unwrap()
        } else {
            q
        };
        let k_padded_storage;
        let v_padded_storage;
        let (k, v): (&MlxArray, &MlxArray) = if pad_k_amt > 0 {
            let kv_dtype = mlxcel_core::array_dtype(k);
            let pad_k = mlxcel_core::full_f32(
                &[b, self.num_kv_heads, pad_k_amt, self.head_dim],
                0.0,
                kv_dtype,
            );
            let pad_v = mlxcel_core::full_f32(
                &[b, self.num_kv_heads, pad_k_amt, self.head_dim],
                0.0,
                kv_dtype,
            );
            k_padded_storage = mlxcel_core::concatenate(k, &pad_k, 2);
            v_padded_storage = mlxcel_core::concatenate(v, &pad_v, 2);
            (
                k_padded_storage.as_ref().unwrap(),
                v_padded_storage.as_ref().unwrap(),
            )
        } else {
            (k, v)
        };

        let k_blocked = mlxcel_core::reshape(
            k,
            &[
                b,
                self.num_kv_heads,
                num_key_blocks,
                self.block_size,
                self.head_dim,
            ],
        );
        let v_blocked = mlxcel_core::reshape(
            v,
            &[
                b,
                self.num_kv_heads,
                num_key_blocks,
                self.block_size,
                self.head_dim,
            ],
        );

        // Per-(kv_head, q_block) gather: each q_block has its own top_k key
        // block indices into the same global set of num_key_blocks key blocks.
        // take_along_axis requires indices to match the source rank with
        // broadcast along non-gather axes. We add a q_block axis to k/v_blocked
        // (broadcast view, not materialized — MLX evaluates lazily) and reshape
        // `selected` to broadcast over block_size and head_dim. take_along_axis
        // on the new k_blocks axis (axis=3) then produces the per-(kv_head,
        // q_block) gather we actually want.
        //
        // Shape walk (asymmetric):
        //   k_blocked:                       [b, kv_h, num_key_blocks, block_size, head_dim]
        //   → reshape (unsqueeze q_block):   [b, kv_h, 1, num_key_blocks, block_size, head_dim]
        //   → broadcast to q_block axis:     [b, kv_h, num_query_blocks, num_key_blocks, block_size, head_dim]
        //   selected:                        [b, kv_h, num_query_blocks, top_k]
        //   → reshape (unsqueeze 2 trailing):[b, kv_h, num_query_blocks, top_k, 1, 1]
        //   → broadcast over inner dims:     [b, kv_h, num_query_blocks, top_k, block_size, head_dim]
        //   take_along_axis on axis 3:       [b, kv_h, num_query_blocks, top_k, block_size, head_dim]
        //   reshape to combine top_k+block:  [b, kv_h, num_query_blocks, top_k*block_size, head_dim]
        //
        // Symmetric history (latent bug fixed cycle 65): the prior `take(...,
        // axis=2)` used global-gather semantics (single indices array applied
        // uniformly across all kv_heads and q_blocks). It produced a
        // num_kv_heads-fold element-count mismatch on the downstream reshape
        // and would have crashed any prompt long enough to make MSA fire
        // (l > 2048 in a single forward call). The per-(kv_head, q_block)
        // gather above replaced it. THIS cycle (#79): the variables previously
        // named `num_blocks` are split into `num_query_blocks` and
        // `num_key_blocks` so cached prefill no longer mismatches K's actual
        // sequence dim (kv_len > q_len when cache_offset > 0).
        let bk_view = [
            b,
            self.num_kv_heads,
            1,
            num_key_blocks,
            self.block_size,
            self.head_dim,
        ];
        let bk_target = [
            b,
            self.num_kv_heads,
            num_query_blocks,
            num_key_blocks,
            self.block_size,
            self.head_dim,
        ];
        let k_expanded =
            mlxcel_core::broadcast_to(&mlxcel_core::reshape(&k_blocked, &bk_view), &bk_target);
        let v_expanded =
            mlxcel_core::broadcast_to(&mlxcel_core::reshape(&v_blocked, &bk_view), &bk_target);

        let sel_view = [b, self.num_kv_heads, num_query_blocks, self.top_k, 1, 1];
        let sel_target = [
            b,
            self.num_kv_heads,
            num_query_blocks,
            self.top_k,
            self.block_size,
            self.head_dim,
        ];
        let sel_broadcast =
            mlxcel_core::broadcast_to(&mlxcel_core::reshape(selected, &sel_view), &sel_target);

        debug!(
            layer = self.layer_idx,
            k_expanded_shape = ?mlxcel_core::array_shape(&k_expanded),
            sel_broadcast_shape = ?mlxcel_core::array_shape(&sel_broadcast),
            "sparse_sdpa.pre_gather (take_along_axis on axis=3)"
        );

        let k_gathered = mlxcel_core::take_along_axis(&k_expanded, &sel_broadcast, 3);
        let v_gathered = mlxcel_core::take_along_axis(&v_expanded, &sel_broadcast, 3);

        debug!(
            layer = self.layer_idx,
            k_gathered_shape = ?mlxcel_core::array_shape(&k_gathered),
            "sparse_sdpa.post_gather"
        );

        let kv_per_q_block = self.top_k * self.block_size;
        let k_flat = mlxcel_core::reshape(
            &k_gathered,
            &[
                b,
                self.num_kv_heads,
                num_query_blocks,
                kv_per_q_block,
                self.head_dim,
            ],
        );
        let v_flat = mlxcel_core::reshape(
            &v_gathered,
            &[
                b,
                self.num_kv_heads,
                num_query_blocks,
                kv_per_q_block,
                self.head_dim,
            ],
        );

        let q_blocked = mlxcel_core::reshape(
            q,
            &[
                b,
                self.num_heads,
                num_query_blocks,
                self.block_size,
                self.head_dim,
            ],
        );

        let n_rep = self.num_heads / self.num_kv_heads;
        // 5D GQA expansion. mlxcel_core::utils::repeat_kv assumes a 4D shape
        // [batch, n_kv_heads, seq_len, head_dim] and reads shape[3] as
        // head_dim. Our tensors here are 5D [b, num_kv_heads, num_query_blocks,
        // kv_per_q_block, head_dim]; calling repeat_kv on them mis-derives
        // head_dim = kv_per_q_block and crashes the downstream reshape. Inline
        // the broadcast pattern for 5D so each kv_head's [num_query_blocks,
        // kv_per_q_block, head_dim] block is repeated n_rep times.
        let kv_5d_with_rep = |x: &MlxArray| -> UniquePtr<MlxArray> {
            let x_view = mlxcel_core::reshape(
                x,
                &[
                    b,
                    self.num_kv_heads,
                    1,
                    num_query_blocks,
                    kv_per_q_block,
                    self.head_dim,
                ],
            );
            let x_broad = mlxcel_core::broadcast_to(
                &x_view,
                &[
                    b,
                    self.num_kv_heads,
                    n_rep,
                    num_query_blocks,
                    kv_per_q_block,
                    self.head_dim,
                ],
            );
            mlxcel_core::reshape(
                &x_broad,
                &[
                    b,
                    self.num_heads,
                    num_query_blocks,
                    kv_per_q_block,
                    self.head_dim,
                ],
            )
        };
        let k_expanded = kv_5d_with_rep(&k_flat);
        let v_expanded = kv_5d_with_rep(&v_flat);

        let scores = mlxcel_core::matmul(
            &q_blocked,
            &mlxcel_core::transpose_axes(&k_expanded, &[0, 1, 2, 4, 3]),
        );
        let scores = mlxcel_core::multiply_scalar(&scores, self.scale);

        // Asymmetric unified causal + padding + sentinel mask.
        // See build_msa_unified_mask_asymmetric below for the rationale. p_q
        // uses absolute coords (cache_offset + i); p_k stays in [0, num_key_
        // blocks * block_size); the `p_k > p_q ⇒ invalid` rule subsumes
        // within-block causality, future-block leakage, divisibility padding,
        // AND cached-prefix causality.
        let scores_dtype = mlxcel_core::array_dtype(&scores);
        let additive = build_msa_unified_mask_asymmetric(
            selected,
            b,
            self.num_kv_heads,
            self.num_heads,
            n_rep,
            num_query_blocks,
            num_key_blocks,
            self.top_k,
            self.block_size,
            cache_offset,
            scores_dtype,
        );
        let scores = mlxcel_core::add(&scores, &additive);

        let weights = mlxcel_core::softmax(&scores, -1);
        let out = mlxcel_core::matmul(&weights, &v_expanded);

        // Reshape to padded q-length first, then slice back to the real q_len.
        let out = mlxcel_core::reshape(&out, &[b, self.num_heads, padded_q_len, self.head_dim]);
        let out = if pad_q_amt > 0 {
            mlxcel_core::slice(
                &out,
                &[0, 0, 0, 0],
                &[b, self.num_heads, q_len, self.head_dim],
            )
        } else {
            out
        };
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[b, q_len, self.num_heads * self.head_dim]);
        self.o_proj.forward(&out)
    }

    fn dense_attention(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        v: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // q shape after transpose: [batch, num_heads, q_len, head_dim].
        // When mask is None and q_len > 1 (prefill of a multi-token prompt),
        // the underlying `fast_scaled_dot_product_attention` runs full
        // bidirectional attention — every query attends to every key, including
        // future positions. The residual stream collapses across positions and
        // every position predicts the same token. Use MLX's "causal" SDPA path
        // in this case so the kernel applies the causal mask itself.
        // (Llama3/Qwen3 sidestep this by returning `true` from
        // `supports_maskless_padded_prefill`; M3 is conservative and accepts a
        // mask from the prefill helper, but during decode (q_len == 1) and
        // certain code paths the mask is None and causal masking is implicit.)
        let q_len = mlxcel_core::array_shape(q)[2];
        // Build an explicit additive causal mask [q_len, k_len] of 0/-inf when
        // mask is None and q_len > 1 (prefill of a multi-token prompt).
        let raw = if mask.is_none() && q_len > 1 {
            let k_len = mlxcel_core::array_shape(k)[2];
            let dtype = mlxcel_core::array_dtype(q);
            // ones[q_len, k_len], tril keeps the lower triangle (causal: q can see k<=q).
            //
            // CHUNKED PREFILL: q is the current chunk only; k is the FULL cache (prior chunks +
            // current). The causal boundary for absolute position `cache_offset + i` (i in [0, q_len))
            // is at k-position `cache_offset + i`, so the tril offset must be `cache_offset`, derivable
            // from shapes as `k_len - q_len`. Using `0` here was correct only for the first chunk
            // (cache_offset=0) and silently truncated every subsequent chunk's attention to the first
            // `q_len` cached tokens — producing the "model can't attend to late prompt content"
            // hallucination signature observed at any prompt length above the chunk size.
            let causal_offset = k_len - q_len;
            let ones = mlxcel_core::full_f32(&[q_len, k_len], 1.0, dtype);
            let lower = mlxcel_core::tril(&ones, causal_offset);
            // additive = (lower == 1) ? 0 : -inf
            let zero = mlxcel_core::full_f32(&[1], 0.0, dtype);
            let neg_inf = mlxcel_core::full_f32(&[1], f32::NEG_INFINITY, dtype);
            let one = mlxcel_core::full_f32(&[1], 1.0, dtype);
            let is_one = mlxcel_core::equal(&lower, &one);
            let additive = mlxcel_core::where_cond(&is_one, &zero, &neg_inf);
            let mask_ref: &MlxArray = additive.as_ref().unwrap();
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    q,
                    k,
                    v,
                    self.scale,
                    mask_ref as *const MlxArray,
                    0.0,
                    0,
                )
            }
        } else {
            let mask_ptr = mask
                .map(|m| m as *const MlxArray)
                .unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(q, k, v, self.scale, mask_ptr, 0.0, 0)
            }
        };
        let shape = mlxcel_core::array_shape(&raw);
        let b = shape[0];
        let l = shape[2];
        let out = mlxcel_core::transpose_axes(&raw, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[b, l, self.num_heads * self.head_dim]);
        self.o_proj.forward(&out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
        has_index: bool,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let g = args.group_size();
        let b = args.bits();
        let cfg = &args.text_config;

        let (q_proj, k_proj, v_proj, o_proj) = (
            load_linear(weights, &format!("{}.q_proj", prefix), g, b)?,
            load_linear(weights, &format!("{}.k_proj", prefix), g, b)?,
            load_linear(weights, &format!("{}.v_proj", prefix), g, b)?,
            load_linear(weights, &format!("{}.o_proj", prefix), g, b)?,
        );

        // Per-head Q/K norm on head_dim
        let (q_norm, k_norm) = if cfg.use_qk_norm {
            let qw = get_weight(weights, &format!("{}.q_norm.weight", prefix))?;
            let kw = get_weight(weights, &format!("{}.k_norm.weight", prefix))?;
            (
                Some(GemmaRMSNorm::new(qw, cfg.rms_norm_eps)),
                Some(GemmaRMSNorm::new(kw, cfg.rms_norm_eps)),
            )
        } else {
            (None, None)
        };

        let (index_q_proj, index_k_proj, index_q_norm, index_k_norm) = if has_index {
            let iqp = load_linear(weights, &format!("{}.index_q_proj", prefix), g, b)?;
            let ikp = load_linear(weights, &format!("{}.index_k_proj", prefix), g, b)?;
            let iqn = get_weight(weights, &format!("{}.index_q_norm.weight", prefix))
                .ok()
                .map(|w| GemmaRMSNorm::new(w, cfg.rms_norm_eps));
            let ikn = get_weight(weights, &format!("{}.index_k_norm.weight", prefix))
                .ok()
                .map(|w| GemmaRMSNorm::new(w, cfg.rms_norm_eps));
            (Some(iqp), Some(ikp), iqn, ikn)
        } else {
            (None, None, None, None)
        };

        let sparse_cfg = &cfg.sparse_attention_config;
        let head_dim = cfg.head_dim as i32;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            index_q_proj,
            index_k_proj,
            index_q_norm,
            index_k_norm,
            num_heads: cfg.num_attention_heads as i32,
            num_kv_heads: cfg.num_key_value_heads as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims: cfg.rotary_dim as i32,
            rope_base: cfg.rope_theta,
            block_size: sparse_cfg.sparse_block_size as i32,
            top_k: sparse_cfg.sparse_topk_blocks as i32,
            index_dim: sparse_cfg.sparse_index_dim as i32,
            sparse_local_block: sparse_cfg.sparse_local_block as i32,
            layer_idx,
        })
    }
}

// ============================================================================
// MSA unified causal+padding+sentinel mask
// ============================================================================

/// Build the unified causal+padding+sentinel mask for MSA's sparse_sdpa.
///
/// For each (q absolute position p_q, gathered K absolute position p_k), the
/// score is valid iff `p_k <= p_q`. Returns an additive mask of shape
/// `[b, num_heads, num_blocks, block_size, top_k * block_size]` containing
/// `-inf` at invalid positions and `0` at valid positions, ready to be added
/// to the score tensor before softmax.
///
/// One mask subsumes three previously separate concerns:
///
///   1. **Within-block causality.** Q at q_block 7 pos 5 cannot attend to K at
///      q_block 7 pos 10 (HF reference encodes this through build_block_mask).
///   2. **Future-block leakage via the top_k -1 sentinel issue.** Early q_blocks
///      have fewer than top_k causally-valid k_blocks; argpartition picks
///      future blocks whose index-branch scores are -inf, but the gather
///      happily reads them. Their absolute K positions are > q's, so they
///      mask out here.
///   3. **Divisibility padding.** Padded K positions in the last block have
///      absolute positions >= l, and every real Q has position < l, so they
///      mask out here.
///
/// Applied unconditionally — causality concerns are independent of
/// divisibility, and the cost is one extra elementwise compare per element.
///
/// Reference: `modeling_minimax_m3_vl.py:632-634` in
/// `MiniMaxM3VLIndexer.build_block_mask` computes the same
/// `k_positions > position_ids` causal mask over the full K span.
///
/// Extracted as a free function (rather than inlined into sparse_sdpa) so the
/// unit tests in `mod tests` can call the exact production code with
/// hand-computed fixtures.
///
/// # Note on bit-exact reproducibility across refactors
///
/// This function was extracted from an inline block in `sparse_sdpa` at commit
/// 75fbe5e (the prior inline implementation was in commit 7da5bfa). The
/// extracted-function form is **algebraically identical** to the inline form —
/// same `mlxcel_core` ops, same arguments, same order — and the unit tests
/// verify the resulting mask is bit-identical against a hand-computed
/// reference. But the post-refactor binary produces **different greedy-decode
/// tokens at temp=0** on the same prompt as the pre-refactor binary.
///
/// Cause: MLX is lazy. Intermediate `UniquePtr<MlxArray>` values drop at this
/// function's return boundary, forcing materialization in a different schedule
/// than the inline version (which kept all intermediates alive until end of
/// `forward()`). Different materialization scheduling → different GPU kernel
/// issue order → different fp accumulation in the downstream attention math.
///
/// **This is a property of mlxcel's lazy-eval surface**, not a correctness
/// concern. The differential against the dense path (verified-correct
/// baseline) showed 49/50 token match for the post-refactor MSA path on a
/// 2339-token prompt at greedy temp=0; the only divergence was at a literal
/// tie. Correctness is established by the unit tests and the dense
/// differential — *not* by bit-exact reproducibility against any prior commit.
///
/// If you refactor this again (or any MSA-path code that lives in a chain of
/// lazy `UniquePtr<MlxArray>` operations), expect inference-output token
/// deltas without correctness regressions. Verify correctness via the unit
/// tests and a dense-differential, not via diffing tokens against the previous
/// commit's output.
///
/// # Dead-code retention (cycle 79, 2026-06-26)
///
/// As of cycle 79's asymmetric refactor, `build_msa_unified_mask_asymmetric`
/// subsumes this function (cache_offset=0 + num_query_blocks==num_key_blocks
/// gives identical output). The symmetric form and its 7 unit tests are
/// retained for regression value — they pin a smaller, hand-computable
/// surface that anyone modifying the asymmetric form can sanity-check
/// against. Slated for deletion after the proper-fix sequence (#78-#83)
/// merges and the asymmetric form has lived in main for a cycle without
/// regressions.
#[allow(dead_code)]
fn build_msa_unified_mask(
    selected: &MlxArray,
    b: i32,
    num_kv_heads: i32,
    num_heads: i32,
    n_rep: i32,
    num_blocks: i32,
    top_k: i32,
    block_size: i32,
    padded_l: i32,
    scores_dtype: i32,
) -> UniquePtr<MlxArray> {
    let kv_len = top_k * block_size;

    let p_q = mlxcel_core::arange_i32(0, padded_l, 1);
    let p_q = mlxcel_core::reshape(&p_q, &[num_blocks, block_size]);

    let selected_5 = mlxcel_core::reshape(selected, &[b, num_kv_heads, num_blocks, top_k, 1]);
    let block_size_scalar = mlxcel_core::from_slice_i32(&[block_size], &[1]);
    let selected_x_bs = mlxcel_core::multiply(&selected_5, &block_size_scalar);
    let k_pos_axis = mlxcel_core::arange_i32(0, block_size, 1);
    let k_pos_axis_5 = mlxcel_core::reshape(&k_pos_axis, &[1, 1, 1, 1, block_size]);
    let p_k = mlxcel_core::add(&selected_x_bs, &k_pos_axis_5);
    let p_k = mlxcel_core::reshape(&p_k, &[b, num_kv_heads, num_blocks, kv_len]);

    let mask_shape = [b, num_kv_heads, num_blocks, block_size, kv_len];
    let p_q_5 = mlxcel_core::reshape(&p_q, &[1, 1, num_blocks, block_size, 1]);
    let p_k_5 = mlxcel_core::reshape(&p_k, &[b, num_kv_heads, num_blocks, 1, kv_len]);
    let p_q_full = mlxcel_core::broadcast_to(&p_q_5, &mask_shape);
    let p_k_full = mlxcel_core::broadcast_to(&p_k_5, &mask_shape);
    let invalid_kv = mlxcel_core::greater(&p_k_full, &p_q_full);

    let invalid_with_rep_dim = mlxcel_core::reshape(
        &invalid_kv,
        &[b, num_kv_heads, 1, num_blocks, block_size, kv_len],
    );
    let invalid_repeated = mlxcel_core::broadcast_to(
        &invalid_with_rep_dim,
        &[b, num_kv_heads, n_rep, num_blocks, block_size, kv_len],
    );
    let invalid_per_head = mlxcel_core::reshape(
        &invalid_repeated,
        &[b, num_heads, num_blocks, block_size, kv_len],
    );

    let neg_inf = mlxcel_core::full_f32(&[1], f32::NEG_INFINITY, scores_dtype);
    let zero = mlxcel_core::full_f32(&[1], 0.0, scores_dtype);
    mlxcel_core::where_cond(&invalid_per_head, &neg_inf, &zero)
}

/// Asymmetric variant of `build_msa_unified_mask` — supports chunked prefill
/// and prompt-cache, where the current chunk's query length differs from the
/// total key length (cache_offset > 0).
///
/// # Coordinate system
///
/// All positions are ABSOLUTE in the full sequence (not relative to the
/// current chunk):
///
///   p_q[q_block, q_pos] = cache_offset + q_block * block_size + q_pos
///   p_k[kv_head, q_block, top_k_idx, k_pos]
///     = selected[kv_head, q_block, top_k_idx] * block_size + k_pos
///
/// `selected[..] ∈ [0, num_key_blocks)` indexes into the FULL cached K's
/// block partition; `k_pos ∈ [0, block_size)` is the position within the
/// selected block. Both produce absolute positions in `[0, num_key_blocks *
/// block_size)`.
///
/// The validity rule `p_k > p_q ⇒ invalid` from the symmetric version still
/// holds — both sides are in the same absolute coordinate system. This rule
/// subsumes within-block causality, future-block leakage, divisibility
/// padding, AND cached-prefix causality (a current-chunk query can attend to
/// any cached key whose absolute position is ≤ the query's absolute
/// position, regardless of which chunk that key came from).
///
/// # Why this isn't a special case of the symmetric mask
///
/// The symmetric `build_msa_unified_mask` builds `p_q = arange(0, padded_l)`
/// where padded_l = num_blocks * block_size. That implicitly assumes q_len
/// == k_len and cache_offset == 0. In a chunked-prefill call with
/// cache_offset > 0:
///   - p_q must start at cache_offset, not 0, so the new chunk's queries
///     have the right absolute coords for the causal comparison against
///     cached keys
///   - num_query_blocks and num_key_blocks differ; the mask shape gains a
///     num_query_blocks axis (outer) while the gather size stays
///     top_k * block_size (per query block)
///   - `selected` has shape [b, num_kv_heads, num_query_blocks, top_k] —
///     one selection per query block, indexing into num_key_blocks
///
/// # Parameters
///
/// `num_key_blocks` doesn't appear in the mask math directly — it's bounded
/// by `selected`'s value range — but it's accepted as a parameter for
/// trace logging and as a contract assertion at the call site.
fn build_msa_unified_mask_asymmetric(
    selected: &MlxArray,
    b: i32,
    num_kv_heads: i32,
    num_heads: i32,
    n_rep: i32,
    num_query_blocks: i32,
    _num_key_blocks: i32,
    top_k: i32,
    block_size: i32,
    cache_offset: i32,
    scores_dtype: i32,
) -> UniquePtr<MlxArray> {
    let padded_q_len = num_query_blocks * block_size;
    let kv_per_q_block = top_k * block_size;

    // Query positions: absolute coords starting at cache_offset.
    let p_q = mlxcel_core::arange_i32(cache_offset, cache_offset + padded_q_len, 1);
    let p_q = mlxcel_core::reshape(&p_q, &[num_query_blocks, block_size]);

    // Key positions: selected_block_idx * block_size + pos_within_block,
    // computed from the per-query-block selected indices. `selected` indexes
    // into num_key_blocks (the full cached K's block partition), so the
    // resulting p_k values are absolute positions in [0, num_key_blocks *
    // block_size), spanning both cached prefix and current chunk.
    let selected_5 = mlxcel_core::reshape(selected, &[b, num_kv_heads, num_query_blocks, top_k, 1]);
    let block_size_scalar = mlxcel_core::from_slice_i32(&[block_size], &[1]);
    let selected_x_bs = mlxcel_core::multiply(&selected_5, &block_size_scalar);
    let k_pos_axis = mlxcel_core::arange_i32(0, block_size, 1);
    let k_pos_axis_5 = mlxcel_core::reshape(&k_pos_axis, &[1, 1, 1, 1, block_size]);
    let p_k = mlxcel_core::add(&selected_x_bs, &k_pos_axis_5);
    let p_k = mlxcel_core::reshape(&p_k, &[b, num_kv_heads, num_query_blocks, kv_per_q_block]);

    let mask_shape = [
        b,
        num_kv_heads,
        num_query_blocks,
        block_size,
        kv_per_q_block,
    ];
    let p_q_5 = mlxcel_core::reshape(&p_q, &[1, 1, num_query_blocks, block_size, 1]);
    let p_k_5 = mlxcel_core::reshape(
        &p_k,
        &[b, num_kv_heads, num_query_blocks, 1, kv_per_q_block],
    );
    let p_q_full = mlxcel_core::broadcast_to(&p_q_5, &mask_shape);
    let p_k_full = mlxcel_core::broadcast_to(&p_k_5, &mask_shape);
    let invalid_kv = mlxcel_core::greater(&p_k_full, &p_q_full);

    // GQA expansion: replicate each kv_head's mask n_rep times into num_heads.
    let invalid_with_rep_dim = mlxcel_core::reshape(
        &invalid_kv,
        &[
            b,
            num_kv_heads,
            1,
            num_query_blocks,
            block_size,
            kv_per_q_block,
        ],
    );
    let invalid_repeated = mlxcel_core::broadcast_to(
        &invalid_with_rep_dim,
        &[
            b,
            num_kv_heads,
            n_rep,
            num_query_blocks,
            block_size,
            kv_per_q_block,
        ],
    );
    let invalid_per_head = mlxcel_core::reshape(
        &invalid_repeated,
        &[b, num_heads, num_query_blocks, block_size, kv_per_q_block],
    );

    let neg_inf = mlxcel_core::full_f32(&[1], f32::NEG_INFINITY, scores_dtype);
    let zero = mlxcel_core::full_f32(&[1], 0.0, scores_dtype);
    mlxcel_core::where_cond(&invalid_per_head, &neg_inf, &zero)
}

// ============================================================================
// Custom SwiGLU activation - verified from Transformers code
// gate = clamp(gate, max=limit)
// up = clamp(up, -limit, limit)
// glu = gate * sigmoid(gate * alpha)
// return down_proj((up + 1.0) * glu)
// ============================================================================

fn swiglu_forward(
    x: &MlxArray,
    gate_proj: &UnifiedLinear,
    up_proj: &UnifiedLinear,
    down_proj: &UnifiedLinear,
    alpha: f32,
    limit: f32,
) -> UniquePtr<MlxArray> {
    let gate = gate_proj.forward(x);
    let up = up_proj.forward(x);

    let gate_clamped = mlxcel_core::minimum(
        &gate,
        &mlxcel_core::full_f32(&[1], limit, mlxcel_core::array_dtype(&gate)),
    );
    let up_clamped = mlxcel_core::minimum(
        &mlxcel_core::maximum(
            &up,
            &mlxcel_core::full_f32(&[1], -limit, mlxcel_core::array_dtype(&up)),
        ),
        &mlxcel_core::full_f32(&[1], limit, mlxcel_core::array_dtype(&up)),
    );

    // glu = gate * sigmoid(gate * alpha)
    let gate_alpha = mlxcel_core::multiply_scalar(&gate_clamped, alpha);
    let sig = mlxcel_core::sigmoid(&gate_alpha);
    let glu = mlxcel_core::multiply(&gate_clamped, &sig);

    // (up + 1.0) * glu
    let up_plus_1 = mlxcel_core::add(
        &up_clamped,
        &mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::array_dtype(&up_clamped)),
    );
    let hidden = mlxcel_core::multiply(&up_plus_1, &glu);
    down_proj.forward(&hidden)
}

// ============================================================================
// M3 SwiGLU activation (pure, no projections) — for routed-expert path
//
// Same formula as `swiglu_forward` above, but takes already-projected
// `up` and `gate` arrays so it can sit between SwitchLinear gate/up and
// SwitchLinear down in the MoE path.
//
//   gate_c = clamp(gate, max=limit)
//   up_c   = clamp(up,   ±limit)
//   glu    = gate_c * sigmoid(alpha * gate_c)
//   out    = (up_c + 1.0) * glu
//
// Fast path: when (alpha, limit) == (1.702, 7.0) — M3's configured values
// and identical to GPT-OSS — dispatch to the existing compiled primitive
// `compiled_gpt_oss_swiglu_activation`, which fuses the whole graph.
// ============================================================================

fn m3_swiglu_activation(
    up: &MlxArray,
    gate: &MlxArray,
    alpha: f32,
    limit: f32,
) -> UniquePtr<MlxArray> {
    if (alpha - 1.702).abs() <= f32::EPSILON && (limit - 7.0).abs() <= f32::EPSILON {
        return mlxcel_core::compiled_gpt_oss_swiglu_activation(up, gate);
    }

    let dtype = mlxcel_core::array_dtype(up);
    let neg_lim = mlxcel_core::full_f32(&[1], -limit, dtype);
    let pos_lim = mlxcel_core::full_f32(&[1], limit, dtype);
    let gate_c = mlxcel_core::minimum(gate, &pos_lim);
    let up_c = mlxcel_core::minimum(&mlxcel_core::maximum(up, &neg_lim), &pos_lim);
    let glu_scaled = mlxcel_core::multiply_scalar(&gate_c, alpha);
    let sig = mlxcel_core::sigmoid(&glu_scaled);
    let out_glu = mlxcel_core::multiply(&gate_c, &sig);
    let one = mlxcel_core::full_f32(&[1], 1.0, dtype);
    let up_plus_1 = mlxcel_core::add(&up_c, &one);
    let result = mlxcel_core::multiply(&out_glu, &up_plus_1);
    mlxcel_core::astype(&result, dtype)
}

// ============================================================================
// M3 routed-expert forward — parallels SwitchGLU::forward but inserts
// `m3_swiglu_activation` between the gate/up and down projections instead
// of `compiled_swiglu_activation` (standard SwiGLU). Same sort/gather/scatter
// shape contract so the surrounding MoE code path is unchanged.
// ============================================================================

fn m3_switchglu_forward(
    x: &MlxArray,
    indices: &MlxArray,
    gate_proj: &SwitchLinear,
    up_proj: &SwitchLinear,
    down_proj: &SwitchLinear,
    alpha: f32,
    limit: f32,
) -> UniquePtr<MlxArray> {
    let indices_shape = mlxcel_core::array_shape(indices);
    let n_tokens = indices_shape[0];
    let top_k = indices_shape[1];
    let total = n_tokens * top_k;
    let do_sort = total >= 64;

    let x_exp = mlxcel_core::expand_dims(x, -2);
    let x_exp = mlxcel_core::expand_dims(&x_exp, -3);

    if do_sort {
        let (sorted_x, sorted_idx, inv_order) = gather_sort(&x_exp, indices);
        let x_gate = gate_proj.forward(&sorted_x, &sorted_idx, true);
        let x_up = up_proj.forward(&sorted_x, &sorted_idx, true);
        let activated = m3_swiglu_activation(&x_up, &x_gate, alpha, limit);
        let output = down_proj.forward(&activated, &sorted_idx, true);

        // Inline scatter_unsort: unsort, reshape to [n_tokens, top_k, ...], drop middle axis
        let unsorted = mlxcel_core::take(&output, &inv_order, 0);
        let x_shape = mlxcel_core::array_shape(&unsorted);
        let reshaped = mlxcel_core::reshape(&unsorted, &[n_tokens, top_k, x_shape[1], x_shape[2]]);
        mlxcel_core::squeeze_axis(&reshaped, 2)
    } else {
        let x_gate = gate_proj.forward(&x_exp, indices, false);
        let x_up = up_proj.forward(&x_exp, indices, false);
        let activated = m3_swiglu_activation(&x_up, &x_gate, alpha, limit);
        let output = down_proj.forward(&activated, indices, false);
        mlxcel_core::squeeze_axis(&output, -2)
    }
}

// ============================================================================
// Shared Experts (SwiGLU, not standard SiLU)
// ============================================================================

pub struct SharedExperts {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl SharedExperts {
    pub fn forward(&self, x: &MlxArray, alpha: f32, limit: f32) -> UniquePtr<MlxArray> {
        swiglu_forward(
            x,
            &self.gate_proj,
            &self.up_proj,
            &self.down_proj,
            alpha,
            limit,
        )
    }

    pub fn from_weights(weights: &WeightMap, prefix: &str, g: i32, b: i32) -> Result<Self, String> {
        Ok(Self {
            gate_proj: load_linear(weights, &format!("{}.gate_proj", prefix), g, b)?,
            up_proj: load_linear(weights, &format!("{}.up_proj", prefix), g, b)?,
            down_proj: load_linear(weights, &format!("{}.down_proj", prefix), g, b)?,
        })
    }
}

// ============================================================================
// MoE Block
// ============================================================================

pub struct SparseMoeBlock {
    pub router: UnifiedLinear,
    // Three separate SwitchLinears so the activation step can be M3's
    // clamped (up+1.0) variant instead of standard silu(gate)*up. The
    // surrounding sort/gather/scatter shape contract is preserved via
    // `m3_switchglu_forward`, which mirrors SwitchGLU::forward.
    pub gate_proj: SwitchLinear,
    pub up_proj: SwitchLinear,
    pub down_proj: SwitchLinear,
    pub shared_experts: Option<SharedExperts>,
    pub e_score_correction_bias: UniquePtr<MlxArray>,
    pub num_experts_per_tok: usize,
    pub routed_scaling_factor: f32,
    pub swiglu_alpha: f32,
    pub swiglu_limit: f32,
}

impl SparseMoeBlock {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden_dim = orig_shape[orig_shape.len() - 1];
        let n: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
        let x_flat = mlxcel_core::reshape(x, &[n, hidden_dim]);

        let x_f32 = mlxcel_core::astype(&x_flat, mlxcel_core::dtype::FLOAT32);
        let logits = self.router.forward(&x_f32);
        let scores = mlxcel_core::sigmoid(&logits);
        let orig_scores = mlxcel_core::copy(&scores);
        let biased = mlxcel_core::add(&scores, &self.e_score_correction_bias);

        let k = self.num_experts_per_tok as i32;
        let neg_biased = mlxcel_core::negative(&biased);
        let part = mlxcel_core::argpartition(&neg_biased, k - 1, -1);
        let s = mlxcel_core::array_shape(&part);
        let topk_idx = mlxcel_core::slice(&part, &[0, 0], &[s[0], k]);
        let topk_scores = mlxcel_core::take_along_axis(&orig_scores, &topk_idx, -1);
        let score_sum = mlxcel_core::sum_axis(&topk_scores, -1, true);
        let eps = mlxcel_core::full_f32(&[1], 1e-20, mlxcel_core::array_dtype(&topk_scores));
        let norm_scores = mlxcel_core::divide(&topk_scores, &mlxcel_core::add(&score_sum, &eps));
        let norm_scores = mlxcel_core::astype(&norm_scores, mlxcel_core::array_dtype(&x_flat));

        let expert_out = m3_switchglu_forward(
            &x_flat,
            &topk_idx,
            &self.gate_proj,
            &self.up_proj,
            &self.down_proj,
            self.swiglu_alpha,
            self.swiglu_limit,
        );

        // Apply routed_scaling_factor - verified from Transformers code
        let expert_out = mlxcel_core::multiply_scalar(&expert_out, self.routed_scaling_factor);

        let mut result = crate::models::switch_layers::moe_weighted_sum(
            &expert_out,
            &norm_scores,
            mlxcel_core::array_dtype(&x_flat),
        );

        // Add shared expert output
        if let Some(ref shared) = self.shared_experts {
            let shared_out = shared.forward(&x_flat, self.swiglu_alpha, self.swiglu_limit);
            result = mlxcel_core::add(&result, &shared_out);
        }

        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&result, &orig_shape)
        } else {
            result
        }
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let g = args.group_size();
        let b = args.bits();
        let cfg = &args.text_config;

        let router =
            UnifiedLinear::from_weights(weights, &format!("{}.gate", prefix), g, args.gate_bits())?;

        // M3 routes through three separate SwitchLinears (gate=w1, up=w3, down=w2)
        // rather than the single SwitchGLU, so the activation step can be M3's
        // clamped (up+1.0) variant. Same w1/w3/w2 convention as before.
        let switch_mlp_prefix = format!("{}.switch_mlp", prefix);
        let gate_proj =
            SwitchLinear::from_weights(weights, &format!("{}.w1", switch_mlp_prefix), g, b)?;
        let up_proj =
            SwitchLinear::from_weights(weights, &format!("{}.w3", switch_mlp_prefix), g, b)?;
        let down_proj =
            SwitchLinear::from_weights(weights, &format!("{}.w2", switch_mlp_prefix), g, b)?;

        let shared = if cfg.n_shared_experts > 0 {
            let shared_prefix = format!("{}.shared_experts", prefix);
            Some(SharedExperts::from_weights(weights, &shared_prefix, g, b)?)
        } else {
            None
        };

        let bias_key = format!("{}.e_score_correction_bias", prefix);
        let e_score_correction_bias = weights
            .get(&bias_key)
            .map(|w| mlxcel_core::copy(w))
            .unwrap_or_else(|| {
                mlxcel_core::full_f32(
                    &[cfg.num_local_experts as i32],
                    0.0,
                    mlxcel_core::dtype::FLOAT32,
                )
            });

        Ok(Self {
            router,
            gate_proj,
            up_proj,
            down_proj,
            shared_experts: shared,
            e_score_correction_bias,
            num_experts_per_tok: cfg.num_experts_per_tok,
            routed_scaling_factor: cfg.routed_scaling_factor,
            swiglu_alpha: cfg.swiglu_alpha,
            swiglu_limit: cfg.swiglu_limit,
        })
    }
}

// ============================================================================
// Dense MLP (SwiGLU, not standard SiLU)
// ============================================================================

pub struct DenseMLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
    pub swiglu_alpha: f32,
    pub swiglu_limit: f32,
}

impl DenseMLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        swiglu_forward(
            x,
            &self.gate_proj,
            &self.up_proj,
            &self.down_proj,
            self.swiglu_alpha,
            self.swiglu_limit,
        )
    }

    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        g: i32,
        b: i32,
        alpha: f32,
        limit: f32,
    ) -> Result<Self, String> {
        Ok(Self {
            gate_proj: load_linear(weights, &format!("{}.gate_proj", prefix), g, b)?,
            up_proj: load_linear(weights, &format!("{}.up_proj", prefix), g, b)?,
            down_proj: load_linear(weights, &format!("{}.down_proj", prefix), g, b)?,
            swiglu_alpha: alpha,
            swiglu_limit: limit,
        })
    }
}

// ============================================================================
// Decoder Layer
// ============================================================================

pub struct DecoderLayer {
    pub self_attn: SparseAttention,
    pub mlp: Option<DenseMLP>,
    pub moe: Option<SparseMoeBlock>,
    pub input_layernorm: GemmaRMSNorm,
    pub post_attention_layernorm: GemmaRMSNorm,
    pub layer_idx: usize,
}

impl DecoderLayer {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        trace!(
            layer = self.layer_idx,
            is_moe = self.moe.is_some(),
            "decoder_layer.forward entry"
        );
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        let normed = self.post_attention_layernorm.forward(&h);
        let ff_out = if let Some(ref moe) = self.moe {
            moe.forward(&normed)
        } else if let Some(ref mlp) = self.mlp {
            mlp.forward(&normed)
        } else {
            unreachable!()
        };
        mlxcel_core::add(&h, &ff_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("language_model.model.layers.{}", layer_idx);
        let use_msa = args.use_msa_for_layer(layer_idx);

        let self_attn = SparseAttention::from_weights(
            weights,
            args,
            &format!("{}.self_attn", prefix),
            use_msa,
            layer_idx,
        )?;

        let is_moe = args.text_config.moe_layer_freq[layer_idx];
        let mlp_prefix = format!("{}.mlp", prefix);

        let (mlp, moe) = if is_moe {
            let moe_prefix = format!("{}.block_sparse_moe", prefix);
            (
                None,
                Some(SparseMoeBlock::from_weights(weights, args, &moe_prefix)?),
            )
        } else {
            let mlp = DenseMLP::from_weights(
                weights,
                &mlp_prefix,
                args.group_size(),
                args.bits(),
                args.text_config.swiglu_alpha,
                args.text_config.swiglu_limit,
            )?;
            (Some(mlp), None)
        };

        let input_norm = get_weight(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let post_norm = get_weight(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;
        let input_layernorm = GemmaRMSNorm::new(input_norm, args.text_config.rms_norm_eps);
        let post_attention_layernorm = GemmaRMSNorm::new(post_norm, args.text_config.rms_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            moe,
            input_layernorm,
            post_attention_layernorm,
            layer_idx,
        })
    }
}

// ============================================================================
// Model
// ============================================================================

pub struct MiniMaxM3Model {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: GemmaRMSNorm,
    pub lm_head: Option<UnifiedLinear>,
}

impl MiniMaxM3Model {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let in_shape = mlxcel_core::array_shape(input_ids);
        debug!(
            input_shape = ?in_shape,
            cache_offset = caches.first().map(|c| c.offset).unwrap_or(-1),
            mask_present = mask.is_some(),
            num_layers = self.layers.len(),
            "model.forward entry"
        );
        let mut h = self.embed_tokens.forward(input_ids);
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
            if (i + 1) % 5 == 0 {
                mlxcel_core::eval(&h);
            }
        }
        let h = self.norm.forward(&h);
        if let Some(ref head) = self.lm_head {
            head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        }
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();
        let config_str = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let full_config: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;
        let text_config = full_config
            .get("text_config")
            .ok_or("Missing text_config")?;
        // Accept both canonical MiniMax-M3 quantized export schemas:
        //   * mlx-community's `mlx_lm.convert` output uses top-level
        //     `quantization` with `group_size` / `bits`.
        //   * MiniMax's own MXFP8 upload and NVIDIA-style MXFP4/NVFP4
        //     exports use HF's top-level `quantization_config` with
        //     `quant_method` / `weight_block_size` / `ignored_layers`.
        // `M3Quantization` deserializes both; here we just have to feed
        // it whichever key is present. Preferring `quantization` first
        // keeps behavior unchanged for existing mlx-community loads.
        let quant_value = full_config
            .get("quantization")
            .or_else(|| full_config.get("quantization_config"));
        let args: ModelArgs = serde_json::from_value(serde_json::json!({
            "text_config": text_config,
            "quantization": quant_value,
        }))
        .map_err(|e| format!("Failed to parse ModelArgs: {}", e))?;
        let weights = crate::models::load_text_weights(model_dir, None)?;
        let model = Self::from_weights_with_prefix(&weights, &args, "language_model.model")?;
        Ok((model, args))
    }

    pub fn from_weights_with_prefix(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let g = args.group_size();
        let b = args.bits();
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, &format!("{}.embed_tokens", prefix), g, b)?;

        let mut layers = Vec::with_capacity(args.text_config.num_hidden_layers);
        for i in 0..args.text_config.num_hidden_layers {
            layers.push(DecoderLayer::from_weights(weights, args, i)?);
        }

        let norm_weight = get_weight(weights, &format!("{}.norm.weight", prefix))?;
        let norm = GemmaRMSNorm::new(norm_weight, args.text_config.rms_norm_eps);

        let lm_head = if !args.text_config.tie_word_embeddings {
            Some(UnifiedLinear::from_weights(
                weights,
                "language_model.lm_head",
                g,
                b,
            )?)
        } else {
            None
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        Self::from_weights_with_prefix(weights, args, "language_model.model")
    }
}

fn get_weight(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

impl LanguageModel for MiniMaxM3Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        MiniMaxM3Model::forward(self, input_ids, caches, mask)
    }
    fn make_caches(&self) -> Vec<KVCache> {
        MiniMaxM3Model::make_caches(self)
    }
    fn num_layers(&self) -> usize {
        self.layers.len()
    }
    fn eos_token_ids(&self) -> Vec<i32> {
        vec![200020]
    }
    /// M3's `dense_attention` applies causal masking implicitly when the
    /// caller passes `mask=None` (via MLX's "causal" SDPA mode). Declaring
    /// `true` here matches the Llama3/Qwen3 pattern and tells the generate
    /// dispatcher to skip materializing an L×L mask during prefill. The
    /// attention layer is the contract holder for causality; mlxcel's M3
    /// follows the SGLang/HF convention of "attention is always causal,
    /// the mask is an implementation detail."
    fn supports_maskless_padded_prefill(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_m3_config_parsing() {
        let config_str = r#"{
            "text_config": {
                "hidden_size": 6144,
                "intermediate_size": 3072,
                "num_hidden_layers": 60,
                "num_attention_heads": 64,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "vocab_size": 200064,
                "rms_norm_eps": 1e-06,
                "rope_theta": 5000000,
                "rotary_dim": 64,
                "partial_rotary_factor": 0.5,
                "use_qk_norm": true,
                "tie_word_embeddings": false,
                "dense_intermediate_size": 12288,
                "shared_intermediate_size": 3072,
                "num_local_experts": 128,
                "num_experts_per_tok": 4,
                "n_shared_experts": 1,
                "scoring_func": "sigmoid",
                "use_routing_bias": true,
                "moe_layer_freq": [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                "qk_norm_type": "per_head",
                "swiglu_alpha": 1.702,
                "swiglu_limit": 7.0,
                "routed_scaling_factor": 2.0,
                "sparse_attention_config": {
                    "use_sparse_attention": true,
                    "sparse_index_dim": 128,
                    "sparse_num_index_heads": 4,
                    "sparse_topk_blocks": 16,
                    "sparse_block_size": 128,
                    "sparse_disable_index_value": [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                    "sparse_attention_freq": [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
                }
            }
        }"#;

        let full_config: serde_json::Value = serde_json::from_str(config_str).unwrap();
        let args: ModelArgs = serde_json::from_value(full_config.clone()).unwrap();

        assert_eq!(args.text_config.hidden_size, 6144);
        assert_eq!(args.text_config.num_attention_heads, 64);
        assert_eq!(args.text_config.num_key_value_heads, 4);
        assert_eq!(args.text_config.head_dim, 128);
        assert_eq!(args.text_config.num_local_experts, 128);
        assert_eq!(args.text_config.num_experts_per_tok, 4);
        assert_eq!(args.text_config.n_shared_experts, 1);
        assert_eq!(args.text_config.swiglu_alpha, 1.702);
        assert_eq!(args.text_config.swiglu_limit, 7.0);
        assert_eq!(args.text_config.routed_scaling_factor, 2.0);
        assert_eq!(
            args.text_config.sparse_attention_config.sparse_block_size,
            128
        );
        assert_eq!(
            args.text_config.sparse_attention_config.sparse_topk_blocks,
            16
        );

        assert!(!args.use_msa_for_layer(0));
        assert!(!args.use_msa_for_layer(2));
        assert!(args.use_msa_for_layer(3));
        assert!(args.use_msa_for_layer(59));
    }

    /// mlx-community MiniMax-M3 exports (produced by `mlx_lm.convert`)
    /// carry the historical top-level `quantization` key with
    /// `group_size` / `bits`. Pre-existing behaviour: `group_size()` and
    /// `bits()` return those values verbatim; loader auto-detect maps
    /// `bits == 8` → mxfp8 quantized load.
    #[test]
    fn m3_quantization_mlx_community_schema_deserializes() {
        let q: M3Quantization = serde_json::from_str(r#"{ "group_size": 32, "bits": 8 }"#)
            .expect("mlx-community quantization schema must deserialize");

        assert_eq!(q.group_size_effective(), Some(32));
        assert_eq!(q.bits_effective(), Some(8));
        assert!(q.quant_method.is_none());
        assert!(q.weight_block_size.is_none());
        assert!(q.ignored_layers.is_none());
    }

    /// MiniMaxAI's own MXFP8 upload
    /// (`MiniMaxAI/MiniMax-M3-MXFP8`) uses the HF `quantization_config`
    /// schema: `quant_method`, `weight_block_size`, `activation_scheme`,
    /// `ignored_layers`. `group_size_effective` derives from
    /// `weight_block_size[1]` and `bits_effective` derives from
    /// `quant_method`, so the loader's mode auto-detect sees
    /// `bits == 8` / `group_size == 32` and correctly selects mxfp8.
    #[test]
    fn m3_quantization_hf_mxfp8_schema_deserializes() {
        let q: M3Quantization = serde_json::from_str(
            r#"{
                "quant_method": "mxfp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [1, 32],
                "ignored_layers": [
                    "lm_head",
                    "model.embed_tokens",
                    "vision_tower",
                    "language_model.model.layers.10.block_sparse_moe.gate"
                ]
            }"#,
        )
        .expect("HF MXFP8 quantization_config schema must deserialize");

        assert_eq!(q.quant_method.as_deref(), Some("mxfp8"));
        assert_eq!(q.weight_block_size.as_deref(), Some(&[1, 32][..]));
        assert_eq!(q.activation_scheme.as_deref(), Some("dynamic"));
        assert_eq!(q.group_size_effective(), Some(32));
        assert_eq!(q.bits_effective(), Some(8));
        assert!(q.group_size.is_none());
        assert!(q.bits.is_none());
        let ignored = q.ignored_layers.expect("ignored_layers must be present");
        assert!(ignored.iter().any(|s| s == "lm_head"));
        assert!(ignored.iter().any(|s| s == "model.embed_tokens"));
        assert_eq!(ignored.len(), 4);
    }

    /// `quant_method` may also carry `mxfp4` or `nvfp4`; both round to
    /// `bits == 4` so the loader's auto-detect selects the block-float
    /// mode from `group_size` (16 → nvfp4, else → mxfp4).
    #[test]
    fn m3_quantization_hf_mxfp4_and_nvfp4_derive_bits_4() {
        let q4: M3Quantization =
            serde_json::from_str(r#"{ "quant_method": "mxfp4", "weight_block_size": [1, 32] }"#)
                .unwrap();
        assert_eq!(q4.bits_effective(), Some(4));
        assert_eq!(q4.group_size_effective(), Some(32));

        let qnv: M3Quantization =
            serde_json::from_str(r#"{ "quant_method": "nvfp4", "weight_block_size": [1, 16] }"#)
                .unwrap();
        assert_eq!(qnv.bits_effective(), Some(4));
        assert_eq!(qnv.group_size_effective(), Some(16));
    }

    /// Explicit `bits` and `group_size` win over `quant_method`
    /// derivations. Guards against a future mixed export that carries
    /// both keys and against silent divergence between the two paths.
    #[test]
    fn m3_quantization_explicit_fields_override_quant_method() {
        let q: M3Quantization = serde_json::from_str(
            r#"{
                "quant_method": "mxfp4",
                "weight_block_size": [1, 32],
                "bits": 8,
                "group_size": 64
            }"#,
        )
        .unwrap();

        assert_eq!(q.bits_effective(), Some(8));
        assert_eq!(q.group_size_effective(), Some(64));
    }

    #[test]
    fn test_swiglu_oai_matches_python() {
        let alpha = 1.702f32;
        let limit = 7.0f32;
        let test_cases = vec![
            (0.0f32, 0.0f32),
            (1.0f32, 1.0f32),
            (-1.0f32, -1.0f32),
            (10.0f32, 10.0f32),
            (-10.0f32, -10.0f32),
        ];

        for (gate_val, up_val) in test_cases {
            let gate_clamped = gate_val.min(limit);
            let up_clamped = up_val.max(-limit).min(limit);
            let glu = gate_clamped * (1.0 / (1.0 + (-(gate_clamped * alpha)).exp()));
            let rust_result = (up_clamped + 1.0) * glu;

            let py_glu = gate_clamped * (1.0 / (1.0 + (-(gate_clamped * alpha)).exp()));
            let py_result = (up_clamped + 1.0) * py_glu;

            assert!(
                (rust_result - py_result).abs() < 1e-6,
                "SwiGLU mismatch gate={} up={}: rust={} py={}",
                gate_val,
                up_val,
                rust_result,
                py_result
            );
        }
    }

    /// Read one element from the mask at (head, q_block, q_pos, k_idx) for batch 0.
    /// Returns 0.0 (valid) or -inf (invalid).
    fn mask_at(mask: &MlxArray, head: i32, q_block: i32, q_pos: i32, k_idx: i32) -> f32 {
        let single = mlxcel_core::slice(
            mask,
            &[0, head, q_block, q_pos, k_idx],
            &[1, head + 1, q_block + 1, q_pos + 1, k_idx + 1],
        );
        mlxcel_core::eval(&single);
        mlxcel_core::item_f32(&single)
    }

    /// Unit fixture for build_msa_unified_mask. Small dimensions so every position
    /// is enumerable by hand, but large enough to exercise GQA replication
    /// (n_rep=2), future-block selection (kv_head 1 picks block 2 from q_block 0),
    /// and the divisibility-padding case (padded_l=6, l=5 → last block holds one
    /// real position and one padded position).
    ///
    /// Layout:
    ///   b=1, num_kv_heads=2, num_heads=4, n_rep=2,
    ///   num_blocks=3, top_k=2, block_size=2, kv_len=4, padded_l=6
    ///
    /// selected[1, 2, 3, 2]:
    ///   kv_head 0 picks blocks [0,1] for every q_block — pure past selections
    ///   kv_head 1 picks [0,2], [1,2], [0,1] — future-block selection at q_blocks 0, 1
    ///
    /// Reference computation (in the test body):
    ///   p_q[q_block, q_pos] = q_block * block_size + q_pos
    ///   p_k[kv_head, q_block, top_k_idx, block_pos]
    ///     = selected[kv_head, q_block, top_k_idx] * block_size + block_pos
    ///   invalid = (p_k > p_q)
    ///
    /// GQA: head 0,1 use kv_head 0; head 2,3 use kv_head 1.
    fn msa_mask_fixture() -> UniquePtr<MlxArray> {
        let selected_data: &[i32] = &[
            // kv_head 0: blocks [0,1] for each q_block (past selections)
            0, 1, 0, 1, 0, 1,
            // kv_head 1: blocks [0,2], [1,2], [0,1] (future-block at q_blocks 0,1)
            0, 2, 1, 2, 0, 1,
        ];
        let selected = mlxcel_core::from_slice_i32(selected_data, &[1, 2, 3, 2]);
        build_msa_unified_mask(
            &selected,
            /* b */ 1,
            /* num_kv_heads */ 2,
            /* num_heads */ 4,
            /* n_rep */ 2,
            /* num_blocks */ 3,
            /* top_k */ 2,
            /* block_size */ 2,
            /* padded_l */ 6,
            /* scores_dtype */ mlxcel_core::dtype::FLOAT32,
        )
    }

    #[test]
    fn test_msa_mask_self_attention_is_valid() {
        // head 0, q_block 0, q_pos 0 (p_q=0), k_idx 0 (k_block 0 pos 0, p_k=0).
        // p_k == p_q → valid. Smallest-stakes sanity check.
        let mask = msa_mask_fixture();
        assert_eq!(
            mask_at(&mask, 0, 0, 0, 0),
            0.0,
            "self-attention must be valid"
        );
    }

    #[test]
    fn test_msa_mask_within_block_causality() {
        // head 0, q_block 0, q_pos 0 (p_q=0), k_idx 1 (k_block 0 pos 1, p_k=1).
        // Same block, K position is future → must be invalid.
        // This was THE bug pre-unified-mask: the prior sparse_sdpa applied
        // only block-level causality and let within-block future K positions
        // contribute to softmax.
        let mask = msa_mask_fixture();
        let v = mask_at(&mask, 0, 0, 0, 1);
        assert!(
            v.is_infinite() && v < 0.0,
            "within-block causality must mask: got {} expected -inf",
            v
        );
    }

    #[test]
    fn test_msa_mask_future_block_leakage_masked() {
        // head 2 (kv_head 1), q_block 0 selects k_blocks [0, 2].
        // q_pos 0 (p_q=0), k_idx 2 = k_block 2 pos 0 (p_k=4).
        // Pre-unified mask, the gather happily read this -inf-scored future
        // block. Unified mask: p_k > p_q → invalid.
        let mask = msa_mask_fixture();
        let v = mask_at(&mask, 2, 0, 0, 2);
        assert!(
            v.is_infinite() && v < 0.0,
            "future-block selection must mask: got {} expected -inf",
            v
        );
    }

    #[test]
    fn test_msa_mask_padded_position_masked() {
        // head 2, q_block 0 selects k_block 2 which contains positions {4, 5}.
        // Position 5 is the padded position (l=5, padded_l=6).
        // q_pos 0 (p_q=0), k_idx 3 = k_block 2 pos 1 (p_k=5) → invalid.
        let mask = msa_mask_fixture();
        let v = mask_at(&mask, 2, 0, 0, 3);
        assert!(
            v.is_infinite() && v < 0.0,
            "padded K position must mask: got {} expected -inf",
            v
        );
    }

    #[test]
    fn test_msa_mask_causally_past_block_valid() {
        // head 0, q_block 2, q_pos 0 (p_q=4). Selected = [0, 1] both in the past.
        // k_idx 0 = k_block 0 pos 0 (p_k=0) → 0 <= 4 → valid.
        // k_idx 3 = k_block 1 pos 1 (p_k=3) → 3 <= 4 → valid.
        let mask = msa_mask_fixture();
        assert_eq!(mask_at(&mask, 0, 2, 0, 0), 0.0, "past block must be valid");
        assert_eq!(
            mask_at(&mask, 0, 2, 0, 3),
            0.0,
            "past block last pos must be valid"
        );
    }

    #[test]
    fn test_msa_mask_gqa_replication_within_kv_group() {
        // head 1 shares kv_head 0 with head 0; head 3 shares kv_head 1 with head 2.
        // The mask MUST be identical inside each GQA group at every (q, k) position.
        let mask = msa_mask_fixture();
        for q_block in 0..3 {
            for q_pos in 0..2 {
                for k_idx in 0..4 {
                    let h0 = mask_at(&mask, 0, q_block, q_pos, k_idx);
                    let h1 = mask_at(&mask, 1, q_block, q_pos, k_idx);
                    let h2 = mask_at(&mask, 2, q_block, q_pos, k_idx);
                    let h3 = mask_at(&mask, 3, q_block, q_pos, k_idx);
                    // Compare as bit patterns so -inf == -inf passes.
                    assert_eq!(
                        h0.to_bits(),
                        h1.to_bits(),
                        "kv_head 0 group (heads 0,1) mismatch at q_block={} q_pos={} k_idx={}",
                        q_block,
                        q_pos,
                        k_idx,
                    );
                    assert_eq!(
                        h2.to_bits(),
                        h3.to_bits(),
                        "kv_head 1 group (heads 2,3) mismatch at q_block={} q_pos={} k_idx={}",
                        q_block,
                        q_pos,
                        k_idx,
                    );
                }
            }
        }
    }

    #[test]
    fn test_msa_mask_full_exhaustive_table() {
        // Ground truth: every (head, q_block, q_pos, k_idx) compared to a hand
        // table. If anything in the mask construction regresses, this catches it.
        // Layout repeats the fixture math: see msa_mask_fixture doc comment.
        let mask = msa_mask_fixture();
        // selected[kv_head][q_block][top_k_idx]:
        let selected = [
            [[0i32, 1], [0, 1], [0, 1]], // kv_head 0
            [[0, 2], [1, 2], [0, 1]],    // kv_head 1
        ];
        let block_size = 2i32;
        for head in 0..4 {
            let kv_head = (head / 2) as usize;
            for q_block in 0..3 {
                for q_pos in 0..2 {
                    let p_q = q_block * block_size + q_pos;
                    for k_idx in 0..4 {
                        let top_k_idx = (k_idx / block_size) as usize;
                        let block_pos = k_idx % block_size;
                        let p_k =
                            selected[kv_head][q_block as usize][top_k_idx] * block_size + block_pos;
                        let expected_invalid = p_k > p_q;
                        let actual = mask_at(&mask, head, q_block, q_pos, k_idx);
                        if expected_invalid {
                            assert!(
                                actual.is_infinite() && actual < 0.0,
                                "expected -inf at head={head} qb={q_block} qp={q_pos} ki={k_idx} (p_q={p_q} p_k={p_k}); got {actual}",
                            );
                        } else {
                            assert_eq!(
                                actual, 0.0,
                                "expected 0.0 at head={head} qb={q_block} qp={q_pos} ki={k_idx} (p_q={p_q} p_k={p_k})",
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_causal_block_mask() {
        let num_blocks = 4;
        for q in 0..num_blocks {
            for k in 0..num_blocks {
                if k > q {
                    assert!(k > q, "Block {} should NOT see block {}", q, k);
                } else {
                    assert!(k <= q, "Block {} should see block {}", q, k);
                }
            }
        }
    }

    #[test]
    fn test_topk_selection() {
        let scores = [0.1f32, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6];
        let k = 3;
        let mut indexed: Vec<(usize, f32)> =
            scores.iter().enumerate().map(|(i, &s)| (i, -s)).collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let selected: Vec<usize> = indexed.iter().take(k).map(|(i, _)| *i).collect();
        assert_eq!(selected, vec![1, 3, 5]);
    }

    #[test]
    fn test_shape_consistency() {
        let _hidden_size = 6144;
        let num_heads = 64;
        let head_dim = 128;
        let block_size = 128;
        let top_k = 16;
        let max_context = 1048576;

        assert_eq!(num_heads * head_dim, 8192);
        assert_eq!(max_context / block_size, 8192);
        assert_eq!(top_k * block_size, 2048);
    }

    #[test]
    fn test_moe_expert_leaf_mapping() {
        let leaf_names = ["w1", "w3", "w2"];
        assert_eq!(leaf_names[0], "w1"); // gate_proj
        assert_eq!(leaf_names[1], "w3"); // up_proj
        assert_eq!(leaf_names[2], "w2"); // down_proj
    }

    // -------------------------------------------------------------------
    // m3_swiglu_activation unit tests
    //
    // The activation feeds the routed-expert path. Wrong math here is what
    // produced "asezasezasez..." — the standard SwitchGLU used silu(gate)*up
    // instead of M3's clamp + (up+1.0) trick. These tests exercise:
    //   1. Fast path matches the compiled primitive bit-for-bit
    //   2. Manual path matches the fast path at M3's config values
    //   3. Hand-computed scalar reference
    //   4. Gate upper-clamping (only upper, not symmetric)
    //   5. Up symmetric clamping
    //   6. gate=0 ⇒ out=0 (the glu-vanishing invariant)
    //   7. Shape preservation
    // -------------------------------------------------------------------

    fn vec_from_array(arr: &mlxcel_core::MlxArray) -> Vec<f32> {
        let shape = mlxcel_core::array_shape(arr);
        let n: i32 = shape.iter().product();
        (0..n)
            .map(|i| {
                let starts: Vec<i32> = shape
                    .iter()
                    .enumerate()
                    .map(|(d, _)| {
                        let stride: i32 = shape[d + 1..].iter().product();
                        (i / stride) % shape[d]
                    })
                    .collect();
                let stops: Vec<i32> = starts.iter().map(|s| s + 1).collect();
                let s = mlxcel_core::slice(arr, &starts, &stops);
                mlxcel_core::eval(&s);
                mlxcel_core::item_f32(&s)
            })
            .collect()
    }

    #[test]
    fn test_m3_swiglu_activation_fast_path_matches_compiled_primitive() {
        // Mixed positive/negative, in-range and over-limit values.
        let up = mlxcel_core::from_slice_f32(&[1.0, -2.0, 8.0, -8.0, 0.5], &[1, 5]);
        let gate = mlxcel_core::from_slice_f32(&[0.5, 1.0, 10.0, -3.0, 0.0], &[1, 5]);

        let mine = m3_swiglu_activation(&up, &gate, 1.702, 7.0);
        mlxcel_core::eval(&mine);
        let theirs = mlxcel_core::compiled_gpt_oss_swiglu_activation(&up, &gate);
        mlxcel_core::eval(&theirs);

        let diff = mlxcel_core::subtract(&mine, &theirs);
        let abs_diff = mlxcel_core::abs(&diff);
        let total = mlxcel_core::sum_all(&abs_diff);
        mlxcel_core::eval(&total);
        let total_val = mlxcel_core::item_f32(&total);
        assert!(
            total_val < 1e-6,
            "fast path differs from compiled primitive: total abs diff = {}",
            total_val
        );
    }

    #[test]
    fn test_m3_swiglu_activation_manual_path_matches_fast_at_config_values() {
        // Force manual path with alpha and limit nudged just past EPSILON,
        // then compare to manual computation at those exact values. This
        // verifies the manual path math is correct (independent of the
        // primitive). We can't compare to the fast path directly because
        // the constants differ.
        let up = mlxcel_core::from_slice_f32(&[1.0, -2.0, 8.0, -8.0], &[1, 4]);
        let gate = mlxcel_core::from_slice_f32(&[0.5, 1.0, 10.0, -3.0], &[1, 4]);

        let alpha = 1.5_f32;
        let limit = 5.0_f32;
        let mine = m3_swiglu_activation(&up, &gate, alpha, limit);
        mlxcel_core::eval(&mine);

        let expected: Vec<f32> = vec![(1.0_f32, 0.5_f32), (-2.0, 1.0), (8.0, 10.0), (-8.0, -3.0)]
            .into_iter()
            .map(|(u, g)| {
                let gc = g.min(limit);
                let uc = u.clamp(-limit, limit);
                let glu = gc * (1.0 / (1.0 + (-(gc * alpha)).exp()));
                (uc + 1.0) * glu
            })
            .collect();

        let actual = vec_from_array(&mine);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-5,
                "elt {} mismatch: actual={} expected={}",
                i,
                a,
                e
            );
        }
    }

    #[test]
    fn test_m3_swiglu_activation_scalar_reference() {
        // gate=1, up=0, alpha=1.702, limit=7
        //   gate_c = 1, up_c = 0
        //   glu = 1 * sigmoid(1.702) ≈ 0.845826
        //   out = (0 + 1) * 0.845826 ≈ 0.845826
        let up = mlxcel_core::from_slice_f32(&[0.0], &[1, 1]);
        let gate = mlxcel_core::from_slice_f32(&[1.0], &[1, 1]);
        let result = m3_swiglu_activation(&up, &gate, 1.702, 7.0);
        mlxcel_core::eval(&result);
        let val = mlxcel_core::item_f32(&result);
        let expected = 1.0 / (1.0 + (-1.702_f32).exp());
        assert!(
            (val - expected).abs() < 1e-5,
            "scalar: actual={} expected={}",
            val,
            expected
        );
    }

    #[test]
    fn test_m3_swiglu_activation_gate_upper_clamp() {
        // gate=10 should clamp to 7 → same output as gate=7
        let up = mlxcel_core::from_slice_f32(&[0.5], &[1, 1]);
        let gate_hi = mlxcel_core::from_slice_f32(&[10.0], &[1, 1]);
        let gate_at = mlxcel_core::from_slice_f32(&[7.0], &[1, 1]);

        let r_hi = m3_swiglu_activation(&up, &gate_hi, 1.702, 7.0);
        let r_at = m3_swiglu_activation(&up, &gate_at, 1.702, 7.0);
        mlxcel_core::eval(&r_hi);
        mlxcel_core::eval(&r_at);

        let v_hi = mlxcel_core::item_f32(&r_hi);
        let v_at = mlxcel_core::item_f32(&r_at);
        assert!(
            (v_hi - v_at).abs() < 1e-5,
            "gate clamp: gate=10 ({}) should match gate=7 ({})",
            v_hi,
            v_at
        );
    }

    #[test]
    fn test_m3_swiglu_activation_up_symmetric_clamp() {
        // up=-10 should clamp to -7; up=+10 should clamp to +7
        let gate = mlxcel_core::from_slice_f32(&[1.0], &[1, 1]);

        let up_neg = mlxcel_core::from_slice_f32(&[-10.0], &[1, 1]);
        let up_neg_c = mlxcel_core::from_slice_f32(&[-7.0], &[1, 1]);
        let r_neg = m3_swiglu_activation(&up_neg, &gate, 1.702, 7.0);
        let r_neg_c = m3_swiglu_activation(&up_neg_c, &gate, 1.702, 7.0);
        mlxcel_core::eval(&r_neg);
        mlxcel_core::eval(&r_neg_c);
        assert!((mlxcel_core::item_f32(&r_neg) - mlxcel_core::item_f32(&r_neg_c)).abs() < 1e-5);

        let up_pos = mlxcel_core::from_slice_f32(&[10.0], &[1, 1]);
        let up_pos_c = mlxcel_core::from_slice_f32(&[7.0], &[1, 1]);
        let r_pos = m3_swiglu_activation(&up_pos, &gate, 1.702, 7.0);
        let r_pos_c = m3_swiglu_activation(&up_pos_c, &gate, 1.702, 7.0);
        mlxcel_core::eval(&r_pos);
        mlxcel_core::eval(&r_pos_c);
        assert!((mlxcel_core::item_f32(&r_pos) - mlxcel_core::item_f32(&r_pos_c)).abs() < 1e-5);
    }

    #[test]
    fn test_m3_swiglu_activation_zero_gate_gives_zero() {
        // gate=0 → glu = 0*sigmoid(0) = 0 → out = (up+1)*0 = 0 for any up
        let up = mlxcel_core::from_slice_f32(&[5.0, -3.0, 7.0, -7.0], &[1, 4]);
        let gate = mlxcel_core::from_slice_f32(&[0.0, 0.0, 0.0, 0.0], &[1, 4]);
        let result = m3_swiglu_activation(&up, &gate, 1.702, 7.0);
        mlxcel_core::eval(&result);

        for (i, v) in vec_from_array(&result).iter().enumerate() {
            assert!(v.abs() < 1e-6, "elt {} should be zero, got {}", i, v);
        }
    }

    #[test]
    fn test_m3_swiglu_activation_shape_preservation() {
        // Output shape must match input shape (no broadcasting collapse)
        let up = mlxcel_core::from_slice_f32(&[1.0; 12], &[2, 3, 2]);
        let gate = mlxcel_core::from_slice_f32(&[0.5; 12], &[2, 3, 2]);
        let result = m3_swiglu_activation(&up, &gate, 1.702, 7.0);
        mlxcel_core::eval(&result);
        assert_eq!(mlxcel_core::array_shape(&result), vec![2, 3, 2]);
    }

    // ========================================================================
    // Asymmetric MSA mask tests (chunked-prefill / cached case)
    // ========================================================================
    //
    // Layout:
    //   b=1, num_kv_heads=2, num_heads=4, n_rep=2
    //   block_size=2, top_k=2
    //   cache_offset=4  → cached prefix has 2 blocks (positions 0-3)
    //   num_query_blocks=2  → current chunk has 2 blocks (positions 4-7)
    //   num_key_blocks=4    → full sequence has 4 blocks (positions 0-7)
    //
    // Mask shape: [b=1, num_heads=4, num_query_blocks=2, block_size=2,
    //              kv_per_q_block=4]
    //
    // selected[kv_head, q_block, k_idx] (one selection per query block,
    // indexing into num_key_blocks=4):
    //   kv_head 0:
    //     q_block 0 (p_q=[4,5]): [0, 2] — cached block 0 + current block 0
    //     q_block 1 (p_q=[6,7]): [1, 3] — cached block 1 + current block 1
    //   kv_head 1:
    //     q_block 0 (p_q=[4,5]): [0, 3] — cached block 0 + FUTURE current block
    //     q_block 1 (p_q=[6,7]): [2, 3] — both current blocks
    //
    // p_q[q_block, q_pos] = cache_offset + q_block * block_size + q_pos
    // p_k[kv_head, q_block, k_idx, block_pos]
    //   = selected[kv_head, q_block, k_idx] * block_size + block_pos
    // invalid = (p_k > p_q)
    //
    // GQA: head 0,1 use kv_head 0; head 2,3 use kv_head 1.

    fn msa_asym_mask_fixture() -> UniquePtr<MlxArray> {
        let selected_data: &[i32] = &[
            // kv_head 0:
            // q_block 0: [0, 2]   q_block 1: [1, 3]
            0, 2, 1, 3, // kv_head 1:
            // q_block 0: [0, 3]   q_block 1: [2, 3]
            0, 3, 2, 3,
        ];
        let selected = mlxcel_core::from_slice_i32(selected_data, &[1, 2, 2, 2]);
        build_msa_unified_mask_asymmetric(
            &selected,
            /* b */ 1,
            /* num_kv_heads */ 2,
            /* num_heads */ 4,
            /* n_rep */ 2,
            /* num_query_blocks */ 2,
            /* num_key_blocks */ 4,
            /* top_k */ 2,
            /* block_size */ 2,
            /* cache_offset */ 4,
            /* scores_dtype */ mlxcel_core::dtype::FLOAT32,
        )
    }

    #[test]
    fn test_msa_asym_mask_cached_prefix_attendable() {
        // THE NEW PROPERTY this asymmetric mask exists to enable.
        // head 0, q_block 0 (p_q=[4,5]), k_idx 0 = selected block 0 pos 0
        // (p_k=0). The query is in the current chunk, the key is in the
        // cached prefix. Past p_q → must be valid.
        let mask = msa_asym_mask_fixture();
        assert_eq!(
            mask_at(&mask, 0, 0, 0, 0),
            0.0,
            "current-chunk query must be able to attend to cached prefix"
        );
        assert_eq!(
            mask_at(&mask, 0, 0, 0, 1),
            0.0,
            "cached prefix position 1 also attendable from p_q=4"
        );
    }

    #[test]
    fn test_msa_asym_mask_within_block_causality_in_current_chunk() {
        // head 0, q_block 0, q_pos 0 (p_q=4), selected block 2 (current
        // chunk first block, positions 4,5).
        // k_idx 2 = selected_idx=1 block_pos=0 (p_k=4): equal → valid.
        // k_idx 3 = selected_idx=1 block_pos=1 (p_k=5): future → invalid.
        let mask = msa_asym_mask_fixture();
        assert_eq!(
            mask_at(&mask, 0, 0, 0, 2),
            0.0,
            "p_k == p_q (self) must be valid"
        );
        let v = mask_at(&mask, 0, 0, 0, 3);
        assert!(
            v.is_infinite() && v < 0.0,
            "within-block causality must mask p_k > p_q in current chunk: got {} expected -inf",
            v
        );
    }

    #[test]
    fn test_msa_asym_mask_future_block_in_current_chunk_masked() {
        // head 2 (kv_head 1), q_block 0 (p_q=[4,5]) selects k_block 3
        // (current chunk's SECOND block, positions 6,7). All p_k > p_q.
        // k_idx 2 = block_pos=0 (p_k=6) > 4 → invalid for q_pos 0.
        // k_idx 3 = block_pos=1 (p_k=7) > 5 → invalid for q_pos 1.
        let mask = msa_asym_mask_fixture();
        for q_pos in 0..2 {
            for k_idx in 2..4 {
                let v = mask_at(&mask, 2, 0, q_pos, k_idx);
                assert!(
                    v.is_infinite() && v < 0.0,
                    "future-block must mask: q_pos={} k_idx={} got {} expected -inf",
                    q_pos,
                    k_idx,
                    v
                );
            }
        }
    }

    #[test]
    fn test_msa_asym_mask_cache_offset_shifts_p_q() {
        // Cached selections from q_block 0 (p_q=[4,5]): selected k_block 0
        // covers positions [0,1]. All p_k < p_q → all valid.
        // This proves p_q starts at cache_offset=4, not 0. If p_q started at
        // 0, p_q[0]=0 and p_k=0 would still be valid by coincidence, but
        // p_q[1]=1 < p_k=1=0 ... actually that ALSO would pass equality.
        // The disambiguating check is q_block 1 attending k_block 1: with
        // correct cache_offset, p_q=[6,7], p_k=[2,3] → all valid. Without
        // cache_offset, p_q=[2,3], p_k=[2,3]: q_pos 0 p_k 1 (p_k=3 > p_q=2)
        // would mask. Run that one.
        let mask = msa_asym_mask_fixture();
        // head 0, q_block 1 (p_q=[6,7]), k_idx 0..1 = selected block 1
        // (cached positions [2,3]). All p_k < p_q → all valid.
        for q_pos in 0..2 {
            for k_idx in 0..2 {
                assert_eq!(
                    mask_at(&mask, 0, 1, q_pos, k_idx),
                    0.0,
                    "second q_block attending cached block 1 must be valid \
                     (proves p_q uses cache_offset): q_pos={} k_idx={}",
                    q_pos,
                    k_idx
                );
            }
        }
    }

    #[test]
    fn test_msa_asym_mask_cross_block_within_chunk_valid() {
        // head 0, q_block 1 (p_q=[6,7]), k_idx 0..1 = selected block 1
        // (cached, p_k=[2,3]); k_idx 2..3 = selected block 3 (current chunk
        // second block, p_k=[6,7]).
        // k_idx 2 (p_k=6): valid for q_pos 0 (p_q=6, equal), valid for
        // q_pos 1 (p_q=7, p_k < p_q).
        // k_idx 3 (p_k=7): invalid for q_pos 0 (p_k=7 > p_q=6), valid for
        // q_pos 1 (equal).
        let mask = msa_asym_mask_fixture();
        assert_eq!(mask_at(&mask, 0, 1, 0, 2), 0.0, "p_k=6, p_q=6 valid");
        assert_eq!(mask_at(&mask, 0, 1, 1, 2), 0.0, "p_k=6, p_q=7 valid");
        let v = mask_at(&mask, 0, 1, 0, 3);
        assert!(
            v.is_infinite() && v < 0.0,
            "p_k=7, p_q=6 must mask: got {} expected -inf",
            v
        );
        assert_eq!(mask_at(&mask, 0, 1, 1, 3), 0.0, "p_k=7, p_q=7 valid");
    }

    #[test]
    fn test_msa_asym_mask_gqa_replication_within_kv_group() {
        // head 0,1 share kv_head 0; head 2,3 share kv_head 1.
        // The mask must be identical inside each GQA group at every
        // (q_block, q_pos, k_idx).
        let mask = msa_asym_mask_fixture();
        for q_block in 0..2 {
            for q_pos in 0..2 {
                for k_idx in 0..4 {
                    let h0 = mask_at(&mask, 0, q_block, q_pos, k_idx);
                    let h1 = mask_at(&mask, 1, q_block, q_pos, k_idx);
                    let h2 = mask_at(&mask, 2, q_block, q_pos, k_idx);
                    let h3 = mask_at(&mask, 3, q_block, q_pos, k_idx);
                    assert_eq!(
                        h0.to_bits(),
                        h1.to_bits(),
                        "kv_head 0 group (heads 0,1) mismatch at q_block={} q_pos={} k_idx={}",
                        q_block,
                        q_pos,
                        k_idx,
                    );
                    assert_eq!(
                        h2.to_bits(),
                        h3.to_bits(),
                        "kv_head 1 group (heads 2,3) mismatch at q_block={} q_pos={} k_idx={}",
                        q_block,
                        q_pos,
                        k_idx,
                    );
                }
            }
        }
    }

    // ========================================================================
    // Integration test: four-chunk chunked-prefill vs single-shot forward
    // ========================================================================
    //
    // Target: prove that SparseAttention::forward produces the same output
    // when called as four chunks with a growing KVCache as when called once
    // on the full input. Same input + same weights + same seed of behavior
    // → outputs match within fp tolerance.
    //
    // Stuart's framing (cycle 79, 2026-06-26): "four chunks, not two."
    // Four exercises accumulated-error across multiple cache extensions, not
    // just the single boundary that two chunks would catch.
    //
    // The integration test is the gate that distinguishes "no crash"
    // (necessary but not sufficient) from "correct attention pattern across
    // cached prefill" (the multi-turn opencode goal).

    fn make_linear(
        out_features: i32,
        in_features: i32,
        scale: f32,
    ) -> mlxcel_core::layers::UnifiedLinear {
        // Deterministic small weights. Pattern is sin-based so values
        // distribute across positive and negative without random.
        let n = (out_features * in_features) as usize;
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let v = (i as f32 * 0.137).sin() * scale;
            data.push(v);
        }
        let weight = mlxcel_core::from_slice_f32(&data, &[out_features, in_features]);
        mlxcel_core::layers::UnifiedLinear::Regular(mlxcel_core::layers::Linear::new(weight, None))
    }

    fn make_gemma_rms_norm(dim: i32) -> GemmaRMSNorm {
        // weight=0 → GemmaRMSNorm effective multiplier = (1 + 0) = 1, so
        // this norm just normalises (no scaling). Eps=1e-6 standard.
        let zeros: Vec<f32> = vec![0.0; dim as usize];
        let weight = mlxcel_core::from_slice_f32(&zeros, &[dim]);
        GemmaRMSNorm::new(weight, 1e-6)
    }

    fn make_test_sparse_attention() -> SparseAttention {
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 4;
        let hidden = num_heads * head_dim; // 16
        let index_dim = 4;
        let block_size = 2;
        let top_k = 2;
        let sparse_local_block = 1;
        let rope_dims = 2;
        let rope_base = 10000.0;

        SparseAttention {
            q_proj: make_linear(num_heads * head_dim, hidden, 0.1),
            k_proj: make_linear(num_kv_heads * head_dim, hidden, 0.1),
            v_proj: make_linear(num_kv_heads * head_dim, hidden, 0.1),
            o_proj: make_linear(hidden, num_heads * head_dim, 0.1),
            q_norm: Some(make_gemma_rms_norm(head_dim)),
            k_norm: Some(make_gemma_rms_norm(head_dim)),
            index_q_proj: Some(make_linear(num_kv_heads * index_dim, hidden, 0.1)),
            index_k_proj: Some(make_linear(index_dim, hidden, 0.1)),
            index_q_norm: Some(make_gemma_rms_norm(index_dim)),
            index_k_norm: Some(make_gemma_rms_norm(index_dim)),
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims,
            rope_base,
            block_size,
            top_k,
            index_dim,
            sparse_local_block,
            layer_idx: 0,
        }
    }

    fn make_test_input(b: i32, l: i32, hidden: i32) -> UniquePtr<MlxArray> {
        // Deterministic input via cos(i * 0.073) for reproducibility.
        let n = (b * l * hidden) as usize;
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            data.push((i as f32 * 0.073).cos() * 0.5);
        }
        mlxcel_core::from_slice_f32(&data, &[b, l, hidden])
    }

    fn output_l2_diff(a: &MlxArray, b: &MlxArray) -> f32 {
        let diff = mlxcel_core::subtract(a, b);
        let sq = mlxcel_core::multiply(&diff, &diff);
        // sum over all axes by reducing repeatedly
        let s = mlxcel_core::array_shape(&sq);
        let mut acc = mlxcel_core::copy(&sq);
        for axis in (0..s.len()).rev() {
            acc = mlxcel_core::sum_axis(&acc, axis as i32, false);
        }
        mlxcel_core::eval(&acc);
        mlxcel_core::item_f32(&acc).sqrt()
    }

    #[test]
    fn test_msa_four_chunk_chunked_vs_single_shot_no_crash() {
        // The minimum gate: four-chunk forward through the cached MSA path
        // completes without crashing, with correct output shapes per chunk.
        // This is the cycle-65 latent comment turned into a live test.
        let layer = make_test_sparse_attention();
        let hidden = layer.num_heads * layer.head_dim;
        let l_chunk = 6; // 3 blocks of 2 per chunk
        let n_chunks = 4;
        let l_total = l_chunk * n_chunks; // 24

        let input = make_test_input(1, l_total, hidden);
        let mut cache = KVCache::new();
        for i in 0..n_chunks {
            let chunk = mlxcel_core::slice(
                &input,
                &[0, i * l_chunk, 0],
                &[1, (i + 1) * l_chunk, hidden],
            );
            let out = layer.forward(&chunk, &mut cache, None);
            mlxcel_core::eval(&out);
            let shape = mlxcel_core::array_shape(&out);
            assert_eq!(
                shape,
                vec![1, l_chunk, hidden],
                "chunk {} output shape mismatch: got {:?} expected {:?}",
                i,
                shape,
                vec![1, l_chunk, hidden]
            );
        }
        // Final state assertions: cache has full sequence length.
        assert_eq!(
            cache.offset, l_total,
            "main K/V offset must equal total tokens"
        );
        assert_eq!(
            cache.m3_idx_offset(),
            l_total,
            "indexer K offset must equal total tokens (lockstep with main K)"
        );
    }

    #[test]
    fn test_msa_four_chunk_matches_single_shot_within_tolerance() {
        // The correctness gate: chunked forward (with growing cache) must
        // produce the same output as single-shot forward (no cache) for
        // positions where both paths take the MSA branch.
        //
        // Note on dispatch: chunk 1 with l_chunk=6 has num_key_blocks=3,
        // top_k=2 → MSA. So all four chunks take MSA, and single-shot also
        // takes MSA. The two paths should match within fp tolerance for ALL
        // positions.
        let layer = make_test_sparse_attention();
        let hidden = layer.num_heads * layer.head_dim;
        let l_chunk = 6;
        let n_chunks = 4;
        let l_total = l_chunk * n_chunks;

        let input = make_test_input(1, l_total, hidden);

        // Chunked path
        let mut cache = KVCache::new();
        let mut chunked_outs: Vec<UniquePtr<MlxArray>> = Vec::new();
        for i in 0..n_chunks {
            let chunk = mlxcel_core::slice(
                &input,
                &[0, i * l_chunk, 0],
                &[1, (i + 1) * l_chunk, hidden],
            );
            chunked_outs.push(layer.forward(&chunk, &mut cache, None));
        }
        let chunked_concat = {
            let mut acc = mlxcel_core::copy(&chunked_outs[0]);
            for out in &chunked_outs[1..] {
                acc = mlxcel_core::concatenate(&acc, out, 1);
            }
            acc
        };
        mlxcel_core::eval(&chunked_concat);

        // Single-shot path
        let mut cache_single = KVCache::new();
        let single_out = layer.forward(&input, &mut cache_single, None);
        mlxcel_core::eval(&single_out);

        let single_shape = mlxcel_core::array_shape(&single_out);
        let chunked_shape = mlxcel_core::array_shape(&chunked_concat);
        assert_eq!(
            single_shape, chunked_shape,
            "single and chunked output shapes must match"
        );

        // L2 norm of difference. Tolerance is loose because chunked path
        // does smaller matmuls (different fp accumulation order); the
        // selection of top-k blocks should be deterministic across paths
        // for matching query positions, so the residual is fp-numerical only.
        let diff = output_l2_diff(&chunked_concat, &single_out);
        // Compute the L2 norm of single_out for relative comparison.
        let single_norm = {
            let sq = mlxcel_core::multiply(&single_out, &single_out);
            let s = mlxcel_core::array_shape(&sq);
            let mut acc = mlxcel_core::copy(&sq);
            for axis in (0..s.len()).rev() {
                acc = mlxcel_core::sum_axis(&acc, axis as i32, false);
            }
            mlxcel_core::eval(&acc);
            mlxcel_core::item_f32(&acc).sqrt()
        };
        let relative = if single_norm > 1e-6 {
            diff / single_norm
        } else {
            diff
        };
        // Tolerance: 5% relative L2. fp differences from kernel/op-order
        // changes typically land < 1%; 5% is generous against subtle
        // selection-determinism issues that would still indicate
        // correctness in spirit.
        assert!(
            relative < 0.05,
            "chunked vs single-shot relative L2 diff = {} (absolute {}, norm {}). \
             Tolerance 0.05. Larger diff indicates the asymmetric path is \
             selecting different blocks than single-shot — a correctness \
             regression.",
            relative,
            diff,
            single_norm
        );
    }

    #[test]
    fn test_msa_dense_saturation_then_msa_transition_matches_single_shot() {
        // Cycle-79.5 fix gate: a session that starts with dense-saturated
        // chunks (`num_key_blocks <= top_k`) and later transitions to MSA
        // (because kv_len grew past the saturation threshold) MUST produce
        // the same output as a single-shot MSA-throughout reference. This
        // requires m3_idx_k to accumulate on EVERY MSA-eligible forward —
        // including dense-saturated chunks — so the post-saturation MSA
        // chunk's selector sees the full prefix.
        //
        // Pre-cycle-79.5 the m3_idx_k_update_and_fetch call lived only in
        // the MSA branch, so the dense-saturated chunks left m3_idx_k empty.
        // The next chunk's lockstep check failed and forced a dense
        // fallback for the remainder of the session — wrong dispatch, but
        // the output still came out plausible because dense is a valid
        // attention mechanism. A correctness test (output match) wouldn't
        // have caught it; only a live multi-turn smoke test exposed the
        // permanent dense fallback. This test now traps the regression at
        // unit level.
        //
        // Config: block_size=2, top_k=2. Chunk 1 l=2 → num_key_blocks=1
        // ≤ top_k, dense-saturated AND l <= block_size (both dense
        // triggers). Chunk 2 l=4 → kv_len=6, num_key_blocks=3 > top_k=2 and
        // l > block_size → MSA dispatches. Single-shot reference l=6,
        // num_key_blocks=3 → MSA.
        let layer = make_test_sparse_attention();
        let hidden = layer.num_heads * layer.head_dim;
        let l_dense = 2;
        let l_msa = 4;
        let l_total = l_dense + l_msa; // 6
        let input = make_test_input(1, l_total, hidden);

        // Reference: single-shot MSA over the full 6 tokens.
        let mut cache_ref = KVCache::new();
        let ref_out = layer.forward(&input, &mut cache_ref, None);
        mlxcel_core::eval(&ref_out);
        // Sanity: single-shot took MSA path — m3_idx_offset advanced to
        // l_total only if MSA fired. Always-update would also advance it,
        // but for this single-shot case both paths agree.
        assert_eq!(cache_ref.m3_idx_offset(), l_total);

        // Chunked: dense-saturated chunk 1 followed by MSA chunk 2.
        let mut cache_chunked = KVCache::new();
        let chunk1 = mlxcel_core::slice(&input, &[0, 0, 0], &[1, l_dense, hidden]);
        let out1 = layer.forward(&chunk1, &mut cache_chunked, None);
        mlxcel_core::eval(&out1);
        // After chunk 1 the always-update keeps lockstep even though
        // dispatch was dense. THIS is the cycle-79.5 invariant.
        assert_eq!(
            cache_chunked.offset, l_dense,
            "main K offset must advance after chunk 1"
        );
        assert_eq!(
            cache_chunked.m3_idx_offset(),
            l_dense,
            "m3_idx_offset must advance to {l_dense} after dense-saturated \
             chunk 1 — the always-update path must populate it regardless \
             of dispatch. If this fails, the indexer K cache is gappy and \
             the next MSA chunk will see an incomplete prefix."
        );

        let chunk2 = mlxcel_core::slice(&input, &[0, l_dense, 0], &[1, l_total, hidden]);
        let out2 = layer.forward(&chunk2, &mut cache_chunked, None);
        mlxcel_core::eval(&out2);
        assert_eq!(cache_chunked.offset, l_total);
        assert_eq!(
            cache_chunked.m3_idx_offset(),
            l_total,
            "m3_idx_offset must equal l_total after chunk 2's MSA forward — \
             this proves MSA actually dispatched (dense_attention never \
             advances m3_idx_offset)"
        );

        // Compare outputs. Chunked path concatenates out1 (dense, l_dense
        // tokens) and out2 (MSA, l_msa tokens). Single-shot ref is MSA over
        // all l_total tokens.
        let chunked_concat = mlxcel_core::concatenate(&out1, &out2, 1);
        mlxcel_core::eval(&chunked_concat);

        let ref_shape = mlxcel_core::array_shape(&ref_out);
        let chunked_shape = mlxcel_core::array_shape(&chunked_concat);
        assert_eq!(
            ref_shape, chunked_shape,
            "single-shot and chunked output shapes must match"
        );

        let diff = output_l2_diff(&chunked_concat, &ref_out);
        let ref_norm = {
            let sq = mlxcel_core::multiply(&ref_out, &ref_out);
            let s = mlxcel_core::array_shape(&sq);
            let mut acc = mlxcel_core::copy(&sq);
            for axis in (0..s.len()).rev() {
                acc = mlxcel_core::sum_axis(&acc, axis as i32, false);
            }
            mlxcel_core::eval(&acc);
            mlxcel_core::item_f32(&acc).sqrt()
        };
        let relative = if ref_norm > 1e-6 {
            diff / ref_norm
        } else {
            diff
        };
        // The chunked path's chunk 1 is dense (l_dense tokens) but the
        // reference's first l_dense tokens come from MSA's output at those
        // positions. For positions where MSA selected all key blocks, dense
        // and MSA produce identical math. For chunk 1 (kv_len=2 at chunk 1
        // boundary in chunked, kv_len=2..6 in ref) the two paths diverge by
        // construction: ref's chunk-1-positions saw the full 6-token K, the
        // chunked's chunk 1 only saw its own 2-token K. They are NOT
        // expected to match for those positions.
        //
        // What IS expected to match: chunked chunk 2's MSA output (positions
        // 2..6) vs ref's MSA output for positions 2..6. Slice both and
        // compare just those positions.
        let chunked_msa_slice =
            mlxcel_core::slice(&chunked_concat, &[0, l_dense, 0], &[1, l_total, hidden]);
        let ref_msa_slice = mlxcel_core::slice(&ref_out, &[0, l_dense, 0], &[1, l_total, hidden]);
        mlxcel_core::eval(&chunked_msa_slice);
        mlxcel_core::eval(&ref_msa_slice);
        let msa_diff = output_l2_diff(&chunked_msa_slice, &ref_msa_slice);
        let ref_msa_norm = {
            let sq = mlxcel_core::multiply(&ref_msa_slice, &ref_msa_slice);
            let s = mlxcel_core::array_shape(&sq);
            let mut acc = mlxcel_core::copy(&sq);
            for axis in (0..s.len()).rev() {
                acc = mlxcel_core::sum_axis(&acc, axis as i32, false);
            }
            mlxcel_core::eval(&acc);
            mlxcel_core::item_f32(&acc).sqrt()
        };
        let msa_relative = if ref_msa_norm > 1e-6 {
            msa_diff / ref_msa_norm
        } else {
            msa_diff
        };
        assert!(
            msa_relative < 0.05,
            "chunked MSA-after-dense vs single-shot MSA relative L2 = {} \
             (absolute {}, norm {}). Tolerance 0.05. A larger diff indicates \
             chunk 2's MSA dispatch was scoring with a gappy idx_k (missing \
             chunk 1's positions), which would happen if the m3_idx_k cache \
             update only fired in the MSA branch. relative is the position- \
             [{}..{}) slice where both paths take MSA. The aggregate \
             chunked-vs-ref diff (which includes chunk 1's dense-vs-MSA \
             divergence) is {:.4} relative; that is expected to be larger.",
            msa_relative,
            msa_diff,
            ref_msa_norm,
            l_dense,
            l_total,
            relative
        );
    }

    #[test]
    fn test_msa_four_chunk_with_cache_detach_adopt_midway_matches_continuous() {
        // Cycle-79 proper-fix gate: exercise the detach/adopt round-trip in
        // the MIDDLE of a chunked-prefill session. Without `m3_idx_k`
        // preservation through `clone_handle`/`install_detached`, the MSA
        // dispatch on chunks 3-4 would either crash (pre-band-aid) or fall
        // back to dense via the lockstep guard (post-band-aid). With the
        // proper fix, MSA fires normally and the output matches a
        // continuous-cache reference within fp tolerance.
        let layer = make_test_sparse_attention();
        let hidden = layer.num_heads * layer.head_dim;
        let l_chunk = 6;
        let n_chunks = 4;
        let l_total = l_chunk * n_chunks;

        let input = make_test_input(1, l_total, hidden);

        // Reference path: forward all four chunks against a single growing
        // cache (no adoption interruption).
        let mut cache_ref = KVCache::new();
        let mut outs_ref: Vec<UniquePtr<MlxArray>> = Vec::new();
        for i in 0..n_chunks {
            let chunk = mlxcel_core::slice(
                &input,
                &[0, i * l_chunk, 0],
                &[1, (i + 1) * l_chunk, hidden],
            );
            outs_ref.push(layer.forward(&chunk, &mut cache_ref, None));
        }

        // Adoption path: forward chunks 1-2, simulate prompt-cache donate +
        // adopt by round-tripping the cache through `clone_handle` /
        // `install_detached` (the same primitives `CachePool::detach`/`adopt`
        // call into), then continue with chunks 3-4 against the restored
        // cache.
        let mut cache_src = KVCache::new();
        let mut outs_adopt: Vec<UniquePtr<MlxArray>> = Vec::new();
        for i in 0..2 {
            let chunk = mlxcel_core::slice(
                &input,
                &[0, i * l_chunk, 0],
                &[1, (i + 1) * l_chunk, hidden],
            );
            outs_adopt.push(layer.forward(&chunk, &mut cache_src, None));
        }
        assert_eq!(cache_src.offset, (2 * l_chunk) as i32);
        assert_eq!(cache_src.m3_idx_offset(), (2 * l_chunk) as i32);

        // Donate.
        let handle = cache_src.clone_handle();
        assert!(cache_src.is_empty(), "source cache must drain after donate");
        assert_eq!(cache_src.m3_idx_offset(), 0);

        // Adopt.
        let mut cache_adopted = KVCache::new();
        cache_adopted
            .install_detached(handle)
            .expect("install_detached must succeed");
        assert_eq!(
            cache_adopted.offset,
            (2 * l_chunk) as i32,
            "main K offset must round-trip"
        );
        assert_eq!(
            cache_adopted.m3_idx_offset(),
            (2 * l_chunk) as i32,
            "indexer K offset must round-trip — the cycle-79 proper fix. A \
             zero here means `m3_idx_k` was dropped on detach/adopt and chunk \
             3's MSA dispatch will fall back to dense via the lockstep guard."
        );
        assert!(
            cache_adopted.has_m3_idx_k_state(),
            "indexer K tensor must round-trip alongside main K/V"
        );

        for i in 2..n_chunks {
            let chunk = mlxcel_core::slice(
                &input,
                &[0, i * l_chunk, 0],
                &[1, (i + 1) * l_chunk, hidden],
            );
            outs_adopt.push(layer.forward(&chunk, &mut cache_adopted, None));
        }
        assert_eq!(cache_adopted.offset, l_total);
        assert_eq!(cache_adopted.m3_idx_offset(), l_total);

        // Compare both paths' concatenated outputs.
        let concat = |outs: &[UniquePtr<MlxArray>]| -> UniquePtr<MlxArray> {
            let mut acc = mlxcel_core::copy(&outs[0]);
            for out in &outs[1..] {
                acc = mlxcel_core::concatenate(&acc, out, 1);
            }
            mlxcel_core::eval(&acc);
            acc
        };
        let ref_concat = concat(&outs_ref);
        let adopt_concat = concat(&outs_adopt);

        let diff = output_l2_diff(&ref_concat, &adopt_concat);
        let ref_norm = {
            let sq = mlxcel_core::multiply(&ref_concat, &ref_concat);
            let s = mlxcel_core::array_shape(&sq);
            let mut acc = mlxcel_core::copy(&sq);
            for axis in (0..s.len()).rev() {
                acc = mlxcel_core::sum_axis(&acc, axis as i32, false);
            }
            mlxcel_core::eval(&acc);
            mlxcel_core::item_f32(&acc).sqrt()
        };
        let relative = if ref_norm > 1e-6 {
            diff / ref_norm
        } else {
            diff
        };
        // Same input, same model, same dispatch path through MSA on both
        // sides. Any non-trivial diff means the adoption disturbed the
        // computation. A regression in m3_idx_k preservation would push
        // chunks 3-4 to the dense fallback while the reference path stays
        // on MSA, producing a much larger relative diff.
        assert!(
            relative < 0.05,
            "detach-adopt-midway vs continuous-cache relative L2 = {} \
             (abs {}, norm {}). Tolerance 0.05. A larger diff indicates the \
             adoption side fell back to dense for chunks 3-4 — i.e. the \
             `m3_idx_k` preservation through detach/adopt regressed.",
            relative,
            diff,
            ref_norm
        );
    }

    /// Read one element from a block-scores tensor at
    /// (kv_head, q_block, k_block) for batch 0.
    fn block_score_at(scores: &MlxArray, head: i32, q_block: i32, k_block: i32) -> f32 {
        let single = mlxcel_core::slice(
            scores,
            &[0, head, q_block, k_block],
            &[1, head + 1, q_block + 1, k_block + 1],
        );
        mlxcel_core::eval(&single);
        mlxcel_core::item_f32(&single)
    }

    /// The local-block guarantee must anchor at the chunk's REAL last query
    /// position when the final query block is partial.
    ///
    /// Fixture geometry (block_size=2, sparse_local_block=1): a dense
    /// prompt-cache adoption resumed prefill at cache_offset=5 with a
    /// single query token (q_len=1). kv_len=6 → key blocks {0,1,2}; the
    /// query's real position 5 lives in block 2. The pre-fix formula
    /// anchored at the padded block end (position 6 → block 3, which does
    /// not exist), so the +inf landed nowhere and NO local block was
    /// forced — the degenerate selection the MSA paper's fixed allocation
    /// exists to prevent. The mutation this test kills: dropping the
    /// `q_len` clamp from `ensure_local_block_score_asymmetric`.
    #[test]
    fn test_local_guarantee_partial_final_query_block_forces_real_newest_block() {
        let layer = make_test_sparse_attention();
        assert_eq!(layer.block_size, 2, "fixture geometry assumption");
        assert_eq!(layer.sparse_local_block, 1, "fixture geometry assumption");

        let num_query_blocks = 1;
        let num_key_blocks = 3;
        let scores = mlxcel_core::zeros(
            &[1, layer.num_kv_heads, num_query_blocks, num_key_blocks],
            mlxcel_core::dtype::FLOAT32,
        );

        let forced = layer.ensure_local_block_score_asymmetric(
            &scores,
            num_query_blocks,
            num_key_blocks,
            5, // cache_offset: adoption resumed mid-block
            1, // q_len: single trailing query token at absolute position 5
        );

        for head in 0..layer.num_kv_heads {
            assert_eq!(
                block_score_at(&forced, head, 0, 2),
                f32::INFINITY,
                "head {head}: the key block containing the query's real \
                 position (block 2) must be forced to +inf"
            );
            assert_eq!(
                block_score_at(&forced, head, 0, 0),
                0.0,
                "head {head}: distant block 0 must stay unforced"
            );
            assert_eq!(
                block_score_at(&forced, head, 0, 1),
                0.0,
                "head {head}: block 1 is outside the local window \
                 (sparse_local_block=1) and must stay unforced"
            );
        }
    }

    /// Full query blocks must be unaffected by the `q_len` anchor: for a
    /// block-aligned chunk the clamp is a no-op and the forced set matches
    /// the original asymmetric formula (the diagonal local block).
    #[test]
    fn test_local_guarantee_full_query_blocks_unchanged_by_q_len_anchor() {
        let layer = make_test_sparse_attention();

        let num_query_blocks = 1;
        let num_key_blocks = 3;
        let scores = mlxcel_core::zeros(
            &[1, layer.num_kv_heads, num_query_blocks, num_key_blocks],
            mlxcel_core::dtype::FLOAT32,
        );

        // cache_offset=4 (block-aligned), q_len=2 (one full block covering
        // absolute positions 4-5, i.e. exactly key block 2).
        let forced = layer.ensure_local_block_score_asymmetric(
            &scores,
            num_query_blocks,
            num_key_blocks,
            4,
            2,
        );

        for head in 0..layer.num_kv_heads {
            assert_eq!(
                block_score_at(&forced, head, 0, 2),
                f32::INFINITY,
                "head {head}: aligned case must force the diagonal block, \
                 exactly as before the q_len anchor"
            );
            assert_eq!(block_score_at(&forced, head, 0, 0), 0.0);
            assert_eq!(block_score_at(&forced, head, 0, 1), 0.0);
        }
    }

    /// Dense prompt-cache adoption resumes prefill at an ARBITRARY token
    /// offset (`DetachedCacheSet::truncate_to` trims to exactly
    /// `matched_len` — e.g. the live 160-token adoption of 2026-07-04,
    /// which is not a multiple of the 128 block size). This is the
    /// layer-level gate for that path: trim a donated cache to an
    /// unaligned boundary, adopt, resume prefill from there, and require
    /// the outputs to match a continuous-cache reference.
    ///
    /// Covers, in one arc: `trim_to` K/V + `m3_idx_k` lockstep at a
    /// non-block boundary, MSA resume with `cache_offset % block_size != 0`
    /// (which exercises the q_len-anchored local-block guarantee), and the
    /// mlx-lm #975 off-by-one class (a one-token trim error shifts every
    /// resumed position and blows the tolerance).
    #[test]
    fn test_msa_trim_adopt_at_block_aligned_boundary_matches_continuous() {
        let layer = make_test_sparse_attention();
        let hidden = layer.num_heads * layer.head_dim;
        let l_chunk = 6;
        let n_chunks = 4;
        let l_total = l_chunk * n_chunks; // 24
        // Block-aligned (but NOT chunk-aligned) trim: cold-equivalence holds
        // and this test pins it. Empirical finding (2026-07-04): at an
        // UNALIGNED trim (7 here, or the live dense-adoption 160 with
        // block_size 128), the resumed output legitimately diverges from a
        // cold run (relative L2 ≈ 1.13 at this scale) because MSA's
        // query-block pooling grid anchors at the chunk start — a shifted
        // grid pools different positions together, selects different top-k
        // blocks, and computes different (valid) attention. Cold-equivalence
        // after partial adoption therefore requires flooring dense adoption
        // to the MSA block size, mirroring the paged path's #225 flooring.
        // Tracked as follow-up work.
        let trim_target: i32 = 8;

        let input = make_test_input(1, l_total, hidden);

        // Reference: all four chunks against one continuous cache.
        let mut cache_ref = KVCache::new();
        let mut outs_ref: Vec<UniquePtr<MlxArray>> = Vec::new();
        for i in 0..n_chunks {
            let chunk = mlxcel_core::slice(
                &input,
                &[0, i * l_chunk, 0],
                &[1, (i + 1) * l_chunk, hidden],
            );
            outs_ref.push(layer.forward(&chunk, &mut cache_ref, None));
        }

        // Donor: two chunks (12 tokens), donate, trim to 7 — the dense
        // adopt path's exact shape (truncate to raw matched_len).
        let mut cache_src = KVCache::new();
        for i in 0..2 {
            let chunk = mlxcel_core::slice(
                &input,
                &[0, i * l_chunk, 0],
                &[1, (i + 1) * l_chunk, hidden],
            );
            let _ = layer.forward(&chunk, &mut cache_src, None);
        }
        let mut handle = cache_src.clone_handle();
        handle
            .trim_to(trim_target)
            .expect("trim to an unaligned boundary must succeed");

        // Adopt and resume prefill from token 7 in one 17-token chunk.
        // The offsets after install are the observable trim contract: a
        // K/V-vs-indexer lockstep failure at the non-block boundary shows
        // up as a mismatch here.
        let mut cache_adopted = KVCache::new();
        cache_adopted
            .install_detached(handle)
            .expect("install_detached must succeed");
        assert_eq!(cache_adopted.offset, trim_target);
        assert_eq!(
            cache_adopted.m3_idx_offset(),
            trim_target,
            "indexer cache must trim in lockstep at a non-block boundary"
        );

        let suffix = mlxcel_core::slice(&input, &[0, trim_target, 0], &[1, l_total, hidden]);
        let out_suffix = layer.forward(&suffix, &mut cache_adopted, None);
        assert_eq!(cache_adopted.offset, l_total);
        assert_eq!(cache_adopted.m3_idx_offset(), l_total);

        // Reference positions 7..24, sliced out of the concatenated
        // reference outputs.
        let mut ref_concat = mlxcel_core::copy(&outs_ref[0]);
        for out in &outs_ref[1..] {
            ref_concat = mlxcel_core::concatenate(&ref_concat, out, 1);
        }
        let ref_suffix =
            mlxcel_core::slice(&ref_concat, &[0, trim_target, 0], &[1, l_total, hidden]);
        mlxcel_core::eval(&ref_suffix);
        mlxcel_core::eval(&out_suffix);

        let diff = output_l2_diff(&ref_suffix, &out_suffix);
        let ref_norm = {
            let sq = mlxcel_core::multiply(&ref_suffix, &ref_suffix);
            let s = mlxcel_core::array_shape(&sq);
            let mut acc = mlxcel_core::copy(&sq);
            for axis in (0..s.len()).rev() {
                acc = mlxcel_core::sum_axis(&acc, axis as i32, false);
            }
            mlxcel_core::eval(&acc);
            mlxcel_core::item_f32(&acc).sqrt()
        };
        let relative = if ref_norm > 1e-6 {
            diff / ref_norm
        } else {
            diff
        };
        assert!(
            relative < 0.05,
            "unaligned trim-adopt resume vs continuous relative L2 = {} \
             (abs {}, norm {}). Tolerance 0.05 (same cross-chunking gate as \
             the chunked-vs-single-shot test). A large diff means the trim \
             left K/V and the indexer cache disagreeing about the resume \
             position, or the resumed MSA path mis-anchored its local-block \
             guarantee.",
            relative,
            diff,
            ref_norm
        );
    }

    #[test]
    fn test_msa_four_chunk_determinism_bit_exact() {
        // Cycle 64 lesson: rule out MLX nondeterminism before declaring an
        // fp-numerical issue real. Run the chunked path twice. If outputs
        // differ bit-exactly, MLX is nondeterministic (not the asymmetric
        // path's fault).
        let layer = make_test_sparse_attention();
        let hidden = layer.num_heads * layer.head_dim;
        let l_chunk = 6;
        let n_chunks = 4;

        let input = make_test_input(1, l_chunk * n_chunks, hidden);

        let run = || -> UniquePtr<MlxArray> {
            let mut cache = KVCache::new();
            let mut chunked_outs: Vec<UniquePtr<MlxArray>> = Vec::new();
            for i in 0..n_chunks {
                let chunk = mlxcel_core::slice(
                    &input,
                    &[0, i * l_chunk, 0],
                    &[1, (i + 1) * l_chunk, hidden],
                );
                chunked_outs.push(layer.forward(&chunk, &mut cache, None));
            }
            let mut acc = mlxcel_core::copy(&chunked_outs[0]);
            for out in &chunked_outs[1..] {
                acc = mlxcel_core::concatenate(&acc, out, 1);
            }
            mlxcel_core::eval(&acc);
            acc
        };

        let run1 = run();
        let run2 = run();
        let diff = output_l2_diff(&run1, &run2);
        assert!(
            diff < 1e-6,
            "chunked path is nondeterministic: L2 diff between two runs = {}",
            diff
        );
    }

    #[test]
    fn test_msa_asym_mask_exhaustive_against_reference() {
        // Brute-force check every (head, q_block, q_pos, k_idx) of the
        // fixture against a hand-computed reference using the same rule
        // the mask documents: p_k > p_q ⇒ invalid.
        // GQA: head h uses kv_head = h / n_rep = h / 2.
        const N_REP: i32 = 2;
        const BLOCK_SIZE: i32 = 2;
        const TOP_K: i32 = 2;
        const CACHE_OFFSET: i32 = 4;
        let selected = [
            // [kv_head][q_block][k_select_idx]
            [[0, 2], [1, 3]],
            [[0, 3], [2, 3]],
        ];
        let mask = msa_asym_mask_fixture();
        let mut checked = 0;
        for head in 0..4i32 {
            let kv_head = (head / N_REP) as usize;
            for q_block in 0..2i32 {
                let p_q_base = CACHE_OFFSET + q_block * BLOCK_SIZE;
                for q_pos in 0..BLOCK_SIZE {
                    let p_q = p_q_base + q_pos;
                    for k_idx in 0..(TOP_K * BLOCK_SIZE) {
                        let select_idx = (k_idx / BLOCK_SIZE) as usize;
                        let block_pos = k_idx % BLOCK_SIZE;
                        let selected_block = selected[kv_head][q_block as usize][select_idx];
                        let p_k = selected_block * BLOCK_SIZE + block_pos;
                        let expected_invalid = p_k > p_q;
                        let v = mask_at(&mask, head, q_block, q_pos, k_idx);
                        if expected_invalid {
                            assert!(
                                v.is_infinite() && v < 0.0,
                                "exhaustive: head={} q_block={} q_pos={} k_idx={} \
                                 p_q={} p_k={} expected -inf got {}",
                                head,
                                q_block,
                                q_pos,
                                k_idx,
                                p_q,
                                p_k,
                                v
                            );
                        } else {
                            assert_eq!(
                                v, 0.0,
                                "exhaustive: head={} q_block={} q_pos={} k_idx={} \
                                 p_q={} p_k={} expected 0.0 got {}",
                                head, q_block, q_pos, k_idx, p_q, p_k, v
                            );
                        }
                        checked += 1;
                    }
                }
            }
        }
        // 4 heads × 2 q_blocks × 2 q_pos × 4 k_idx = 64 positions exhaustively.
        assert_eq!(checked, 64, "expected 64 mask positions checked");
    }
}
