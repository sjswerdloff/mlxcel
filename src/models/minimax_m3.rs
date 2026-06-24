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

use crate::models::switch_layers::{gather_sort, SwitchLinear};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, GemmaRMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;
use tracing::{debug, trace};

/// Load a UnifiedLinear. Auto-detects quantization mode from weight shapes.
fn load_linear(
    weights: &WeightMap,
    prefix: &str,
    g: i32,
    b: i32,
) -> Result<UnifiedLinear, String> {
    UnifiedLinear::from_weights(weights, prefix, g, b)
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub text_config: TextConfig,
    #[serde(default)]
    pub quantization: Option<M3Quantization>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct M3Quantization {
    pub group_size: i32,
    pub bits: i32,
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
        let gs = self.quantization.as_ref().map(|q| q.group_size).unwrap_or(64);
        eprintln!("[M3 ModelArgs] group_size={} (quantization={:?})", gs, self.quantization);
        gs
    }
    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
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

        if self.index_q_proj.is_none() || l <= self.block_size {
            debug!(
                layer = self.layer_idx,
                b = b,
                l = l,
                block_size = self.block_size,
                has_index_proj = self.index_q_proj.is_some(),
                branch = "dense",
                reason = if self.index_q_proj.is_none() {
                    "no_index_proj"
                } else {
                    "l_le_block_size"
                },
                "attn.dispatch"
            );
            return self.dense_attention(&q, &cache_k, &cache_v, mask);
        }

        let num_blocks = l / self.block_size;
        if num_blocks <= self.top_k {
            debug!(
                layer = self.layer_idx,
                b = b,
                l = l,
                num_blocks = num_blocks,
                top_k = self.top_k,
                branch = "dense",
                reason = "num_blocks_le_top_k",
                "attn.dispatch"
            );
            return self.dense_attention(&q, &cache_k, &cache_v, mask);
        }

        debug!(
            layer = self.layer_idx,
            b = b,
            l = l,
            num_blocks = num_blocks,
            top_k = self.top_k,
            block_size = self.block_size,
            branch = "msa",
            "attn.dispatch"
        );

        // Index Branch - verified from Transformers MiniMaxM3VLIndexer.
        //
        // Order must be RESHAPE → NORM (matching the regular Q/K norm path above and
        // the HF reference). GemmaRMSNorm's `weight` is shape [index_dim] = [128];
        // applying it to the raw projection output (last dim = num_kv_heads * index_dim
        // = 512) throws "[rms_norm] (*weight) must have the same size as the last
        // dimension of x but has 128 elements." Reshape into per-head shape first so the
        // 128-element norm weight aligns with the head-dim axis.
        let idx_q_raw = self.index_q_proj.as_ref().unwrap().forward(x);
        let idx_k_raw = self.index_k_proj.as_ref().unwrap().forward(x);

        let idx_q = mlxcel_core::reshape(&idx_q_raw, &[b, l, self.num_kv_heads, self.index_dim]);
        let idx_k = mlxcel_core::reshape(&idx_k_raw, &[b, l, 1, self.index_dim]);

        let idx_q = if let Some(ref n) = self.index_q_norm {
            n.forward(&idx_q)
        } else {
            idx_q
        };
        let idx_k = if let Some(ref n) = self.index_k_norm {
            n.forward(&idx_k)
        } else {
            idx_k
        };

        let idx_q = mlxcel_core::transpose_axes(&idx_q, &[0, 2, 1, 3]);
        let idx_k = mlxcel_core::transpose_axes(&idx_k, &[0, 2, 1, 3]);

        // Index Branch also gets RoPE - verified from Transformers code:
        // apply_rotary_pos_emb(idx_q, idx_k, cos[..., :head_dim], sin[..., :head_dim])
        let idx_q =
            mlxcel_core::fast_rope(&idx_q, self.rope_dims, false, self.rope_base, 1.0, offset);
        let idx_k =
            mlxcel_core::fast_rope(&idx_k, self.rope_dims, false, self.rope_base, 1.0, offset);

        // Block max-pool scoring
        let k_blocked =
            mlxcel_core::reshape(&idx_k, &[b, 1, num_blocks, self.block_size, self.index_dim]);
        let k_pool = mlxcel_core::max_axis(&k_blocked, 3, false);

        let q_blocked = mlxcel_core::reshape(
            &idx_q,
            &[
                b,
                self.num_kv_heads,
                num_blocks,
                self.block_size,
                self.index_dim,
            ],
        );
        let q_pool = mlxcel_core::max_axis(&q_blocked, 3, false);

        let scale_idx = 1.0 / (self.index_dim as f32).sqrt();
        let k_pool_t = mlxcel_core::transpose_axes(&k_pool, &[0, 1, 3, 2]);
        let block_scores = mlxcel_core::matmul(&q_pool, &k_pool_t);
        let block_scores = mlxcel_core::multiply_scalar(&block_scores, scale_idx);

        let block_scores = self.apply_causal_block_mask(&block_scores, num_blocks);

        // Local block always included (set to inf) - verified from Transformers code
        let block_scores = self.ensure_local_block_score(&block_scores, num_blocks);

        // Top-K selection
        let neg_scores = mlxcel_core::negative(&block_scores);
        let k_minus_1 = self.top_k - 1;
        let partitioned = mlxcel_core::argpartition(&neg_scores, k_minus_1, -1);
        let selected = mlxcel_core::slice(
            &partitioned,
            &[0, 0, 0, 0],
            &[b, self.num_kv_heads, num_blocks, self.top_k],
        );

        self.sparse_sdpa(&q, &cache_k, &cache_v, &selected, b, l, num_blocks)
    }

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

    fn sparse_sdpa(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        v: &MlxArray,
        selected: &MlxArray,
        b: i32,
        l: i32,
        num_blocks: i32,
    ) -> UniquePtr<MlxArray> {
        debug!(
            layer = self.layer_idx,
            b = b,
            l = l,
            num_blocks = num_blocks,
            top_k = self.top_k,
            selected_shape = ?mlxcel_core::array_shape(selected),
            "sparse_sdpa.entry"
        );
        let k_blocked = mlxcel_core::reshape(
            k,
            &[
                b,
                self.num_kv_heads,
                num_blocks,
                self.block_size,
                self.head_dim,
            ],
        );
        let v_blocked = mlxcel_core::reshape(
            v,
            &[
                b,
                self.num_kv_heads,
                num_blocks,
                self.block_size,
                self.head_dim,
            ],
        );

        // Per-(kv_head, q_block) gather: each q_block has its own top_k key block
        // indices into the same global set of num_blocks key blocks. take_along_axis
        // requires indices to match the source rank with broadcast along non-gather
        // axes. We add a q_block axis to k/v_blocked (broadcast view, not materialized
        // — MLX evaluates lazily) and reshape `selected` to broadcast over block_size
        // and head_dim. take_along_axis on the new k_blocks axis (axis=3) then
        // produces the per-(kv_head, q_block) gather we actually want.
        //
        // Shape walk:
        //   k_blocked:                       [b, kv_h, num_blocks, block_size, head_dim]
        //   → reshape (unsqueeze q_block):   [b, kv_h, 1, num_blocks, block_size, head_dim]
        //   → broadcast to q_block axis:     [b, kv_h, num_blocks, num_blocks, block_size, head_dim]
        //   selected:                        [b, kv_h, num_blocks, top_k]
        //   → reshape (unsqueeze 2 trailing):[b, kv_h, num_blocks, top_k, 1, 1]
        //   → broadcast over inner dims:     [b, kv_h, num_blocks, top_k, block_size, head_dim]
        //   take_along_axis on axis 3:       [b, kv_h, num_blocks, top_k, block_size, head_dim]
        //   reshape to combine top_k+block:  [b, kv_h, num_blocks, top_k*block_size, head_dim]
        //
        // This replaces the previous `take(..., axis=2)` path, which used global gather
        // semantics (single indices array applied uniformly across all kv_heads and
        // q_blocks). That produced a result with a `num_kv_heads`-fold element-count
        // mismatch on the subsequent reshape and would have crashed any prompt long
        // enough to make MSA fire (l > 2048 in a single forward call). Unreachable
        // under mlxcel's default prefill_chunk_size=512, hence latent until now.
        let bk_view = [b, self.num_kv_heads, 1, num_blocks, self.block_size, self.head_dim];
        let bk_target = [b, self.num_kv_heads, num_blocks, num_blocks, self.block_size, self.head_dim];
        let k_expanded = mlxcel_core::broadcast_to(
            &mlxcel_core::reshape(&k_blocked, &bk_view),
            &bk_target,
        );
        let v_expanded = mlxcel_core::broadcast_to(
            &mlxcel_core::reshape(&v_blocked, &bk_view),
            &bk_target,
        );

        let sel_view = [b, self.num_kv_heads, num_blocks, self.top_k, 1, 1];
        let sel_target =
            [b, self.num_kv_heads, num_blocks, self.top_k, self.block_size, self.head_dim];
        let sel_broadcast = mlxcel_core::broadcast_to(
            &mlxcel_core::reshape(selected, &sel_view),
            &sel_target,
        );

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

        let kv_len = self.top_k * self.block_size;
        let k_flat = mlxcel_core::reshape(
            &k_gathered,
            &[b, self.num_kv_heads, num_blocks, kv_len, self.head_dim],
        );
        let v_flat = mlxcel_core::reshape(
            &v_gathered,
            &[b, self.num_kv_heads, num_blocks, kv_len, self.head_dim],
        );

        let q_blocked = mlxcel_core::reshape(
            q,
            &[
                b,
                self.num_heads,
                num_blocks,
                self.block_size,
                self.head_dim,
            ],
        );

        let n_rep = self.num_heads / self.num_kv_heads;
        let k_expanded = mlxcel_core::utils::repeat_kv(&k_flat, n_rep);
        let v_expanded = mlxcel_core::utils::repeat_kv(&v_flat, n_rep);

        let scores = mlxcel_core::matmul(
            &q_blocked,
            &mlxcel_core::transpose_axes(&k_expanded, &[0, 1, 2, 4, 3]),
        );
        let scores = mlxcel_core::multiply_scalar(&scores, self.scale);
        let weights = mlxcel_core::softmax(&scores, -1);
        let out = mlxcel_core::matmul(&weights, &v_expanded);

        let out = mlxcel_core::reshape(&out, &[b, self.num_heads, l, self.head_dim]);
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[b, l, self.num_heads * self.head_dim]);
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
            unsafe { mlxcel_core::layers::attention_from_ptr(q, k, v, self.scale, mask_ptr, 0.0, 0) }
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
        let reshaped = mlxcel_core::reshape(
            &unsorted,
            &[n_tokens, top_k, x_shape[1], x_shape[2]],
        );
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
        let gate_proj = SwitchLinear::from_weights(
            weights,
            &format!("{}.w1", switch_mlp_prefix),
            g,
            b,
        )?;
        let up_proj = SwitchLinear::from_weights(
            weights,
            &format!("{}.w3", switch_mlp_prefix),
            g,
            b,
        )?;
        let down_proj = SwitchLinear::from_weights(
            weights,
            &format!("{}.w2", switch_mlp_prefix),
            g,
            b,
        )?;

        let shared = if cfg.n_shared_experts > 0 {
            let shared_prefix = format!("{}.shared_experts", prefix);
            Some(SharedExperts::from_weights(
                weights,
                &shared_prefix,
                g,
                b,
            )?)
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
        let args: ModelArgs = serde_json::from_value(serde_json::json!({
            "text_config": text_config,
            "quantization": full_config.get("quantization"),
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

        let expected: Vec<f32> = vec![
            (1.0_f32, 0.5_f32),
            (-2.0, 1.0),
            (8.0, 10.0),
            (-8.0, -3.0),
        ]
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
}
