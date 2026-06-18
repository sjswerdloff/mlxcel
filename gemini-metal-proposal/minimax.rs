use mlx_rs::nn::{QuantizedLinear, Module};
use mlx_rs::array::Array;
use crate::ffi::{sparse_topk, block_sparse_sdpa};

pub struct MiniMaxSparseAttentionBlock {
    pub q_proj: QuantizedLinear,
    pub k_proj: QuantizedLinear,
    pub v_proj: QuantizedLinear,
    pub o_proj: QuantizedLinear,
    pub num_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
    pub top_k: i32,
}

impl MiniMaxSparseAttentionBlock {
    pub fn new(config: &MiniMaxConfig) -> Self {
        Self {
            // mlxcel's QuantizedLinear automatically handles the fp8_e8m0fnu scale tensors
            q_proj: QuantizedLinear::new(config.hidden_size, config.hidden_size, false, 8),
            k_proj: QuantizedLinear::new(config.hidden_size, config.kv_hidden_size, false, 8),
            v_proj: QuantizedLinear::new(config.hidden_size, config.kv_hidden_size, false, 8),
            o_proj: QuantizedLinear::new(config.hidden_size, config.hidden_size, false, 8),
            num_heads: config.num_attention_heads,
            head_dim: config.hidden_size / config.num_attention_heads,
            block_size: 128,
            top_k: config.msa_top_k,
        }
    }

    /// The Forward Pass logic triggered by the mlxcel execution scheduler
    pub fn forward(&self, x: &Array, mask: Option<&Array>, cache: Option<&mut KVCache>) -> Array {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        // 1. MatMul via MXFP8 Dequantization
        // Rust routes this to MLX, which seamlessly dequantizes FP8 -> FP16 in SRAM
        let q = self.q_proj.forward(x).reshape(&[batch_size, seq_len, self.num_heads, self.head_dim]);
        let mut k = self.k_proj.forward(x).reshape(&[batch_size, seq_len, -1, self.head_dim]);
        let mut v = self.v_proj.forward(x).reshape(&[batch_size, seq_len, -1, self.head_dim]);

        // 2. Cache management (Append new tokens to unified memory cache)
        if let Some(c) = cache {
            let (new_k, new_v) = c.update(&k, &v);
            k = new_k;
            v = new_v;
        }

        // 3. Block relevance metric pooling
        // Using MLX native ops to mean-pool the sequence dimension into block sizes
        let q_pool = q.reshape(&[batch_size, -1, self.block_size, self.num_heads, self.head_dim]).mean(&[2]);
        let k_pool = k.reshape(&[batch_size, -1, self.block_size, k.shape()[2], self.head_dim]).mean(&[2]);
        let block_metrics = mlx_rs::ops::matmul(&q_pool, &k_pool.transpose(&[0, 1, 3, 2]));

        // 4. FFI Call to Custom Metal Indexer (Parallel Bitonic Sort)
        let active_indices = sparse_topk(&block_metrics, self.top_k);

        // 5. FFI Call to Custom Metal Block-Sparse SDPA
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let out = block_sparse_sdpa(&q, &k, &v, &active_indices, scale);

        // 6. Final output projection
        self.o_proj.forward(&out.reshape(&[batch_size, seq_len, -1]))
    }
}
