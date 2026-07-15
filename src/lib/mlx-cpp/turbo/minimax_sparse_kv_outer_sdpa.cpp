// Copyright 2026 Lablup Inc. and Jeongkyu Shin
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

// MSA KV-Outer Block-Sparse Decode Attention.
//
// Two-phase kernel following the paper's KV-outer iteration.
//
// Phase 1: Each threadgroup loads ONE KV block into threadgroup memory
// (SRAM) in chunks, then iterates over the inverted index to process all
// queries that attend to this block. Writes partial max, sum_exp, and
// weighted-V accumulators to global memory.
//
// Phase 2: For each query, merges all Phase 1 partials into the final
// normalized output using flash-attention rescale.
//
// Threadgroup memory budget (Apple M3: 32 KB per threadgroup):
//   tg_k[ChunkSize * Dim] in fp16 — staged K chunk
//   tg_v[ChunkSize * Dim] in fp16 — staged V chunk
//   tg_m[NumSims] + tg_l[NumSims] + tg_acc[NumSims * Dim] — combine scratch
//
// With ChunkSize=64, Dim=128: tg_k=16 KB, tg_v=16 KB, combine=~1.5 KB.
// Total: ~33.5 KB — tight but fits with Dim=128. For Dim=64, ChunkSize=128
// (full block in one load).

#include "minimax_sparse_kv_outer.h"

#include <mlx/fast.h>
#include <mlx/ops.h>

#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mlxcel::turbo {

namespace {

// Phase 1: Per-KV-block partial attention.
//
// Thread topology:
//   Grid Z = KV block index (0 .. num_key_blocks - 1)
//   Grid Y = KV head (0 .. Hkv - 1)
//   Grid X = 32 lanes × NumSims SIMD groups
//
// Each threadgroup handles ONE (kv_block, kv_head) pair. The KV block is
// loaded into threadgroup memory in chunks. For each chunk, all queries
// in the inverted index are processed — computing per-query-head Q·K dot
// products via simd_sum, online softmax, and weighted-V accumulation.
//
// The inverted index tells us which query positions attend to this block.
// For decode (L=1), there's typically 1 query position per block. For
// prefill, there can be many.
constexpr const char* KV_OUTER_PHASE1_SOURCE = R"(
    uint lane = thread_position_in_threadgroup.x;    // 0 .. 31
    uint sg = thread_position_in_threadgroup.y;      // 0 .. NumSims-1
    // Grid.z is a flattened (kv_head, kv_block_idx) pair.
    uint flat_idx = threadgroup_position_in_grid.z;
    uint n_key_blocks = (uint)k_blocked_shape[2];
    uint kv_head = flat_idx / n_key_blocks;
    uint kv_block_idx = flat_idx % n_key_blocks;

    uint dim = (uint)Dim;
    uint dpt = (uint)DimsPerThread;
    uint d0 = lane * dpt;
    uint block_size = (uint)BlockSize;
    uint chunk_size = (uint)ChunkSize;
    uint nrep = (uint)NRep;
    uint max_qpb = (uint)MaxQueriesPerBlock;

    // Absolute block ID for this compact index. Used for causal masking.
    // block_ids maps compact index → absolute block ID.
    uint abs_block_id = (uint)block_ids[kv_block_idx];

    // Query counts and inverted index for this block.
    // query_counts is [B, Hkv, n_selected], inv_index is [B, Hkv, n_selected, MaxQPB].
    // Must offset by kv_head to read the correct head's data.
    uint head_block_idx = kv_head * n_key_blocks + kv_block_idx;
    uint num_queries = (uint)query_counts[head_block_idx];
    uint idx_offset = head_block_idx * max_qpb;

    // Base offset in the blocked KV layout: [B, Hkv, num_key_blocks, BlockSize, Dim]
    uint kv_base = kv_head * n_key_blocks * block_size * dim
                 + kv_block_idx * block_size * dim;

    float scale_v = scale[0];

    // Threadgroup memory for chunked KV load.
    threadgroup float tg_k[ChunkSize * Dim];
    threadgroup float tg_v[ChunkSize * Dim];

    // Per-SIMD-group online-softmax accumulators (Dim-wide for flash combine).
    threadgroup float tg_acc[NumSims * Dim];

    // This SIMD group's query head range.
    uint heads_per_simd = (nrep + (uint)NumSims - 1) / (uint)NumSims;
    uint h_start = sg * heads_per_simd;
    uint h_end = min(h_start + heads_per_simd, nrep);

    // If this block has no queries for this head, write neutral partials
    // (m=-inf, l=0) so Phase 2 contributes nothing from this block.
    if (num_queries == 0) {
        for (uint rh = h_start; rh < h_end; rh++) {
            uint q_head_idx = kv_head * nrep + rh;
            uint partial_base = 0 * n_key_blocks + kv_block_idx;
            if (lane == 0u) {
                partial_m[q_head_idx * n_key_blocks + partial_base] = -INFINITY;
                partial_l[q_head_idx * n_key_blocks + partial_base] = 0.0f;
            }
        }
        return;
    }

    // Initialize accumulators for each query head in this SIMD group's range.
    // We process query heads sequentially — for each, we sweep all chunks.
    for (uint rh = h_start; rh < h_end; rh++) {
        uint q_head_idx = kv_head * nrep + rh;

        // Load this query head's Q slice into registers.
        float q_reg[DimsPerThread];
        for (uint j = 0; j < dpt; j++) {
            uint d = d0 + j;
            q_reg[j] = (d < dim) ? q[q_head_idx * dim + d] : 0.0f;
        }

        // Online softmax state.
        float m = -INFINITY;
        float l = 0.0f;
        float acc[DimsPerThread];
        for (uint j = 0; j < dpt; j++) acc[j] = 0.0f;

        // Process each query in the inverted index for this block.
        for (uint qi = 0; qi < num_queries; qi++) {
            uint q_pos = (uint)inverted_index[idx_offset + qi];

            // Sweep the KV block in chunks.
            for (uint chunk_start = 0; chunk_start < block_size; chunk_start += chunk_size) {
                uint chunk_end = min(chunk_start + chunk_size, block_size);
                uint actual_chunk = chunk_end - chunk_start;

                // Cooperatively load this chunk into threadgroup memory.
                uint chunk_elems = actual_chunk * dim;
                uint chunk_base = kv_base + chunk_start * dim;
                for (uint i = lane; i < chunk_elems; i += 32) {
                    uint tok = i / dim;
                    uint d = i % dim;
                    uint off = chunk_base + tok * dim + d;
                    tg_k[tok * dim + d] = (float)k_blocked[off];
                    tg_v[tok * dim + d] = (float)v_blocked[off];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Process all tokens in this chunk.
                for (uint t = 0; t < actual_chunk; t++) {
                    uint tok_base = t * dim;

                    // Q · K_t via simd_sum across 32 lanes.
                    float partial = 0.0f;
                    for (uint j = 0; j < dpt; j++) {
                        uint d = d0 + j;
                        float kd = (d < dim) ? tg_k[tok_base + d] : 0.0f;
                        partial += q_reg[j] * kd;
                    }
                    float score = simd_sum(partial) * scale_v;

                    // Causal mask: mask future tokens to -inf instead of
                    // skipping (continue would deadlock on threadgroup_barrier).
                    uint abs_tok_pos = abs_block_id * block_size + chunk_start + t;
                    if (abs_tok_pos > (uint)q_pos) {
                        score = -INFINITY;
                    }

                    // Online softmax with rescale.
                    // Skip contribution entirely for masked (-inf) tokens.
                    if (isfinite(score)) {
                        float m_new = fmax(m, score);
                        float corr = fast::exp(m - m_new);
                        float p = fast::exp(score - m_new);
                        l = l * corr + p;
                        for (uint j = 0; j < dpt; j++) {
                            uint d = d0 + j;
                            float vd = (d < dim) ? tg_v[tok_base + d] : 0.0f;
                            acc[j] = acc[j] * corr + p * vd;
                        }
                        m = m_new;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Write partials to global memory for this (q_head, q_pos, kv_block).
            // partial_m/l/v shapes: [B, Hq, L, n_key_blocks]
            // For decode (L=1), the partial buffer index is 0 (not the absolute q_pos).
            // Bounds check: q_head_idx must be < Hq to prevent OOB writes.
            uint hq_limit = (uint)q_shape[1];
            if (q_head_idx < hq_limit) {
                uint partial_buf_idx = 0;
                uint partial_base = partial_buf_idx * n_key_blocks + kv_block_idx;
                uint partial_vec_base = partial_buf_idx * n_key_blocks * dim + kv_block_idx * dim;

                for (uint j = 0; j < dpt; j++) {
                    uint d = d0 + j;
                    if (d < dim) {
                        partial_v[q_head_idx * n_key_blocks * dim + partial_vec_base + d] = acc[j];
                    }
                }
                if (lane == 0u) {
                    partial_m[q_head_idx * n_key_blocks + partial_base] = m;
                    partial_l[q_head_idx * n_key_blocks + partial_base] = l;
                }
            }

            // Reset for next query head (only if more to process).
            if (rh + 1 < h_end) {
                m = -INFINITY;
                l = 0.0f;
                for (uint j = 0; j < dpt; j++) acc[j] = 0.0f;
            }
        }
    }
)";

// Phase 2: Global softmax reduction.
//
// Thread topology:
//   Grid Z = Query Head (0 .. Hq - 1)
//   Grid Y = Query Position (0 .. L - 1)
//   Grid X = lanes partitioning head dimension (Dim)
//
// For each query, sweeps across all num_key_blocks partials from Phase 1,
// computes the global maximum, rescales partials, and produces the final
// normalized attention output.
constexpr const char* KV_OUTER_PHASE2_SOURCE = R"(
    uint d = thread_position_in_grid.x;
    uint q_pos = thread_position_in_grid.y;
    uint hq = thread_position_in_grid.z;

    uint num_blocks = (uint)num_key_blocks_arr[0];
    uint dim = (uint)Dim;

    uint q_len = (uint)q_shape[2];

    // Strides for partial buffers: [B, Hq, L, num_key_blocks]
    uint scalar_stride = num_blocks;
    uint vec_stride = num_blocks * dim;

    uint base_scalar = hq * q_len * scalar_stride + q_pos * scalar_stride;
    uint base_vec = hq * q_len * vec_stride + q_pos * vec_stride;

    // Pass 2a: find global max across all partial blocks.
    float global_max = -INFINITY;
    for (uint b = 0; b < num_blocks; b++) {
        float pm = (float)partial_m[base_scalar + b];
        if (pm > global_max) global_max = pm;
    }
    if (!isfinite(global_max)) global_max = 0.0f;

    // Pass 2b: compute global denominator.
    float global_sum_exp = 0.0f;
    for (uint b = 0; b < num_blocks; b++) {
        float pm = (float)partial_m[base_scalar + b];
        float pl = (float)partial_l[base_scalar + b];
        global_sum_exp += pl * metal::fast::exp(pm - global_max);
    }
    if (!(global_sum_exp > 0.0f) || !isfinite(global_sum_exp)) global_sum_exp = 1.0f;
    float inv_denom = 1.0f / global_sum_exp;

    // Pass 2c: accumulate weighted V and write output.
    if (d < dim) {
        float v_acc = 0.0f;
        for (uint b = 0; b < num_blocks; b++) {
            float pm = (float)partial_m[base_scalar + b];
            float pv = (float)partial_v[base_vec + b * dim + d];
            v_acc += pv * metal::fast::exp(pm - global_max);
        }
        uint out_offset = hq * q_len * dim + q_pos * dim + d;
        final_output[out_offset] = v_acc * inv_denom;
    }
)";

constexpr int SIMD_WIDTH = 32;

// Thread-safe lazy-init holders.
struct Phase1KernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;
    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = mlx::core::fast::metal_kernel(
                "msa_kv_outer_phase1",
                {"q", "k_blocked", "v_blocked", "inverted_index", "query_counts", "block_ids", "scale"},
                {"partial_m", "partial_l", "partial_v"},
                std::string(KV_OUTER_PHASE1_SOURCE));
        });
        return *kernel;
    }
};

struct Phase2KernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;
    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = mlx::core::fast::metal_kernel(
                "msa_kv_outer_phase2",
                {"partial_m", "partial_l", "partial_v", "num_key_blocks_arr", "q_shape"},
                {"final_output"},
                std::string(KV_OUTER_PHASE2_SOURCE));
        });
        return *kernel;
    }
};

inline Phase1KernelHolder& get_phase1_kernel() {
    static Phase1KernelHolder holder;
    return holder;
}

inline Phase2KernelHolder& get_phase2_kernel() {
    static Phase2KernelHolder holder;
    return holder;
}

} // namespace

// Phase 1 launcher.
KvOuterPartials minimax_sparse_kv_outer_sdpa(
    const mlx::core::array& q,
    const mlx::core::array& k_blocked,
    const mlx::core::array& v_blocked,
    const mlx::core::array& inverted_index,
    const mlx::core::array& query_counts,
    const mlx::core::array& block_ids,
    float scale,
    int block_size,
    int max_queries_per_block) {
    using mlx::core::Dtype;
    using mlx::core::Shape;
    using mlx::core::fast::TemplateArg;

    const auto& q_shape = q.shape();         // [B, Hq, L, Dim]
    const auto& k_shape = k_blocked.shape(); // [B, Hkv, num_key_blocks, BlockSize, Dim]

    int batch = q_shape[0];
    int hq = q_shape[1];
    int q_len = q_shape[2];
    int dim = q_shape[3];
    int hkv = k_shape[1];
    int num_key_blocks = k_shape[2];

    int n_rep = hkv > 0 ? hq / hkv : 1;
    if (n_rep < 1) n_rep = 1;

    auto& kernel = get_phase1_kernel().get();

    int dims_per_thread = (dim + SIMD_WIDTH - 1) / SIMD_WIDTH;

    // ChunkSize: how many tokens we load into threadgroup memory at once.
    // Budget: 2 * ChunkSize * Dim * sizeof(float) + NumSims * Dim * 4 + overhead
    // With Dim=128, NumSims=4: tg_acc = 2048 bytes. Remaining: 30720 bytes.
    // ChunkSize = 30720 / (2 * 128 * 4) = 30. Use 32 (power of 2).
    // For Dim=64: ChunkSize = 30720 / (2 * 64 * 4) = 60. Use 64.
    int chunk_size = 30720 / (2 * dim * 4);
    if (chunk_size > block_size) chunk_size = block_size;
    if (chunk_size < 1) chunk_size = 1;
    // Round down to power of 2 for clean division.
    int cs = 1;
    while (cs * 2 <= chunk_size) cs *= 2;
    chunk_size = cs;

    int num_sims = 4;  // SIMD groups per threadgroup.

    std::vector<std::pair<std::string, TemplateArg>> template_args = {
        {"Dim", dim},
        {"BlockSize", block_size},
        {"ChunkSize", chunk_size},
        {"MaxQueriesPerBlock", max_queries_per_block},
        {"NRep", n_rep},
        {"DimsPerThread", dims_per_thread},
        {"NumSims", num_sims},
    };

    auto scale_arr = mlx::core::full(mlx::core::Shape{1}, scale, mlx::core::float32);

    std::vector<mlx::core::array> inputs = {
        q,              // [B, Hq, L, Dim]                           f32
        k_blocked,      // [B, Hkv, n_selected, BlockSize, Dim]      f16
        v_blocked,      // [B, Hkv, n_selected, BlockSize, Dim]      f16
        inverted_index, // [B, Hkv, n_selected, MaxQPB]              i32
        query_counts,   // [B, Hkv, n_selected]                      i32
        block_ids,      // [n_selected]                               i32 — compact → absolute
        scale_arr,      // [1]                                        f32
    };

    // Partial output shapes.
    std::vector<Shape> output_shapes = {
        Shape{batch, hq, q_len, num_key_blocks},        // partial_m
        Shape{batch, hq, q_len, num_key_blocks},        // partial_l
        Shape{batch, hq, q_len, num_key_blocks, dim},   // partial_v
    };
    std::vector<Dtype> output_dtypes = {
        mlx::core::float32,
        mlx::core::float32,
        mlx::core::float32,
    };

    // Grid: (32, NumSims, Hkv * num_key_blocks) — 3D (API limit).
    // Metal shader decodes Grid.z into (kv_head, kv_block_idx).
    int total_tiles = hkv * num_key_blocks;
    auto results = kernel(
        inputs,
        output_shapes,
        output_dtypes,
        std::make_tuple(SIMD_WIDTH, num_sims, total_tiles),
        std::make_tuple(SIMD_WIDTH, num_sims, 1),
        template_args,
        std::optional<float>(0.0f),  // Zero-initialize all outputs (partial_v must be 0 for zero-query slots)
        false,
        {});

    KvOuterPartials out;
    out.partial_m = std::make_unique<mlx::core::array>(std::move(results[0]));
    out.partial_l = std::make_unique<mlx::core::array>(std::move(results[1]));
    out.partial_v = std::make_unique<mlx::core::array>(std::move(results[2]));
    return out;
}

// Phase 2 launcher.
std::unique_ptr<mlx::core::array> minimax_sparse_kv_outer_reduction(
    const mlx::core::array& q,
    const mlx::core::array& partial_m,
    const mlx::core::array& partial_l,
    const mlx::core::array& partial_v,
    int num_key_blocks) {
    using mlx::core::Dtype;
    using mlx::core::Shape;
    using mlx::core::fast::TemplateArg;

    const auto& q_shape = q.shape();
    int batch = q_shape[0];
    int hq = q_shape[1];
    int q_len = q_shape[2];
    int dim = q_shape[3];

    auto& kernel = get_phase2_kernel().get();

    std::vector<std::pair<std::string, TemplateArg>> template_args = {
        {"Dim", dim},
    };

    auto blocks_arr = mlx::core::full(mlx::core::Shape{1}, num_key_blocks, mlx::core::int32);
    std::vector<int32_t> q_shape_raw = {batch, hq, q_len, dim};
    auto q_shape_arr = mlx::core::array(q_shape_raw.data(), mlx::core::Shape{4}, mlx::core::int32);

    std::vector<mlx::core::array> inputs = {
        partial_m,
        partial_l,
        partial_v,
        blocks_arr,
        q_shape_arr,
    };

    std::vector<Shape> output_shapes = {Shape{batch, hq, q_len, dim}};
    std::vector<Dtype> output_dtypes = {mlx::core::float32};

    int tg_x = dim;
    if (tg_x > 1024) tg_x = 1024;

    auto results = kernel(
        inputs,
        output_shapes,
        output_dtypes,
        std::make_tuple(dim, q_len, hq),
        std::make_tuple(tg_x, 1, 1),
        template_args,
        std::nullopt,
        false,
        {});

    return std::make_unique<mlx::core::array>(std::move(results[0]));
}

} // namespace mlxcel::turbo
