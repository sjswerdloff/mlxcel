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

// MSA KV-Outer Phase 1: Per-KV-block partial attention.
//
// Each threadgroup loads ONE KV block into threadgroup memory in chunks.
// The chunk loop is outermost: each chunk is loaded ONCE cooperatively
// by all SIMD groups, then all assigned query heads process it while
// it's resident. This is the paper's KV-outer IO amortization.
//
// Thread topology:
//   Grid: (SIMD_WIDTH, NumSims, Hq * n_selected) — 3D
//   Threadgroup: (SIMD_WIDTH, NumSims, 1)
//   Grid.z flattens (q_head, block_idx): q_head = flat / n_selected,
//                                         block_idx = flat % n_selected
//
// Each SIMD group independently processes its query head. No cross-SIMD
// barriers needed — accumulators live in registers, partials are written
// directly to global memory. Phase 2 handles the global reduction.
//
// Threadgroup memory: tg_k[ChunkSize * Dim] + tg_v[ChunkSize * Dim]
// For Dim=128, ChunkSize=16: 16 KB each, 32 KB total (Apple M3 limit).

#include "minimax_sparse_kv_outer.h"

#include <mlx/fast.h>
#include <mlx/ops.h>

#include <algorithm>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mlxcel::turbo {

namespace {

constexpr int SIMD_WIDTH = 32;

// Phase 1 kernel: chunk-outermost KV-outer block-sparse attention.
constexpr const char* KV_OUTER_PHASE1_SOURCE = R"(
    uint lane = thread_position_in_threadgroup.x;    // 0 .. 31
    uint sg = thread_position_in_threadgroup.y;      // 0 .. NumSims-1
    uint flat_idx = threadgroup_position_in_grid.z;

    uint n_selected = (uint)k_blocked_shape[2];
    uint q_head_idx = flat_idx / n_selected;
    uint kv_block_idx = flat_idx % n_selected;

    uint dim = (uint)Dim;
    uint dpt = (uint)DimsPerThread;
    uint d0 = lane * dpt;
    uint block_size = (uint)BlockSize;
    uint chunk_size = (uint)ChunkSize;
    uint hq = (uint)q_shape[1];

    // Bounds guard.
    if (q_head_idx >= hq) return;

    // Per-head query count from the per-query-head inverted index.
    uint head_block_idx = q_head_idx * n_selected + kv_block_idx;
    uint num_queries = (uint)query_counts[head_block_idx];
    uint idx_offset = head_block_idx * (uint)MaxQueriesPerBlock;

    // If no queries for this (head, block), write neutral partials.
    if (num_queries == 0) {
        uint partial_base = 0 * n_selected + kv_block_idx;
        partial_m[q_head_idx * n_selected + partial_base] = -INFINITY;
        partial_l[q_head_idx * n_selected + partial_base] = 0.0f;
        // partial_v stays at init_value (0).
        return;
    }

    // KV base offset in blocked layout: [B, Hkv, n_selected, BlockSize, Dim]
    // Hkv is derived from the GQA mapping: kv_head = q_head_idx / NRep.
    uint nrep = (uint)NRep;
    uint kv_head = q_head_idx / nrep;
    uint kv_base = kv_head * n_selected * block_size * dim
                 + kv_block_idx * block_size * dim;

    // Absolute block ID for causal masking.
    uint abs_block_id = (uint)block_ids[kv_block_idx];

    float scale_v = scale[0];

    // Load this query head's Q slice into registers.
    float q_reg[DimsPerThread];
    for (uint j = 0; j < dpt; j++) {
        uint d = d0 + j;
        q_reg[j] = (d < dim) ? q[q_head_idx * dim + d] : 0.0f;
    }

    // Online softmax accumulators (in registers, per SIMD group).
    float m = -INFINITY;
    float l = 0.0f;
    float acc[DimsPerThread];
    for (uint j = 0; j < dpt; j++) acc[j] = 0.0f;

    // Linear thread ID across all SIMD groups for cooperative load.
    uint linear_tid = sg * 32 + lane;
    uint threads_per_tg = (uint)NumSims * 32;

    // Threadgroup memory for chunked KV load.
    threadgroup float tg_k[ChunkSize * Dim];
    threadgroup float tg_v[ChunkSize * Dim];

    // Outer loop: over chunks of the KV block.
    // Each chunk is loaded ONCE, then all heads process it.
    for (uint chunk_start = 0; chunk_start < block_size; chunk_start += chunk_size) {
        uint chunk_end = min(chunk_start + chunk_size, block_size);
        uint actual_chunk = chunk_end - chunk_start;
        uint chunk_elems = actual_chunk * dim;

        // Cooperatively load this chunk into threadgroup memory.
        // All threads in the threadgroup participate (linear_tid).
        uint chunk_base = kv_base + chunk_start * dim;
        for (uint i = linear_tid; i < chunk_elems; i += threads_per_tg) {
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

            // Causal mask: mask future tokens to -inf.
            uint abs_tok_pos = abs_block_id * block_size + chunk_start + t;
            if (abs_tok_pos > (uint)inverted_index[idx_offset]) {
                score = -INFINITY;
            }

            // Online softmax — skip masked tokens entirely.
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

    // Write partials to global memory.
    // For decode (L=1), partial_buf_idx = 0.
    uint partial_buf_idx = 0;
    uint partial_base = partial_buf_idx * n_selected + kv_block_idx;

    for (uint j = 0; j < dpt; j++) {
        uint d = d0 + j;
        if (d < dim) {
            partial_v[q_head_idx * n_selected * dim + partial_base * dim + d] = acc[j];
        }
    }
    if (lane == 0u) {
        partial_m[q_head_idx * n_selected + partial_base] = m;
        partial_l[q_head_idx * n_selected + partial_base] = l;
    }
)";

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

inline Phase1KernelHolder& get_phase1_kernel() {
    static Phase1KernelHolder holder;
    return holder;
}

} // namespace

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

    const auto& q_shape = q.shape();
    const auto& k_shape = k_blocked.shape();

    int batch = q_shape[0];
    int hq = q_shape[1];
    int q_len = q_shape[2];
    int dim = q_shape[3];
    int hkv = k_shape[1];
    int n_selected = k_shape[2];

    int n_rep = hkv > 0 ? hq / hkv : 1;
    if (n_rep < 1) n_rep = 1;

    auto& kernel = get_phase1_kernel().get();

    int dims_per_thread = (dim + SIMD_WIDTH - 1) / SIMD_WIDTH;

    // ChunkSize: how many tokens fit in threadgroup memory alongside tg_k/tg_v.
    // Budget: ChunkSize * Dim * 2 * sizeof(float) ≤ 32 KB
    // For Dim=128: ChunkSize ≤ 32. For Dim=64: ChunkSize ≤ 64.
    int chunk_size = 32768 / (2 * dim * 4);
    if (chunk_size > block_size) chunk_size = block_size;
    if (chunk_size < 1) chunk_size = 1;
    // Round down to power of 2.
    int cs = 1;
    while (cs * 2 <= chunk_size) cs *= 2;
    chunk_size = cs;

    int num_sims = 4;

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
        inverted_index, // [B, Hq, n_selected, MaxQPB]               i32
        query_counts,   // [B, Hq, n_selected]                       i32
        block_ids,      // [n_selected]                               i32
        scale_arr,      // [1]                                        f32
    };

    std::vector<Shape> output_shapes = {
        Shape{batch, hq, q_len, n_selected},        // partial_m
        Shape{batch, hq, q_len, n_selected},        // partial_l
        Shape{batch, hq, q_len, n_selected, dim},   // partial_v
    };
    std::vector<Dtype> output_dtypes = {
        mlx::core::float32,
        mlx::core::float32,
        mlx::core::float32,
    };

    // Grid: (SIMD_WIDTH, NumSims, Hq * n_selected) — 3D.
    // Each threadgroup handles one (q_head, block) pair.
    int total_tiles = hq * n_selected;
    auto results = kernel(
        inputs,
        output_shapes,
        output_dtypes,
        std::make_tuple(SIMD_WIDTH, num_sims, total_tiles),
        std::make_tuple(SIMD_WIDTH, num_sims, 1),
        template_args,
        std::optional<float>(0.0f),  // Zero-init outputs (partial_v must be 0 for zero-query slots).
        false,
        {});

    KvOuterPartials out;
    out.partial_m = std::make_unique<mlx::core::array>(std::move(results[0]));
    out.partial_l = std::make_unique<mlx::core::array>(std::move(results[1]));
    out.partial_v = std::make_unique<mlx::core::array>(std::move(results[2]));
    return out;
}

} // namespace mlxcel::turbo
