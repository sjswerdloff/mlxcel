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

// MSA KV-Outer Phase 1: Per-(kv_head, kv_block) partial attention.
//
// Each threadgroup handles ONE (kv_head, selected_block) pair. It loads
// chunks of that KV block into threadgroup memory cooperatively, then
// processes ALL G query heads in that kv_head's GQA group while each
// chunk is resident.
//
// This is the paper's KV-outer IO amortization: KV loaded once per chunk,
// processed by all G replicated query heads.
//
// Thread topology:
//   Grid: (SIMD_WIDTH, NumSims, Hkv * n_selected) — 3D
//   Threadgroup: (SIMD_WIDTH, NumSims, 1)
//   Grid.z = kv_head * n_selected + kv_block_idx
//
// Each SIMD group processes ceil(G / NumSims) query heads from the
// kv_head's GQA group. All SIMD groups share the same kv_head, so
// the cooperative KV load is correct.
//
// Threadgroup memory: tg_k[ChunkSize * Dim] + tg_v[ChunkSize * Dim]
// For Dim=128, ChunkSize=32: 16 KB each, 32 KB total (Apple M3 limit).

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

constexpr const char* KV_OUTER_PHASE1_SOURCE = R"(
    uint lane = thread_position_in_threadgroup.x;    // 0 .. 31
    uint sg = thread_position_in_threadgroup.y;      // 0 .. NumSims-1
    uint tile_idx = threadgroup_position_in_grid.z;

    uint n_selected = (uint)k_blocked_shape[2];
    uint kv_head = tile_idx / n_selected;
    uint kv_block_idx = tile_idx % n_selected;

    uint dim = (uint)Dim;
    uint dpt = (uint)DimsPerThread;
    uint d0 = lane * dpt;
    uint block_size = (uint)BlockSize;
    uint chunk_size = (uint)ChunkSize;
    uint nrep = (uint)NRep;
    uint num_sims = (uint)NumSims;
    uint heads_per_simd = (uint)MaxHeadsPerSimd;

    // This kv_head's G replicated query heads are: kv_head * nrep .. (kv_head+1) * nrep
    // Each SIMD group processes a slice of those G heads.
    uint g_start = kv_head * nrep;
    uint head_start = g_start + sg * heads_per_simd;
    uint head_end = min(head_start + heads_per_simd, g_start + nrep);

    // KV base offset: [B, Hkv, n_selected, BlockSize, Dim]
    uint kv_base = kv_head * n_selected * block_size * dim
                 + kv_block_idx * block_size * dim;

    // Absolute block ID for causal masking.
    uint abs_block_id = (uint)block_ids[kv_block_idx];

    float scale_v = scale[0];

    // Load assigned query heads' Q slices into registers.
    // Only load heads that are within this GQA group's range.
    float q_reg[MaxHeadsPerSimd][DimsPerThread];
    uint num_heads = (head_start < g_start + nrep) ? (head_end - head_start) : 0;
    for (uint hi = 0; hi < num_heads; hi++) {
        uint qh = head_start + hi;
        for (uint j = 0; j < dpt; j++) {
            uint d = d0 + j;
            q_reg[hi][j] = (d < dim) ? q[qh * dim + d] : 0.0f;
        }
    }

    // Online softmax accumulators (in registers, per head).
    // Initialize to neutral values — inactive heads keep these.
    float m[MaxHeadsPerSimd];
    float l[MaxHeadsPerSimd];
    float acc[MaxHeadsPerSimd][DimsPerThread];
    for (uint hi = 0; hi < (uint)MaxHeadsPerSimd; hi++) {
        m[hi] = -INFINITY;
        l[hi] = 0.0f;
        for (uint j = 0; j < dpt; j++) acc[hi][j] = 0.0f;
    }

    // Linear thread ID across all SIMD groups for cooperative load.
    uint linear_tid = sg * 32 + lane;
    uint threads_per_tg = num_sims * 32;

    // Threadgroup memory for chunked KV load.
    threadgroup float tg_k[ChunkSize * Dim];
    threadgroup float tg_v[ChunkSize * Dim];

    // Activity gate: check if ANY head in this GQA group selected this block.
    // Selection is GQA-group-shared (API contract), so checking the first head
    // suffices. The value is uniform across all SIMD groups (same g_start,
    // same kv_block_idx). The barrier ensures all groups have entered.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool tile_active = (query_counts[g_start * n_selected + kv_block_idx] > 0);

    // Inactive tile: write neutral partials and return immediately.
    // SAFE because tile_active is uniform across the entire threadgroup —
    // all SIMD groups agree, so no divergent barrier issue.
    if (!tile_active) {
        // Neutral partials are already set by the zero-init output.
        // Just need to write m=-INF for lane 0 of each head.
        for (uint hi = 0; hi < num_heads; hi++) {
            uint qh = head_start + hi;
            uint partial_base = qh * n_selected + kv_block_idx;
            if (lane == 0u) {
                partial_m[partial_base] = -INFINITY;
                partial_l[partial_base] = 0.0f;
            }
            // partial_v stays at init_value (0) — no write needed.
        }
        return;
    }

    // Active tile: load KV chunks and process query heads.
    for (uint chunk_start = 0; chunk_start < block_size; chunk_start += chunk_size) {
        uint chunk_end = min(chunk_start + chunk_size, block_size);
        uint actual_chunk = chunk_end - chunk_start;
        uint chunk_elems = actual_chunk * dim;

        // Cooperatively load this chunk into threadgroup memory.
        uint chunk_base = kv_base + chunk_start * dim;
        for (uint i = linear_tid; i < chunk_elems; i += threads_per_tg) {
            uint tok = i / dim;
            uint d = i % dim;
            tg_k[tok * dim + d] = (float)k_blocked[chunk_base + tok * dim + d];
            tg_v[tok * dim + d] = (float)v_blocked[chunk_base + tok * dim + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Process assigned query heads against this chunk.
        for (uint hi = 0; hi < num_heads; hi++) {
            uint qh = head_start + hi;

            // Check if this (head, block) pair is actually selected.
            uint head_block_idx = qh * n_selected + kv_block_idx;
            uint count = (uint)query_counts[head_block_idx];
            if (count == 0) continue;

            for (uint t = 0; t < actual_chunk; t++) {
                uint tok_base = t * dim;

                // Q · K_t via simd_sum across 32 lanes.
                float partial = 0.0f;
                for (uint j = 0; j < dpt; j++) {
                    uint d = d0 + j;
                    float kd = (d < dim) ? tg_k[tok_base + d] : 0.0f;
                    partial += q_reg[hi][j] * kd;
                }
                float score = simd_sum(partial) * scale_v;

                // Causal mask.
                uint abs_tok_pos = abs_block_id * block_size + chunk_start + t;
                if (abs_tok_pos > (uint)inverted_index[head_block_idx]) {
                    score = -INFINITY;
                }

                // Online softmax — skip masked tokens entirely.
                if (isfinite(score)) {
                    float m_new = fmax(m[hi], score);
                    float corr = fast::exp(m[hi] - m_new);
                    float p = fast::exp(score - m_new);
                    l[hi] = l[hi] * corr + p;
                    for (uint j = 0; j < dpt; j++) {
                        uint d = d0 + j;
                        float vd = (d < dim) ? tg_v[tok_base + d] : 0.0f;
                        acc[hi][j] = acc[hi][j] * corr + p * vd;
                    }
                    m[hi] = m_new;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partials to global memory for all assigned query heads.
    for (uint hi = 0; hi < num_heads; hi++) {
        uint qh = head_start + hi;
        uint partial_base = qh * n_selected + kv_block_idx;

        for (uint j = 0; j < dpt; j++) {
            uint d = d0 + j;
            if (d < dim) {
                partial_v[partial_base * dim + d] = acc[hi][j];
            }
        }
        if (lane == 0u) {
            partial_m[partial_base] = m[hi];
            partial_l[partial_base] = l[hi];
        }
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

    // Validate decode-only contract.
    if (batch != 1 || q_len != 1) {
        throw std::runtime_error(
            "kv_outer_phase1: decode only (B=1, L=1), got B="
            + std::to_string(batch) + " L=" + std::to_string(q_len));
    }
    if (hkv == 0 || hq % hkv != 0) {
        throw std::runtime_error(
            "kv_outer_phase1: Hq must be divisible by Hkv, got Hq="
            + std::to_string(hq) + " Hkv=" + std::to_string(hkv));
    }
    if (max_queries_per_block != 1) {
        throw std::runtime_error(
            "kv_outer_phase1: max_queries_per_block must be 1, got "
            + std::to_string(max_queries_per_block));
    }

    int n_rep = hq / hkv;

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
    int heads_per_simd = (n_rep + num_sims - 1) / num_sims;

    std::vector<std::pair<std::string, TemplateArg>> template_args = {
        {"Dim", dim},
        {"BlockSize", block_size},
        {"ChunkSize", chunk_size},
        {"MaxQueriesPerBlock", max_queries_per_block},
        {"NRep", n_rep},
        {"DimsPerThread", dims_per_thread},
        {"NumSims", num_sims},
        {"MaxHeadsPerSimd", heads_per_simd},
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

    // Output shapes: [B, Hq, L, n_selected] for decode (B=1, L=1).
    std::vector<Shape> output_shapes = {
        Shape{1, hq, 1, n_selected},        // partial_m
        Shape{1, hq, 1, n_selected},        // partial_l
        Shape{1, hq, 1, n_selected, dim},   // partial_v
    };
    std::vector<Dtype> output_dtypes = {
        mlx::core::float32,
        mlx::core::float32,
        mlx::core::float32,
    };

    // Grid: (SIMD_WIDTH, NumSims, Hkv * n_selected) — one threadgroup per (kv_head, block).
    // Each SIMD group processes ceil(G / NumSims) query heads from that kv_head's GQA group.
    int total_tiles = hkv * n_selected;
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
