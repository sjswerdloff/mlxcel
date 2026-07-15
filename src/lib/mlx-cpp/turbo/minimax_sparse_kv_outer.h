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

// MSA KV-Outer: shared types for Phase 1 and Phase 2.

#pragma once

#include <mlx/array.h>
#include <memory>

namespace mlxcel::turbo {

// Phase 1 output: partial max, sum_exp, and weighted-V accumulators.
struct KvOuterPartials {
    std::unique_ptr<mlx::core::array> partial_m;  // [B, Hq, L, n_selected] f32
    std::unique_ptr<mlx::core::array> partial_l;  // [B, Hq, L, n_selected] f32
    std::unique_ptr<mlx::core::array> partial_v;  // [B, Hq, L, n_selected, Dim] f32
};

// Phase 1: per-KV-block partial attention with chunked KV loading.
KvOuterPartials minimax_sparse_kv_outer_sdpa(
    const mlx::core::array& q,
    const mlx::core::array& k_blocked,
    const mlx::core::array& v_blocked,
    const mlx::core::array& inverted_index,
    const mlx::core::array& query_counts,
    const mlx::core::array& block_ids,
    float scale,
    int block_size,
    int max_queries_per_block);

// Phase 2: global softmax reduction across partials.
std::unique_ptr<mlx::core::array> minimax_sparse_kv_outer_reduction(
    const mlx::core::array& q,
    const mlx::core::array& partial_m,
    const mlx::core::array& partial_l,
    const mlx::core::array& partial_v,
    int num_key_blocks);

} // namespace mlxcel::turbo
