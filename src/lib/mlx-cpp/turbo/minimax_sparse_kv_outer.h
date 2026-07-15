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
// Two-phase kernel following the paper's KV-outer iteration:
//   Phase 1: Per-KV-block partial attention. Each threadgroup loads ONE KV
//            block into threadgroup memory (SRAM) in chunks, then iterates
//            over the inverted index to process all queries that attend to
//            this block. Writes partial max, sum_exp, and weighted-V
//            accumulators to global memory.
//   Phase 2: Global softmax reduction. For each query, merges all partials
//            from Phase 1 into the final normalized output.

#pragma once

#include <mlx/array.h>
#include <memory>

namespace mlxcel::turbo {

struct KvOuterPartials {
    std::unique_ptr<mlx::core::array> partial_m;
    std::unique_ptr<mlx::core::array> partial_l;
    std::unique_ptr<mlx::core::array> partial_v;
};

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

std::unique_ptr<mlx::core::array> minimax_sparse_kv_outer_reduction(
    const mlx::core::array& q,
    const mlx::core::array& partial_m,
    const mlx::core::array& partial_l,
    const mlx::core::array& partial_v,
    int num_key_blocks);

} // namespace mlxcel::turbo
