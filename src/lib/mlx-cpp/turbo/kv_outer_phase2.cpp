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

// MSA KV-Outer Phase 2: Global softmax reduction.
//
// For each query, sweeps across all partial blocks from Phase 1,
// computes the global maximum, rescales partials, and produces the
// final normalized attention output.

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

constexpr int SIMD_WIDTH = 32;

// Phase 2 kernel: global softmax reduction.
//
// Thread topology:
//   Grid Z = Query Head (0 .. Hq - 1)
//   Grid Y = Query Position (0 .. L - 1)
//   Grid X = lanes partitioning head dimension (Dim)
//
// For each query, sweeps across num_key_blocks partials from Phase 1,
// computes the global max and sum_exp, then produces normalized output.
constexpr const char* KV_OUTER_PHASE2_SOURCE = R"(
    uint d = thread_position_in_grid.x;
    uint q_pos = thread_position_in_grid.y;
    uint hq = thread_position_in_grid.z;

    uint num_blocks = (uint)num_key_blocks_arr[0];
    uint dim = (uint)Dim;
    uint q_len = (uint)q_shape[2];

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

inline Phase2KernelHolder& get_phase2_kernel() {
    static Phase2KernelHolder holder;
    return holder;
}

} // namespace

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

    // Validate decode-only contract (grid and offsets omit batch).
    if (batch != 1) {
        throw std::runtime_error(
            "kv_outer_phase2: decode only (B=1), got B=" + std::to_string(batch));
    }

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
