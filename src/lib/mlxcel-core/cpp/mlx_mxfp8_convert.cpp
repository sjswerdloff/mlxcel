// MXFP8 → MLX Affine format converter
// Converts unpacked uint8 weights + uint8 scales to MLX's native packed uint32
// format, enabling MLX's optimized Metal quantized matmul kernels.
//
// MXFP8 format: uint8 weights [M, N], uint8 scales [M, N/32] (E4M3)
// MLX affine:   uint32 packed weights [M, N/4], float16 scales [M, N/gs], float16 biases [M, N/gs]
//
// This is a one-time cost during model construction.

#include "mlx_cxx_internal.h"

namespace mlx_cxx {

// Pack 4 consecutive uint8 values into one uint32
inline uint32_t pack_u8x4(const uint8_t* data) {
    return uint32_t(data[0]) |
           (uint32_t(data[1]) << 8) |
           (uint32_t(data[2]) << 16) |
           (uint32_t(data[3]) << 24);
}

// Convert MXFP8 E4M3 uint8 to float16
inline mlx::core::float16_t mxfp8_to_f16(uint8_t val) {
    if (val == 0) return mlx::core::float16_t(0.0f);
    uint sign = (val >> 7) & 1;
    uint exponent = (val >> 3) & 0xF;
    uint mantissa = val & 0x7;
    float exp_val;
    if (exponent == 0) {
        exp_val = ldexp(1.0f + (float)mantissa / 8.0f, -6);
    } else {
        exp_val = ldexp(1.0f + (float)mantissa / 8.0f, (int)exponent - 7);
    }
    return mlx::core::float16_t(sign ? -exp_val : exp_val);
}

// Convert MXFP8 weights to MLX affine format
// Input:  weight_uint8 [M, N], scales_uint8 [M, N/32]
// Output: weight_packed [M, N/4] uint32, scales_f16 [M, N/32] float16, biases_f16 [M, N/32] float16 (zeros)
std::tuple<mlx::core::array, mlx::core::array, mlx::core::array>
mxfp8_to_affine(const mlx::core::array& weight_u8, const mlx::core::array& scales_u8) {
    using namespace mlx::core;

    auto w_shape = weight_u8.shape();
    int M = w_shape[0];
    int N = w_shape[1];
    int N_packed = N / 4;
    int N_groups = N / 32;

    // Get raw data pointers
    auto w_data = weight_u8.data<uint8_t>();
    auto s_data = scales_u8.data<uint8_t>();

    // Pack weights: 4 uint8 → 1 uint32
    std::vector<uint32_t> packed(M * N_packed);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N_packed; n++) {
            packed[m * N_packed + n] = pack_u8x4(&w_data[m * N + n * 4]);
        }
    }

    // Convert scales: uint8 E4M3 → float16
    std::vector<float16_t> scales_f16(M * N_groups);
    for (int i = 0; i < M * N_groups; i++) {
        scales_f16[i] = mxfp8_to_f16(s_data[i]);
    }

    // Biases: zeros (MXFP8 has no zero-point)
    std::vector<float16_t> biases_f16(M * N_groups, 0.0f);

    auto weight_packed = array(packed.data(), {M, N_packed}, uint32);
    auto scales_out = array(scales_f16.data(), {M, N_groups}, float16);
    auto biases_out = array(biases_f16.data(), {M, N_groups}, float16);

    return std::make_tuple(weight_packed, scales_out, biases_out);
}

} // namespace mlx_cxx
