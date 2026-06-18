#include <metal_stdlib>
using namespace metal;

// MXFP8 E4M3 to float32 conversion
// Format: 1 sign bit, 4 exponent bits, 3 mantissa bits
float e4m3_to_f32(uint8_t val) {
    if (val == 0) return 0.0f;
    uint sign = (val >> 7) & 1;
    uint exponent = (val >> 3) & 0xF;
    uint mantissa = val & 0x7;

    // E4M3 bias is 7, no inf/nan, max exponent is 15
    float exp_val;
    if (exponent == 0) {
        // Subnormal
        exp_val = ldexp(1.0f + (float)mantissa / 8.0f, -6);
    } else {
        exp_val = ldexp(1.0f + (float)mantissa / 8.0f, (int)exponent - 7);
    }
    return sign ? -exp_val : exp_val;
}

// MXFP8 dequant kernel: weight[i] * scale[i / block_size] -> f16
// Each thread handles one output element.
kernel void mxfp8_dequant_kernel(
    device const uint8_t* weight    [[buffer(0)]],   // [M, N] uint8
    device const uint8_t* scales    [[buffer(1)]],    // [M, N/32] uint8 (one scale per 32 elements)
    device half* out                 [[buffer(2)]],    // [M, N] f16
    constant int& N                  [[buffer(3)]],    // inner dimension
    constant int& block_size         [[buffer(4)]],    // 32
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= (uint)(out - out)) return; // bounds check done via grid size
    // gid indexes into the flat [M*N] output
    int col = gid % N;
    int row = gid / N;

    float w = e4m3_to_f32(weight[gid]);
    int scale_idx = row * (N / block_size) + (col / block_size);
    float s = e4m3_to_f32(scales[scale_idx]);
    out[gid] = half(w * s);
}

// Alternative: dequant with scales stored as [M, ceil(N/32)]
// scales is indexed by (row, col / block_size)
kernel void mxfp8_dequant_v2_kernel(
    device const uint8_t* weight    [[buffer(0)]],   // [M, N] uint8
    device const uint8_t* scales    [[buffer(1)]],    // [M, N_blocks] uint8
    device half* out                 [[buffer(2)]],    // [M, N] f16
    constant int& M                  [[buffer(3)]],
    constant int& N                  [[buffer(4)]],
    constant int& N_blocks           [[buffer(5)]],    // ceil(N / 32)
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= (uint)(M * N)) return;
    int col = gid % N;
    int row = gid / N;

    float w = e4m3_to_f32(weight[gid]);
    int scale_idx = row * N_blocks + (col / 32);
    float s = e4m3_to_f32(scales[scale_idx]);
    out[gid] = half(w * s);
}
