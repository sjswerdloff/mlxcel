# MiniMax-M3 MXFP8 Inference: Technical Insights

## 1. MLX JIT Metal Kernel Compilation Rules

MLX's `metal_kernel()` JIT places your source string **inside** a generated kernel body. This means:

### What DOESN'T work
- `kernel void my_kernel(...)` — MLX auto-generates the `kernel` signature. Putting one in your source causes `error: expected expression` or `function definition is not allowed here`.
- Standalone function definitions (`float my_func(...) { ... }`) — Metal doesn't allow function definitions inside a kernel body. MLX's JIT puts your source inside a function context where this is illegal.
- Using `gid` or custom thread variable names — MLX doesn't generate these. You must use `thread_position_in_grid`, `thread_position_in_threadgroup`, `threadgroup_position_in_grid`.
- `constant int& M` for scalar args — MLX maps `input_names` to buffer slots. Scalar args must be passed as `device const int*` arrays and dereferenced with `*d_M`.

### What WORKS
- Your source string is the **kernel body only** — no function signature, no `kernel void`.
- Variables from `input_names` are available as device pointers (e.g., `weight`, `scales`).
- Variables from `output_names` are available as device pointers (e.g., `out`).
- `thread_position_in_grid.x` for global thread ID.
- `constexpr int` declarations inside the source string work (they're Metal constants, not C++ function definitions).
- Scalar arguments (M, N, K) must be passed as `array({value})` in the inputs list and dereferenced as `*d_M` in the kernel.

### The compilation context matters
When `eval()` is called at the end of the full forward pass, MLX compiles the kernel library successfully. When `eval()` is called mid-graph (e.g., during chunked eval at layer 10), the compilation context is different and standalone function definitions fail. Inlining the conversion fixes this for both contexts.

## 2. MXFP8 Format: MLX Native vs Custom Kernel

### MLX's `quantized_matmul` with `mode="mxfp8"`
- Expects **packed uint32** weights (4 uint8 elements per uint32)
- MXFP8 checkpoints store **unpacked uint8** (1 element per byte)
- These are **fundamentally incompatible formats** — cannot use MLX's native path without repacking
- The FFI bridge (`quantized_linear_forward`) dispatches to `mxfp8_matmul` when `mode="mxfp8"` and `biases` is None

### MXFP8 E4M3 format
- 1 sign bit, 4 exponent bits, 3 mantissa bits
- Bias = 7, no inf/nan, max exponent = 15
- Scale blocks: one uint8 scale per 32 weight elements
- Scales are stored as `[K/32, N]` for weight `[K, N]`

### Memory implications
- MXFP8 weights on disk: ~413 GB for 428B params
- Dequantized to f16: ~856 GB (exceeds 512 GB unified RAM)
- **Cannot eagerly dequant all MoE experts** — 57 layers × 128 experts × 3 projections = ~832 GB needed

## 3. Metal Command Buffer Timeout

### The problem
Apple Silicon Metal command buffers have a ~5 second GPU timeout. MLX uses lazy evaluation — all operations queue in one command buffer until `eval()` is called. For a 430B model with 60 layers of dequants + matmuls, the queue exceeds this timeout.

### Why chunked eval is necessary
Calling `mlxcel_core::eval(&h)` every N layers during prefill flushes the command buffer, keeping each batch under the timeout. Without it, the entire forward pass queues as one command.

### The eval() compilation interaction
- `eval()` at end of full graph: compiles Metal kernel library successfully (different context)
- `eval()` mid-graph: compiles in a context where standalone function definitions fail
- Fix: inline all conversion functions directly in the kernel body

### Load time indicator
- Without eager eval: ~0.9s (lazy, mmap-backed)
- With eager eval of dequants: ~8.5s (forces Metal kernel compilation + execution)
- The 8.5s load means dequants are being materialized during construction

## 4. Model Architecture: What Burns Memory

### Layer composition (per layer)
- **Dense layers (0-2)**: 4 attention projections (q,k,v,o) + SwiGLU MLP (gate,up,down)
- **MoE layers (3-59)**: 4 attention projections + index projections (q,k) + router + 128 experts × 3 projections + shared experts × 3 projections

### Where the command buffer time goes
- Per-layer: ~10 matmuls for dense, ~400+ matmuls for MoE (128 experts × 3)
- Each matmul involves weight access (lazy if dequant, eager if fused)
- MoE expert routing: `gather_qmm` or `gather_mm` handles per-expert slicing
- The MoE experts dominate the computation budget

### Why fused kernel helps for Linear but not MoE
- Fused `mxfp8_matmul` replaces `quantized_linear_forward` — handles q/k/v/o projections directly
- MoE experts use `SwitchLinear` → `gather_qmm` — different code path, needs a separate gather variant
- A full fused path would need `mxfp8_gather_qmm` that does expert routing + dequant + matmul

## 5. The `dense_attention` Shape Bug

### Root cause
`attention_from_ptr` (MLX's built-in SDPA) returns raw attention output in shape `[batch, heads, seq, head_dim]`. It does NOT apply `o_proj` or reshape to `[batch, seq, hidden_dim]`. This is the standard MLX SDPA behavior — the caller is expected to handle the projection and reshape.

### The fix pattern (from `sparse_sdpa`)
```rust
let out = mlxcel_core::transpose_axes(&raw, &[0, 2, 1, 3]);
let out = mlxcel_core::reshape(&out, &[b, l, self.num_heads * self.head_dim]);
self.o_proj.forward(&out)
```

### Why it only manifests at layer 3+
Layers 0-2 are dense (no MoE), layers 3-59 have index branches. The `dense_attention` path is used when `index_q_proj.is_none() || l <= block_size`. The bug exists in all layers but the broadcast error only triggers when the shapes mismatch at the `add(x, &attn_out)` in `DecoderLayer::forward`.

## 6. MXFP8 Detection Heuristics

### Correct detection (for `load_linear`)
```rust
let is_mxfp8 = weights.get(&scales_key).is_some()     // scales present
    && weights.get(&biases_key).is_none()               // no biases
    && weights.get(&weight_key)
        .map(|w| mlxcel_core::array_dtype(w) == mlxcel_core::dtype::UINT8)
        .unwrap_or(false);                               // uint8 weight dtype
```

### Why `bits == 8` detection fails
`UnifiedLinear::from_weights` auto-detects mode based on `bits == 8 → "mxfp8"`. But M3's config may report `bits = 4` (the default quantization bits) even though MXFP8 weights are 8-bit. The weight dtype (uint8) is the reliable signal, not the config bits.

### Why `UnifiedLinear::from_weights_with_mode` is the right call
Instead of `from_weights` (which may detect wrong mode), explicitly pass `"mxfp8"` when MXFP8 is detected. This bypasses the heuristic and goes directly to the right code path.

## 7. Fused Kernel Performance Characteristics

### Current naive implementation
- BM=64, BN=64, BK=32, TM=4, TN=4
- 16×16 = 256 threads per threadgroup
- Threadgroup memory: ~10 KB (a_tile 8KB + b_tile 2KB + s_tile 64B)
- Accumulation: scalar (no SIMD)
- Input: f32 (loaded from f32 activations), output: f16
- All E4M3 dequant is inline (no function calls)

### Why it's slow for 430B
- Each layer has 4+ matmuls (q,k,v,o projections) × 60 layers = 240+ kernel launches
- Naive tiling doesn't utilize Apple Silicon's SIMD units
- No pipelining — each BK block loads, computes, then loads next (no overlap)
- The kernel is compute-bound but doesn't saturate the ALUs

### Optimization priorities (T5.2-T5.4)
1. **SIMD reduce** — Apple Silicon has 32-wide SIMD. Use `simd_sum` to reduce 32 partial products per thread, reducing threadgroup syncs by 32×.
2. **Larger tiles** — BM=128 or BM=256 for better compute/memory ratio. Need to balance against threadgroup memory limits (32 KB on M3).
3. **Double buffering** — Load next BK block while computing current one. Requires 2× threadgroup memory but hides load latency.

## 8. SwitchLinear MXFP8 Handling

### The gather_qmm problem
`SwitchLinear::Quantized` calls `gather_qmm` which requires biases for affine mode. MXFP8 has no biases → crash: `Biases must be provided for affine quantization`.

### Current workaround
Dequantize MXFP8 experts to f16 in `SwitchLinear::from_stacked_parts`, returning `SwitchLinear::Regular` which uses `gather_mm` (no biases needed).

### Future: mxfp8_gather_qmm
Need a per-expert fused kernel that:
1. Sorts tokens by expert index (existing `gather_sort`)
2. For each expert, loads its uint8 weight + scales
3. Dequants on-the-fly during matmul
4. Scatter results back to original token order

This eliminates the dequant memory overhead for experts and avoids the command buffer timeout.
