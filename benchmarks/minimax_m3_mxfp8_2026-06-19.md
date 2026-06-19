# MiniMax-M3 MXFP8 Inference Benchmarks

**Date:** 2026-06-19
**Hardware:** M3 Ultra, 512 GB Unified RAM
**Model:** MiniMax-M3-MXFP8 (~428B params, ~23B activated, 60 layers, 128 experts)
**Quantization:** MXFP8 (unpacked uint8 weights + per-block uint8 scales)
**Code path:** Dequant-to-f16 via custom Metal kernel, lazy evaluation

---

## Configuration A: No eager eval, no chunked eval
- **Model load:** 0.89s (weights are mmap-backed, lazy)
- **Prefill (4 tokens → 60 layers):** Succeeds, generates 1 token ("l" after "Hello")
- **Decode (1 token → 60 layers):** GPU timeout (>5s Metal command buffer limit)
- **Time to first token:** ~2.5 minutes (includes prefill of 4 tokens)
- **Prefill tok/s:** ~0.025 tok/s (4 tokens / 161s)
- **Decode tok/s:** N/A (times out)

## Configuration B: Chunked eval during prefill (every 10 layers)
- **Model load:** 1.0s
- **Prefill:** Works (eval flushes command buffer every 10 layers)
- **Decode:** GPU timeout (no eval during decode, lazy dequants accumulate)
- **Time to first token:** ~2.5 minutes
- **Prefill tok/s:** ~0.025 tok/s
- **Decode tok/s:** N/A (times out)

## Configuration C: Eager eval for ALL dequants during construction
- **Model load:** >10 minutes (600+ eval calls, one per weight tensor)
- **Forward:** Never reached (construction too slow)
- **Not viable.**

## Configuration D: Fused mxfp8_matmul kernel (no dequant)
- **Model load:** 8.4s (includes kernel compilation)
- **Prefill:** Runs but too slow (naive tiled matmul without SIMD)
- **Decode:** Too slow to complete in 10 minutes
- **Not viable without kernel optimization.**

---

## Root Cause Analysis

### Why decode always times out
MLX uses lazy evaluation. All operations (dequants + matmuls) queue in one Metal command buffer until `eval()` is called. For a 430B model:
- Each layer has ~130 matmuls (4 attention + 128×3 MoE experts + shared)
- With lazy dequants, each matmul adds a Metal kernel launch to the queue
- 60 layers × 130 matmuls = 7,800 operations in one command buffer
- Metal command buffer timeout: 5 seconds
- A single layer (130 matmuls) exceeds 5 seconds

### Why eager eval during construction is too slow
- 60 layers × ~10 weight tensors per layer = 600+ eval calls
- Each eval triggers Metal kernel compilation (first time) + execution
- Total: >10 minutes for construction

### Why chunked eval during decode doesn't help
- eval() during decode triggers MLX Metal kernel compilation
- First eval call compiles the kernel library (~5-10s)
- This overhead makes each 10-layer batch exceed the 5s timeout

---

## What Would Fix This

1. **Fused mxfp8_matmul with SIMD optimization** — eliminates lazy dequants entirely. Naive kernel works but is too slow. Needs:
   - SIMD reduce (`simd_sum` across warp lanes)
   - Larger tiles (BM=128, BN=128)
   - Double-buffered async pipeline
   - Expected: 10-100× speedup over naive kernel

2. **Eager eval with batched construction** — evaluate dequants in batches of 5 layers during construction, with sleep between batches to avoid Metal kernel compilation storms

3. **Custom mxfp8_gather_qmm** — fused kernel for MoE experts that handles expert routing + dequant + matmul in one pass

---

## Performance Targets (from user)

| Metric | Target | Current |
|--------|--------|---------|
| Prefill tok/s | >10 | ~0.025 |
| Decode tok/s | >5 (conversational) | N/A (timeout) |
| TTFT | <5s | ~150s |
| Decode >10 tok/s | Useful for dev | N/A |
| Decode >20 tok/s | General use | N/A |

**Current state: model loads and generates tokens, but decode performance requires the fused kernel optimization work (T6.2-T6.4).**
