#!/usr/bin/env python3
"""Hypothesis B verification: does mlxcel's MoE gather_qmm path compute the
same per-expert output that MLX's `mx.dequantize` + manual matmul produces?

Reads dumps from layer 3's first MoE invocation:
  layer3_moe_input.bin           - x_flat[0, :]      (hidden=6144, bf16)
  layer3_moe_topk.bin            - topk_idx[0, :]    (k=4, i32)
  layer3_moe_scores.bin          - norm_scores[0, :] (k=4, bf16)
  layer3_moe_router_logits.bin   - router_logits[0, :] (num_experts=128, f32)
  layer3_moe_expert_out.bin      - expert_out[0, :, :] (k=4, hidden=6144, bf16)

For each of the 4 experts selected for token 0:
  1. Load w1/w2/w3 packed weights + scales from the safetensors for that expert
  2. Use MLX's dequantize(mode="nvfp4", group_size=16, bits=4) to recover BF16 weights
  3. Compute SwiGLU manually:
       gate = input @ w1.T  (NB: HF stores [out, in], so matmul uses transpose)
       up   = input @ w3.T
       gate_c = clamp(gate, max=7)
       up_c   = clamp(up,   ±7)
       glu    = gate_c * sigmoid(1.702 * gate_c)
       hidden = (up_c + 1) * glu
       down   = hidden @ w2.T
  4. Compare to mlxcel's expert_out[0, k_slot, :]

Usage: uv run --with safetensors --with numpy --with ml_dtypes \\
       --with 'mlx>=0.18' python3 scripts/verify_expert.py
"""
import json
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path
from safetensors import safe_open

MODEL_DIR = Path("/Volumes/T7 Shield/models/huggingface_cache_hub/models--sjswerdloff--MiniMax-M3-NVFP4-mlx")
PROBE_DIR = Path("/tmp")
HIDDEN = 6144
INTER = 3072  # intermediate_size for routed experts
LAYER = 3     # first MoE layer
K = 4         # num_experts_per_tok
ALPHA = 1.702
LIMIT = 7.0


def load_bf16(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype=np.uint16).view(bfloat16).astype(np.float32)


def load_i32(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype=np.int32)


def load_f32(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype=np.float32)


_SHARD_CACHE: dict = {}


def get_shard_mlx(shard_name: str):
    """Cache: load a single safetensors shard with MLX (handles f8_e4m3fn)."""
    import mlx.core as mx
    if shard_name not in _SHARD_CACHE:
        path = MODEL_DIR / shard_name
        _SHARD_CACHE[shard_name] = mx.load(str(path))
    return _SHARD_CACHE[shard_name]


def load_mlx_array(idx_map: dict, key: str):
    """Return the mx.array for a given safetensors key, loaded via MLX."""
    shard = get_shard_mlx(idx_map[key])
    return shard[key]


def dequantize_nvfp4(packed_w, scales_w, group_size: int = 16) -> np.ndarray:
    """Dequantize NVFP4 via mx.dequantize. Both inputs are mx.array."""
    import mlx.core as mx
    out = mx.dequantize(packed_w, scales_w, biases=None, group_size=group_size, bits=4, mode="nvfp4")
    # Cast to f32 inside MLX (avoids numpy bf16 buffer ambiguity), then materialize.
    out_f32 = out.astype(mx.float32)
    mx.eval(out_f32)
    return np.array(out_f32, copy=True)


def compute_expert_swiglu(x: np.ndarray, w1: np.ndarray, w2: np.ndarray, w3: np.ndarray) -> np.ndarray:
    """One expert's full forward: input -> SwiGLU MLP output.
    Shapes: x[hidden], w1[inter, hidden] (gate), w3[inter, hidden] (up), w2[hidden, inter] (down).
    """
    gate = x @ w1.T              # [inter]
    up = x @ w3.T                # [inter]
    gate_c = np.minimum(gate, LIMIT)
    up_c   = np.clip(up, -LIMIT, LIMIT)
    glu    = gate_c * (1.0 / (1.0 + np.exp(-ALPHA * gate_c)))
    hidden = (up_c + 1.0) * glu
    return hidden @ w2.T         # [hidden]


def main():
    # Load the probes
    inp     = load_bf16(PROBE_DIR / "m3_probe_layer3_moe_input.bin")
    topk    = load_i32(PROBE_DIR / "m3_probe_layer3_moe_topk.bin")
    scores  = load_bf16(PROBE_DIR / "m3_probe_layer3_moe_scores.bin")
    logits  = load_f32(PROBE_DIR / "m3_probe_layer3_moe_router_logits.bin")
    raw     = (PROBE_DIR / "m3_probe_layer3_moe_expert_out.bin").read_bytes()
    expert_out = np.frombuffer(raw, dtype=np.uint16).view(bfloat16).astype(np.float32).reshape(K, HIDDEN)

    assert inp.shape == (HIDDEN,), f"input shape {inp.shape}"
    assert topk.shape == (K,),     f"topk shape {topk.shape}"
    assert scores.shape == (K,),   f"scores shape {scores.shape}"

    print(f"Token 0 routed to experts: {topk.tolist()}")
    print(f"Normalized scores:         {scores.tolist()}")
    print(f"Router logits[top experts]: {logits[topk].tolist()}")
    print(f"Input vec stats: mean|x|={np.abs(inp).mean():.4f} max|x|={np.abs(inp).max():.4f}")

    # Load the safetensors index
    with open(MODEL_DIR / "model.safetensors.index.json") as f:
        idx = json.load(f)["weight_map"]

    prefix = f"language_model.model.layers.{LAYER}.block_sparse_moe.experts"

    print("\n--- Per-expert verification ---")
    for k_slot in range(K):
        expert_id = int(topk[k_slot])
        print(f"\nSlot {k_slot}: expert {expert_id}")

        # Load packed weights and scales for this expert (via MLX so f8_e4m3fn works)
        w1_packed = load_mlx_array(idx, f"{prefix}.{expert_id}.w1.weight")
        w1_scales = load_mlx_array(idx, f"{prefix}.{expert_id}.w1.scales")
        w2_packed = load_mlx_array(idx, f"{prefix}.{expert_id}.w2.weight")
        w2_scales = load_mlx_array(idx, f"{prefix}.{expert_id}.w2.scales")
        w3_packed = load_mlx_array(idx, f"{prefix}.{expert_id}.w3.weight")
        w3_scales = load_mlx_array(idx, f"{prefix}.{expert_id}.w3.scales")

        print(f"  w1 packed shape: {w1_packed.shape} dtype={w1_packed.dtype}, scales shape: {w1_scales.shape} dtype={w1_scales.dtype}")

        w1 = dequantize_nvfp4(w1_packed, w1_scales)   # [inter, hidden]
        w2 = dequantize_nvfp4(w2_packed, w2_scales)   # [hidden, inter]
        w3 = dequantize_nvfp4(w3_packed, w3_scales)   # [inter, hidden]

        print(f"  w1 dequant shape: {w1.shape}, mean|w|={np.abs(w1).mean():.4f}, max|w|={np.abs(w1).max():.4f}")

        # Run the expert forward manually
        py_out = compute_expert_swiglu(inp, w1, w2, w3)
        ml_out = expert_out[k_slot]

        diff = py_out - ml_out
        abs_err = np.abs(diff)
        rel_err = abs_err / (np.abs(ml_out) + 1e-6)

        py_norm = float(np.linalg.norm(py_out))
        ml_norm = float(np.linalg.norm(ml_out))
        cos = float(np.dot(py_out, ml_out) / (py_norm * ml_norm + 1e-12))

        print(f"  py_out  : mean|x|={np.abs(py_out).mean():.4f} max|x|={np.abs(py_out).max():.4f} norm={py_norm:.2f}")
        print(f"  mlxcel  : mean|x|={np.abs(ml_out).mean():.4f} max|x|={np.abs(ml_out).max():.4f} norm={ml_norm:.2f}")
        print(f"  diff    : mean|err|={abs_err.mean():.4f} max|err|={abs_err.max():.4f}")
        print(f"  cos_sim(py, mlxcel) = {cos:.6f}")
        if cos > 0.999:
            print(f"  ✓ MATCH (cos > 0.999)")
        elif cos > 0.95:
            print(f"  ~ close but not exact (NVFP4 precision?)")
        else:
            print(f"  ✗ DIFFERENT — investigate gather_qmm vs dequantize+matmul")


if __name__ == "__main__":
    main()
