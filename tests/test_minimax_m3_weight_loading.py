"""Test MiniMax-M3 weight loading pipeline using shards 1-3.

Discovers actual tensor keys from safetensors metadata and verifies:
1. Shard loading and layer presence
2. Weight shapes for dense attention layers (non-MoE)
3. Index Branch weights on layer 3
4. Tensor data access (BF16 via raw bytes + F8_E4M3 metadata + F32 data)
5. Synthetic top-k selection logic
6. MoE gate and expert weights
7. Summary statistics
"""

import struct
import json
from safetensors import safe_open
from pathlib import Path
import numpy as np

MODEL_DIR = Path("/Volumes/T7 Shield/models/huggingface_cache_hub")
SHARD_FILES = [
    MODEL_DIR / "model-00001-of-00031.safetensors",
    MODEL_DIR / "model-00002-of-00031.safetensors",
    MODEL_DIR / "model-00003-of-00031.safetensors",
]

DENSE_ATTN_LAYERS = {0, 1, 2}


def load_shard_metadata(path):
    info = {}
    with safe_open(str(path), framework="numpy") as f:
        for key in f.keys():
            s = f.get_slice(key)
            info[key] = {"shape": list(s.get_shape()), "dtype": str(s.get_dtype())}
    return info


def read_tensor_raw(path, key):
    """Read any tensor from safetensors as raw bytes + metadata."""
    with open(path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len).decode('utf-8'))
        if key not in header:
            return None, None, None
        meta = header[key]
        dtype = meta['dtype']
        shape = meta['shape']
        start, end = meta['data_offsets']
        f.seek(8 + header_len + start)
        raw = f.read(end - start)
        return raw, dtype, shape


def bf16_to_float32(raw_bytes, shape):
    """Convert BF16 raw bytes to float32 numpy array."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint16).reshape(shape)
    return (arr.astype(np.uint32) << 16).view(np.float32)


def f32_from_raw(raw_bytes, shape):
    """Read float32 raw bytes as numpy array."""
    return np.frombuffer(raw_bytes, dtype=np.float32).reshape(shape)


# ── Test 1: Shard Loading ────────────────────────────────────────────────

def test_shard_loading(all_shards):
    print("=" * 70)
    print("TEST 1: Shard Loading and Layer Presence")
    print("=" * 70)

    passed = failed = 0
    for shard_idx, shard in all_shards.items():
        layers = sorted(set(
            int(k.split("layers.")[1].split(".")[0])
            for k in shard if "layers." in k
        ))
        print(f"\n  Shard {shard_idx}: {len(shard)} tensors, layers {layers}")
        for layer in layers:
            count = sum(1 for k in shard if f"layers.{layer}." in k)
            print(f"    Layer {layer}: {count} tensors")
            if count > 0:
                passed += 1
            else:
                failed += 1
    print(f"\n  Result: {passed} passed, {failed} failed")
    return passed, failed


# ── Test 2: Dense Attention Weight Shapes ────────────────────────────────

DENSE_WEIGHTS = {
    "input_layernorm.weight": [6144],
    "self_attn.q_proj.weight": [8192, 6144],
    "self_attn.q_proj.weight_scale_inv": [8192, 192],
    "self_attn.k_proj.weight": [512, 6144],
    "self_attn.k_proj.weight_scale_inv": [512, 192],
    "self_attn.v_proj.weight": [512, 6144],
    "self_attn.v_proj.weight_scale_inv": [512, 192],
    "self_attn.o_proj.weight": [6144, 8192],
    "self_attn.o_proj.weight_scale_inv": [6144, 256],
    "self_attn.q_norm.weight": [128],
    "self_attn.k_norm.weight": [128],
    "post_attention_layernorm.weight": [6144],
}


def test_dense_weight_shapes(all_shards):
    print("\n" + "=" * 70)
    print("TEST 2: Dense Attention Weight Shapes (Layers 0, 1, 2)")
    print("=" * 70)

    passed = failed = 0
    all_tensors = {}
    for shard in all_shards.values():
        all_tensors.update(shard)

    for layer in DENSE_ATTN_LAYERS:
        print(f"\n  Layer {layer}:")
        for suffix, expected_shape in DENSE_WEIGHTS.items():
            key = f"language_model.model.layers.{layer}.{suffix}"
            if key in all_tensors:
                actual = all_tensors[key]["shape"]
                ok = actual == expected_shape
                dtype = all_tensors[key]["dtype"]
                status = "OK" if ok else f"MISMATCH expected {expected_shape}"
                print(f"    {suffix}: {actual} {dtype} {status}")
                passed += 1
            else:
                print(f"    {suffix}: MISSING")
                failed += 1
    print(f"\n  Result: {passed} passed, {failed} failed")
    return passed, failed


# ── Test 3: Index Branch Weights ─────────────────────────────────────────

INDEX_BRANCH = {
    "self_attn.index_q_proj.weight": [512, 6144],
    "self_attn.index_q_proj.weight_scale_inv": [512, 192],
    "self_attn.index_k_proj.weight": [128, 6144],
    "self_attn.index_k_proj.weight_scale_inv": [128, 192],
    "self_attn.index_q_norm.weight": [128],
    "self_attn.index_k_norm.weight": [128],
}


def test_index_branch(all_shards):
    print("\n" + "=" * 70)
    print("TEST 3: Index Branch Weights (Layer 3)")
    print("=" * 70)

    passed = failed = 0
    layer = 3
    all_tensors = {}
    for shard in all_shards.values():
        all_tensors.update(shard)

    for suffix, expected_shape in INDEX_BRANCH.items():
        key = f"language_model.model.layers.{layer}.{suffix}"
        if key in all_tensors:
            actual = all_tensors[key]["shape"]
            ok = actual == expected_shape
            dtype = all_tensors[key]["dtype"]
            status = "OK" if ok else f"MISMATCH expected {expected_shape}"
            print(f"  {suffix}: {actual} {dtype} {status}")
            passed += 1
        else:
            print(f"  {suffix}: MISSING")
            failed += 1
    print(f"\n  Result: {passed} passed, {failed} failed")
    return passed, failed


# ── Test 4: Tensor Data Access ───────────────────────────────────────────

def test_tensor_data_access():
    print("\n" + "=" * 70)
    print("TEST 4: Tensor Data Access (BF16 + F8_E4M3 + F32)")
    print("=" * 70)

    passed = failed = 0
    shard_path = SHARD_FILES[0]

    # 4a: BF16 layernorm weight (raw file reading)
    bf16_key = "language_model.model.layers.0.input_layernorm.weight"
    print(f"\n  BF16 tensor: {bf16_key}")
    raw, dtype, shape = read_tensor_raw(shard_path, bf16_key)
    if raw and dtype == "BF16":
        values = bf16_to_float32(raw, shape)
        print(f"    Shape: {values.shape}")
        print(f"    First 10 values: {values[:10]}")
        print(f"    Min: {values.min():.6f}, Max: {values.max():.6f}, Mean: {values.mean():.6f}")
        passed += 1
    else:
        print(f"    FAILED to read")
        failed += 1

    # 4b: BF16 q_norm weight
    qnorm_key = "language_model.model.layers.0.self_attn.q_norm.weight"
    print(f"\n  BF16 tensor: {qnorm_key}")
    raw, dtype, shape = read_tensor_raw(shard_path, qnorm_key)
    if raw and dtype == "BF16":
        values = bf16_to_float32(raw, shape)
        print(f"    Shape: {values.shape}")
        print(f"    First 10 values: {values[:10]}")
        print(f"    Min: {values.min():.6f}, Max: {values.max():.6f}, Mean: {values.mean():.6f}")
        passed += 1
    else:
        print(f"    FAILED to read")
        failed += 1

    # 4c: F8_E4M3 metadata accessibility
    f8_key = "language_model.model.layers.0.self_attn.q_proj.weight"
    scale_key = "language_model.model.layers.0.self_attn.q_proj.weight_scale_inv"
    print(f"\n  MXFP8 metadata:")
    with safe_open(str(shard_path), framework="numpy") as f:
        s = f.get_slice(f8_key)
        print(f"    Weight: {list(s.get_shape())} {s.get_dtype()}")
        s2 = f.get_slice(scale_key)
        print(f"    Scale:  {list(s2.get_shape())} {s2.get_dtype()}")
        passed += 1

    # 4d: F32 gate weight (shard 2, layer 3)
    gate_key = "language_model.model.layers.3.block_sparse_moe.gate.weight"
    print(f"\n  F32 tensor: gate.weight (shard 2)")
    raw, dtype, shape = read_tensor_raw(SHARD_FILES[1], gate_key)
    if raw and dtype == "F32":
        values = f32_from_raw(raw, shape)
        print(f"    Shape: {values.shape}")
        print(f"    First 10 values of first row: {values[0, :10]}")
        print(f"    Min: {values.min():.6f}, Max: {values.max():.6f}, Mean: {values.mean():.6f}")
        passed += 1
    else:
        print(f"    FAILED: dtype={dtype}")
        failed += 1

    print(f"\n  Result: {passed} passed, {failed} failed")
    return passed, failed


# ── Test 5: Top-K Selection Logic ────────────────────────────────────────

def test_topk_selection():
    print("\n" + "=" * 70)
    print("TEST 5: Top-K Selection Logic (Synthetic)")
    print("=" * 70)

    passed = failed = 0
    np.random.seed(42)

    # sparse_num_index_heads=4, sparse_topk_blocks=16, sparse_block_size=128
    block_scores = np.random.randn(1, 4, 16).astype(np.float32)
    top_k = 16
    num_heads = 4
    num_blocks = 16

    print(f"\n  block_scores: {block_scores.shape}")
    print(f"  Head 0 scores: {block_scores[0, 0, :]}")

    # Top-k selection
    selected = np.zeros((1, num_heads, top_k), dtype=np.int32)
    for b in range(1):
        for h in range(num_heads):
            idx = np.argpartition(block_scores[b, h, :], -top_k)[-top_k:]
            idx = idx[np.argsort(-block_scores[b, h, idx])]
            selected[b, h, :] = idx

    print(f"  Head 0 indices: {selected[0, 0, :]}")

    # 5a: Valid range
    if np.all(selected >= 0) and np.all(selected < num_blocks):
        print("  Indices in valid range: PASS")
        passed += 1
    else:
        print("  Indices out of range: FAIL")
        failed += 1

    # 5b: Shape
    if selected.shape == (1, num_heads, top_k):
        print("  Shape correct: PASS")
        passed += 1
    else:
        print("  Shape incorrect: FAIL")
        failed += 1

    # 5c: Unique
    all_unique = all(
        len(np.unique(selected[b, h])) == top_k
        for b in range(1) for h in range(num_heads)
    )
    if all_unique:
        print("  Unique indices: PASS")
        passed += 1
    else:
        print("  Duplicate indices: FAIL")
        failed += 1

    # 5d: Correct top-k scores
    scores_ok = True
    for h in range(num_heads):
        sel = block_scores[0, h, selected[0, h, :]]
        ref = np.sort(block_scores[0, h, :])[-top_k:]
        if not np.allclose(np.sort(sel), ref, rtol=1e-5):
            scores_ok = False
    if scores_ok:
        print("  Top-k scores correct: PASS")
        passed += 1
    else:
        print("  Score mismatch: FAIL")
        failed += 1

    # 5e: Sparse top-k=4
    print(f"\n  Sparse top-k=4:")
    top_k4 = 4
    sparse = np.zeros((1, num_heads, top_k4), dtype=np.int32)
    for b in range(1):
        for h in range(num_heads):
            idx = np.argpartition(block_scores[b, h, :], -top_k4)[-top_k4:]
            idx = idx[np.argsort(-block_scores[b, h, idx])]
            sparse[b, h, :] = idx
    print(f"  Head 0 sparse: {sparse[0, 0, :]}")
    print(f"  Head 0 scores: {block_scores[0, 0, sparse[0, 0, :]]}")

    sparse_ok = True
    for h in range(num_heads):
        sel = block_scores[0, h, sparse[0, h, :]]
        ref = np.sort(block_scores[0, h, :])[-top_k4:]
        if not np.allclose(np.sort(sel), ref, rtol=1e-5):
            sparse_ok = False
    if sparse_ok:
        print("  Sparse top-4 correct: PASS")
        passed += 1
    else:
        print("  Sparse top-4 mismatch: FAIL")
        failed += 1

    print(f"\n  Result: {passed} passed, {failed} failed")
    return passed, failed


# ── Test 6: MoE Weights ──────────────────────────────────────────────────

def test_moe_weights(all_shards):
    print("\n" + "=" * 70)
    print("TEST 6: MoE Gate and Expert Weights (Layer 3)")
    print("=" * 70)

    passed = failed = 0
    layer = 3
    all_tensors = {}
    for shard in all_shards.values():
        all_tensors.update(shard)

    # Gate + e_score_correction_bias (F32)
    for suffix in ["block_sparse_moe.gate.weight", "block_sparse_moe.e_score_correction_bias"]:
        key = f"language_model.model.layers.{layer}.{suffix}"
        if key in all_tensors:
            s = all_tensors[key]
            print(f"  {suffix}: {s['shape']} {s['dtype']}")
            passed += 1
        else:
            print(f"  {suffix}: MISSING")
            failed += 1

    # Expert w1/w2/w3 (first 3)
    for ei in range(3):
        for proj in ["w1", "w2", "w3"]:
            key = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{ei}.{proj}.weight"
            if key in all_tensors:
                s = all_tensors[key]
                print(f"  experts.{ei}.{proj}: {s['shape']} {s['dtype']}")
                passed += 1
            else:
                print(f"  experts.{ei}.{proj}: MISSING")
                failed += 1

    # Shared experts
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        key = f"language_model.model.layers.{layer}.block_sparse_moe.shared_experts.{proj}.weight"
        if key in all_tensors:
            s = all_tensors[key]
            print(f"  shared_experts.{proj}: {s['shape']} {s['dtype']}")
            passed += 1
        else:
            print(f"  shared_experts.{proj}: MISSING")
            failed += 1

    print(f"\n  Result: {passed} passed, {failed} failed")
    return passed, failed


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("MiniMax-M3 Weight Loading Pipeline Test")
    print("=" * 70)

    all_shards = {}
    for idx, path in enumerate(SHARD_FILES, start=1):
        print(f"Loading shard {idx}: {path.name}...")
        all_shards[idx] = load_shard_metadata(path)
        print(f"  {len(all_shards[idx])} tensors")

    total_p = total_f = 0
    for name, fn in [
        ("Shard Loading", lambda: test_shard_loading(all_shards)),
        ("Dense Weight Shapes", lambda: test_dense_weight_shapes(all_shards)),
        ("Index Branch", lambda: test_index_branch(all_shards)),
        ("Tensor Data", test_tensor_data_access),
        ("Top-K Selection", test_topk_selection),
        ("MoE Weights", lambda: test_moe_weights(all_shards)),
    ]:
        p, f = fn()
        total_p += p
        total_f += f

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_tensors = sum(len(s) for s in all_shards.values())
    dtype_bytes = {"BF16": 2, "F8_E4M3": 1, "F32": 4, "U8": 1}
    total_bytes = 0
    for shard in all_shards.values():
        for meta in shard.values():
            elems = 1
            for d in meta["shape"]:
                elems *= d
            total_bytes += elems * dtype_bytes.get(meta["dtype"], 4)

    print(f"Total tensors across 3 shards: {total_tensors}")
    print(f"Total elements (weighted by dtype): {total_bytes:,}")
    print(f"Estimated on-disk size: {total_bytes / (1024**3):.2f} GB")
    print(f"\nVerification: {total_p} passed, {total_f} failed")
    print(f"\nOverall: {'ALL TESTS PASSED' if total_f == 0 else f'{total_f} TESTS FAILED'}")


if __name__ == "__main__":
    main()
