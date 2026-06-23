#!/usr/bin/env python3
"""Layer 0 attention bug isolation.

Reads the attention substep dumps and answers three questions:

  1. Does Q/K NORM collapse position direction?
     -> compare cos_sim(pos_a, pos_b) before and after q_norm/k_norm.

  2. Does PARTIAL RoPE rotate the right dims?
     -> q_after_rope[..., 64:] should EQUAL q_after_transpose[..., 64:] exactly.
        (M3 config: head_dim=128, rotary_dim=64. First 64 should rotate, last 64 pass through.)

  3. Does SOFTMAX(QK^T) produce a meaningful attention distribution, or
     is it near-uniform / dominated by an attention sink?
     -> Compute the attention pattern for head 0 manually in numpy and inspect.

Usage:
  uv run --with numpy --with ml_dtypes python3 scripts/verify_attention.py
"""
import math
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

PROBE_DIR = Path("/tmp")
HEAD_DIM = 128
ROTARY_DIM = 64        # partial RoPE first half
NUM_HEADS = 64
NUM_KV_HEADS = 4
# These positions are sampled (start, middle, end) by the Rust instrumentation
# at the layer level; we want to compare attention behavior at multiple query
# positions for the head-level analysis.


def load(label: str, shape: tuple) -> np.ndarray:
    path = PROBE_DIR / f"m3_probe_{label}.bin"
    if not path.exists():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    # bf16 storage; cast through ml_dtypes
    u16 = np.frombuffer(raw, dtype=np.uint16)
    arr = u16.view(bfloat16).astype(np.float32)
    return arr.reshape(shape)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a.flatten(), b.flatten()) / (na * nb))


def positions_table(label: str, tensor: np.ndarray, position_axis: int):
    """Print cos_sim between three sampled positions of `tensor`.
    tensor shape is [1, ..., position_axis_dim, ..., head_dim] etc.
    We flatten everything except `position_axis` for the comparison."""
    seq = tensor.shape[position_axis]
    positions = [0, seq // 2, seq - 1]
    # Slice each position along `position_axis`
    slices = [np.take(tensor, p, axis=position_axis) for p in positions]
    # Flatten remaining dims
    flat = [s.flatten() for s in slices]
    print(f"  {label:30s}  cos(p0,p1)={cos_sim(flat[0], flat[1]):.4f}  "
          f"cos(p0,p2)={cos_sim(flat[0], flat[2]):.4f}  "
          f"cos(p1,p2)={cos_sim(flat[1], flat[2]):.4f}  "
          f"L2(p0,p2)={float(np.linalg.norm(flat[0] - flat[2])):.2f}")


def main():
    # Determine seq_len from file size (Q after reshape: [1, L, 64, 128] bf16)
    raw_size = (PROBE_DIR / "m3_probe_attn_q_after_reshape.bin").stat().st_size
    L = raw_size // (1 * NUM_HEADS * HEAD_DIM * 2)
    print(f"Detected seq_len = {L}")
    print()

    # Load all the dumps with known shapes
    q_reshape   = load("attn_q_after_reshape",   (1, L, NUM_HEADS, HEAD_DIM))
    q_qnorm     = load("attn_q_after_qnorm",     (1, L, NUM_HEADS, HEAD_DIM))
    q_transpose = load("attn_q_after_transpose", (1, NUM_HEADS, L, HEAD_DIM))
    q_rope      = load("attn_q_after_rope",      (1, NUM_HEADS, L, HEAD_DIM))

    k_reshape   = load("attn_k_after_reshape",   (1, L, NUM_KV_HEADS, HEAD_DIM))
    k_qnorm     = load("attn_k_after_knorm",     (1, L, NUM_KV_HEADS, HEAD_DIM))
    k_transpose = load("attn_k_after_transpose", (1, NUM_KV_HEADS, L, HEAD_DIM))
    k_rope      = load("attn_k_after_rope",      (1, NUM_KV_HEADS, L, HEAD_DIM))

    v_reshape   = load("attn_v_after_reshape",   (1, L, NUM_KV_HEADS, HEAD_DIM))

    # ------------------------------------------------------------------
    # Question 1: Does Q/K norm collapse positions?
    # ------------------------------------------------------------------
    print("=" * 78)
    print("QUESTION 1: Does Q/K norm collapse position direction?")
    print("=" * 78)
    print("Q tensor (positions in axis 1 before transpose, axis 2 after):")
    positions_table("after reshape   (raw projection)", q_reshape,   position_axis=1)
    positions_table("after q_norm                    ", q_qnorm,     position_axis=1)
    positions_table("after transpose                 ", q_transpose, position_axis=2)
    positions_table("after RoPE                      ", q_rope,      position_axis=2)
    print()
    print("K tensor:")
    positions_table("after reshape   (raw projection)", k_reshape,   position_axis=1)
    positions_table("after k_norm                    ", k_qnorm,     position_axis=1)
    positions_table("after transpose                 ", k_transpose, position_axis=2)
    positions_table("after RoPE                      ", k_rope,      position_axis=2)
    print()
    print("V tensor (passes through unchanged):")
    positions_table("after reshape", v_reshape, position_axis=1)
    print()
    print("Reading: cos_sim near 1.0 = positions are collapsed.")
    print("If cos_sim jumps from low (raw) to high (after norm), q_norm/k_norm is the culprit.")
    print()

    # ------------------------------------------------------------------
    # Question 2: Does partial RoPE rotate only the right dims?
    # ------------------------------------------------------------------
    print("=" * 78)
    print("QUESTION 2: Does partial RoPE leave dims [64:] unchanged?")
    print("=" * 78)
    # q_rope and q_transpose are both shape [1, num_heads, L, head_dim]
    # Take the last (head_dim - rotary_dim) = 64 dims at every position/head.
    # They should be IDENTICAL between transpose and rope outputs.
    q_pass_pre  = q_transpose[..., ROTARY_DIM:]  # [1, 64, L, 64]
    q_pass_post = q_rope[..., ROTARY_DIM:]
    diff_q = np.abs(q_pass_pre - q_pass_post)
    print(f"Q: max abs diff in dims [{ROTARY_DIM}:] = {diff_q.max():.6f}, "
          f"mean abs diff = {diff_q.mean():.6f}")
    if diff_q.max() < 1e-4:
        print(f"  PASS: partial RoPE preserves Q dims [{ROTARY_DIM}:]")
    else:
        print(f"  FAIL: partial RoPE is mutating Q dims [{ROTARY_DIM}:] — likely rotating wrong slice")

    k_pass_pre  = k_transpose[..., ROTARY_DIM:]
    k_pass_post = k_rope[..., ROTARY_DIM:]
    diff_k = np.abs(k_pass_pre - k_pass_post)
    print(f"K: max abs diff in dims [{ROTARY_DIM}:] = {diff_k.max():.6f}, "
          f"mean abs diff = {diff_k.mean():.6f}")
    if diff_k.max() < 1e-4:
        print(f"  PASS: partial RoPE preserves K dims [{ROTARY_DIM}:]")
    else:
        print(f"  FAIL: partial RoPE is mutating K dims [{ROTARY_DIM}:] — likely rotating wrong slice")

    # Also: dims [:64] SHOULD change (i.e. RoPE actually does something on the first half)
    q_rot_pre  = q_transpose[..., :ROTARY_DIM]
    q_rot_post = q_rope[..., :ROTARY_DIM]
    diff_rot = np.abs(q_rot_pre - q_rot_post)
    print(f"Q: max abs diff in dims [:{ROTARY_DIM}] = {diff_rot.max():.6f} "
          f"(should be > 0 — RoPE should change these)")
    print()

    # ------------------------------------------------------------------
    # Question 3: Is the attention distribution near-uniform / collapsed?
    # ------------------------------------------------------------------
    print("=" * 78)
    print("QUESTION 3: Is softmax(Q @ K.T) producing a sane attention pattern?")
    print("=" * 78)
    # Use head 0 of Q and the corresponding KV head (head 0 since 64 Q heads share 4 KV heads,
    # so Q head 0 maps to KV head 0).
    Q0 = q_rope[0, 0, :, :]              # [L, head_dim]
    K0 = k_rope[0, 0, :, :]              # [L, head_dim]
    scale = 1.0 / math.sqrt(HEAD_DIM)
    scores = Q0 @ K0.T * scale           # [L, L]
    # Causal mask
    mask = np.triu(np.ones((L, L), dtype=bool), k=1)
    scores[mask] = -np.inf
    # Softmax row-wise
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)

    def describe_row(p):
        row = weights[p]                          # [L]
        valid = row[: p + 1]                      # only first p+1 are allowed (causal)
        invalid = row[p + 1 :]
        # Concentration: entropy + top-3
        entropy = -float(np.sum(valid * np.log(valid + 1e-12)))
        max_uniform = math.log(p + 1) if p > 0 else 0.0
        top3 = np.argsort(-valid)[:3]
        top3_vals = valid[top3]
        leak_to_future = float(invalid.sum())
        print(f"  query pos {p:>4}: entropy={entropy:.3f} (uniform={max_uniform:.3f})  "
              f"max_weight={float(valid.max()):.4f} at key {int(top3[0])}  "
              f"top3=[{top3.tolist()}] vals={top3_vals.tolist()}  "
              f"sum_future={leak_to_future:.2e}")

    print("For head 0, sampling queries at start / middle / end:")
    sample_positions = [0, L // 4, L // 2, 3 * L // 4, L - 3, L - 2, L - 1]
    for p in sample_positions:
        describe_row(p)
    print()
    print("Reading:")
    print("  entropy near uniform means softmax is flat (no meaningful selection).")
    print("  same top-3 keys across very different queries = attention sink / collapsed.")
    print("  sum_future > ~1e-6 means causal mask is leaking (bug).")
    print()


if __name__ == "__main__":
    main()
