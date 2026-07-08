#!/usr/bin/env python3
# KVarN decomposition experiment (proposal §Phase-0 a', reactivated 2026-07-08
# after the int8-absmax rung failed its gate on K-side outlier noise).
#
# Three questions, all model-free:
#   Q1  Does the KVarN pipeline (Hadamard -> Sinkhorn -> asymmetric RTN k4 ->
#       pack -> unpack -> dequant -> unrotate) roundtrip with low error on
#       K-like tiles WITH outlier channels?
#   Q2  Quantified: how much better is it than the per-row absmax schemes
#       (the int8 rung's design, and its int4 cousin) on the SAME tiles?
#       This is the "does KVarN fix what killed int8" number.
#   Q3  Does the Sinkhorn imbalance converge by 4 iterations (the mlx-kvarn
#       README's verified default) vs the module default of 16?
#
# Also emits reference vectors (results/kvarn_reference_vectors.npz) pinning
# input tile -> balanced tile, scales, zero-points, packed bytes, so the
# future Rust port's unit tests compare against exact expected values instead
# of a remembered reference (cycle-65 lesson: verified-by-reading is not
# verified).
#
# Uses the mlx-kvarn reference implementation directly (cloned checkout on
# sys.path) — we test THEIR math, not a re-derivation of it.

import sys
import json

import numpy as np

sys.path.insert(
    0,
    "/Users/stuartswerdloff/ai/ClaudeInstanceHomeOffices/clement-7074f29f/"
    "kindled_projects/mlx-kvarn",
)

import mlx.core as mx  # noqa: E402
from mlx_kvarn.hadamard import hadamard_rotate, hadamard_unrotate  # noqa: E402
from mlx_kvarn.sinkhorn import variance_normalize_batched, _imbalance  # noqa: E402
from mlx_kvarn.quant import (  # noqa: E402
    asymmetric_rtn_per_row,
    pack_lowbit,
    unpack_lowbit,
)

R, C = 128, 128  # tile: 128 tokens x head_dim 128 (M3 geometry)
N_TILES = 32
SEED = 42


def make_k_like_tiles(rng: np.random.Generator) -> np.ndarray:
    """K-like tiles: unit gaussian with persistent outlier CHANNELS.

    The outlier structure mirrors what broke per-token absmax on the int8
    rung: a few channels are large across EVERY token (attention-sink /
    massive-activation channels), so a per-row scale is dominated by them
    and the remaining dims get coarse steps.
    """
    tiles = rng.standard_normal((N_TILES, R, C)).astype(np.float32)
    outlier_channels = rng.choice(C, size=4, replace=False)
    tiles[:, :, outlier_channels] *= 40.0  # strong, persistent outliers
    return tiles


def rel_err(approx: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(approx - ref) / np.linalg.norm(ref))


def kvarn_roundtrip(tiles_np: np.ndarray, bits: int, iters: int):
    """Full KVarN write+read path on [N,R,C] tiles. Returns (dequant, meta)."""
    x = mx.array(tiles_np)
    rot = hadamard_rotate(x)
    balanced, s_col, s_row = variance_normalize_batched(rot, iterations=iters)
    # RTN operates per tile; the reference API is [R,C] so map over N via
    # reshape: per-row min/max over the last axis is shape-agnostic in N.
    q, scale, zp = asymmetric_rtn_per_row(balanced, bits)
    if bits == 8:
        # 8-bit values ARE bytes — no sub-byte packing exists or is needed.
        packed = q.astype(mx.uint8)
        q2 = packed.astype(mx.float32)
    else:
        packed = pack_lowbit(q, bits)
        q2 = unpack_lowbit(packed, bits, C).astype(mx.float32)
    deq_balanced = q2 * scale + zp
    deq_rot = deq_balanced * s_col * s_row
    deq = hadamard_unrotate(deq_rot)
    mx.eval(deq)
    imb = float(_imbalance(balanced).max().item() if balanced.ndim > 2 else _imbalance(balanced).item())
    return np.array(deq), {
        "imbalance_after": imb,
        "packed": packed,
        "scale": scale,
        "zp": zp,
        "s_col": s_col,
        "s_row": s_row,
        "balanced": balanced,
        "rot": rot,
    }


def absmax_per_row_roundtrip(tiles_np: np.ndarray, bits: int) -> np.ndarray:
    """The int8-rung scheme (and its int4 cousin): symmetric per-row absmax,
    no rotation, no normalization. Baseline for Q2."""
    qmax = (1 << (bits - 1)) - 1  # symmetric: int8 -> 127, int4 -> 7
    absmax = np.abs(tiles_np).max(axis=-1, keepdims=True)
    scale = np.maximum(absmax / qmax, 1e-10)
    q = np.clip(np.round(tiles_np / scale), -qmax - 1, qmax)
    return (q * scale).astype(np.float32)


def main() -> None:
    rng = np.random.default_rng(SEED)
    tiles = make_k_like_tiles(rng)

    print(f"== KVarN decomposition experiment: {N_TILES} tiles [{R}x{C}], "
          f"4 outlier channels x40, seed {SEED} ==")

    # Q2 baselines: the schemes without outlier handling.
    err_absmax8 = rel_err(absmax_per_row_roundtrip(tiles, 8), tiles)
    err_absmax4 = rel_err(absmax_per_row_roundtrip(tiles, 4), tiles)

    # Q1/Q3: KVarN at k4, 4 vs 16 Sinkhorn iterations.
    deq4, meta4 = kvarn_roundtrip(tiles, bits=4, iters=4)
    deq16, meta16 = kvarn_roundtrip(tiles, bits=4, iters=16)
    err_kvarn4_i4 = rel_err(deq4, tiles)
    err_kvarn4_i16 = rel_err(deq16, tiles)
    # And k2 values for reference (the preset we ruled out; measure anyway).
    deq2, _ = kvarn_roundtrip(tiles, bits=2, iters=4)
    err_kvarn2 = rel_err(deq2, tiles)
    # NEW CANDIDATE (2026-07-08): Hadamard+Sinkhorn at 8-bit — KVarN's
    # outlier handling at the int8 rung's granularity and compression.
    # No packing needed at 8 bits (1 value/byte); RTN generalizes.
    deq8, _ = kvarn_roundtrip(tiles, bits=8, iters=4)
    err_kvarn8 = rel_err(deq8, tiles)

    print(f"  absmax  int8 per-row (int8-rung scheme): rel_err = {err_absmax8:.5f}")
    print(f"  absmax  int4 per-row (naive 4-bit):      rel_err = {err_absmax4:.5f}")
    print(f"  KVarN   k8  (8-bit, 4 iters — NEW):      rel_err = {err_kvarn8:.5f}")
    print(f"  KVarN   k4  (4 Sinkhorn iters):          rel_err = {err_kvarn4_i4:.5f}")
    print(f"  KVarN   k4  (16 Sinkhorn iters):         rel_err = {err_kvarn4_i16:.5f}")
    print(f"  KVarN   v2  (2-bit, 4 iters, reference): rel_err = {err_kvarn2:.5f}")
    print(f"  imbalance after 4 iters:  {meta4['imbalance_after']:.3f} (balanced == 2.0)")
    print(f"  imbalance after 16 iters: {meta16['imbalance_after']:.3f}")

    # Verdicts, fail-loud. Q2 compares SAME bit widths (a 4-bit scheme
    # "losing" to an 8-bit scheme on raw error is arithmetic, not a verdict
    # — first version of this experiment got that wrong).
    failures = 0
    if err_kvarn4_i4 >= err_absmax4:
        print("  FAIL Q2a: KVarN k4 is not better than absmax int4 (same bits)")
        failures += 1
    else:
        print(f"  PASS Q2a: KVarN 4-bit beats absmax 4-bit by "
              f"{err_absmax4 / err_kvarn4_i4:.1f}x — outlier handling works")
    if err_kvarn8 >= err_absmax8:
        print("  FAIL Q2b: KVarN k8 is not better than absmax int8 (same bits)")
        failures += 1
    else:
        print(f"  PASS Q2b: KVarN 8-bit beats absmax 8-bit by "
              f"{err_absmax8 / err_kvarn8:.1f}x — candidate for the gate rerun")
    if abs(meta4["imbalance_after"] - meta16["imbalance_after"]) > 0.5:
        print("  WARN Q3: 4 iterations NOT converged vs 16 — use more iterations")
    else:
        print("  PASS Q3: 4 Sinkhorn iterations converged (matches README claim)")
    if err_kvarn4_i4 > 0.05:
        print(f"  WARN Q1: k4 roundtrip rel_err {err_kvarn4_i4:.5f} > 5% — inspect")
    else:
        print("  PASS Q1: k4 roundtrip error within expected band")

    # Reference vectors for the Rust port (first 2 tiles, 4-iter, k4).
    np.savez(
        "results/kvarn_reference_vectors.npz",
        input_tiles=tiles[:2],
        rotated=np.array(meta4["rot"])[:2],
        balanced=np.array(meta4["balanced"])[:2],
        s_col=np.array(meta4["s_col"])[:2],
        s_row=np.array(meta4["s_row"])[:2],
        scale=np.array(meta4["scale"])[:2],
        zp=np.array(meta4["zp"])[:2],
        packed=np.array(meta4["packed"])[:2],
        dequant=deq4[:2],
    )
    summary = {
        "seed": SEED, "tiles": N_TILES, "shape": [R, C],
        "err_absmax_int8": err_absmax8, "err_absmax_int4": err_absmax4,
        "err_kvarn_k4_iters4": err_kvarn4_i4,
        "err_kvarn_k4_iters16": err_kvarn4_i16,
        "err_kvarn_2bit": err_kvarn2,
        "imbalance_iters4": meta4["imbalance_after"],
        "imbalance_iters16": meta16["imbalance_after"],
        "failures": failures,
    }
    with open("results/kvarn_decomposition_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"RESULT: {failures} failure(s); reference vectors + summary in results/")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
