#!/usr/bin/env python3
# Phase K0 — reference equivalence for dequant-after-gather
# (DESIGN_fused_msa_kvarn_decode_plan_2026-07-10.md; design §8.7).
#
# Core assumption under test, per Xander's structural note ("if K0 fails,
# K1 and K2 need redesign"):
#
#   Q1  DEQUANT-AFTER-GATHER == DEQUANT-ALL-THEN-GATHER, bit-identical.
#       Per-tile dequant is independent — no cross-tile interaction, no
#       accumulation-order dependency — so gathering selected tiles first
#       and dequantizing only those must produce identical values.
#   Q2  The Q/O ROTATION TRICK is numerically sound at op level:
#       scores(H·q, H·k) == scores(q, k) and the output un-rotation
#       recovers attention output computed in the standard frame, within
#       fp32 tolerance (exact equality is not expected: matmul in a
#       rotated basis sums the same products in a different order).
#   Q3  MIXED-FORMAT assembly (fp16 sink block + u8 tiles + fp16 tail)
#       through the gather path matches the full-window reference.
#
# Model-free, seconds to run. Emits fixtures pinning gather+dequant outputs
# for K1's Rust tests: results/kvarn_gather_reference_fixtures.npz

import sys

sys.path.insert(
    0,
    "/Users/stuartswerdloff/ai/ClaudeInstanceHomeOffices/clement-7074f29f/"
    "kindled_projects/mlx-kvarn",
)

import json

import numpy as np
import mlx.core as mx  # noqa: E402
from mlx_kvarn.hadamard import hadamard_rotate, hadamard_unrotate  # noqa: E402
from mlx_kvarn.sinkhorn import variance_normalize_batched  # noqa: E402
from mlx_kvarn.quant import asymmetric_rtn_per_row  # noqa: E402

# M3 geometry, matching kvarn.rs and the MSA config:
# sparse_block_size == KVARN_TILE_TOKENS == sink length == 128.
R, C = 128, 128  # tile: 128 tokens x head_dim 128
N_TILES = 16  # interior quantized tiles (blocks 1..16)
TAIL_LEN = 83  # partial local block (fp16, always attended)
SEED = 42
ITERS = 4
SELECTED_TILES = [2, 7, 11, 15]  # interior block indices selected by "MSA"


def make_k_like(rng, n_tok):
    x = rng.standard_normal((n_tok, C)).astype(np.float32)
    outlier_channels = rng.choice(C, size=4, replace=False)
    x[:, outlier_channels] *= 40.0
    return x


def quantize_tiles(tiles_np):
    """KVarN write path on [N,R,C] rotated tiles -> quantized parts."""
    x = mx.array(tiles_np)
    balanced, s_col, s_row = variance_normalize_batched(x, iterations=ITERS)
    q, scale, zp = asymmetric_rtn_per_row(balanced, 8)
    return {
        "q": q.astype(mx.uint8),
        "scale": scale,
        "zp": zp,
        "s_col": s_col,
        "s_row": s_row,
    }


def dequant_tiles(parts, idx=None):
    """Dequant (all tiles, or only tiles at `idx`) -> rotated-frame fp32."""
    sel = (lambda a: a[mx.array(idx)]) if idx is not None else (lambda a: a)
    q = sel(parts["q"]).astype(mx.float32)
    de = (q * sel(parts["scale"]) + sel(parts["zp"])) * sel(parts["s_row"])
    return de * sel(parts["s_col"])


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Build a rotated-frame cache the way update_kvarn8 does: rotate in the
    # standard frame, then sink stays fp16-rotated, interior quantizes,
    # tail stays fp16-rotated.
    sink_std = make_k_like(rng, R)
    tiles_std = make_k_like(rng, N_TILES * R).reshape(N_TILES, R, C)
    tail_std = make_k_like(rng, TAIL_LEN)

    rot = lambda a: hadamard_rotate(mx.array(a))  # noqa: E731
    sink_rot = rot(sink_std)
    tiles_rot = mx.array(
        np.stack([np.array(rot(tiles_std[i])) for i in range(N_TILES)])
    )
    tail_rot = rot(tail_std)
    parts = quantize_tiles(np.array(tiles_rot))

    # ── Q1: dequant-after-gather == dequant-all-then-gather ────────────────
    all_then_gather = np.array(dequant_tiles(parts))[SELECTED_TILES]
    gather_then_dequant = np.array(dequant_tiles(parts, idx=SELECTED_TILES))
    q1_bitwise = np.array_equal(all_then_gather, gather_then_dequant)

    # ── Q3: mixed-format assembly through the gather path ──────────────────
    # "MSA selected": block 0 (sink, fp16) + SELECTED_TILES + local (tail).
    full_window = np.concatenate(
        [np.array(sink_rot), np.array(dequant_tiles(parts)).reshape(-1, C),
         np.array(tail_rot)]
    )
    def block_slice(b):  # full-window token range of block b
        return full_window[b * R : b * R + (TAIL_LEN if b == N_TILES + 1 else R)]
    ref_blocks = [block_slice(0)] + [block_slice(t + 1) for t in SELECTED_TILES] \
        + [block_slice(N_TILES + 1)]
    gathered = [np.array(sink_rot)] + list(gather_then_dequant) + [np.array(tail_rot)]
    q3_bitwise = all(
        np.array_equal(a, b) for a, b in zip(ref_blocks, gathered)
    )

    # ── Q2: rotation-trick numerics on the gathered window ──────────────────
    # Standard frame reference: un-rotate everything, score with standard q.
    k_rot = np.concatenate([g.reshape(-1, C) for g in gathered])
    k_std = np.array(hadamard_unrotate(mx.array(k_rot)))
    q_std = make_k_like(rng, 1)
    q_rot = np.array(hadamard_rotate(mx.array(q_std)))
    scores_std = q_std @ k_std.T
    scores_rot = q_rot @ k_rot.T
    q2_score_maxerr = float(np.abs(scores_std - scores_rot).max()
                            / max(np.abs(scores_std).max(), 1e-9))
    # Output side: softmax(scores)·V, V rotated; un-rotate output once.
    v_rot = k_rot  # reuse as V-like data
    w = np.exp(scores_std - scores_std.max())
    w = w / w.sum()
    out_std = w @ np.array(hadamard_unrotate(mx.array(v_rot)))
    out_rot_then_unrot = np.array(hadamard_unrotate(mx.array(w @ v_rot)))
    q2_out_maxerr = float(np.abs(out_std - out_rot_then_unrot).max()
                          / max(np.abs(out_std).max(), 1e-9))

    q2_ok = q2_score_maxerr < 1e-5 and q2_out_maxerr < 1e-5

    results = {
        "q1_gather_dequant_bitwise_identical": q1_bitwise,
        "q3_mixed_format_assembly_bitwise": q3_bitwise,
        "q2_rotation_scores_rel_maxerr": q2_score_maxerr,
        "q2_rotation_output_rel_maxerr": q2_out_maxerr,
        "q2_within_fp32_tolerance": q2_ok,
        "selected_tiles": SELECTED_TILES,
        "geometry": {"tile": R, "head_dim": C, "n_tiles": N_TILES,
                     "tail": TAIL_LEN, "seed": SEED},
    }
    print(json.dumps(results, indent=2))

    np.savez(
        "results/kvarn_gather_reference_fixtures.npz",
        q_codes=np.array(parts["q"]),
        scale=np.array(parts["scale"]),
        zp=np.array(parts["zp"]),
        s_col=np.array(parts["s_col"]),
        s_row=np.array(parts["s_row"]),
        sink_rot=np.array(sink_rot),
        tail_rot=np.array(tail_rot),
        selected=np.array(SELECTED_TILES, dtype=np.int32),
        gathered_dequant=gather_then_dequant,
    )
    print("fixtures: results/kvarn_gather_reference_fixtures.npz")

    verdict = "K0 PASS — dequant-after-gather is semantics-preserving" if (
        q1_bitwise and q3_bitwise and q2_ok
    ) else "K0 FAIL — core assumption broken, K1/K2 need redesign"
    print(f"VERDICT: {verdict}")
    sys.exit(0 if (q1_bitwise and q3_bitwise and q2_ok) else 1)


if __name__ == "__main__":
    main()
