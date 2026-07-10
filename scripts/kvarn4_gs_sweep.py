#!/usr/bin/env python3
# KVarN4 stage-1c: GROUP-SIZE sweep — the lever the kills never tested.
#
# Stuart's directive (2026-07-10 night): "If there is a way to make KVarN4
# work, explore it. It would double the number of Kindled who could use it
# on one machine. And even if it's only valid for shorter contexts, that
# could be used for subagents."
#
# Both prior kills (stage 1: k4-K, 14.5% flips; stage 1b: K8V4, marginal
# gate) quantized at ONE group per row — per-row RTN over d=128 is
# gs=128. Finer groups are the standard 4-bit fix: each group's RTN spans
# half (gs=64) or a quarter (gs=32) of the dynamic range. C's gather_qmm
# serves gs∈{32,64,128} natively, so a passing variant rides the existing
# serve path; only the write path (grouped RTN) is engine work, gated
# behind this screen as always.
#
# Variants (pre-registered): k4 × gs∈{128,64,32} at Sinkhorn iters=4 (the
# verified setting), plus gs∈{64,32} at iters=16 (coarser codes may
# benefit from tighter balancing — the iters=4-vs-16 no-difference result
# was measured AT k8). k8/gs128 is the anchor. Pipeline order is the
# reference's exactly (rotate → Sinkhorn → RTN → pack/unpack → dequant →
# un-rotate); only RTN granularity varies. One lever; Sinkhorn×group
# interaction is recorded as follow-up, not smuggled in.
#
# Gates per variant (all pre-registered, same classes as stages 1/1b):
#   PRODUCTION tier (the vessel's continuous cognition):
#     K-side:  recon p95 < 0.10 AND argmax flips < 0.05
#     V-side (composed K8V4'): out p95 < 0.10 AND marginal-over-K8V8 < 0.05
#     full (composed K4'V4'):  out p95 < 0.10 AND marginal-over-K8V8 < 0.05
#   SUBAGENT tier (short context ≤ ~32K ≈ 250 tiles; task-scoped quality —
#   compounding bounded, so the marginal gate is EXCLUDED by design):
#     K-side flips < 0.05 AND composed out p95 < 0.10
#   Tile screens remain necessary-not-sufficient: any pass buys the
#   real-tile re-screen at engine boot, then engine A/B. Nothing here
#   authorizes engine time.
#
# Memory arithmetic is printed per variant (codes + per-group scalars +
# s_col + m3_idx) so every pass/fail pairs with its capacity price —
# "double the Kindled" gets numbers, not vibes.

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx  # noqa: E402

from kvarn4_tile_screen import (  # noqa: E402
    C,
    N_QUERIES,
    QUERY_SEED,
    R,
    TILE_SEED,
    build_population,
    kvarn_roundtrip,
    per_tile_rel_err,
    score_metrics,
    softmax,
)
from mlx_kvarn.hadamard import hadamard_rotate, hadamard_unrotate  # noqa: E402
from mlx_kvarn.quant import pack_lowbit, unpack_lowbit  # noqa: E402
from mlx_kvarn.sinkhorn import variance_normalize_batched  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESULTS_JSON = REPO / "results" / "kvarn4_gs_sweep_summary.json"

GATE_RECON_P95 = 0.10
GATE_FLIP_RATE = 0.05
GATE_OUT_P95 = 0.10
GATE_MARGINAL_P95 = 0.05

KV_HEADS, IDX_DIM = 4, 128  # M3 geometry, for the memory table


def rtn_grouped(balanced: mx.array, bits: int, gs: int):
    """Reference asymmetric RTN, per GROUP of gs along the last axis.

    Identical math to mlx_kvarn.quant.asymmetric_rtn_per_row (min/max
    range, eps 1e-10, round-clip) — gs == C reproduces it exactly.
    Returns q [N,R,C] int32 and per-group scale/zp broadcast back to
    [N,R,C] so the dequant chain below stays elementwise.
    """
    n, r, c = balanced.shape
    g = balanced.reshape(n, r, c // gs, gs)
    qmax = (1 << bits) - 1
    lo = mx.min(g, axis=-1, keepdims=True)
    hi = mx.max(g, axis=-1, keepdims=True)
    scale = mx.maximum((hi - lo) / qmax, 1e-10)
    zp = lo
    q = mx.clip(mx.round((g - zp) / scale), 0, qmax).astype(mx.int32)
    bcast = (n, r, c // gs, gs)
    return (
        q.reshape(n, r, c),
        mx.broadcast_to(scale, bcast).reshape(n, r, c),
        mx.broadcast_to(zp, bcast).reshape(n, r, c),
    )


def roundtrip_grouped(tiles_np: np.ndarray, bits: int, gs: int, iters: int) -> np.ndarray:
    """Stage-1 pipeline with grouped RTN; mirrors kvarn_roundtrip exactly."""
    x = mx.array(tiles_np)
    rot = hadamard_rotate(x)
    balanced, s_col, s_row = variance_normalize_batched(rot, iterations=iters)
    q, scale, zp = rtn_grouped(balanced, bits, gs)
    q2 = unpack_lowbit(pack_lowbit(q, bits), bits, C).astype(mx.float32)
    deq_rot = (q2 * scale + zp) * s_row * s_col
    deq = hadamard_unrotate(deq_rot)
    mx.eval(deq)
    return np.array(deq)


def bytes_per_token_layer(bits: int, gs: int) -> dict:
    """Per token per layer (both K+V sides, KV_HEADS heads) + m3_idx."""
    codes = KV_HEADS * C * bits // 8 * 2
    groups = C // gs
    scalars = KV_HEADS * (2 * groups + 1) * 4 * 2  # scale+zp per group, s_row per row
    s_col = KV_HEADS * C * 4 * 2 // R  # per-tile, amortized over R tokens
    idx = IDX_DIM * 2
    return {"codes": codes, "scalars": scalars, "s_col": s_col, "idx": idx,
            "total": codes + scalars + s_col + idx}


def out_err(out_hat, out_clean):
    n = out_clean.shape[0]
    d = np.linalg.norm((out_hat - out_clean).reshape(n, -1), axis=1)
    r = np.linalg.norm(out_clean.reshape(n, -1), axis=1)
    return d / np.maximum(r, 1e-12)


def main() -> None:
    rng = np.random.default_rng(TILE_SEED)
    tiles, _names, _sizes = build_population(rng)
    n_tiles = tiles.shape[0]
    qrng = np.random.default_rng(QUERY_SEED)
    queries = (
        qrng.standard_normal((n_tiles, N_QUERIES, C))
        .astype(np.float16)
        .astype(np.float32)
    )
    print(
        f"== KVarN4 stage-1c group-size sweep: {n_tiles} tiles [{R}x{C}], "
        f"{N_QUERIES} fp16 queries/tile, seeds ({TILE_SEED},{QUERY_SEED}) =="
    )

    s_clean = np.einsum("nqd,ntd->nqt", queries, tiles) / np.sqrt(C)
    w_clean = softmax(s_clean)
    top1_clean = s_clean.argmax(axis=-1)
    out_clean = np.einsum("nqt,ntd->nqd", w_clean, tiles)

    # Anchor: validated production k8 (reference roundtrip, gs=128, iters=4).
    deq8 = kvarn_roundtrip(tiles, 8)
    w_k8 = softmax(np.einsum("nqd,ntd->nqt", queries, deq8) / np.sqrt(C))
    anchor_out = np.einsum("nqt,ntd->nqd", w_k8, deq8)
    anchor_p95 = float(np.percentile(out_err(anchor_out, out_clean), 95))
    mem8 = bytes_per_token_layer(8, 128)
    print(f"anchor K8V8 composed p95 {anchor_p95:.5f}; k8 {mem8['total']}B/token/layer")

    variants = [(128, 4), (64, 4), (32, 4), (64, 16), (32, 16)]
    rows, results = [], {}
    for gs, iters in variants:
        tag = f"k4/gs{gs}/it{iters}"
        deq4 = roundtrip_grouped(tiles, 4, gs, iters)
        rel = per_tile_rel_err(deq4, tiles)
        sm, _d, _f, _v = score_metrics(queries, tiles, deq4, w_clean, s_clean, top1_clean)
        w_k4 = softmax(np.einsum("nqd,ntd->nqt", queries, deq4) / np.sqrt(C))
        asym = out_err(np.einsum("nqt,ntd->nqd", w_k8, deq4), out_clean)
        full = out_err(np.einsum("nqt,ntd->nqd", w_k4, deq4), out_clean)
        mem = bytes_per_token_layer(4, gs)
        m = {
            "recon_p95": float(np.percentile(rel, 95)),
            "flips": sm["flip_rate"],
            "asym_p95": float(np.percentile(asym, 95)),
            "full_p95": float(np.percentile(full, 95)),
            "bytes": mem["total"],
            "vs_k8_mem": mem8["total"] / mem["total"],
        }
        k_ok = m["recon_p95"] < GATE_RECON_P95 and m["flips"] < GATE_FLIP_RATE
        asym_prod = (
            m["asym_p95"] < GATE_OUT_P95
            and (m["asym_p95"] - anchor_p95) < GATE_MARGINAL_P95
        )
        full_prod = (
            k_ok
            and m["full_p95"] < GATE_OUT_P95
            and (m["full_p95"] - anchor_p95) < GATE_MARGINAL_P95
        )
        sub = k_ok and m["full_p95"] < GATE_OUT_P95
        # Subagent tier for the ASYM composition: K stays k8, so the tier's
        # flips gate is k8's own (0.88%, passes by construction); only the
        # composed absolute gate applies. This evaluates the pre-registered
        # tier for the K8V4 role — it was defined above, just never scored.
        asym_sub = m["asym_p95"] < GATE_OUT_P95
        m["verdicts"] = {
            "K_production": k_ok,
            "K8V4_production": asym_prod,
            "K8V4_subagent": asym_sub,
            "K4V4_production": full_prod,
            "K4V4_subagent": sub,
        }
        results[tag] = m
        rows.append(
            f"| {tag:<14} | {m['recon_p95']:.5f} | {m['flips']*100:6.3f}% | "
            f"{m['asym_p95']:.5f} | {m['full_p95']:.5f} | {mem['total']}B "
            f"({m['vs_k8_mem']:.2f}x) | K:{'PASS' if k_ok else 'fail'} "
            f"asymP:{'PASS' if asym_prod else 'fail'} "
            f"fullP:{'PASS' if full_prod else 'fail'} "
            f"sub:{'PASS' if sub else 'fail'} asymSub:{'PASS' if asym_sub else 'fail'} |"
        )

    print("\n| variant | recon p95 | flips | K8V4' p95 | K4'V4' p95 | B/tok/layer | verdicts |")
    print("|---|---|---|---|---|---|---|")
    for r_ in rows:
        print(r_)
    print(
        f"\ngates: recon<{GATE_RECON_P95} flips<{GATE_FLIP_RATE} out<{GATE_OUT_P95} "
        f"marginal<{GATE_MARGINAL_P95} (subagent tier: no marginal gate); "
        f"anchor p95 {anchor_p95:.5f}"
    )

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(
        json.dumps(
            {
                "seeds": {"tile": TILE_SEED, "query": QUERY_SEED},
                "anchor_p95": anchor_p95,
                "k8_bytes": mem8,
                "variants": results,
            },
            indent=2,
        )
    )
    print(f"summary JSON: {RESULTS_JSON}")
    any_pass = any(
        v["verdicts"]["K4V4_subagent"] or v["verdicts"]["K8V4_production"] or v["verdicts"]["K8V4_subagent"]
        for v in results.values()
    )
    sys.exit(0 if any_pass else 1)


if __name__ == "__main__":
    main()
