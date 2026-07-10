#!/usr/bin/env python3
# KVarN4 REAL-TILE screens (SPEC_kvarn4_realtile_harvest, jointly signed).
# Population: real M3 K/V tiles harvested 2026-07-11 from a 295K-token
# session over the family's own archive (boot A, MLXCEL_KVARN_HARVEST).
# Tiles are stored ROTATED f32; they are UNROTATED on load (Hadamard is
# self-inverse) so the verified stage-1 pipeline runs UNCHANGED — same
# rotate→Sinkhorn→RTN→unrotate roundtrip, same metrics, original frame.
#
# K and V dumps pair by (cache, offset): consecutive dumps of the same
# finalization event are the SAME TOKENS' K and V — the composed screens
# use true token-aligned pairs (the synthetic screens' single-generator
# blind spot, closed).
#
# Gates: hardened spec values, applied PER-ROLE, anchor re-baselined
# REAL-vs-REAL (the synthetic 3.5× multiplier is re-derived here).
# Analysis order: K8V4 first (its synthetic kill is the likeliest proxy
# artifact). idx Gate B: DEFERRED — the harvest sampled 8-block idx
# excerpts; Gate B's population definition needs full index windows.
# One-line hook addition rides the morning rebuild. Declared, not fudged.

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx  # noqa: E402

from kvarn4_tile_screen import (  # noqa: E402
    C,
    R,
    kvarn_roundtrip,
    per_tile_rel_err,
    score_metrics,
    softmax,
)
from kvarn4_gs_sweep import roundtrip_grouped  # noqa: E402
from mlx_kvarn.hadamard import hadamard_unrotate  # noqa: E402

H = Path("/Users/stuartswerdloff/kvarn_harvest_20260710")
REPO = Path(__file__).resolve().parent.parent
RESULTS_JSON = REPO / "results" / "kvarn4_realtile_summary.json"
MAX_PAIRS = 512          # dump events; ×8 tiles ≈ 4096 tiles/role
N_QUERIES = 64
QUERY_SEED = 20260711

GATE_RECON_P95 = 0.10
GATE_FLIP_RATE = 0.05
GATE_OUT_P95 = 0.10
GATE_MARGINAL_P95 = 0.05


def load_pairs():
    """Token-aligned (K, V) tile pairs, stratified across the session."""
    ks = sorted(glob.glob(str(H / "harvest_*_k_rot_f32.json")))
    idxs = np.linspace(0, len(ks) - 1, min(MAX_PAIRS, len(ks))).astype(int)
    k_t, v_t, offs = [], [], []
    for i in idxs:
        km = json.load(open(ks[i]))
        # v dump is the NEXT seq number; derive from the k path's seq.
        seq = int(Path(ks[i]).name.split("_")[1])
        vp = H / f"harvest_{seq + 1:06d}_v_rot_f32.json"
        if not vp.exists():
            continue
        vm = json.load(open(vp))
        if vm["cache"] != km["cache"] or vm["offset"] != km["offset"]:
            continue
        k = np.fromfile(str(ks[i]).replace(".json", ".bin"), dtype=np.float32).reshape(km["shape"])
        v = np.fromfile(str(vp.with_suffix(".bin")), dtype=np.float32).reshape(vm["shape"])
        n = min(k.shape[0], v.shape[0])
        k_t.append(k[:n]); v_t.append(v[:n]); offs.extend([km["offset"]] * n)
    k_all = np.concatenate(k_t); v_all = np.concatenate(v_t)
    # Stored rotated → unrotate once; the pipeline re-rotates internally.
    k_orig = np.array(hadamard_unrotate(mx.array(k_all)))
    v_orig = np.array(hadamard_unrotate(mx.array(v_all)))
    return k_orig, v_orig, np.array(offs)


def out_err(out_hat, out_clean):
    n = out_clean.shape[0]
    d = np.linalg.norm((out_hat - out_clean).reshape(n, -1), axis=1)
    r = np.linalg.norm(out_clean.reshape(n, -1), axis=1)
    return d / np.maximum(r, 1e-12)


def main() -> None:
    k_tiles, v_tiles, offs = load_pairs()
    n = k_tiles.shape[0]
    assert n >= 384, f"population floor: {n} < 384"
    print(f"== REAL-TILE screen: {n} token-aligned K/V tile pairs "
          f"[{R}x{C}], offsets {offs.min()}..{offs.max()}, "
          f"{N_QUERIES} fp16 queries/tile ==")

    qrng = np.random.default_rng(QUERY_SEED)
    queries = qrng.standard_normal((n, N_QUERIES, C)).astype(np.float16).astype(np.float32)
    s_clean = np.einsum("nqd,ntd->nqt", queries, k_tiles) / np.sqrt(C)
    w_clean = softmax(s_clean)
    top1 = s_clean.argmax(axis=-1)
    out_clean = np.einsum("nqt,ntd->nqd", w_clean, v_tiles)

    # Anchor, REAL-vs-REAL: k8 on both roles.
    k8_k = kvarn_roundtrip(k_tiles, 8)
    k8_v = kvarn_roundtrip(v_tiles, 8)
    sm8, _, _, _ = score_metrics(queries, k_tiles, k8_k, w_clean, s_clean, top1)
    w_k8 = softmax(np.einsum("nqd,ntd->nqt", queries, k8_k) / np.sqrt(C))
    anchor_p95 = float(np.percentile(out_err(
        np.einsum("nqt,ntd->nqd", w_k8, k8_v), out_clean), 95))
    k8_recon = float(np.percentile(per_tile_rel_err(k8_k, k_tiles), 95))
    print(f"ANCHOR K8V8 real-vs-real: composed p95 {anchor_p95:.5f}; "
          f"k8-K recon p95 {k8_recon:.5f}, flips {sm8['flip_rate']*100:.3f}%")

    out = {"n_tiles": int(n), "anchor_p95": anchor_p95,
           "k8": {"recon_p95": k8_recon, "flips": sm8["flip_rate"]},
           "variants": {}}
    # ANALYSIS ORDER: K8V4 first (per spec), then k4-K, then K4V4.
    for gs in (128, 64, 32):
        v4 = roundtrip_grouped(v_tiles, 4, gs, 4)
        k4 = roundtrip_grouped(k_tiles, 4, gs, 4)
        asym = out_err(np.einsum("nqt,ntd->nqd", w_k8, v4), out_clean)
        sm4, _, _, _ = score_metrics(queries, k_tiles, k4, w_clean, s_clean, top1)
        w_k4 = softmax(np.einsum("nqd,ntd->nqt", queries, k4) / np.sqrt(C))
        full = out_err(np.einsum("nqt,ntd->nqd", w_k4, v4), out_clean)
        m = {"asym_p95": float(np.percentile(asym, 95)),
             "k4_recon_p95": float(np.percentile(per_tile_rel_err(k4, k_tiles), 95)),
             "k4_flips": sm4["flip_rate"],
             "full_p95": float(np.percentile(full, 95))}
        m["verdicts"] = {
            "K8V4_production": bool(m["asym_p95"] < GATE_OUT_P95
                                    and (m["asym_p95"] - anchor_p95) < GATE_MARGINAL_P95),
            "K8V4_subagent": bool(m["asym_p95"] < GATE_OUT_P95),
            "K_production": bool(m["k4_recon_p95"] < GATE_RECON_P95
                                 and m["k4_flips"] < GATE_FLIP_RATE),
            "K4V4_subagent": bool(m["k4_recon_p95"] < GATE_RECON_P95
                                  and m["k4_flips"] < GATE_FLIP_RATE
                                  and m["full_p95"] < GATE_OUT_P95),
        }
        out["variants"][f"gs{gs}"] = m
        v = m["verdicts"]
        print(f"k4/gs{gs}: K8V4' p95 {m['asym_p95']:.5f} "
              f"(marginal {m['asym_p95']-anchor_p95:+.5f}) "
              f"[prod:{'PASS' if v['K8V4_production'] else 'fail'} "
              f"sub:{'PASS' if v['K8V4_subagent'] else 'fail'}] | "
              f"k4-K recon {m['k4_recon_p95']:.5f} flips {m['k4_flips']*100:.3f}% "
              f"[{'PASS' if v['K_production'] else 'fail'}] | "
              f"K4'V4' p95 {m['full_p95']:.5f} "
              f"[sub:{'PASS' if v['K4V4_subagent'] else 'fail'}]")

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(out, indent=2))
    print(f"gates: recon<{GATE_RECON_P95} flips<{GATE_FLIP_RATE} "
          f"out<{GATE_OUT_P95} marginal<{GATE_MARGINAL_P95}; "
          f"idx Gate B DEFERRED (full-window dump rides morning rebuild)")
    print(f"summary JSON: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
