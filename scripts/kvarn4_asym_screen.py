#!/usr/bin/env python3
# KVarN4 stage-1b: ASYMMETRIC composition screen — K8 scores × V4 values.
#
# Stage 1 (kvarn4_tile_screen.py) killed SYMMETRIC k4 on the score side
# (14.5% argmax flips vs the 5% gate) but its V-side gate PASSED (V-sum
# p95 0.0885 < 0.1). That V-sum used softmax weights from CLEAN scores —
# an upper-bound-friendly proxy. This screen composes the ACTUAL
# asymmetric pipeline a K8V4 cache mode would run:
#
#   weights = softmax(q · dequant_k8(K)^T / sqrt(C))   ← real k8 score noise
#   out     = weights · dequant_k4(V)                  ← real k4 value noise
#
# measured against out_clean = softmax(q·K^T/√C) · V per tile, with the
# K8V8 composition as the ANCHOR (tonight's validated production shape) so
# the candidate's number is read as marginal-cost-over-anchor, not just an
# absolute. K4V4 is included as dead-context.
#
# Same population, same seeds, same reference pipeline as stage 1 —
# imported from it, not copied, so the two screens cannot drift.
#
# Gates (kill-test thresholds, same class as stage 1's V gate):
#   composed K8V4 out-err p95 < 0.10          (absolute, stage-1 gate class)
#   composed K8V4 p95 − K8V8 p95 < 0.05       (marginal cost of V4 bounded)
#
# PASS graduates the asymmetric candidate to the next stage (real-tile
# confirmation + engine A/B per Violet's staging plan — NO engine time is
# authorized by this screen alone; the stage-1 kill record for k4-K BINDS
# and is untouched by this result either way).
#
# Scope note (why this is script-only): C's V-side gather_qmm serves
# bits=4 natively, but STORAGE at 4 bits means nibble-packed cache fields
# and a bits=4 layout-identity verification (the fold agent proved bits=8
# only) — that is engine work gated behind this screen.

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kvarn4_tile_screen import (  # noqa: E402
    C,
    N_QUERIES,
    QUERY_SEED,
    R,
    SINKHORN_ITERS,
    TILE_SEED,
    build_population,
    kvarn_roundtrip,
    softmax,
)

REPO = Path(__file__).resolve().parent.parent
RESULTS_MD = REPO / "RESULTS_kvarn4_asym_screen_2026-07-10.md"
RESULTS_JSON = REPO / "results" / "kvarn4_asym_screen_summary.json"

GATE_OUT_P95 = 0.10        # absolute composed-output rel-err gate
GATE_MARGINAL_P95 = 0.05   # candidate p95 minus anchor p95


def per_tile_out_err(out_hat: np.ndarray, out_clean: np.ndarray) -> np.ndarray:
    n = out_clean.shape[0]
    d = np.linalg.norm((out_hat - out_clean).reshape(n, -1), axis=1)
    r = np.linalg.norm(out_clean.reshape(n, -1), axis=1)
    return d / np.maximum(r, 1e-12)


def main() -> None:
    rng = np.random.default_rng(TILE_SEED)
    tiles, group_names, group_sizes = build_population(rng)
    n_tiles = tiles.shape[0]
    qrng = np.random.default_rng(QUERY_SEED)
    queries = (
        qrng.standard_normal((n_tiles, N_QUERIES, C))
        .astype(np.float16)
        .astype(np.float32)
    )
    print(
        f"== KVarN4 stage-1b asymmetric screen: {n_tiles} tiles [{R}x{C}], "
        f"Sinkhorn {SINKHORN_ITERS} iters, {N_QUERIES} fp16 queries/tile, "
        f"tile_seed={TILE_SEED}, query_seed={QUERY_SEED} =="
    )

    s_clean = np.einsum("nqd,ntd->nqt", queries, tiles) / np.sqrt(C)
    out_clean = np.einsum("nqt,ntd->nqd", softmax(s_clean), tiles)

    deq8 = kvarn_roundtrip(tiles, 8)
    deq4 = kvarn_roundtrip(tiles, 4)
    w_k8 = softmax(np.einsum("nqd,ntd->nqt", queries, deq8) / np.sqrt(C))
    w_k4 = softmax(np.einsum("nqd,ntd->nqt", queries, deq4) / np.sqrt(C))

    comps = {
        "K8V8 (anchor)": np.einsum("nqt,ntd->nqd", w_k8, deq8),
        "K8V4 (candidate)": np.einsum("nqt,ntd->nqd", w_k8, deq4),
        "K4V4 (dead, context)": np.einsum("nqt,ntd->nqd", w_k4, deq4),
    }

    summary, per_tile = {}, {}
    print("\n| composition | out-err mean | p95 | max |")
    print("|---|---|---|---|")
    for name, out in comps.items():
        e = per_tile_out_err(out, out_clean)
        per_tile[name] = e
        summary[name] = {
            "mean": float(e.mean()),
            "p95": float(np.percentile(e, 95)),
            "max": float(e.max()),
        }
        m = summary[name]
        print(f"| {name:<20} | {m['mean']:.5f} | {m['p95']:.5f} | {m['max']:.5f} |")

    print("\nPer-group breakdown (out-err mean):")
    print("| group | n | K8V8 | K8V4 | K4V4 |")
    print("|---|---|---|---|---|")
    ofs = 0
    groups_json = []
    for nm, sz in zip(group_names, group_sizes):
        row = {"group": nm, "n": sz}
        cells = []
        for cname in comps:
            v = float(per_tile[cname][ofs : ofs + sz].mean())
            row[cname] = v
            cells.append(f"{v:.5f}")
        groups_json.append(row)
        print(f"| {nm} | {sz} | {' | '.join(cells)} |")
        ofs += sz

    anchor_p95 = summary["K8V8 (anchor)"]["p95"]
    cand_p95 = summary["K8V4 (candidate)"]["p95"]
    marginal = cand_p95 - anchor_p95
    gate_abs = cand_p95 < GATE_OUT_P95
    gate_marg = marginal < GATE_MARGINAL_P95
    survives = gate_abs and gate_marg
    print(
        f"\nGATES: candidate p95 {cand_p95:.5f} "
        f"{'<' if gate_abs else '>='} {GATE_OUT_P95} "
        f"[{'PASS' if gate_abs else 'FAIL'}]; "
        f"marginal-over-anchor {marginal:.5f} "
        f"{'<' if gate_marg else '>='} {GATE_MARGINAL_P95} "
        f"[{'PASS' if gate_marg else 'FAIL'}]"
    )
    print(f"VERDICT: {'K8V4 SURVIVES stage-1b' if survives else 'K8V4 KILLED at stage-1b'}")

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(
        json.dumps(
            {
                "seeds": {"tile": TILE_SEED, "query": QUERY_SEED},
                "gates": {
                    "out_p95": GATE_OUT_P95,
                    "marginal_p95": GATE_MARGINAL_P95,
                },
                "summary": summary,
                "groups": groups_json,
                "survives": survives,
            },
            indent=2,
        )
    )
    print(f"summary JSON: {RESULTS_JSON}")
    sys.exit(0 if survives else 1)


if __name__ == "__main__":
    main()
