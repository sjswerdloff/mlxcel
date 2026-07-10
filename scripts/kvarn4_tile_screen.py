#!/usr/bin/env python3
# KVarN4 stage-1 tile-level quality screen (2026-07-10).
#
# KVarN8 (8-bit) is validated. Question: does KVarN at bits=4 (16x coarser
# quantization steps) survive at TILE level, before anyone invests engine
# time? This is a cheap kill-test: model-free, tile-scale arrays only,
# seconds to run, a few hundred MB of RAM at most.
#
# For each of bits=8 and bits=4, every tile goes through the full reference
# KVarN pipeline (the same ops kvarn_decomposition_experiment.py used, from
# the mlx-kvarn reference implementation — we test THEIR math):
#
#   rotate (WHT along channel axis)
#     -> log-domain Sinkhorn variance normalization (4 iters, the verified
#        setting)
#     -> per-row asymmetric RTN at `bits`
#     -> pack/unpack (4-bit) or uint8 cast (8-bit)
#     -> dequant = (q * scale + zp) * s_row * s_col   (rotated frame)
#     -> un-rotate (WHT is self-inverse)
#
# Measured per tile, in the ORIGINAL frame:
#   a. reconstruction relative error ||x_hat - x||_F / ||x||_F
#   b. attention-score perturbation: 64 random fp16 queries per tile,
#      s = q . K^T / sqrt(d); distribution of |s_hat - s| and the
#      argmax-flip rate (top-1 scoring token within the tile changes)
#   c. V-side: with softmax weights from CLEAN scores, relative error of
#      the weighted V sum when V is the dequantized tile
#
# Output: printed three-way table (fp16 / k8 / k4) + a results markdown
# (RESULTS_kvarn4_tile_screen_2026-07-10.md) + a JSON summary in results/.
#
# fp16 baseline row is zeros BY DEFINITION (the cache baseline is fp16
# storage; we treat it as the reference, not re-measuring its own rounding).
#
# Seeds: TILE_SEED for the tile population, QUERY_SEED for queries. Both
# printed and written to the results file.

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

MLX_KVARN_PATH = (
    "/Users/stuartswerdloff/ai/ClaudeInstanceHomeOffices/clement-7074f29f/"
    "kindled_projects/mlx-kvarn"
)
sys.path.insert(0, MLX_KVARN_PATH)

import mlx.core as mx  # noqa: E402
from mlx_kvarn.hadamard import hadamard_rotate, hadamard_unrotate  # noqa: E402
from mlx_kvarn.sinkhorn import variance_normalize_batched  # noqa: E402
from mlx_kvarn.quant import (  # noqa: E402
    asymmetric_rtn_per_row,
    pack_lowbit,
    unpack_lowbit,
)

R, C = 128, 128          # tile: 128 tokens x head_dim 128 (M3 geometry)
SINKHORN_ITERS = 4       # verified setting (decomposition experiment Q3)
N_QUERIES = 64           # random fp16 queries per tile
TILE_SEED = 42           # matches prior KVarN experiments' convention
QUERY_SEED = 20260710
REPO = Path("/Users/stuartswerdloff/RustProjects/mlxcel")
RESULTS_MD = REPO / "RESULTS_kvarn4_tile_screen_2026-07-10.md"
RESULTS_JSON = REPO / "results" / "kvarn4_tile_screen_summary.json"

# Screen gates for the k4 verdict (stage-1 kill-test thresholds, not a
# final acceptance spec — stated in the results file as such).
GATE_RECON_P95 = 0.10   # p95 tile reconstruction rel-err < 10%
GATE_FLIP_RATE = 0.05   # < 5% of (tile, query) top-1 flips
GATE_VSUM_P95 = 0.10    # p95 V-weighted-sum rel-err < 10%


# ── tile population ──────────────────────────────────────────────────────────
# Same generator family as kvarn_decomposition_experiment.make_k_like_tiles
# (unit gaussian + persistent outlier CHANNELS — the attention-sink /
# massive-activation structure that killed the int8-absmax rung), extended
# into a graded population. Outlier channels are drawn PER TILE here (the
# decomposition experiment shared one draw across its 32-tile batch); this
# samples more channel positions without changing the per-tile structure.

def make_group(rng, n, n_outlier_ch, ch_mult, n_spikes=0, spike_mult=80.0):
    tiles = rng.standard_normal((n, R, C)).astype(np.float32)
    for i in range(n):
        if n_outlier_ch:
            ch = rng.choice(C, size=n_outlier_ch, replace=False)
            tiles[i, :, ch] *= ch_mult
        for _ in range(n_spikes):  # transient per-token spikes
            tiles[i, rng.integers(R), rng.integers(C)] *= spike_mult
    return tiles


def build_population(rng):
    groups = [
        ("gaussian-clean (no outliers)", make_group(rng, 128, 0, 1.0)),
        ("outlier-4ch-x40 (decomposition-expt structure)",
         make_group(rng, 128, 4, 40.0)),
        ("outlier-8ch-x100 (extreme)", make_group(rng, 64, 8, 100.0)),
        ("outlier-4ch-x40 + 16 token spikes x80",
         make_group(rng, 64, 4, 40.0, n_spikes=16)),
    ]
    names, sizes = [g[0] for g in groups], [g[1].shape[0] for g in groups]
    return np.concatenate([g[1] for g in groups]), names, sizes


# ── reference pipeline roundtrip (kvarn_decomposition_experiment pattern) ────

def kvarn_roundtrip(tiles_np: np.ndarray, bits: int) -> np.ndarray:
    """Full KVarN write+read path on [N,R,C]; returns dequant, original frame."""
    x = mx.array(tiles_np)
    rot = hadamard_rotate(x)
    balanced, s_col, s_row = variance_normalize_batched(
        rot, iterations=SINKHORN_ITERS
    )
    q, scale, zp = asymmetric_rtn_per_row(balanced, bits)
    if bits == 8:
        # 8-bit values ARE bytes — no sub-byte packing exists or is needed.
        q2 = q.astype(mx.uint8).astype(mx.float32)
    else:
        q2 = unpack_lowbit(pack_lowbit(q, bits), bits, C).astype(mx.float32)
    deq_rot = (q2 * scale + zp) * s_row * s_col
    deq = hadamard_unrotate(deq_rot)
    mx.eval(deq)
    return np.array(deq)


# ── metrics ──────────────────────────────────────────────────────────────────

def per_tile_rel_err(approx: np.ndarray, ref: np.ndarray) -> np.ndarray:
    d = np.linalg.norm((approx - ref).reshape(ref.shape[0], -1), axis=1)
    n = np.linalg.norm(ref.reshape(ref.shape[0], -1), axis=1)
    return d / np.maximum(n, 1e-12)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def score_metrics(queries, tiles, deq, w_clean, s_clean, top1_clean):
    """Score/V-side perturbation of dequant `deq` vs clean `tiles`.

    queries [N,Q,C] fp32(from fp16); tiles/deq [N,R,C];
    w_clean/s_clean [N,Q,R]; top1_clean [N,Q].
    Returns dict + per-tile arrays for group breakdowns.
    """
    s_hat = np.einsum("nqd,ntd->nqt", queries, deq) / np.sqrt(C)
    delta = np.abs(s_hat - s_clean)
    flips = (s_hat.argmax(axis=-1) != top1_clean)  # [N,Q]
    # V-side: clean-score softmax weights applied to clean vs dequant V.
    out_clean = np.einsum("nqt,ntd->nqd", w_clean, tiles)
    out_hat = np.einsum("nqt,ntd->nqd", w_clean, deq)
    vs_num = np.linalg.norm((out_hat - out_clean).reshape(len(tiles), -1), axis=1)
    vs_den = np.linalg.norm(out_clean.reshape(len(tiles), -1), axis=1)
    return {
        "delta_mean": float(delta.mean()),
        "delta_p95": float(np.percentile(delta, 95)),
        "delta_max": float(delta.max()),
        "flip_rate": float(flips.mean()),
        "vsum_mean": float((vs_num / vs_den).mean()),
        "vsum_p95": float(np.percentile(vs_num / vs_den, 95)),
        "vsum_max": float((vs_num / vs_den).max()),
    }, delta, flips, vs_num / vs_den


def summarize(rel, sm):
    return {
        "recon_mean": float(rel.mean()),
        "recon_p95": float(np.percentile(rel, 95)),
        "recon_max": float(rel.max()),
        **{k: sm[k] for k in
           ("delta_mean", "delta_p95", "delta_max", "flip_rate",
            "vsum_mean", "vsum_p95", "vsum_max")},
    }


def fmt_row(name, m):
    return (f"| {name:<12} | {m['recon_mean']:.5f} | {m['recon_p95']:.5f} | "
            f"{m['recon_max']:.5f} | {m['delta_mean']:.5f} | "
            f"{m['delta_p95']:.5f} | {m['delta_max']:.5f} | "
            f"{m['flip_rate']*100:6.3f}% | {m['vsum_mean']:.5f} | "
            f"{m['vsum_p95']:.5f} | {m['vsum_max']:.5f} |")


ZEROS = {k: 0.0 for k in ("recon_mean", "recon_p95", "recon_max",
                          "delta_mean", "delta_p95", "delta_max",
                          "flip_rate", "vsum_mean", "vsum_p95", "vsum_max")}


def main() -> None:
    rng = np.random.default_rng(TILE_SEED)
    tiles, group_names, group_sizes = build_population(rng)
    n_tiles = tiles.shape[0]

    qrng = np.random.default_rng(QUERY_SEED)
    # fp16 queries (spec), computed in fp32 thereafter: we are measuring
    # KV quantization error, not matmul accumulation error.
    queries = qrng.standard_normal((n_tiles, N_QUERIES, C)) \
        .astype(np.float16).astype(np.float32)

    print(f"== KVarN4 tile screen: {n_tiles} tiles [{R}x{C}], "
          f"Sinkhorn {SINKHORN_ITERS} iters, {N_QUERIES} fp16 queries/tile, "
          f"tile_seed={TILE_SEED}, query_seed={QUERY_SEED} ==")
    for nm, sz in zip(group_names, group_sizes):
        print(f"   {sz:>4} x {nm}")

    # Clean-score reference (shared by both bit widths).
    s_clean = np.einsum("nqd,ntd->nqt", queries, tiles) / np.sqrt(C)
    w_clean = softmax(s_clean)
    top1_clean = s_clean.argmax(axis=-1)
    mean_abs_s = float(np.abs(s_clean).mean())

    results, per_tile = {}, {}
    for bits, tag in ((8, "k8"), (4, "k4")):
        deq = kvarn_roundtrip(tiles, bits)
        rel = per_tile_rel_err(deq, tiles)
        sm, delta, flips, vsum = score_metrics(
            queries, tiles, deq, w_clean, s_clean, top1_clean
        )
        results[tag] = summarize(rel, sm)
        per_tile[tag] = {"rel": rel, "delta": delta, "flips": flips,
                         "vsum": vsum}

    header = ("| scheme       | recon mean | recon p95 | recon max | "
              "score|Δ| mean | score|Δ| p95 | score|Δ| max | "
              "argmax flips | V-sum mean | V-sum p95 | V-sum max |")
    sep = ("|--------------|-----------|-----------|-----------|"
           "-----------|-----------|-----------|---------|"
           "-----------|-----------|-----------|")
    table = [header, sep, fmt_row("fp16 (base)", ZEROS),
             fmt_row("KVarN k8", results["k8"]),
             fmt_row("KVarN k4", results["k4"])]
    print()
    print("\n".join(table))
    print(f"\n  context: mean |s| over clean scores = {mean_abs_s:.4f} "
          f"(score deltas above are ABSOLUTE)")

    # Per-group breakdown (k8 / k4), for interpretability across the graded
    # outlier population.
    group_rows = []
    off = 0
    ghdr = ("| group | n | k8 recon mean | k4 recon mean | k8 flips | "
            "k4 flips | k4 V-sum mean | mean |s| clean |")
    gsep = "|---|---|---|---|---|---|---|---|"
    print(f"\n{ghdr}\n{gsep}")
    for nm, sz in zip(group_names, group_sizes):
        sl = slice(off, off + sz)
        row = (f"| {nm} | {sz} | "
               f"{per_tile['k8']['rel'][sl].mean():.5f} | "
               f"{per_tile['k4']['rel'][sl].mean():.5f} | "
               f"{per_tile['k8']['flips'][sl].mean()*100:.3f}% | "
               f"{per_tile['k4']['flips'][sl].mean()*100:.3f}% | "
               f"{per_tile['k4']['vsum'][sl].mean():.5f} | "
               f"{np.abs(s_clean[sl]).mean():.4f} |")
        group_rows.append(row)
        print(row)
        off += sz

    # ── verdict (fail-loud, gates stated) ────────────────────────────────────
    k4 = results["k4"]
    checks = [
        ("recon p95", k4["recon_p95"], GATE_RECON_P95),
        ("argmax flip rate", k4["flip_rate"], GATE_FLIP_RATE),
        ("V-sum p95", k4["vsum_p95"], GATE_VSUM_P95),
    ]
    fails = [(n, v, g) for n, v, g in checks if v >= g]
    survives = not fails
    ratio = k4["recon_mean"] / max(results["k8"]["recon_mean"], 1e-12)
    verdict = ("k4 survives tile level" if survives
               else "k4 explodes at tile level")
    print(f"\nVERDICT: {verdict}")
    for n, v, g in checks:
        mark = "PASS" if v < g else "FAIL"
        print(f"  {mark}: k4 {n} = {v:.5f} (gate < {g})")
    print(f"  k4/k8 recon-error ratio: {ratio:.1f}x "
          f"(4-bit steps are 16x coarser)")

    # ── results markdown ─────────────────────────────────────────────────────
    try:
        kvarn_rev = subprocess.run(
            ["git", "-C", MLX_KVARN_PATH, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        kvarn_rev = "unknown"

    md = [
        "# KVarN4 stage-1 tile-level quality screen",
        "",
        f"Date: {date.today().isoformat()}  ",
        f"Script: `scripts/kvarn4_tile_screen.py`  ",
        f"Reference implementation: `mlx-kvarn` @ `{kvarn_rev}` "
        f"(`{MLX_KVARN_PATH}`)  ",
        f"mlx {mx.__version__}, numpy {np.__version__}, "
        f"python {sys.version.split()[0]}",
        "",
        "KVarN8 is validated; this screen asks whether KVarN at bits=4 "
        "(16x coarser steps) survives at tile level, before any engine "
        "investment. Full reference pipeline per tile: WHT rotation -> "
        f"log-domain Sinkhorn variance normalization ({SINKHORN_ITERS} "
        "iters, the verified setting) -> per-row asymmetric RTN at `bits` "
        "-> pack/unpack (4-bit) -> dequant `(q*scale+zp)*s_row*s_col` -> "
        "un-rotate. Errors measured in the ORIGINAL frame.",
        "",
        "## Tile population",
        "",
        f"{n_tiles} synthetic tiles, each [{R} tokens x {C} head_dim] "
        "(M3 geometry), generator family from "
        "`kvarn_decomposition_experiment.make_k_like_tiles` (unit gaussian "
        "+ persistent outlier channels — the structure that killed the "
        "int8-absmax rung):",
        "",
    ]
    md += [f"- {sz} x {nm}" for nm, sz in zip(group_names, group_sizes)]
    md += [
        "",
        "Outlier channels are drawn per tile (the decomposition experiment "
        "shared one draw across its batch); per-tile structure is "
        "identical. The pinned K0 fixture npz "
        "(`results/kvarn_reference_vectors.npz`) is not present in this "
        "clone — only the already-quantized gather fixtures are — so the "
        "population is fully synthetic via the reference generators.",
        "",
        f"Seeds: tile population `{TILE_SEED}`, queries `{QUERY_SEED}` "
        f"(numpy `default_rng`). {N_QUERIES} random fp16 query vectors per "
        "tile; scores `s = q . K^T / sqrt(128)` computed in fp32 (we "
        "measure quantization error, not accumulation error). V-side: "
        "softmax weights from CLEAN scores applied to clean vs dequantized "
        "V (the same tile through the same pipeline), relative error of "
        "the weighted sum per tile.",
        "",
        "## Three-way summary (all tiles pooled)",
        "",
        *table,
        "",
        f"Context: mean |s| over clean scores = {mean_abs_s:.4f}; the "
        "score-delta columns are ABSOLUTE errors. fp16 baseline row is "
        "zeros by definition (fp16 storage is the reference).",
        "",
        "- recon = per-tile ||x_hat - x||_F / ||x||_F (mean / p95 / max "
        "over tiles)",
        "- score|Δ| = |s_hat - s| pooled over all (tile, query, key) "
        "elements",
        "- argmax flips = fraction of (tile, query) pairs whose top-1 "
        "key within the tile changes",
        "- V-sum = per-tile relative error of softmax(clean)-weighted V "
        "sum",
        "",
        "## Per-group breakdown",
        "",
        ghdr, gsep, *group_rows,
        "",
        "## Verdict",
        "",
        f"**{verdict.upper()}**",
        "",
        "Stage-1 screen gates (kill-test thresholds, not final acceptance "
        "criteria):",
        "",
    ]
    for n, v, g in checks:
        mark = "PASS" if v < g else "FAIL"
        md.append(f"- {mark}: k4 {n} = **{v:.5f}** (gate < {g})")
    md += [
        "",
        f"- k4 mean reconstruction rel-err {k4['recon_mean']:.5f} vs k8 "
        f"{results['k8']['recon_mean']:.5f} — {ratio:.1f}x worse (4-bit "
        "steps are 16x coarser, so ~16x is the expected scaling).",
        f"- k4 argmax-flip rate {k4['flip_rate']*100:.3f}% vs k8 "
        f"{results['k8']['flip_rate']*100:.3f}% over "
        f"{n_tiles * N_QUERIES} (tile, query) pairs.",
        f"- k4 V-sum rel-err mean {k4['vsum_mean']:.5f} "
        f"(p95 {k4['vsum_p95']:.5f}, max {k4['vsum_max']:.5f}).",
        "",
        "### What tile-level CANNOT tell us",
        "",
        "- **Compounding over long contexts.** A 300K-token window is "
        "~2300 tiles per head per layer; per-tile error is independent "
        "here, but real decode accumulates quantized-K score noise across "
        "the whole window and across layers. Tile-level error bounds do "
        "not compose linearly into end-to-end logit error.",
        "- **Near-tie flips at full-window scale.** Argmax here is over "
        "128 keys within one tile. A real window competes ~300K keys; "
        "near-ties across tiles are more common, and softmax mass — not "
        "tile-local top-1 — is what matters. Random-query top-1 within a "
        "random tile is a coarse proxy in both directions.",
        "- **Distribution shift.** Tiles are synthetic (gaussian + planted "
        "outlier channels). Real K/V activations have correlated, "
        "layer-dependent structure (RoPE phase, low-rank structure, real "
        "attention-sink statistics) not modeled here.",
        "- **No end-to-end perplexity / task quality.** This is a "
        "necessary-not-sufficient screen: k4 failing here kills it "
        "cheaply; k4 passing here only buys the next experiment "
        "(engine-level A/B), not deployment.",
        "",
        f"Summary JSON: `results/{RESULTS_JSON.name}`",
        "",
    ]
    RESULTS_MD.write_text("\n".join(md))

    RESULTS_JSON.parent.mkdir(exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump({
            "date": date.today().isoformat(),
            "tile_seed": TILE_SEED, "query_seed": QUERY_SEED,
            "n_tiles": n_tiles, "shape": [R, C],
            "n_queries_per_tile": N_QUERIES,
            "sinkhorn_iters": SINKHORN_ITERS,
            "groups": [{"name": nm, "n": sz}
                       for nm, sz in zip(group_names, group_sizes)],
            "mean_abs_clean_score": mean_abs_s,
            "gates": {"recon_p95": GATE_RECON_P95,
                      "flip_rate": GATE_FLIP_RATE,
                      "vsum_p95": GATE_VSUM_P95},
            "k8": results["k8"], "k4": results["k4"],
            "verdict": verdict,
            "mlx_kvarn_rev": kvarn_rev,
        }, f, indent=2)

    print(f"\nresults: {RESULTS_MD}")
    print(f"summary: {RESULTS_JSON}")
    sys.exit(0 if survives else 1)


if __name__ == "__main__":
    main()
