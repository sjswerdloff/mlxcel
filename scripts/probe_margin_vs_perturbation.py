#!/usr/bin/env python3
# PROBE — margin vs perturbation decomposition of the Gate B tail
# (exploratory; informs the idx8 engine A/B design; GATES NOTHING).
#
# QUESTION (Xander, 2026-07-11 16:43): is the deep-layer selection
# fragility (Gate B per-layer tail, worst at layers 58/59) a property of
#   H1: the index vectors' GEOMETRY — near-ties at the top-k boundary
#       (thin margins), which ANY perturbation flips: the tail would
#       persist under any quantization scheme; or
#   H2: the QUANTIZATION scheme — idx8 per-row scales too coarse at
#       depth (wide row ranges → large score perturbation): a better
#       scheme would help.
#
# METHOD: both quantities live in the same units (block score, pre-
# softmax) and are separately measurable per sample from the BANKED
# harvest — no engine time:
#   margin_i = clean s_sorted[K-1] − s_sorted[K]   (top-k boundary gap)
#   delta_i  = p95 over window blocks of |quant_score − clean_score|
#   ratio_i  = delta_i / margin_i                  ("flip pressure")
# plus, per layer, the idx8 mechanism variable measured directly from
# the window: row range (hi−lo) per index-frame row (= 255 × quant
# scale). Cross-layer Spearman rank correlations of banked per-layer
# err_p95 against margin (H1 predicts NEGATIVE) and against delta /
# row-range (H2 predicts POSITIVE) separate the hypotheses; 57 layers.
#
# VALIDATION: reuses the reviewed screen's machinery VERBATIM
# (gate_b_idx8_screen.py @ 3cde7e8, sha1 10cff8fa) — loaders, bridge,
# dtype anchor, production-order scoring, idx8 roundtrip — and
# RECOMPUTES every per-layer err_p95, asserting EXACT match against the
# banked summary (results/gate_b_idx8_screen_summary.json @ 80addc7):
# the margins/deltas provably ride the same scoring path that passed
# the sel cross-check at overlap 1.000.
#
# PROVENANCE NOTE: exploratory-unreviewed-before-run (unlike the
# three-seat-gated screen). Self-test is the first act, screen
# discipline inherited. Results are DESIGN INPUT for the engine A/B,
# never a gate.

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_b_idx8_screen as gb  # noqa: E402  (reviewed @ 3cde7e8)

REPO = Path(__file__).resolve().parent.parent
OUT_JSON = REPO / "results" / "probe_margin_vs_perturbation_20260711.json"
ERR_MATCH_TOL = 1e-12  # same code path + data ⇒ exact modulo JSON repr


def sample_margin(clean_bs_h: np.ndarray, k: int) -> float:
    """Clean top-k boundary gap: s_sorted[k-1] − s_sorted[k] (ties → 0)."""
    s = np.sort(clean_bs_h)[::-1]
    return float(s[k - 1] - s[k])


def sample_delta_p95(clean_bs_h: np.ndarray, quant_bs_h: np.ndarray) -> float:
    return float(np.percentile(np.abs(quant_bs_h - clean_bs_h), 95))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation, average ranks for ties (no scipy dependency)."""
    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=float)
        # average tied ranks
        vals, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(vals))
        np.add.at(sums, inv, r)
        return sums[inv] / cnt[inv]
    rx, ry = ranks(x), ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if den == 0.0:
        gb.fail("spearman: zero-variance rank vector")
    return float((rx * ry).sum() / den)


def self_test() -> None:
    # margin: hand-built scores, k=3 → boundary gap is s[2]-s[3] = 5-3 = 2
    bs = np.array([9.0, 7.0, 5.0, 3.0, 1.0], dtype=np.float32)
    assert sample_margin(bs, 3) == 2.0, "margin extraction wrong"
    # tie at the boundary → margin 0 (maximal fragility)
    bt = np.array([9.0, 7.0, 5.0, 5.0, 1.0], dtype=np.float32)
    assert sample_margin(bt, 3) == 0.0, "tie margin must be 0"
    # delta: known perturbation of +0.5 everywhere → p95 = 0.5
    assert sample_delta_p95(bs, bs + 0.5) == 0.5, "delta wrong"
    # spearman: monotone up → +1; monotone down → −1; ties averaged
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(spearman(a, a * 3 + 1) - 1.0) < 1e-12
    assert abs(spearman(a, -a) + 1.0) < 1e-12
    print("probe self-test: 5/5")


def main() -> None:
    self_test()
    if len(sys.argv) != 3:
        gb.fail("usage: probe_margin_vs_perturbation.py <harvest_dir> <banked_summary.json>")
    hdir = Path(sys.argv[1]).expanduser()
    with open(sys.argv[2]) as f:
        banked = json.load(f)["per_layer"]

    # ---- load population EXACTLY as the screen does (same guards) ----
    from collections import defaultdict
    q_sidecars, sel_sidecars = {}, {}
    for j in hdir.glob("harvest_*_idx_q.json"):
        sc = gb.load_sidecar(j)
        key = (int(sc["cache"], 16), sc["offset"])
        if key in q_sidecars:
            gb.fail(f"duplicate idx_q for {key}")
        q_sidecars[key] = (sc, j.with_suffix(".bin"))
    for j in hdir.glob("harvest_*_sel.json"):
        sc = gb.load_sidecar(j)
        key = (int(sc["cache"], 16), sc["offset"])
        sel_sidecars[key] = (sc, j.with_suffix(".bin"))
    if set(q_sidecars) != set(sel_sidecars):
        gb.fail("idx_q/sel pairing mismatch")
    queries_by_layer = defaultdict(list)
    for key, (qsc, qbin) in q_sidecars.items():
        queries_by_layer[key[0]].append((qsc, qbin))

    windows = {}
    for j in hdir.glob("harvest_latest_*_idx_k_win.json"):
        sc = gb.load_sidecar(j)
        windows[sc["cache"]] = (sc, j.with_suffix(".bin"))
    addr2layer = gb.build_layer_map(hdir)
    layer2addr = {v: k for k, v in addr2layer.items()}

    per_layer = {}
    for layer_addr in sorted(queries_by_layer):
        qs = queries_by_layer[layer_addr]
        # layer index via bridge: idx_q sidecar 'cache' is layer_idx hex
        layer = layer_addr
        addr = layer2addr.get(layer)
        if addr is None or addr not in windows:
            print(f"layer {layer}: no window — skipped (mirrors screen)")
            continue
        wsc, wbin = windows[addr]
        win = gb.load_bin(wbin, wsc["dtype"], wsc["shape"]).reshape(-1, gb.INDEX_DIM)
        win_q = gb.rtn8_roundtrip_per_row(win)
        row_range = (win.max(axis=-1) - win.min(axis=-1))  # = 255 × idx8 scale

        errs, margins, deltas, ratios = [], [], [], []
        for qsc, qbin in qs:
            idx_q = gb.load_bin(qbin, qsc["dtype"], qsc["shape"]).reshape(
                gb.N_IDX_HEADS, gb.INDEX_DIM)
            cb = gb.window_block_scores(idx_q, win)
            qb = gb.window_block_scores(idx_q, win_q)
            csets = gb.top_k_sets(cb, gb.TOP_K)
            qsets = gb.top_k_sets(qb, gb.TOP_K)
            for h in range(gb.N_IDX_HEADS):
                err, _ = gb.sample_error(cb[h], csets[h], qsets[h])
                errs.append(err)
                m = sample_margin(cb[h], gb.TOP_K)
                d = sample_delta_p95(cb[h], qb[h])
                margins.append(m)
                deltas.append(d)
                ratios.append(d / max(m, 1e-12))
        del win, win_q

        e = np.array(errs)
        err_p95 = float(np.percentile(e, 95))
        bk = banked[str(layer)]["err_p95"]
        if abs(err_p95 - bk) > ERR_MATCH_TOL:
            gb.fail(
                f"layer {layer}: recomputed err_p95 {err_p95!r} != banked "
                f"{bk!r} — probe is NOT on the screen's scoring path, refusing"
            )
        per_layer[layer] = {
            "err_p95": err_p95,
            "margin_p05": float(np.percentile(margins, 5)),
            "margin_p50": float(np.percentile(margins, 50)),
            "delta_p50": float(np.percentile(deltas, 50)),
            "delta_p95": float(np.percentile(deltas, 95)),
            "ratio_p50": float(np.percentile(ratios, 50)),
            "ratio_p90": float(np.percentile(ratios, 90)),
            "row_range_p50": float(np.percentile(row_range, 50)),
            "row_range_p95": float(np.percentile(row_range, 95)),
            "n_samples": len(errs),
        }

    if len(per_layer) != gb.N_SPARSE_LAYERS:
        gb.fail(f"probe covered {len(per_layer)} layers, expected {gb.N_SPARSE_LAYERS}")
    print(f"err_p95 recomputation: {len(per_layer)}/{gb.N_SPARSE_LAYERS} layers "
          f"EXACT vs banked (tol {ERR_MATCH_TOL})")

    layers = sorted(per_layer)
    col = lambda k: np.array([per_layer[L][k] for L in layers])
    err = col("err_p95")
    corr = {
        "err_vs_margin_p50": spearman(err, col("margin_p50")),   # H1 ⇒ negative
        "err_vs_delta_p95": spearman(err, col("delta_p95")),     # H2 ⇒ positive
        "err_vs_row_range_p95": spearman(err, col("row_range_p95")),  # H2 mech
        "delta_vs_row_range": spearman(col("delta_p95"), col("row_range_p95")),
    }

    print("\nlayer  err_p95  margin_p50  delta_p95  ratio_p50  rowrange_p95")
    for L in layers:
        r = per_layer[L]
        print(f"{L:>5}  {r['err_p95']:.5f}  {r['margin_p50']:.5f}    "
              f"{r['delta_p95']:.5f}   {r['ratio_p50']:.3f}     "
              f"{r['row_range_p95']:.4f}")
    print("\nSpearman (57 layers):")
    for k, v in corr.items():
        print(f"  {k}: {v:+.3f}")

    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"per_layer": {str(L): per_layer[L] for L in layers},
                   "spearman": corr}, f, indent=1)
    print(f"\nbanked: {OUT_JSON}")


if __name__ == "__main__":
    main()
