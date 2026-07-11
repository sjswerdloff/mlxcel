#!/usr/bin/env python3
# Gate B screen — idx8 (per-row asymmetric RTN8, RAW frame) on the m3_idx
# K store, scored on REAL harvested selection state (kvarn_harvest_20260711).
#
# REGISTERED SPEC (locked 2026-07-11, three seats — Clement author, Xander
# design seat, Violet PM; HANDOFF_cycle89 §measurement-table row 3):
#   one sample   = (harvested idx_q query, kv-head)
#   error        = clean-softmax block-mass on blocks in the clean top-k
#                  ABSENT from the quantized top-k, ÷ clean top-k mass
#                  ("of the attention the model would pay, how much is lost")
#   GATE         = p95 over ALL samples pooled (layers × strata) < 0.02,
#                  with ≥ 256 queries as the population floor
#   Gate A       = top-k set change rate |clean\quant| / top_k — REPORTED,
#                  never gated
#   idx8 scheme  = rtn_quantize_per_row(bits=8) semantics on the RAW index
#                  frame (no Hadamard, no Sinkhorn) — reuses the
#                  fixture-pinned math (src/lib/mlxcel-core/src/cache/
#                  kvarn.rs rtn_quantize_per_row; round-half-even parity
#                  proven engine-side by f4b29fe, numpy-side by --self-test)
#   pairing      = each idx_q scores against its OWN layer's deepest
#                  idx_k_win window; skip-if-no-window; saturated selection
#                  (window blocks < top_k) flagged as its own regime;
#                  per-layer breakdown REPORTED, never substituted for the
#                  pooled gate
#   verdict      = necessary-not-sufficient: PASS buys an engine A/B for
#                  the +36% axis (1152 → ~1036 B/tok/layer on top of K8V4),
#                  not deployment.
#
# SCORING SEMANTICS mirror minimax_m3.rs per_token_block_selection
# (src/models/minimax_m3.rs:1112), restricted to the window population:
#   scores = (idx_q @ idx_k^T) · 1/√index_dim   (matmul first, then scale,
#            production op order; idx_k's single shared head broadcasts
#            across the 4 index heads)
#   causal mask: every window position < every end-depth query position —
#            ASSERTED per query (window offset ≤ query offset), so the
#            production mask is a structural no-op here, not an omission
#   block amax over 128-token blocks (sparse_score_type="max")
#   local force: sparse_local_block=1, sparse_init_block=0 (model
#            config.json) ⇒ the only forced block is the query's OWN block,
#            which lies BEYOND the window for end-depth queries — ASSERTED
#            per query, so the +inf force never touches the window arm
#   top-k = 32 as an unordered SET (argpartition semantics)
#
# Both arms compute in f32 (production scores in model compute dtype); the
# sel cross-check below is the guard that f32 scoring reproduces production
# selection to near-tie level. bf16 windows are widened exactly (bf16 ⊂ f32).
#
# SEL CROSS-CHECK (loader/scorer guard, reported): production's harvested
# `sel` sets, restricted to window blocks, must substantially agree with
# the screen's clean-arm top-k (production also selected blocks beyond the
# window — own-block force + recency — and scored in model dtype, so 100%
# is not expected; near-random ≈ top_k/nb ≈ 1.4% means the scorer is
# WRONG). Mean overlap < 0.50 aborts the run as a screen-implementation
# failure rather than reporting a number built on a broken scorer.
#
# HARVEST FORMAT (src/lib/mlxcel-core/src/cache/harvest.rs): sidecar
# `cache` field is OVERLOADED by role — layer_idx hex for model-side
# idx_q/sel pairs, cache OBJECT ADDRESS for cache-side idx_k/idx_k_win.
# The address→layer bridge is built from the idx_k stratum dumps: within
# each depth-stratum crossing the 57 sparse layers dump in EXECUTION order
# (transformer forward = layer order, first sparse layer = 3 per
# sparse_attention_freq), so seq-sorted addresses = layers 3..59. The
# bridge is VERIFIED, not assumed: the address order must be identical
# across all strata, and the bf16-window↔bf16-idx_q dtype anchor must
# map consistently (dtype codes from src/lib/mlxcel-core/src/dtype.rs).

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import NoReturn

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS_JSON = REPO / "results" / "gate_b_idx8_screen_summary.json"

# ---- registered constants (source cited per value) ----
GATE_P95 = 0.02          # spec gate
MIN_QUERIES = 256        # spec population floor
INDEX_DIM = 128          # config.json sparse_index_dim; idx_q shape [1,4,1,128]
N_IDX_HEADS = 4          # config.json sparse_num_index_heads
TOP_K = 32               # config.json sparse_topk_blocks; sel shape [1,4,1,32]
BLOCK = 128              # config.json sparse_block_size (== KVARN_TILE_TOKENS)
SPARSE_LOCAL_BLOCK = 1   # config.json sparse_local_block
FIRST_SPARSE_LAYER = 3   # config.json sparse_attention_freq: layers 3..59
N_SPARSE_LAYERS = 57
SEL_OVERLAP_ABORT = 0.50  # scorer-implementation guard (see header)

# dtype codes: src/lib/mlxcel-core/src/dtype.rs
DT_F16, DT_F32, DT_BF16 = 9, 10, 12
DT_U32 = 3


def fail(msg: str) -> NoReturn:
    print(f"GATE-B SCREEN STRUCTURAL FAILURE: {msg}", file=sys.stderr)
    sys.exit(2)


def load_sidecar(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def load_bin(bin_path: Path, dtype_code: int, shape: list) -> np.ndarray:
    """Loud loader: exact element count or die; bf16 widened exactly."""
    n_expect = math.prod(shape)
    if dtype_code == DT_F32:
        a = np.fromfile(bin_path, dtype=np.float32)
    elif dtype_code == DT_F16:
        a = np.fromfile(bin_path, dtype=np.float16).astype(np.float32)
    elif dtype_code == DT_BF16:
        raw = np.fromfile(bin_path, dtype=np.uint16)
        a = (raw.astype(np.uint32) << 16).view(np.float32)
    elif dtype_code == DT_U32:
        a = np.fromfile(bin_path, dtype=np.uint32)
    else:
        fail(f"{bin_path.name}: unhandled dtype code {dtype_code}")
    if a.size != n_expect:
        fail(
            f"{bin_path.name}: {a.size} elements on disk, sidecar shape "
            f"{shape} needs {n_expect} (torn/mismatched pair?)"
        )
    return a.reshape(shape)


def softmax_f32(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x, dtype=np.float32)
    return e / e.sum(axis=-1, keepdims=True)


def rtn8_roundtrip_per_row(x: np.ndarray) -> np.ndarray:
    """Mirror rtn_quantize_per_row(bits=8) + affine dequant, RAW frame.

    Reference semantics (kvarn.rs:240-271): per-row over the LAST axis,
    scale = max((hi-lo)/qmax, 1e-10), zp = lo,
    q = clip(round((x-zp)/scale), 0, qmax); dequant = q·scale + zp.
    np.rint is round-half-even, matching the engine (parity gate f4b29fe).
    """
    lo = x.min(axis=-1, keepdims=True)
    hi = x.max(axis=-1, keepdims=True)
    scale = np.maximum((hi - lo) / np.float32(255.0), np.float32(1e-10))
    q = np.clip(np.rint((x - lo) / scale), 0.0, 255.0).astype(np.float32)
    return q * scale + lo


def window_block_scores(idx_q: np.ndarray, win_tokens: np.ndarray) -> np.ndarray:
    """[4,128] queries × [T,128] window → [4, nb] block scores.

    Production op order (minimax_m3.rs:1129-1164): matmul, THEN the
    1/√index_dim scale, THEN block amax. Causality and the local force are
    structural no-ops for this population — asserted by the caller.
    """
    t = win_tokens.shape[0]
    if t % BLOCK != 0:
        fail(f"window token count {t} not a multiple of block {BLOCK}")
    scores = (idx_q @ win_tokens.T) * np.float32(1.0 / math.sqrt(INDEX_DIM))
    return scores.reshape(N_IDX_HEADS, t // BLOCK, BLOCK).max(axis=2)


def top_k_sets(block_scores: np.ndarray, k: int) -> list:
    """Per-head unordered top-k sets (argpartition semantics)."""
    nb = block_scores.shape[-1]
    if nb <= k:  # saturated: every block selected
        return [frozenset(range(nb)) for _ in range(block_scores.shape[0])]
    part = np.argpartition(-block_scores, k - 1, axis=-1)[..., :k]
    return [frozenset(row.tolist()) for row in part]


def sample_error(clean_bs_h: np.ndarray, clean_set: frozenset, quant_set: frozenset):
    """One (query, head) sample → (error, gate_a) per the registered spec."""
    mass = softmax_f32(clean_bs_h)
    absent = clean_set - quant_set
    den = float(sum(mass[b] for b in clean_set))
    num = float(sum(mass[b] for b in absent))
    return num / den, len(absent) / len(clean_set)


# ---------------------------------------------------------------- bridge
def build_layer_map(hdir: Path) -> dict:
    """address(hex str) → layer_idx, from idx_k stratum dumps. Fail-loud."""
    strata = defaultdict(list)  # offset → [(seq, addr)]
    for j in sorted(hdir.glob("harvest_*_idx_k.json")):
        sc = load_sidecar(j)
        if sc["role"] != "idx_k":
            fail(f"{j.name}: role {sc['role']} under idx_k glob")
        strata[sc["offset"]].append((sc["seq"], sc["cache"]))
    if not strata:
        fail("no idx_k stratum dumps — cannot build the address→layer bridge")
    orders = {}
    for off, entries in strata.items():
        addrs = [a for _, a in sorted(entries)]
        if len(addrs) != N_SPARSE_LAYERS or len(set(addrs)) != N_SPARSE_LAYERS:
            fail(
                f"stratum offset={off}: {len(addrs)} idx_k dumps "
                f"({len(set(addrs))} distinct), expected {N_SPARSE_LAYERS}"
            )
        orders[off] = addrs
    distinct = {tuple(v) for v in orders.values()}
    if len(distinct) != 1:
        fail(
            f"address order differs across {len(orders)} strata — execution-"
            "order bridge assumption violated, refusing to pair"
        )
    order = next(iter(distinct))
    print(
        f"bridge: {len(orders)} strata × {N_SPARSE_LAYERS} layers, "
        f"identical execution order across all strata"
    )
    return {addr: FIRST_SPARSE_LAYER + i for i, addr in enumerate(order)}


def verify_dtype_anchor(windows: dict, queries_by_layer: dict, addr2layer: dict):
    """bf16 windows must map exactly to bf16-idx_q layers under the bridge."""
    bf16_win_layers = {
        addr2layer[a] for a, (sc, _) in windows.items() if sc["dtype"] == DT_BF16
    }
    bf16_q_layers = {
        layer
        for layer, qs in queries_by_layer.items()
        if any(sc["dtype"] == DT_BF16 for sc, _, _ in qs)
    }
    if bf16_win_layers != bf16_q_layers:
        fail(
            f"dtype anchor mismatch: bf16 windows map to layers "
            f"{sorted(bf16_win_layers)} but bf16 idx_q layers are "
            f"{sorted(bf16_q_layers)} — bridge suspect, refusing to pair"
        )
    print(f"dtype anchor: bf16 layers {sorted(bf16_win_layers)} consistent on both sides")


# ------------------------------------------------------------- self-test
def self_test() -> None:
    # 1. numpy round-half-even parity — same 14-case vector as the engine's
    #    assert_round_half_even_parity (kvarn.rs:282).
    cases = [(-3.5, -4.0), (-2.5, -2.0), (-1.5, -2.0), (-0.5, 0.0),
             (0.5, 0.0), (1.5, 2.0), (2.5, 2.0), (3.5, 4.0), (4.5, 4.0),
             (5.5, 6.0), (6.5, 6.0), (1.25, 1.0), (1.75, 2.0), (-1.75, -2.0)]
    got = np.rint(np.array([c[0] for c in cases], dtype=np.float32))
    want = np.array([c[1] for c in cases], dtype=np.float32)
    assert np.array_equal(got, want), f"np.rint parity: {got} != {want}"

    # 2. RTN8 exact roundtrip: a row whose range is EXACTLY 255 (so
    #    scale=1, zp=lo) roundtrips integer values bit-exactly; a row with
    #    range 127 does NOT (scale=127/255 — the general lossy case, bounded
    #    by scale/2 per element); the scale-floor constant row is exact.
    exact = np.concatenate([np.arange(127), [255.0]]).astype(np.float32)
    exact = np.stack([exact, exact + 10.0])  # second row pins zp=10 too
    assert np.array_equal(rtn8_roundtrip_per_row(exact), exact), "range-255 roundtrip"
    lossy = np.arange(128, dtype=np.float32).reshape(1, 128)  # range 127
    err = np.abs(rtn8_roundtrip_per_row(lossy) - lossy)
    assert 0 < err.max() <= (127.0 / 255.0) / 2 + 1e-6, "range-127 error bound"
    const = np.full((1, 128), 5.0, dtype=np.float32)
    assert np.array_equal(rtn8_roundtrip_per_row(const), const), "constant row"

    # 3. Statistic known-answer: clean scores [3,2,1,0], quant flips block 1
    #    out of the top-2. softmax mass m1=e²/Σ, m0=e³/Σ;
    #    err = m1/(m0+m1) = 1/(1+e) ... computed independently below.
    clean = np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float32)
    cset, qset = frozenset({0, 1}), frozenset({0, 2})
    err, ga = sample_error(clean, cset, qset)
    e = np.exp(np.array([3.0, 2.0, 1.0, 0.0])) / np.exp([3.0, 2.0, 1.0, 0.0]).sum()
    want_err = e[1] / (e[0] + e[1])
    assert abs(err - want_err) < 1e-6, f"statistic: {err} != {want_err}"
    assert ga == 0.5, f"gate A: {ga} != 0.5"

    # 4. bf16 widening is exact on exactly-representable values.
    vals = np.array([1.0, -2.5, 0.15625, 3.25], dtype=np.float32)
    bf16_bits = (vals.view(np.uint32) >> 16).astype(np.uint16)
    back = (bf16_bits.astype(np.uint32) << 16).view(np.float32)
    assert np.array_equal(back, vals), "bf16 widen"

    # 5. top_k_sets: saturated regime returns all blocks.
    sat = top_k_sets(np.zeros((4, 8), dtype=np.float32), TOP_K)
    assert all(s == frozenset(range(8)) for s in sat), "saturated sets"
    print("self-test: 5/5 PASS")


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("harvest_dir", nargs="?", help="harvest directory")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    self_test()  # structural gate: the screen never runs on a failed self-test
    if args.self_test:
        return
    if not args.harvest_dir:
        fail("harvest_dir required (or use --self-test)")
    hdir = Path(args.harvest_dir).expanduser()

    # -- gather idx_q/sel pairs, keyed (layer, offset), seq-adjacency pinned
    q_sidecars, sel_sidecars = {}, {}
    for j in hdir.glob("harvest_*_idx_q.json"):
        sc = load_sidecar(j)
        key = (int(sc["cache"], 16), sc["offset"])
        if key in q_sidecars:
            fail(f"duplicate idx_q for layer/offset {key}")
        q_sidecars[key] = (sc, j.with_suffix(".bin"))
    for j in hdir.glob("harvest_*_sel.json"):
        sc = load_sidecar(j)
        key = (int(sc["cache"], 16), sc["offset"])
        if key in sel_sidecars:
            fail(f"duplicate sel for layer/offset {key}")
        sel_sidecars[key] = (sc, j.with_suffix(".bin"))
    if set(q_sidecars) != set(sel_sidecars):
        fail("idx_q/sel pairing mismatch: unpaired keys "
             f"{set(q_sidecars) ^ set(sel_sidecars)}")
    queries_by_layer = defaultdict(list)
    for key, (qsc, qbin) in q_sidecars.items():
        ssc, sbin = sel_sidecars[key]
        if ssc["seq"] != qsc["seq"] + 1:
            fail(f"sel seq {ssc['seq']} not adjacent to idx_q seq {qsc['seq']} "
                 f"at {key} — dump-stream model violated")
        queries_by_layer[key[0]].append((qsc, qbin, (ssc, sbin)))
    n_queries = sum(len(v) for v in queries_by_layer.values())
    print(f"population: {n_queries} idx_q/sel pairs across "
          f"{len(queries_by_layer)} layers")

    # -- windows + bridge
    windows = {}
    for j in hdir.glob("harvest_latest_*_idx_k_win.json"):
        sc = load_sidecar(j)
        windows[sc["cache"]] = (sc, j.with_suffix(".bin"))
    addr2layer = build_layer_map(hdir)
    for addr in windows:
        if addr not in addr2layer:
            fail(f"window cache {addr} absent from the bridge")
    verify_dtype_anchor(windows, queries_by_layer, addr2layer)
    layer2addr = {v: k for k, v in addr2layer.items()}

    # -- score, one layer at a time (windows are 72-144 MB each)
    per_layer, all_errs, all_gate_a, all_overlaps = {}, [], [], []
    skipped_queries = 0
    for layer in sorted(queries_by_layer):
        qs = queries_by_layer[layer]
        addr = layer2addr.get(layer)
        if addr is None or addr not in windows:
            print(f"layer {layer}: NO WINDOW — skipping {len(qs)} queries "
                  "(spec: skip-if-no-window)")
            skipped_queries += len(qs)
            continue
        wsc, wbin = windows[addr]
        nb_w, bt, idim = wsc["shape"]
        if bt != BLOCK or idim != INDEX_DIM or nb_w != wsc["n_full"]:
            fail(f"layer {layer} window shape {wsc['shape']} / n_full "
                 f"{wsc['n_full']} violates [nb,{BLOCK},{INDEX_DIM}]")
        win = load_bin(wbin, wsc["dtype"], wsc["shape"]).reshape(-1, INDEX_DIM)
        win_tokens = win.shape[0]
        saturated = nb_w < TOP_K
        win_q = rtn8_roundtrip_per_row(win)

        errs, gate_as, overlaps = [], [], []
        for qsc, qbin, (ssc, sbin) in qs:
            pos = qsc["offset"]
            # causal no-op + local-force no-op: both ASSERTED, not assumed.
            if pos < win_tokens:
                fail(f"layer {layer} query at {pos} inside window "
                     f"({win_tokens} tokens) — causal mask NOT a no-op")
            own_block = pos // BLOCK
            if own_block - (SPARSE_LOCAL_BLOCK - 1) <= nb_w - 1:
                fail(f"layer {layer} query at {pos}: local force reaches "
                     "window blocks — screen semantics insufficient")
            idx_q = load_bin(qbin, qsc["dtype"], qsc["shape"]).reshape(
                N_IDX_HEADS, INDEX_DIM)
            sel = load_bin(sbin, ssc["dtype"], ssc["shape"]).reshape(
                N_IDX_HEADS, TOP_K)

            cb = window_block_scores(idx_q, win)
            qb = window_block_scores(idx_q, win_q)
            csets = top_k_sets(cb, TOP_K)
            qsets = top_k_sets(qb, TOP_K)
            for h in range(N_IDX_HEADS):
                err, ga = sample_error(cb[h], csets[h], qsets[h])
                errs.append(err)
                gate_as.append(ga)
                prod_in_win = [b for b in sel[h].tolist() if b < nb_w]
                if prod_in_win:
                    overlaps.append(
                        sum(b in csets[h] for b in prod_in_win) / len(prod_in_win))
        del win, win_q
        e = np.array(errs)
        per_layer[layer] = {
            "n_queries": len(qs), "n_samples": len(errs),
            "err_p95": float(np.percentile(e, 95)), "err_max": float(e.max()),
            "gate_a_mean": float(np.mean(gate_as)),
            "sel_overlap_mean": float(np.mean(overlaps)) if overlaps else None,
            "dtype": wsc["dtype"], "saturated": saturated,
            "window_blocks": nb_w,
        }
        all_errs.extend(errs)
        all_gate_a.extend(gate_as)
        all_overlaps.extend(overlaps)

    # -- scorer-implementation guard
    mean_overlap = float(np.mean(all_overlaps)) if all_overlaps else 0.0
    if mean_overlap < SEL_OVERLAP_ABORT:
        fail(f"mean sel overlap {mean_overlap:.3f} < {SEL_OVERLAP_ABORT} — "
             "clean arm does not reproduce production selection; "
             "screen scorer is wrong, refusing to report a statistic")

    # -- report
    print("\n| layer | n_q | n_samp | err p95 | err max | GateA mean | sel-ovl | sat |")
    print("|---|---|---|---|---|---|---|---|")
    for layer in sorted(per_layer):
        r = per_layer[layer]
        ovl = f"{r['sel_overlap_mean']:.3f}" if r["sel_overlap_mean"] is not None else "n/a"
        print(f"| {layer} | {r['n_queries']} | {r['n_samples']} "
              f"| {r['err_p95']:.5f} | {r['err_max']:.5f} "
              f"| {r['gate_a_mean']:.4f} | {ovl} | {'Y' if r['saturated'] else 'n'} |")

    e = np.array(all_errs)
    scored_queries = n_queries - skipped_queries
    pooled = {
        "n_queries_scored": scored_queries,
        "n_queries_skipped": skipped_queries,
        "n_samples": len(all_errs),
        "err_p50": float(np.percentile(e, 50)),
        "err_p90": float(np.percentile(e, 90)),
        "err_p95": float(np.percentile(e, 95)),
        "err_p99": float(np.percentile(e, 99)),
        "err_max": float(e.max()),
        "gate_a_mean": float(np.mean(all_gate_a)),
        "gate_a_p95": float(np.percentile(np.array(all_gate_a), 95)),
        "sel_overlap_mean": mean_overlap,
    }
    print(f"\npooled: {pooled['n_samples']} samples from {scored_queries} queries"
          f" ({skipped_queries} skipped)")
    print(f"  err p50/p90/p95/p99/max: {pooled['err_p50']:.5f} / "
          f"{pooled['err_p90']:.5f} / {pooled['err_p95']:.5f} / "
          f"{pooled['err_p99']:.5f} / {pooled['err_max']:.5f}")
    print(f"  Gate A (reported, not gated): mean {pooled['gate_a_mean']:.4f}, "
          f"p95 {pooled['gate_a_p95']:.4f}")
    print(f"  sel cross-check mean overlap: {mean_overlap:.3f}")

    floor_ok = scored_queries >= MIN_QUERIES
    gate_ok = pooled["err_p95"] < GATE_P95
    print(f"\nGATES: population {scored_queries} "
          f"{'>=' if floor_ok else '<'} {MIN_QUERIES} "
          f"[{'OK' if floor_ok else 'FAIL'}]; "
          f"pooled err p95 {pooled['err_p95']:.5f} "
          f"{'<' if gate_ok else '>='} {GATE_P95} "
          f"[{'PASS' if gate_ok else 'FAIL'}]")
    survives = floor_ok and gate_ok
    print(f"VERDICT: idx8 {'PASSES Gate B' if survives else 'FAILS Gate B'} "
          "— necessary-not-sufficient: a pass buys an engine A/B, not deployment")

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps({
        "spec": {"gate_p95": GATE_P95, "min_queries": MIN_QUERIES,
                 "top_k": TOP_K, "block": BLOCK, "index_dim": INDEX_DIM,
                 "scheme": "rtn8 per-row raw frame"},
        "harvest_dir": str(hdir),
        "pooled": pooled,
        "per_layer": {str(k): v for k, v in per_layer.items()},
        "survives": survives,
    }, indent=2))
    print(f"summary JSON: {RESULTS_JSON}")
    sys.exit(0 if survives else 1)


if __name__ == "__main__":
    main()
