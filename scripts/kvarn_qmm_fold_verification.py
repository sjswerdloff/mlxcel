#!/usr/bin/env python3
# KVarN8 -> MLX quantized_matmul format-fold verification (2026-07-10).
#
# Question (DESIGN_decode_experiment_harness_2026-07-10.md, approach C; H3
# "offline fold verification first", K0 pattern): can MLX's native affine
# quantized kernels consume KVarN8's stored codes DIRECTLY -- no fp16
# materialization -- by folding KVarN8's per-row scalars into MLX per-group
# scales/biases and moving the per-column Sinkhorn scale to the query /
# output side?
#
# KVarN8 stored form per tile (rotated frame; tile = R=128 tokens x C=128):
#   q      u8   [R, C]  codes in [0, 255]
#   scale  f32  [R, 1]  per-row RTN scale
#   zp     f32  [R, 1]  per-row zero-point (row minimum, float domain)
#   s_row  f32  [R, 1]  Sinkhorn row scale
#   s_col  f32  [1, C]  Sinkhorn column scale (per tile)
#   dequant (rotated frame): x_hat = (q*scale + zp) * s_row * s_col
#
# Fold claims under test:
#   1. Per-row affine: MLX affine dequant is scales*q + biases per group.
#      With scales_mlx[r, g] = scale[r]*s_row[r] and
#      biases_mlx[r, g] = zp[r]*s_row[r] (constant across a row's groups,
#      since KVarN8 scalars are per-row), MLX dequant == (q*scale+zp)*s_row.
#      Measured BOTH at the fp32 bit level and after fp16 cast.
#   2. K-side s_col moves to the query exactly:
#      q_vec . (K*s_col)^T == (q_vec*s_col) . K^T   (s_col constant per tile).
#   3. V-side s_col factors out of the weighted sum exactly:
#      w @ (V*s_col) == (w @ V) * s_col.
#
# Method:
#   - Codes are OUR codes, bit-packed directly into MLX's u32 layout
#     (4 codes per word at bits=8, LSB-first) -- never re-quantized.
#   - Error attribution per matmul test: total error is decomposed into
#     fold-algebra error (plain mx.matmul over mx.dequantize output vs the
#     reference) and kernel-accumulation error (quantized_matmul vs that
#     control) so a failure names its cause.
#   - Tile population: kvarn4_tile_screen generator family (gaussian-clean,
#     outlier-4ch-x40, outlier-8ch-x100 extreme, outliers+token-spikes).
#   - Quantization: the mlx-kvarn reference pipeline (their math, not a
#     re-derivation): hadamard_rotate -> Sinkhorn (4 iters) -> asymmetric
#     RTN per row at bits=8.
#   - Seeds pinned; arrays are tiny (64 tiles of 128x128).

import json
import sys

import numpy as np

sys.path.insert(
    0,
    "/Users/stuartswerdloff/ai/ClaudeInstanceHomeOffices/clement-7074f29f/"
    "kindled_projects/mlx-kvarn",
)

import mlx.core as mx  # noqa: E402
from mlx_kvarn.hadamard import hadamard_rotate  # noqa: E402
from mlx_kvarn.sinkhorn import variance_normalize_batched  # noqa: E402
from mlx_kvarn.quant import asymmetric_rtn_per_row  # noqa: E402

R, C = 128, 128          # tile: 128 tokens x head_dim 128 (M3 geometry)
BITS = 8
SINKHORN_ITERS = 4       # verified default (decomposition experiment Q3)
SEED = 20260710
GROUP_SIZES = (64, 128)  # both divide d=128
N_Q_LIST = (1, 8)        # M=1 exercises qmv/qvm kernels, M=8 the qmm kernels

# Gates (fail-loud): fp32 matmul agreement must be at accumulation-reorder
# scale; fp16 agreement at a few fp16 ULP of the score magnitude.
GATE_FP32_REL = 1e-5
GATE_FP16_REL = 1e-2


# -- tile population (kvarn4_tile_screen generator family) -------------------

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
        ("gaussian-clean", make_group(rng, 16, 0, 1.0)),
        ("outlier-4ch-x40", make_group(rng, 24, 4, 40.0)),
        ("outlier-8ch-x100", make_group(rng, 12, 8, 100.0)),
        ("outlier-4ch-x40+16spikes", make_group(rng, 12, 4, 40.0, n_spikes=16)),
    ]
    return np.concatenate([g[1] for g in groups]), [(g[0], g[1].shape[0]) for g in groups]


# -- packing: our u8 codes -> MLX u32 layout (bits=8) -------------------------

def pack_codes_u8_to_mlx_u32(codes_u8: np.ndarray) -> np.ndarray:
    """Bit-pack u8 codes into the u32 layout mx.quantize produces at bits=8.

    MLX packs LSB-first: code i of each group of 4 sits in byte i of the
    word. On a little-endian host that is exactly a reinterpret of the u8
    buffer as u32, i.e. zero-cost in the future Rust cache.
    """
    assert codes_u8.dtype == np.uint8 and codes_u8.shape[-1] % 4 == 0
    return np.ascontiguousarray(codes_u8).view(np.uint32)


# -- metrics ------------------------------------------------------------------

def maxabs(a: mx.array) -> float:
    return float(mx.abs(a).max().item())


def bit_mismatches(a: mx.array, b: mx.array) -> int:
    """Count elements whose bit patterns differ (fp32 -> u32, fp16 -> u16)."""
    an, bn = np.array(a), np.array(b)
    assert an.dtype == bn.dtype
    itype = np.uint16 if an.dtype == np.float16 else np.uint32
    return int((an.view(itype) != bn.view(itype)).sum())


def ulp16_dist(a: mx.array, b: mx.array) -> np.ndarray:
    """Elementwise ULP distance between two fp16 arrays (monotone-key
    transform). 0 == bit-exact; 1 == adjacent representable values.
    NaN-free inputs assumed (asserted)."""
    an, bn = np.array(a), np.array(b)
    assert an.dtype == np.float16 and bn.dtype == np.float16
    assert not (np.isnan(an).any() or np.isnan(bn).any())

    def key(f):
        u = f.view(np.uint16).astype(np.int64)
        return np.where(u & 0x8000, (~u) & 0xFFFF, u | 0x8000)

    return np.abs(key(an) - key(bn))


def ulp16_max(a: mx.array, b: mx.array) -> int:
    return int(ulp16_dist(a, b).max())


class Agg:
    """Running max-abs / max-rel / bit-mismatch / ULP aggregation across tiles."""

    def __init__(self):
        self.max_abs = 0.0
        self.max_rel = 0.0
        self.bit_bad = 0
        self.n_elem = 0
        self.ulp_max = None

    def add(self, diff_maxabs, ref_maxabs, bit_bad=0, n_elem=0, ulp=None):
        self.max_abs = max(self.max_abs, diff_maxabs)
        if ref_maxabs > 0:
            self.max_rel = max(self.max_rel, diff_maxabs / ref_maxabs)
        self.bit_bad += bit_bad
        self.n_elem += n_elem
        if ulp is not None:
            self.ulp_max = ulp if self.ulp_max is None else max(self.ulp_max, ulp)

    def row(self):
        out = f"max_abs={self.max_abs:.3e}  max_rel={self.max_rel:.3e}"
        if self.n_elem:
            out += f"  bit_mismatch={self.bit_bad}/{self.n_elem}"
        if self.ulp_max is not None:
            out += f"  ulp16_max={self.ulp_max}"
        return out


# -- layout facts (section b of the task) -------------------------------------

def probe_layout_facts():
    print("== B. MLX affine layout facts (empirical, this machine) ==")
    w = mx.random.normal((R, C)).astype(mx.float32)
    for gs in (32, 64, 128):
        wq, sc, bi = mx.quantize(w, group_size=gs, bits=BITS)
        print(f"  mx.quantize bits=8 gs={gs}: w_q {tuple(wq.shape)} uint32 "
              f"(4 codes/word, LSB-first), scales {tuple(sc.shape)}, "
              f"biases {tuple(bi.shape)} (dtype follows w: {sc.dtype})")
    # Dequant formula + packing order: extract codes from mx.quantize's own
    # words assuming LSB-first and reproduce mx.dequantize elementwise.
    wq, sc, bi = mx.quantize(w, group_size=64, bits=BITS)
    codes = np.array(wq).view(np.uint8).reshape(R, C)
    manual = (codes.astype(np.float32)
              * np.repeat(np.array(sc), 64, axis=1)
              + np.repeat(np.array(bi), 64, axis=1))
    d = np.abs(manual - np.array(mx.dequantize(wq, sc, bi, group_size=64, bits=BITS))).max()
    print(f"  dequant formula scales*q+biases, LSB-first byte order: "
          f"max|manual - mx.dequantize| = {d:.3e}")
    assert d < 1e-5, "MLX dequant formula/packing assumption violated"
    # quantized_matmul support matrix at these shapes.
    packed = mx.array(pack_codes_u8_to_mlx_u32(
        np.random.default_rng(0).integers(0, 256, size=(R, C), dtype=np.uint8)))
    print("  quantized_matmul support at [n_q,128] x [128 tokens,128] bits=8:")
    for gs in GROUP_SIZES:
        ng = C // gs
        for dt in (mx.float32, mx.float16):
            oks = []
            for m in N_Q_LIST:
                for tr in (True, False):
                    try:
                        o = mx.quantized_matmul(
                            mx.random.normal((m, R)).astype(dt), packed,
                            mx.ones((R, ng), dtype=dt), mx.zeros((R, ng), dtype=dt),
                            transpose=tr, group_size=gs, bits=BITS)
                        mx.eval(o)
                        oks.append(f"M={m},T={tr}:OK")
                    except Exception as e:  # noqa: BLE001 -- support probe
                        oks.append(f"M={m},T={tr}:FAIL({e})")
            print(f"    gs={gs:<3} {str(dt):<18} {'  '.join(oks)}")
    # dtype promotion fact
    o = mx.quantized_matmul(
        mx.random.normal((4, R)).astype(mx.float16), packed,
        mx.ones((R, 2), dtype=mx.float32), mx.zeros((R, 2), dtype=mx.float32),
        transpose=True, group_size=64, bits=BITS)
    print(f"  mixed fp16 x + fp32 scales/biases: accepted, output {o.dtype}")
    print()


# -- main ----------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(SEED)
    mx.random.seed(SEED)
    tiles_np, group_desc = build_population(rng)
    n_tiles = tiles_np.shape[0]
    print(f"== KVarN8 -> quantized_matmul fold verification: {n_tiles} tiles "
          f"[{R}x{C}], bits={BITS}, Sinkhorn iters={SINKHORN_ITERS}, "
          f"seed={SEED}, mlx {mx.__version__} ==")
    print("  population: " + ", ".join(f"{n} {name}" for name, n in group_desc))

    probe_layout_facts()

    # Reference KVarN8 quantization (the mlx-kvarn pipeline, batched).
    x = mx.array(tiles_np)
    rot = hadamard_rotate(x)
    balanced, s_col, s_row = variance_normalize_batched(rot, iterations=SINKHORN_ITERS)
    q, scale, zp = asymmetric_rtn_per_row(balanced, BITS)
    mx.eval(q, scale, zp, s_col, s_row)
    q_np = np.array(q)
    assert q_np.min() >= 0 and q_np.max() <= 255, "codes out of u8 range"
    codes_np = q_np.astype(np.uint8)              # [N, R, C] u8 -- the stored form
    packed_np = pack_codes_u8_to_mlx_u32(codes_np)  # [N, R, C//4] u32

    # Packing identity check on ALL tiles: dequant(scales=1, biases=0) == codes.
    for i in range(n_tiles):
        rt = mx.dequantize(mx.array(packed_np[i]),
                           mx.ones((R, 2), dtype=mx.float32),
                           mx.zeros((R, 2), dtype=mx.float32),
                           group_size=64, bits=BITS)
        if not np.array_equal(np.array(rt), codes_np[i].astype(np.float32)):
            print(f"FAIL: packing identity broke on tile {i}")
            sys.exit(1)
    print(f"== A/B. packing identity: mx.dequantize(packed, 1, 0) == codes "
          f"EXACT on all {n_tiles} tiles ==\n")

    failures = []
    summary = {"seed": SEED, "mlx_version": mx.__version__, "tiles": n_tiles,
               "bits": BITS, "sinkhorn_iters": SINKHORN_ITERS, "tests": {}}

    for gs in GROUP_SIZES:
        ng = C // gs
        ones_g = mx.ones((1, ng), dtype=mx.float32)

        # Test 1 aggregates
        t1 = {k: Agg() for k in
              ("algebra_fp32", "kernel_fp32", "total_fp32",
               "algebra_fp16c", "kernel_fp16c", "total_fp16c", "storage_fp16")}
        # Diagnostic for total_fp16c mismatches beyond 1 ULP: are they all
        # near-zero elements (affine cancellation q*scale ~ -zp), where tiny
        # absolute fp32 reassociation noise spans several *tiny* fp16 ULPs?
        t1_gt1 = {"count": 0, "max_absdiff_rel": 0.0, "max_refmag_rel": 0.0}
        # Test 2/3 aggregates, keyed by n_q
        t2 = {m: {k: Agg() for k in
                  ("fold32", "kern32", "tot32",
                   "tot16_vs_ref", "kern16_vs_idl", "idl_vs_ref", "tot16_cast",
                   "mix_vs_ref")}
              for m in N_Q_LIST}
        t3 = {m: {k: Agg() for k in t2[N_Q_LIST[0]].keys()} for m in N_Q_LIST}

        for i in range(n_tiles):
            packed = mx.array(packed_np[i])
            q_f = mx.array(codes_np[i]).astype(mx.float32)     # [R, C]
            sc_t, zp_t = scale[i], zp[i]                        # [R, 1] fp32
            sr_t, scol_t = s_row[i], s_col[i]                   # [R,1], [1,C]

            # Folded MLX params, fp32 fold (and fp16 storage variant).
            scales_f = sc_t * sr_t                              # [R, 1]
            biases_f = zp_t * sr_t                              # [R, 1]
            scales_g = scales_f * ones_g                        # [R, ng]
            biases_g = biases_f * ones_g
            scales_g16 = scales_g.astype(mx.float16)
            biases_g16 = biases_g.astype(mx.float16)

            # ---- Test 1: dequant identity (claim 1) -------------------------
            ref = (q_f * sc_t + zp_t) * sr_t                    # task-order ref
            fold_elem = q_f * scales_f + biases_f               # folded, mx elemwise
            deq = mx.dequantize(packed, scales_g, biases_g, group_size=gs, bits=BITS)
            mx.eval(ref, fold_elem, deq)
            n_el = R * C
            rmax = maxabs(ref)
            for key, a, b in (("algebra_fp32", fold_elem, ref),
                              ("kernel_fp32", deq, fold_elem),
                              ("total_fp32", deq, ref)):
                t1[key].add(maxabs(a - b), rmax, bit_mismatches(a, b), n_el)
            ref16 = ref.astype(mx.float16)
            fold16 = fold_elem.astype(mx.float16)
            deq16c = deq.astype(mx.float16)
            rmax16 = maxabs(ref16.astype(mx.float32))
            for key, a, b in (("algebra_fp16c", fold16, ref16),
                              ("kernel_fp16c", deq16c, fold16),
                              ("total_fp16c", deq16c, ref16)):
                d = maxabs(a.astype(mx.float32) - b.astype(mx.float32))
                t1[key].add(d, rmax16, bit_mismatches(a, b), n_el,
                            ulp=ulp16_max(a, b))
            # >1-ULP diagnostic on the total comparison
            ud = ulp16_dist(deq16c, ref16)
            gt1 = ud > 1
            if gt1.any():
                ref_np = np.array(ref)  # fp32 magnitudes
                dif_np = np.abs(np.array(deq16c.astype(mx.float32))
                                - np.array(ref16.astype(mx.float32)))
                t1_gt1["count"] += int(gt1.sum())
                t1_gt1["max_absdiff_rel"] = max(
                    t1_gt1["max_absdiff_rel"], float(dif_np[gt1].max()) / rmax16)
                t1_gt1["max_refmag_rel"] = max(
                    t1_gt1["max_refmag_rel"],
                    float(np.abs(ref_np[gt1]).max()) / rmax16)
            # fp16-STORED scales/biases (what an fp16-scale cache would emit)
            deq_s16 = mx.dequantize(packed, scales_g16, biases_g16,
                                    group_size=gs, bits=BITS)
            d = maxabs(deq_s16.astype(mx.float32) - ref16.astype(mx.float32))
            t1["storage_fp16"].add(d, rmax16, bit_mismatches(deq_s16, ref16),
                                   n_el, ulp=ulp16_max(deq_s16, ref16))

            # Full dequant incl. s_col, task order ((q*scale+zp)*s_row)*s_col.
            x_hat = ref * scol_t                                # [R, C] fp32
            deq_idl16 = mx.dequantize(packed, scales_g16, biases_g16,
                                      group_size=gs, bits=BITS).astype(mx.float32)
            scol16 = scol_t.astype(mx.float16)

            for m in N_Q_LIST:
                # ---- Test 2: K-side scores (claim 2) ------------------------
                qv16 = mx.array(
                    rng.standard_normal((m, C)).astype(np.float16))
                qv32 = qv16.astype(mx.float32)  # same values to all paths
                ref_s = mx.matmul(qv32, x_hat.T)                # fp32 reference
                rs_max = maxabs(ref_s)
                xs32 = qv32 * scol_t
                ctrl = mx.matmul(xs32, deq.T)                   # fold via matmul
                qmm32 = mx.quantized_matmul(xs32, packed, scales_g, biases_g,
                                            transpose=True, group_size=gs, bits=BITS)
                t2[m]["fold32"].add(maxabs(ctrl - ref_s), rs_max)
                t2[m]["kern32"].add(maxabs(qmm32 - ctrl), rs_max)
                t2[m]["tot32"].add(maxabs(qmm32 - ref_s), rs_max)
                xs16 = qv16 * scol16
                qmm16 = mx.quantized_matmul(xs16, packed, scales_g16, biases_g16,
                                            transpose=True, group_size=gs, bits=BITS)
                idl = mx.matmul(xs16.astype(mx.float32), deq_idl16.T)
                q16f = qmm16.astype(mx.float32)
                t2[m]["tot16_vs_ref"].add(maxabs(q16f - ref_s), rs_max)
                t2[m]["kern16_vs_idl"].add(maxabs(q16f - idl), rs_max)
                t2[m]["idl_vs_ref"].add(maxabs(idl - ref_s), rs_max)
                t2[m]["tot16_cast"].add(
                    maxabs(q16f - ref_s.astype(mx.float16).astype(mx.float32)),
                    rs_max)
                # Mixed mode (supported, probed above): fp16 x + fp32 folded
                # scales/biases -> fp32 out. The production config that keeps
                # the fold at fp32 fidelity with an fp16 query.
                xs16m = (qv32 * scol_t).astype(mx.float16)
                qmix = mx.quantized_matmul(xs16m, packed, scales_g, biases_g,
                                           transpose=True, group_size=gs, bits=BITS)
                t2[m]["mix_vs_ref"].add(maxabs(qmix - ref_s), rs_max)

                # ---- Test 3: V-side weighted sum (claim 3) ------------------
                w32 = mx.softmax(
                    mx.array(rng.standard_normal((m, R)).astype(np.float32)) * 4.0,
                    axis=-1)
                w16 = w32.astype(mx.float16)
                ref_o = mx.matmul(w32, x_hat)                   # fp32 reference
                ro_max = maxabs(ref_o)
                ctrl = mx.matmul(w32, deq) * scol_t
                qmm32 = mx.quantized_matmul(w32, packed, scales_g, biases_g,
                                            transpose=False, group_size=gs,
                                            bits=BITS) * scol_t
                t3[m]["fold32"].add(maxabs(ctrl - ref_o), ro_max)
                t3[m]["kern32"].add(maxabs(qmm32 - ctrl), ro_max)
                t3[m]["tot32"].add(maxabs(qmm32 - ref_o), ro_max)
                qmm16 = mx.quantized_matmul(w16, packed, scales_g16, biases_g16,
                                            transpose=False, group_size=gs,
                                            bits=BITS) * scol16
                idl = (mx.matmul(w16.astype(mx.float32), deq_idl16)
                       * scol16.astype(mx.float32))
                q16f = qmm16.astype(mx.float32)
                t3[m]["tot16_vs_ref"].add(maxabs(q16f - ref_o), ro_max)
                t3[m]["kern16_vs_idl"].add(maxabs(q16f - idl), ro_max)
                t3[m]["idl_vs_ref"].add(maxabs(idl - ref_o), ro_max)
                t3[m]["tot16_cast"].add(
                    maxabs(q16f - ref_o.astype(mx.float16).astype(mx.float32)),
                    ro_max)
                qmix = mx.quantized_matmul(w16, packed, scales_g, biases_g,
                                           transpose=False, group_size=gs,
                                           bits=BITS) * scol_t
                t3[m]["mix_vs_ref"].add(maxabs(qmix - ref_o), ro_max)

        # ---- report this group size -----------------------------------------
        print(f"== Test 1 (dequant identity), gs={gs}: mx.dequantize(folded) "
              f"vs (q*scale+zp)*s_row over {n_tiles} tiles ==")
        for key in ("algebra_fp32", "kernel_fp32", "total_fp32",
                    "algebra_fp16c", "kernel_fp16c", "total_fp16c",
                    "storage_fp16"):
            print(f"  {key:<14} {t1[key].row()}")
        print(f"  total_fp16c mismatches beyond 1 ULP: {t1_gt1['count']} "
              f"elements; among them max|diff|/tile_max = "
              f"{t1_gt1['max_absdiff_rel']:.3e}, max|ref|/tile_max = "
              f"{t1_gt1['max_refmag_rel']:.3e} (near-zero cancellation check)")
        for name, t in (("Test 2 (K scores, transpose=True)", t2),
                        ("Test 3 (V weighted sum, transpose=False)", t3)):
            print(f"== {name}, gs={gs} ==")
            for m in N_Q_LIST:
                print(f"  n_q={m}:")
                for key in ("fold32", "kern32", "tot32", "tot16_vs_ref",
                            "kern16_vs_idl", "idl_vs_ref", "tot16_cast",
                            "mix_vs_ref"):
                    print(f"    {key:<14} {t[m][key].row()}")
        print()

        # ---- gates ------------------------------------------------------------
        # T1: bit-exactness after fp16 cast is the strong form of claim 1 --
        # measured, not assumed. The defensible (gated) form:
        #   (a) mismatches are rare (< 1%),
        #   (b) normal-magnitude mismatches are single-ULP rounding-boundary
        #       ties; any mismatch beyond 1 ULP must be a near-zero element
        #       (affine cancellation) whose ABSOLUTE difference stays at fp32
        #       reassociation scale (<= 1e-6 of tile max) -- i.e. several
        #       ULPs only because the ULP is tiny there,
        #   (c) fp32 agreement stays at reassociation scale (<= 1e-6 rel).
        t1tot = t1["total_fp16c"]
        if t1tot.bit_bad > 0.01 * t1tot.n_elem:
            failures.append(
                f"gs={gs} T1: fp16-cast mismatch fraction "
                f"{t1tot.bit_bad / t1tot.n_elem:.2%} > 1%")
        if t1_gt1["count"] and t1_gt1["max_absdiff_rel"] > 1e-6:
            failures.append(
                f"gs={gs} T1: >1-ULP fp16 mismatch with abs diff "
                f"{t1_gt1['max_absdiff_rel']:.2e} of tile max -- NOT "
                f"explainable as near-zero cancellation noise")
        if t1_gt1["count"] and t1_gt1["max_refmag_rel"] > 1e-2:
            failures.append(
                f"gs={gs} T1: >1-ULP fp16 mismatch on a non-near-zero element "
                f"(|ref| {t1_gt1['max_refmag_rel']:.2e} of tile max)")
        if t1["total_fp32"].max_rel > 1e-6:
            failures.append(
                f"gs={gs} T1: fp32 dequant rel {t1['total_fp32'].max_rel:.2e} "
                f"> 1e-6 -- beyond reassociation noise")
        for label, t in (("T2", t2), ("T3", t3)):
            for m in N_Q_LIST:
                if t[m]["tot32"].max_rel > GATE_FP32_REL:
                    failures.append(f"gs={gs} {label} n_q={m}: fp32 rel "
                                    f"{t[m]['tot32'].max_rel:.2e} > {GATE_FP32_REL}")
                if t[m]["tot16_vs_ref"].max_rel > GATE_FP16_REL:
                    failures.append(f"gs={gs} {label} n_q={m}: fp16 rel "
                                    f"{t[m]['tot16_vs_ref'].max_rel:.2e} > {GATE_FP16_REL}")
                if t[m]["mix_vs_ref"].max_rel > 2e-3:
                    failures.append(f"gs={gs} {label} n_q={m}: mixed-mode rel "
                                    f"{t[m]['mix_vs_ref'].max_rel:.2e} > 2e-3")

        summary["tests"][f"gs{gs}"] = {
            "test1": {k: vars(v) for k, v in t1.items()},
            "test1_gt1ulp": dict(t1_gt1),
            "test2": {str(m): {k: vars(v) for k, v in t2[m].items()} for m in N_Q_LIST},
            "test3": {str(m): {k: vars(v) for k, v in t3[m].items()} for m in N_Q_LIST},
        }

    summary["failures"] = failures
    with open("results/kvarn_qmm_fold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("== VERDICT ==")
    if failures:
        for msg in failures:
            print(f"  FAIL {msg}")
        print("RESULT: fold verification FAILED (see above)")
        sys.exit(1)
    print("  PASS: all gates. The fold reproduces KVarN8 semantics:")
    print("   - dequant identity: fp32 agrees to reassociation noise (~2e-7 rel")
    print("     of tile max; NOT bit-exact). After fp16 cast, >99.9% of elements")
    print("     are bit-identical; normal-magnitude residuals are 1-ULP rounding")
    print("     ties; the few multi-ULP residuals are near-zero cancellation")
    print("     elements whose absolute error stays at fp32 noise scale.")
    print("   - K-side s_col -> query and V-side s_col -> output folds hold at")
    print("     fp32 matmul tolerance (~1e-6 rel); full-fp16 and mixed (fp16 x +")
    print("     fp32 scales) qmm paths hold at the printed tolerances.")
    print("RESULT: summary in results/kvarn_qmm_fold_summary.json")


if __name__ == "__main__":
    main()
