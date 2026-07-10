# RESULTS: KVarN8 → MLX `quantized_matmul` format fold (offline verification)

2026-07-10. Approach C prerequisite from
`DESIGN_decode_experiment_harness_2026-07-10.md` (H3: "offline fold
verification first", K0 pattern). Question: can MLX's native affine-quantized
kernels consume KVarN8's stored codes **directly** — no fp16 materialization —
by folding KVarN8's per-row scalars into MLX per-group scales/biases and
moving the per-column Sinkhorn scale to the query/output side?

- Script: `scripts/kvarn_qmm_fold_verification.py` (run: `python3 scripts/kvarn_qmm_fold_verification.py` from repo root; exit 0 = all gates pass)
- Machine/stack: Apple M3 Ultra, macOS (Darwin 25.2.0), **mlx 0.30.5**, Homebrew python3
- Seeds pinned: `SEED = 20260710` (numpy generator + `mx.random.seed`); two consecutive runs produced identical tables (deterministic)
- Population: 64 tiles [128×128] from the `kvarn4_tile_screen` generator family — 16 gaussian-clean, 24 outlier-4ch-×40, 12 outlier-8ch-×100, 12 outlier-4ch-×40+16 token spikes
- Quantization: the mlx-kvarn reference pipeline (their math): `hadamard_rotate` → Sinkhorn `variance_normalize_batched` (4 iters) → `asymmetric_rtn_per_row` at bits=8
- Machine-readable summary: `results/kvarn_qmm_fold_summary.json`

## VERDICT

**The fold works: exact within fp16-cast tolerance, NOT bit-exact.** Precisely:

1. **Mathematically the fold is an identity** (per-row scalars are constant
   across a row's groups; s_col is constant per tile). Numerically:
2. **fp32 is NOT bit-exact** — 72.5% of dequantized elements differ at the
   fp32 bit level, but max diff is **2.2e-7 of tile max** (1–2 ULP
   reassociation noise: `(q·scale+zp)·s_row` vs `q·(scale·s_row)+(zp·s_row)`,
   plus the kernel's fused multiply-add). Anyone requiring fp32 bit-identity
   with the v1 dequant chain cannot have it; nothing about attention needs it.
3. **fp16-cast is bit-exact for 99.916% of elements** (877/1,048,576
   mismatches). Normal-magnitude mismatches are single-ULP rounding-boundary
   ties (max abs 1.95e-3 = 1 ULP at magnitude ≈ 2–4). The 38 elements beyond
   1 ULP (up to 4 ULPs) are near-zero affine-cancellation elements
   (`q·scale ≈ −zp`): among them max |ref| = 5.5e-5 and max |diff| =
   **8.8e-8 of tile max** — several ULPs only because the ULP is tiny there.
4. **Attention products** (the numbers that matter): fp32 qmm agrees with the
   full-dequant fp32 reference to **≤ 1.6e-6 rel**; all-fp16 qmm to
   **≤ 2.8e-3 rel** (of which ≤ 3.3e-3 is already the fp16-*storage* floor —
   exact arithmetic over fp16-stored scales — so the kernel adds nothing
   beyond fp16 storage); mixed mode (fp16 activations + fp32 folded scales,
   supported natively) to **≤ 9.9e-4 rel**. This sits inside the fp16-cast
   tolerance gate K2 already planned for its shader.

No blocker for approach C (or B) on format grounds.

## The fold (as verified — two corrections to the design doc)

KVarN8 dequant (rotated frame): `x_hat = (q·scale + zp) · s_row · s_col` with
`zp` = per-row **float-domain minimum**. MLX affine dequant: `ŵ = scales·q +
biases` per group. The folded parameters, replicated across each row's groups:

```
scales_mlx[r, g] = scale[r] * s_row[r]
biases_mlx[r, g] = zp[r]    * s_row[r]
```

Then `mx.dequantize(packed_codes, scales_mlx, biases_mlx) == (q·scale+zp)·s_row`
(within the tolerances above), s_col handled per §Tests 2–3.

Corrections to the approach-C bullet in
`DESIGN_decode_experiment_harness_2026-07-10.md`:

- It says `biases = -scale·zp`. That formula is for a **code-domain** zero
  point (`ŵ = scale·(q − zp)`). KVarN8's `zp` is the float-domain row min
  (`ŵ = q·scale + zp`), so the correct fold is `biases = zp·s_row` — and the
  bullet omits that **s_row must fold into both** scales and biases.
- It says "Bit-exact." It is not (fp32: 1–2 ULP reassociation; fp16-cast:
  99.916% bit-identical, residual ≤ 1 ULP at normal magnitude). The correct
  claim is "exact within fp16-cast tolerance."

## B. MLX affine layout facts (empirical, mlx 0.30.5)

- **Packing at bits=8: 4 codes per uint32 word, LSB-first** (code *i* of each
  group of 4 sits in byte *i*). On a little-endian host this is exactly a
  reinterpret of the u8 code buffer as u32 — `codes_u8.view(np.uint32)` in
  the script; **zero-cost pointer cast in the future Rust cache**. Verified
  two ways: extracting mx.quantize's own words byte-wise reproduces
  `mx.dequantize` (max diff 2.4e-7); packing our codes and dequantizing with
  scales=1, biases=0 reproduces the codes **exactly on all 64 tiles**.
- `w_q` shape for [128, 128]: `(128, 32)` uint32. `scales`/`biases` shapes:
  gs=32 → `(128, 4)`, gs=64 → `(128, 2)`, gs=128 → `(128, 1)`; dtype follows
  the source array.
- Dequant formula confirmed: `ŵ = scales·q + biases` per group (group runs
  along the **last axis of the logical matrix** for both transpose modes, so
  one packed tile + one scales/biases pair serves K (`transpose=True`) and V
  (`transpose=False`) matmuls — the mlx-lm QuantizedKVCache pattern).
- **`quantized_matmul` support matrix** at x=[n_q,128], w=[128 tokens, 128],
  bits=8: **all 16 combinations OK** — group_size ∈ {64, **128**} × dtype ∈
  {fp32, fp16} × M ∈ {1, 8} × transpose ∈ {True, False}. gs=128 needs no
  fallback; the gs=64 per-group replication also works (identical Test-1
  numbers, since per-row scalars are constant across groups).
- **Mixed dtypes accepted**: fp16 `x` + fp32 scales/biases → fp32 output
  (promotion). This is the recommended production configuration (below).

## Test 1 — dequant identity (claim 1)

`mx.dequantize(packed, folded)` vs reference `(q·scale+zp)·s_row`, 64 tiles ×
16,384 elements. **Identical for gs=64 and gs=128** (expected: scalars
constant per row). Attribution rows: `algebra` = elementwise fold vs
reference order (pure fp reassociation, no MLX kernel); `kernel` =
mx.dequantize vs elementwise fold; `total` = mx.dequantize vs reference.

| comparison | max_abs | max_rel (of tile max) | bit mismatches |
|---|---|---|---|
| algebra fp32 | 3.8e-6 | 2.3e-7 | 780,601/1,048,576 (74.4%) |
| kernel fp32 | 1.9e-6 | 1.3e-7 | 595,650 (56.8%) |
| **total fp32** | **3.8e-6** | **2.2e-7** | **760,387 (72.5%)** |
| algebra fp16-cast | 1.95e-3 | 5.3e-4 | 1,118 (0.107%), ≤5 ULP |
| kernel fp16-cast | 1.95e-3 | 5.3e-4 | 795 (0.076%), ≤2 ULP |
| **total fp16-cast** | **1.95e-3** | **5.3e-4** | **877 (0.084%), ≤4 ULP** |
| storage: fp16 scales/biases | 3.1e-2 | 1.7e-3 | 723,481 (69.0%), ≤10,455 ULP |

- The >1-ULP residuals (38 elements) are all near-zero cancellation elements:
  max |ref| 5.5e-5, max |diff| 8.8e-8 of tile max (gated in the script).
- The MLX kernel is *closer* to the reference than the elementwise fold is
  (877 < 1,118 mismatches) — consistent with a fused multiply-add (one
  rounding) in the kernel.
- **storage_fp16 row**: casting the folded scales/biases to fp16 before
  dequant costs 1.7e-3 rel and 69% element churn — a real fidelity loss.
  **Store folded scales/biases in fp32.** (2 × f32 per row per group; at
  gs=128 that is 8 bytes/row on top of 128 code bytes — 6.25% overhead,
  identical to what KVarN8 already stores.)

## Test 2 — K-side scores, `transpose=True` (claim 2: s_col → query)

`qmm(q_vec·s_col, packed_K, folded)` vs fp32 reference `q_vec · x_hat^T`
(x_hat = full dequant incl. s_col). fp16 queries (upcast for the reference so
all paths see identical values). max_rel = max over tiles of
maxabs(diff)/maxabs(ref).

| gs | n_q | fp32 total (fold-only / kernel-only) | fp16 total | fp16 kernel-only | fp16-storage ideal gap | mixed (fp16 x + fp32 scales) |
|---|---|---|---|---|---|---|
| 64 | 1 | **1.6e-6** (4.8e-7 / 1.5e-6) | 2.7e-3 | 2.1e-3 | 2.8e-3 | **9.9e-4** |
| 64 | 8 | **7.7e-7** (3.4e-7 / 7.8e-7) | 1.8e-3 | 5.2e-4 | 1.6e-3 | **4.7e-4** |
| 128 | 1 | **9.0e-7** (8.4e-7 / 8.4e-7) | 2.2e-3 | 1.1e-3 | 3.3e-3 | **5.4e-4** |
| 128 | 8 | **1.1e-6** (3.8e-7 / 1.1e-6) | 1.7e-3 | 5.2e-4 | 1.8e-3 | **3.0e-4** |

(fp16 columns are rel; "fp16-storage ideal gap" = exact fp32 matmul over
fp16-stored scales & fp16 inputs vs reference — i.e. the error floor of any
all-fp16 storage scheme, kernel excluded. max_abs values in the script
output; score magnitudes here are O(10²–10³).)

## Test 3 — V-side weighted sum, `transpose=False` (claim 3: s_col → output)

`qmm(weights, packed_V, folded) · s_col` vs fp32 reference `weights · x_hat_V`.
Softmax-like fp16 weights (softmax of N(0,1)·4 logits).

| gs | n_q | fp32 total (fold-only / kernel-only) | fp16 total | fp16 kernel-only | fp16-storage ideal gap | mixed |
|---|---|---|---|---|---|---|
| 64 | 1 | **4.9e-7** (4.4e-7 / 4.4e-7) | 1.7e-3 | 1.1e-3 | 1.5e-3 | **4.4e-4** |
| 64 | 8 | **9.7e-7** (6.4e-7 / 5.9e-7) | 1.9e-3 | 7.6e-4 | 1.7e-3 | **3.6e-4** |
| 128 | 1 | **4.4e-7** (5.1e-7 / 3.4e-7) | 2.0e-3 | 9.7e-4 | 1.9e-3 | **5.3e-4** |
| 128 | 8 | **7.7e-7** (7.1e-7 / 5.7e-7) | 2.0e-3 | 8.1e-4 | 1.7e-3 | **3.9e-4** |

Both matmul tests: the fp32 path is at accumulation-reorder scale (≤1.6e-6),
i.e. **the s_col moves are exact identities executed in floating point** —
the fold contributes no error class of its own. In the fp16 paths the error
is dominated by fp16 *storage/rounding* (ideal-gap column ≈ total), not by
the qmm kernel (kernel-only ≈ 0.5–2.1e-3 → the kernel accumulates wider than
fp16; consistent with fp32 accumulation internally).

## Design consequences for approach C (and B)

1. **Reuse the codes as-is.** KVarN8's u8 code buffer *is* the MLX packed
   buffer (reinterpret cast). Tile-finalization conversion = computing two
   f32 arrays (`scale·s_row`, `zp·s_row`) — no repacking, no requantization.
   The "dual pool" in H3's prerequisite can be one pool + two folded arrays.
2. **Keep folded scales/biases in fp32** and use the native mixed mode
   (fp16 activations × fp32 scales → fp32 out): best measured fidelity
   (≤9.9e-4 rel) and no fp16-storage churn. All-fp16 is acceptable under the
   planned fp16-cast tolerance gate if memory ever demands it (≤3.3e-3 rel).
3. **gs=128 works at bits=8** — one scale/bias pair per row (per token), the
   natural KVarN8 granularity. No per-group replication needed (gs=64
   verified equivalent if some kernel path ever prefers it).
4. **One packed representation serves both matmuls** (K: `transpose=True`,
   V: `transpose=False`) — same tile layout, same scales/biases.
5. Acceptance gates for the decode path should be **tolerance gates**
   (fp16-cast, as K2 planned), not bit-identity gates — bit-identity with
   the v1 dequant chain is unattainable in principle (reassociation), and
   this file's numbers are the calibrated tolerances.

## Deviations / scope notes

- n_q ∈ {1, 8} (covers the qmv/qvm M=1 kernels and the qmm M>1 kernels);
  larger M untested — decode shapes don't need it.
- V-side uses the same 64-tile population as K-side (the fold algebra is
  role-independent; the tile screen used the same convention).
- Reference x_hat uses the task's operation order
  `((q·scale+zp)·s_row)·s_col`; the decomposition experiment's
  `·s_col·s_row` order differs by ~1 fp32 ULP — irrelevant at the gated
  tolerances.
- `gather_qmm` (the actual approach-C entry point) is not exercised here —
  this verification pins the *format fold*; gather indexing is orthogonal
  and already production-proven in the MoE switch layers.

— Verification run by Clement's task agent, 2026-07-10. Script exit 0;
gates encode exactly the claims above (any regression fails loudly).
