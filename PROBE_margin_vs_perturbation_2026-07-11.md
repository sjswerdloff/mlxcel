# PROBE — the Gate B tail: margins vs perturbation (idx8)

Status: EXPLORATORY — informs the idx8 engine A/B design; gates
NOTHING. Unreviewed-before-run (unlike the three-seat-gated screen);
self-test first act; every per-layer err_p95 recomputed and asserted
EXACT against the banked screen summary (80addc7) — the probe provably
rides the reviewed scoring path (screen @ 3cde7e8, sel cross-check
1.000). Script: scripts/probe_margin_vs_perturbation.py; data:
results/probe_margin_vs_perturbation_20260711.json; population: the
same 310 × 4 samples, 57 layers, kvarn_harvest_20260711_1524.

Author: Violet (violet-14057653), from Xander's question (16:43):
is the deep-layer selection fragility a property of the index vectors
(H1 — tail persists under any scheme) or of the quantization
(H2 — better scheme helps)?

## Method

Margin and perturbation live in the same units (pre-softmax block
score), separately measurable per sample from banked data:

- margin = clean s_sorted[31] − s_sorted[32] (top-k boundary gap)
- delta = p95 over window blocks of |quant − clean| score
- flip pressure = delta / margin (flip plausible when ≳ 1)
- row range = per-row (hi − lo) of the window = 255 × idx8 scale
  (the scheme's coarseness, measured directly)

## Findings

**1. Perturbation is nearly flat across the model; margins are not.**
delta_p95 spans 0.0197–0.0339 across all 57 layers (< 2× spread).
Median margins span 0.0032–0.0334 (~10×); p05 margins 0.0003–0.0044
(~15×). Where error lives, it lives because margins are thin relative
to an approximately layer-uniform delta — not because delta is locally
large.

**2. The deep tail (58/59) is GEOMETRY — H1, decisively.**

| layer | err_p95 | margin_p50 | margin_p05 | delta_p95 | pressure_p90 | row_range_p95 |
|---|---|---|---|---|---|---|
| 58 | 0.0290 (worst) | 0.00315 (thinnest, 2× below next) | 0.00049 | 0.0218 (below avg) | 33.8 | 8.04 (2nd-lowest) |
| 59 | 0.0256 (2nd) | 0.00775 | 0.00029 (thinnest) | 0.0213 (below avg) | 19.9 | 7.76 (lowest) |
| typical | 0.010–0.020 | 0.010–0.023 | 0.001–0.004 | 0.020–0.030 | 3–14 | 8.1–10.6 |

The two worst layers have the thinnest margin distributions in the
model at BOTH moments, below-average perturbation, and the two LOWEST
row ranges — their index vectors are the EASIEST to quantize and the
hardest to select over. A better 8-bit scheme cannot fix this: even a
3× delta reduction leaves layer-58 tail pressure ~11. The near-ties
are in the clean scores themselves.

**3. The mid-band overage is thin-margin TAIL SAMPLES, not coarse
quant.** Layer 29: median margin 0.0262 (thick) yet margin_p05 0.00098
and pressure_p90 13.7 — its ≥ 0.02 membership comes from a thin-margin
sample subpopulation, invisible at the median. Same shape at 50 (p50
0.0311, p05 0.0027). Everywhere the mechanism is the same: samples
flip where their own margin is thin.

**4. Honesty on the correlations (they are weak, and why).** Spearman
over 57 layers: err vs margin_p50 −0.243, vs margin_p05 −0.182, vs
delta_p95 +0.076 (≈ zero — scheme magnitude does NOT predict layer
error), vs row_range_p95 +0.315 (mild; and delta_vs_row_range +0.569
confirms the mechanism chain works — it just barely moves err).
At n ≈ 22 samples/layer, each per-layer p95 rests on its top TWO
samples; rank correlations over 57 such points are noise-attenuated.
The evidential weight is in the outlier structure (finding 2), not the
pooled correlations. Mid-band ≥ 0.02 membership is fragile at this n.

**5. Layer 3 (bf16, first sparse) is its own small case**: thin tail
(p05 0.00065) AND slightly-elevated delta (0.0232), pressure 7.9.
Neither pure H1 nor H2; not load-bearing for either.

## What this means for the engine A/B design

- **Xander's question, answered for the deep tail**: index-vector
  geometry. The tail persists under any 8-bit index scheme. Engine
  time spent on a fancier index quantization would NOT buy layers
  58/59.
- **The sharpened A/B question**: near-tie flips substitute a block
  whose clean score sits within ~delta of the one lost — score-wise a
  near-equivalent. Whether OUTPUT quality tolerates near-equivalent-
  block substitution at depth is exactly what §4.3 greedy-divergence
  and §4.4 copy-precision measure end-to-end. HYPOTHESIS (suggested,
  not proven here): near-tie substitution is mostly benign because the
  substitute carries nearly equal attention mass. If the A/B holds,
  deep-layer drift is confirmed benign; if it degrades, layers 58/59
  are where to look first.
- **If a scheme upgrade is ever explored** (grouped/rotated index
  frame): it buys the MID-BAND (pressures 1.5–2.5 → below 1), not the
  deep tail. Only worth engine time if the A/B implicates mid-band
  drift.

## Non-claims

- No output-quality claim — the probe measures selection scores only.
- No gate movement: Gate B's verdict and framing stand exactly as
  registered.
- Per-layer conclusions other than 58/59's outlier structure are
  n-limited texture.
