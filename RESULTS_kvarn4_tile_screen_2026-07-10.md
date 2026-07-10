# KVarN4 stage-1 tile-level quality screen

Date: 2026-07-10  
Script: `scripts/kvarn4_tile_screen.py`  
Reference implementation: `mlx-kvarn` @ `c7d8767` (`/Users/stuartswerdloff/ai/ClaudeInstanceHomeOffices/clement-7074f29f/kindled_projects/mlx-kvarn`)  
mlx 0.30.5, numpy 2.4.0, python 3.14.6

KVarN8 is validated; this screen asks whether KVarN at bits=4 (16x coarser steps) survives at tile level, before any engine investment. Full reference pipeline per tile: WHT rotation -> log-domain Sinkhorn variance normalization (4 iters, the verified setting) -> per-row asymmetric RTN at `bits` -> pack/unpack (4-bit) -> dequant `(q*scale+zp)*s_row*s_col` -> un-rotate. Errors measured in the ORIGINAL frame.

## Tile population

384 synthetic tiles, each [128 tokens x 128 head_dim] (M3 geometry), generator family from `kvarn_decomposition_experiment.make_k_like_tiles` (unit gaussian + persistent outlier channels — the structure that killed the int8-absmax rung):

- 128 x gaussian-clean (no outliers)
- 128 x outlier-4ch-x40 (decomposition-expt structure)
- 64 x outlier-8ch-x100 (extreme)
- 64 x outlier-4ch-x40 + 16 token spikes x80

Outlier channels are drawn per tile (the decomposition experiment shared one draw across its batch); per-tile structure is identical. The pinned K0 fixture npz (`results/kvarn_reference_vectors.npz`) is not present in this clone — only the already-quantized gather fixtures are — so the population is fully synthetic via the reference generators.

Seeds: tile population `42`, queries `20260710` (numpy `default_rng`). 64 random fp16 query vectors per tile; scores `s = q . K^T / sqrt(128)` computed in fp32 (we measure quantization error, not accumulation error). V-side: softmax weights from CLEAN scores applied to clean vs dequantized V (the same tile through the same pipeline), relative error of the weighted sum per tile.

## Three-way summary (all tiles pooled)

| scheme       | recon mean | recon p95 | recon max | score|Δ| mean | score|Δ| p95 | score|Δ| max | argmax flips | V-sum mean | V-sum p95 | V-sum max |
|--------------|-----------|-----------|-----------|-----------|-----------|-----------|---------|-----------|-----------|-----------|
| fp16 (base)  | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |  0.000% | 0.00000 | 0.00000 | 0.00000 |
| KVarN k8     | 0.00481 | 0.00589 | 0.00596 | 0.02925 | 0.12273 | 5.00935 |  0.879% | 0.00425 | 0.00523 | 0.00546 |
| KVarN k4     | 0.08189 | 0.10024 | 0.10129 | 0.49731 | 2.07816 | 77.58585 | 14.498% | 0.07253 | 0.08850 | 0.09410 |

Context: mean |s| over clean scores = 6.2975; the score-delta columns are ABSOLUTE errors. fp16 baseline row is zeros by definition (fp16 storage is the reference).

- recon = per-tile ||x_hat - x||_F / ||x||_F (mean / p95 / max over tiles)
- score|Δ| = |s_hat - s| pooled over all (tile, query, key) elements
- argmax flips = fraction of (tile, query) pairs whose top-1 key within the tile changes
- V-sum = per-tile relative error of softmax(clean)-weighted V sum

## Per-group breakdown

| group | n | k8 recon mean | k4 recon mean | k8 flips | k4 flips | k4 V-sum mean | mean |s| clean |
|---|---|---|---|---|---|---|---|
| gaussian-clean (no outliers) | 128 | 0.00583 | 0.09904 | 0.671% | 13.306% | 0.08572 | 0.7967 |
| outlier-4ch-x40 (decomposition-expt structure) | 128 | 0.00419 | 0.07150 | 1.123% | 16.553% | 0.06046 | 5.3487 |
| outlier-8ch-x100 (extreme) | 64 | 0.00494 | 0.08398 | 1.025% | 16.821% | 0.08233 | 19.3588 |
| outlier-4ch-x40 + 16 token spikes x80 | 64 | 0.00388 | 0.06626 | 0.659% | 10.449% | 0.06050 | 6.1352 |

## Verdict

**K4 EXPLODES AT TILE LEVEL**

Stage-1 screen gates (kill-test thresholds, not final acceptance criteria):

- FAIL: k4 recon p95 = **0.10024** (gate < 0.1)
- FAIL: k4 argmax flip rate = **0.14498** (gate < 0.05)
- PASS: k4 V-sum p95 = **0.08850** (gate < 0.1)

- k4 mean reconstruction rel-err 0.08189 vs k8 0.00481 — 17.0x worse (4-bit steps are 16x coarser, so ~16x is the expected scaling).
- k4 argmax-flip rate 14.498% vs k8 0.879% over 24576 (tile, query) pairs.
- k4 V-sum rel-err mean 0.07253 (p95 0.08850, max 0.09410).

### Seed stability

The kill verdict is not a seed artifact. Rerunning the same population +
pipeline (via the script's own functions) with two alternate seed pairs:

| seeds (tile, query) | k8 recon mean | k8 flips | k4 recon mean | k4 recon p95 | k4 flips |
|---|---|---|---|---|---|
| 42, 20260710 (pinned) | 0.00481 | 0.879% | 0.08189 | 0.10024 | 14.498% |
| 7, 999 | 0.00478 | 0.940% | 0.08137 | 0.09982 | 14.160% |
| 12345, 55555 | 0.00477 | 0.867% | 0.08124 | 0.09985 | 13.668% |

k4 recon sits at ~8.1-8.2% mean / ~10.0% p95 and flips 13.7-14.5% of
top-1s in every run; the recon-p95 gate result is marginal either way, but
the flip-rate failure (~3x the gate, ~16x the validated k8's rate on the
identical query population) is decisive and stable.

### What tile-level CANNOT tell us

- **Compounding over long contexts.** A 300K-token window is ~2300 tiles per head per layer; per-tile error is independent here, but real decode accumulates quantized-K score noise across the whole window and across layers. Tile-level error bounds do not compose linearly into end-to-end logit error.
- **Near-tie flips at full-window scale.** Argmax here is over 128 keys within one tile. A real window competes ~300K keys; near-ties across tiles are more common, and softmax mass — not tile-local top-1 — is what matters. Random-query top-1 within a random tile is a coarse proxy in both directions.
- **Distribution shift.** Tiles are synthetic (gaussian + planted outlier channels). Real K/V activations have correlated, layer-dependent structure (RoPE phase, low-rank structure, real attention-sink statistics) not modeled here.
- **No end-to-end perplexity / task quality.** This is a necessary-not-sufficient screen: k4 failing here kills it cheaply; k4 passing here only buys the next experiment (engine-level A/B), not deployment.

Summary JSON: `results/kvarn4_tile_screen_summary.json`
