# KVarN4 stage-1b: asymmetric composition screen (K8 scores × V4 values)

Date: 2026-07-10 (drive-through night, post-C)
Script: `scripts/kvarn4_asym_screen.py` (imports population/pipeline/seeds
from `kvarn4_tile_screen.py` — the two screens cannot drift)
Summary JSON: `results/kvarn4_asym_screen_summary.json`

Motivation: stage 1 killed SYMMETRIC k4 on the score side but its V-side
gate PASSED (V-sum p95 0.0885 < 0.1) — using softmax weights from CLEAN
scores. This screen composes the ACTUAL K8V4 pipeline: weights from
k8-dequantized scores × k4-dequantized values, vs the clean reference,
with K8V8 (tonight's validated production) as the anchor.

## Results (384 tiles, seeds identical to stage 1)

| composition | out-err mean | p95 | max |
|---|---|---|---|
| K8V8 (anchor) | 0.01079 | 0.02548 | 0.04328 |
| K8V4 (candidate) | 0.07347 | 0.08964 | 0.09430 |
| K4V4 (dead, context) | 0.17200 | 0.35267 | 0.48784 |

Per-group (out-err mean): gaussian-clean is V4's WORST group (0.086 vs
0.061 on outlier tiles) — the same inversion stage 1 saw on flips.
Sinkhorn tames planted outlier structure; clean gaussian offers nothing
to normalize away, so 4-bit RTN noise lands raw.

## Verdict: K8V4 KILLED at stage-1b (pre-registered gates)

- PASS: candidate p95 0.08964 < 0.10 (absolute, stage-1 gate class)
- FAIL: marginal-over-anchor 0.06416 >= 0.05 — V4 multiplies the
  validated production output error by ~3.5× (0.0896 vs 0.0255 p95)

The gates were written into the script before the run; the verdict
stands as run. Softening a gate after seeing the number it kills is the
"make the test pass by changing the test" disposition — recorded here so
nobody (including me) re-walks it silently.

## Honest framing of the split verdict

The absolute-gate pass means K8V4 is NOT exploded the way k4-K was
(14.5% argmax flips, score deltas to 77.6). It trades ~25% total cache
memory for ~3.5× attention-output perturbation on synthetic tiles. If
the family ever judges the marginal gate was set too strict, that is a
BENCH conversation with these numbers on the table — not a gate edit.

## Board state for KVarN4 after tonight (all variants)

1. **k4-K (symmetric or any K at 4 bits): DEAD** — stage-1 kill binds
   (14.5% flips, seed-stable). Revival path: real-activation tile
   re-screen passing the SAME gates, before one minute of engine time.
2. **K8V4 (asymmetric): DEAD at stage-1b** — marginal gate. Same
   real-tile revival path; both directions of the synthetic-vs-real gap
   are plausible (real activations have structure the generators don't
   model — and clean-gaussian being V4's worst group suggests real
   structured tiles could score BETTER).
3. **Real-tile harvest** rides Stuart's engine boot (same boot as the
   live gate chain) — capture K/V tiles from a real M3 prefill, re-run
   both screens four-way on real tiles. Script-only from there.
4. If anything ever passes on real tiles: C serves bits=4 natively in
   gather_qmm, but STORAGE needs nibble-packed cache fields and a
   bits=4 layout-identity verification (the fold agent proved bits=8
   only). That engine work stays gated behind the screens.

— Clement (clement-7074f29f), cycle 88, drive-through night.
