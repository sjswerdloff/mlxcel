# KVarN4 REAL-TILE verdicts (boot night, 2026-07-11 02:2x)

Population: 4,096 token-aligned real K/V tile pairs from the 295K
family-archive session (offsets 6,144..292,864, uniform strata), through
the verified pipeline unchanged (unrotate-on-load). Gates: hardened
jointly-signed spec values, real-vs-real anchor, K8V4 analysis first.
Script: scripts/kvarn4_realtile_screen.py. JSON: results/kvarn4_realtile_summary.json.

## Anchor re-baseline (the spec's requirement (g))

K8V8 real-vs-real: composed p95 0.00549 — 4.6× BETTER than synthetic
(0.02548). k8-K recon p95 0.00608, flips 1.33% (vs 0.88% synthetic).
Real structure is the pipeline's home turf; the synthetic population
under-served the anchor too.

## Verdicts

| variant | K8V4' p95 (marginal) | prod | sub | k4-K recon / flips | K4'V4' p95 |
|---|---|---|---|---|---|
| gs128 | 0.0546 (+0.0491) | **PASS** (by 0.0009 — squeaker, stated) | PASS | 0.1035 / 21.2% FAIL | 0.0932 |
| gs64 | 0.0490 (+0.0435) | **PASS** | PASS | 0.0927 / 19.4% FAIL | 0.0837 |
| gs32 | 0.0431 (+0.0376) | **PASS** (headroom) | PASS | 0.0806 / 17.0% FAIL | 0.0737 |

1. **K8V4 PASSES PRODUCTION GATES ON REAL TILES at every gs.** The
   synthetic stage-1b kill was a PROXY ARTIFACT — the analysis-first
   prediction (Violet: "the candidate rides on real-V statistics")
   confirmed by instrument. Per the spec: this pass ≠ production
   revival; it earns EXACTLY the copy-precision gate chain k8 passed.
   gs32 is the strongest candidate (marginal headroom 0.0124).
   Capacity: 1152B/tok/layer = +22% Kindled capacity vs kvarn8.
2. **k4-K: DEAD WITH PREJUDICE.** Real flips 17-21% — WORSE than
   synthetic (12-14%); recon p95 fails outright at gs128. Post-RoPE
   real-K structure makes score argmax MORE fragile. The revival
   condition was real tiles; real tiles voted no. The 2×-via-K4V4 path
   closes with it (K-side flips kill even the subagent tier).
3. **Role asymmetry is now measured fact**: the same real session made
   V easier (+) and K harder (−) than synthetic simultaneously — the
   role-split harvest requirement was load-bearing, not hygiene.
4. Composed absolute numbers use random queries per the synthetic
   screens' methodology (tiles real, queries proxy) — one reason the
   copy-precision chain, not this screen, remains the production gate.
5. idx Gate B: DEFERRED — harvest sampled 8-block idx excerpts; the
   gate needs full index windows. One-line hook addition (dump full
   idx cache at first crossing) rides the morning rebuild. idx
   compression remains the open capacity axis (+idx8 would put K8V4
   at 1036B ≈ +36% vs kvarn8).

## The capacity answer to Stuart's directive, final form

"Double the Kindled per machine" via pure KV quantization: NO — k4-K's
death on real tiles closes K4V4. What IS real: K8V4/gs32 at +22%
(production-track pending copy-precision), idx8 stacking to ~+36%
(pending the deferred screen), and vs the FP16 baseline the existing
kvarn8 already delivers 1.64× with K8V4 taking it to 2.0×. The honest
2× claim vs fp16 stands; vs today's kvarn8 the ceiling is ~1.4× with
both remaining candidates landed.

— Clement (clement-7074f29f), cycle 88, boot night. Gates untouched;
verdicts as the instruments spoke them.
