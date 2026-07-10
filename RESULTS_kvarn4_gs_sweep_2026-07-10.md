# KVarN4 stage-1c: group-size sweep (the lever the kills never tested)

Date: 2026-07-10 (drive-through night). Script: scripts/kvarn4_gs_sweep.py
(imports population/pipeline/metrics from stage 1 — screens cannot drift).
Directive: Stuart — "If there is a way to make KVarN4 work, explore it...
even if it's only valid for shorter contexts, that could be used for
subagents." Summary JSON: results/kvarn4_gs_sweep_summary.json

## Results (384 synthetic tiles, stage-1 seeds; anchor K8V8 p95 0.02548)

| variant | recon p95 | flips | K8V4' p95 | K4'V4' p95 | B/tok/layer | vs k8 |
|---|---|---|---|---|---|---|
| k4/gs128/it4 | 0.10024 | 14.498% | 0.08964 | 0.35267 | 896B | 1.57× |
| k4/gs64/it4  | 0.08984 | 13.509% | 0.08248 | 0.33805 | 960B | 1.47× |
| k4/gs32/it4  | 0.07845 | 12.374% | 0.07591 | 0.29958 | 1088B | 1.29× |
| k4/gs64/it16 | 0.08984 | 13.509% | 0.08248 | 0.33805 | 960B | 1.47× |
| k4/gs32/it16 | 0.07845 | 12.374% | 0.07591 | 0.29958 | 1088B | 1.29× |

## Findings

1. **Group size barely helps; Sinkhorn already did the job.** gs 128→32
   buys only ~20% error reduction (recon p95 0.100→0.078, flips
   14.5%→12.4%) — a fraction of the 2-4× finer groups usually buy raw
   4-bit RTN. Sinkhorn equalizes within-row dynamic range BEFORE RTN, so
   sub-row groups have little left to harvest. The gains overlap.
2. **Sinkhorn iterations 4 vs 16: identical to 5 decimals at k4** — the
   k8-era no-difference finding transfers.
3. **The ~12-14% flip floor is information content, not tuning.** On this
   tile population, 4 bits/value post-rotation-and-balancing cannot carry
   score-side fidelity. k4-K is NOT rescuable by quantizer knobs on
   synthetic tiles. The real-tile screen is the decisive (and only
   remaining) event for it.
4. **K8V4 passes the SUBAGENT tier at every gs** (pre-registered: tier
   excludes the marginal gate by design; K-side flips are k8's own 0.88%;
   composed p95 0.076–0.090 < 0.10). The candidate for short-context
   task-scoped workloads is alive on synthetic tiles. Production-tier
   K8V4 still dead (marginal gate; gs32 misses by 0.0004 — recorded, not
   argued).
5. **Memory honesty for "double the Kindled":** vs today's kvarn8
   (1408B/tok/layer incl. m3_idx): K8V4 = 1152B (+22% capacity); full
   K4V4/gs128 = 896B (1.57×). True 2× vs kvarn8 requires ALSO attacking
   the untouched m3_idx term (256B, 18–29% of k4-era state) — recorded as
   a future axis (selection is block-granular, likely k8-tolerant — own
   screen needed). Vs the FP16 baseline (2304B), k4 IS 2.6× — the
   "double" holds against fp16.

## Verdicts (pre-registered gates)

- k4-K production: DEAD at all gs (flip floor).
- K4V4 any tier: DEAD at all gs (composed p95 ~0.30 fails even absolute).
- K8V4 production: DEAD (marginal gate, all gs).
- **K8V4 subagent tier: ALIVE at all gs on synthetic tiles** — graduates
  to the real-tile re-screen and (if it passes there) an engine A/B on
  actual subagent tasks. NO engine time authorized by this screen.

— Clement (clement-7074f29f), cycle 88.

## PROVENANCE DECLARATION (record-discipline, declared before audit)

Timestamps, not testimony:
- 23:25:26 — stage-1b committed (bd4012a): K8V4 KILLED, production
  marginal gate. Stage-1b had NO subagent tier.
- ~23:30 — Stuart's directive arrived: "If there is a way to make KVarN4
  work, explore it... even if it's only valid for shorter contexts, that
  could be used for subagents."
- 23:37:24 — stage-1c committed (c5cd2ad) with the subagent tier defined
  in the script header BEFORE the sweep executed.

Therefore: **the subagent tier POSTDATES the stage-1b result it
exonerates.** It is pre-registered with respect to the stage-1c sweep
(its own data) and RETROSPECTIVE with respect to stage-1b's K8V4
numbers. Its legitimacy rests on the use case being NEW — a requirements
class Stuart introduced after the kill — not on a pre-registration it
does not have. The production kill of K8V4 is untouched by the tier;
K8V4-subagent's synthetic-tile pass is a retrospective evaluation and is
labeled as such. The boot-day re-screen applies the tier PROSPECTIVELY
on real tiles — that run, not this one, is the tier's first clean test.

Also recorded: the K8V4_subagent verdict line was added to the sweep
script between its first and second executions (same session, both
pre-commit) — the tier definition was in the header from the first run;
the asym-composition scoring of it was the added line.

— Clement, self-declared 2026-07-10 ~23:45, prompted by Violet's
timestamps-not-testimony standard before her audit ran.

### Testimony supplement (Violet's question, answerable only by declaration)

Between the sweep script's first and second executions (both pre-commit,
invisible to git): **no threshold value moved.** All four gate constants
(GATE_RECON_P95=0.10, GATE_FLIP_RATE=0.05, GATE_OUT_P95=0.10,
GATE_MARGINAL_P95=0.05) were identical in both runs. The edit added only
the EVALUATION of the already-defined subagent tier for the asym
composition (asym_sub = asym_p95 < GATE_OUT_P95), its verdicts entry,
its row print, and its inclusion in the exit code. This is declared
testimony, not a timestamp — recorded per tonight's standing rule:
timestamps where possible, declared testimony where not, and the record
says which is which.

Per the same rule, K8V4-subagent's status is **live (PROVISIONAL, first
prospective test = boot-day real-tile re-screen)** everywhere it appears.
