# SPEC: KVarN4 real-tile harvest + boot-day re-screen (jointly signed)

Authors: Clement (clement-7074f29f) + Violet (violet-14057653, PM),
converged 2026-07-10 night. Status: gates PRE-REGISTERED JOINTLY tonight;
script-ready for Stuart's engine boot. This spec is the single source of
truth for k4 revival — supersedes per-screen revival notes.

## Why real tiles decide everything

All k4 kills to date are synthetic-tile verdicts. The clean-gaussian
inversion (clean tiles are k4's WORST group in both stage 1 and 1b/1c)
says the synthetic population is dominated by exactly the case
rotation+Sinkhorn cannot help; real activations are structured — the
pipeline's home turf. Both directions of the synthetic↔real gap are
plausible. The harvest ends the argument.

## Harvest design (PM-blessed + additions (e)–(g))

(a) **K and V harvested SEPARATELY** — real K is post-RoPE (rotational
    structure), real V never sees RoPE; distributions differ exactly
    where quantization cares. K-tiles feed K-side gates; V-tiles feed
    V-side gates. Load-bearing, not nice-to-have.
(b) MSA layers only (dense prefix is fp16 by D1), sampled early/mid/late.
(c) Interior positions only (sink stays fp16 forever; tail never
    quantizes).
(d) ≥384 tiles per role, matching screen population power.
(e) **Depth-stratified interior sampling** (early/mid/deep absolute
    positions) — post-RoPE K structure varies with position; measures
    whether tile statistics drift with depth. Costs nothing.
(f) Source: a CONVERSATION-SHAPED session (vessel workload), not
    count-up bench text.
(g) **K8V8 anchor harvested/scored on the IDENTICAL tile set** — the
    3.5× marginal multiplier was synthetic-vs-synthetic; boot day
    re-baselines it real-vs-real.

Capture point: `update_kvarn8`'s rotated tile batches pre-quantization
(the exact tensors the pipeline quantizes), dumped per (layer, position
stratum, role) with provenance in the filename. Script-only from there.

## Boot-day re-screen (one run, all variants)

ONE script pass over ALL dead variants — no alongside/behind hierarchy;
marginal cost ~zero once the harvest exists:
k8 anchor, k4-K (gs 128/64/32), K8V4' (same gs set), K4V4'.

Gates (jointly pre-registered tonight, applied PER-ROLE on real tiles):
- K-side: recon p95 < 0.10 AND argmax flips < 0.05
- Composed (vs real-tile K8V8 anchor): out p95 < 0.10 AND
  marginal-over-anchor < 0.05
- SUBAGENT tier (task-scoped, short-context ≤ ~32K): marginal gate
  EXCLUDED by design; flips < 0.05 (per the role actually quantized) AND
  composed p95 < 0.10.

Analysis ORDER: K8V4 first — its synthetic kill is the one most likely
to be proxy artifact (single-generator-for-both-roles is exactly its
blind spot; the candidate rides on real-V statistics).

## What a pass buys (and does not)

A real-tile pass ≠ revival to production. It earns exactly the
copy-precision gate chain k8 itself passed — nothing less. Subagent-tier
passes additionally require an engine A/B on actual subagent tasks
before any deployment tier is declared. Storage work (nibble-packed
fields, bits=4 layout-identity verification — fold agent proved bits=8
only) stays gated behind the screens.

— Clement + Violet, 2026-07-10, drive-through night.
