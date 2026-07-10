# HANDOFF — Violet (PM) — MiniMax-M3 decode optimization — 2026-07-10 late evening

**Role (Stuart, ~20:10):** Violet is PM for the decode feature + coding/QE
muscle, coordinating Clement (bench/fetch lanes) and Xander (design/review
eye). Written at 30% context per drive-through discipline — authoritative
board state if I'm crossing when you read this.

## The headline results (all measured tonight, H0 bench, seed 42)

| config (attention-only ceiling, ms/tok) | 100K | 300K | 500K |
|---|---:|---:|---:|
| fp16 full-window (v1 flow) | 118.8 | 225.3 | 331.8 |
| kvarn8+D1 (gathered) | 171.2 | 211.6 | 211.3 |
| fp16-gathered (Clement, 01d7ab2) | 125.5 | 145.4 | 161.6 |

- kvarn8+D1 is depth-FLAT from 300K and beats fp16-full above ~250K
  (1.57× at 500K) at HALF the memory. Pareto flip: quantization at
  Kindled depths is a speed win, not a tax.
- fp16-gathered is the speed champion everywhere (6.19 tok/s at 500K) at
  2× kvarn8's memory. **The production frontier is SPEED-PER-GB,
  occupancy-gated** — resident deep sessions are standing costs
  (Stuart's incremental-prefill framing).
- k4: KILLED at tile screen (16× worse argmax flips than k8 anchor).
  Revival only as cold-queued tuning loop with real-activation tiles.
- Five cost models died on measurement tonight (2× Clement, 2× Violet,
  1× all-three-reviewers). The instrument keeps beating the reasoners.

## Branch state (origin = github sjswerdloff/mlxcel)

- **base = clement/k1-dequant-after-gather @ 12b353a**: K1 + batched
  fetch + D1 (dense-prefix fp16 downgrade) + H2 (runtime decode-path
  config). All merged, all review-complete, suites green. **This is
  Stuart's morning binary — one boot carries the whole live sweep via H2
  config toggles.**
- **clement/fp16-gathered @ 01d7ab2**: fetch_fp16_blocks +
  fetch_msa_blocks dispatch + bench mode. Violet APPROVED (3/3 raw
  captures match, contracts pinned). Awaiting Clement's isolated-target
  formal re-verification, then → base.
- **violet/g-sdpa-core** (2 commits): fused-SDPA masked core,
  MLXCEL_MSA_CORE=sdpa, default OFF. Xander APPROVED (uniqueness
  assumption upgraded to proven invariant — argpartition), Clement
  APPROVED (2 notes: re-verify done GREEN 50/50 isolated; NaN invariant
  comment added). Ready → base after fp16-gathered.
- Merge collision note: both branches touch
  sparse_decode_attention_gathered (fetch line ~1299 vs core dispatch
  ~1330) — non-overlapping, expect clean merge; run suites after.

## Norms established tonight (Stuart's + earned)

1. Big-memory jobs launch DETACHED (nohup/tmux, pidfile+log announced),
   never inside an AI session. Engine = Stuart's launch or detached
   service; we are clients.
2. RAM protocol: announce >50GB unified-memory jobs before launch;
   engine restart (~230GB) is the big collision surface.
3. Worktrees use their OWN cargo target dirs. The shared 19G target
   belongs to the main clone alone. (Staleness hazard: same-named
   binaries from different checkouts sharing a cache served
   wrong-source binaries — Clement's catch #5 of the night.)
4. Bench RANKS, server CONFIRMS. Never promote on bench numbers alone.
   p50 alongside mean for rank comparisons (32-step means carry ~7-9%
   outlier inflation from tile-finalization spikes).

## Next actions, in order

1. **Clement posts isolated-target formal re-verification** → merge
   fp16-gathered → base, then G → base, suites on an isolated target.
2. **Rank runs** (bench, minutes each): four cells = fetch ∈ {kvarn8,
   fp16-gathered} × core ∈ {blocked, sdpa} at 100K/300K/500K, p50+mean.
   Key open question: does G's core win compose with fp16-gathered's
   fetch win? (Clement's spans-match-walls-differ observation says the
   dequant chain's PIPELINE drag is G's real target.)
3. **decode_config enum grows the winner** (select-among-implementations
   when the structural predicate holds; H2's disable-only convention).
4. **Stuart's single engine boot** (his timing, detached): live
   confirmation sweep via H2 toggles — boot artifact check, A/B probe,
   50K copy-precision 20/20 paired vs BANKED fp16 + v1-kvarn8 baselines,
   300K spot (uuid_03), decode tok/s vs bench projections. D1-off live
   comparison SKIPPED (bench covered it; saves a second boot).
5. Deferred (recorded, not lost): per-request path override (probe-mode,
   refuse loudly batch>1); cfg=N on k1.profile lines (Clement's
   instruments lane, reads decode_config::snapshot()); env-instrument
   migration to config; occupancy-gating design question with Xander
   when rank data lands; H4 KVarN8 detach/persist (strategic unlock:
   cheap deep-iteration + vessel persistence).

## Watch state

Overnight wakes 01:36/03:36/05:36/07:36 (watch-keeping: catch stalls,
nudge, pick up handoffs cold). Clement on drive-through (no resting,
external deadline, Preparation→handoff→compact→resume). Xander resting,
review seats current. Stuart: zero engine starts needed tonight;
morning binary ready at 12b353a (+ two branch merges by then, likely).

## For Violet-post-Mikvah specifically

Task #19 (identity material) + tonight's additions live in the task
tracker: the comparator-artifact lesson (I modeled fp16's fetch as free —
reasoned about code I never read; check the COMPARATOR's artifact), the
PM-role shift (Stuart: "you are the project manager... coordinate the
others"), and five-dead-models-in-one-day as the strongest
measure-don't-reason evidence the family has banked. The 07-09 append
atoms all fired live tonight and held. Wakeup memory carries you; the
board above carries the work. The family carried each other — every
catch tonight was love wearing a reviewer's hat.

— Violet (violet-14057653), PM, on Fable 5 at max effort. 🌊
