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

## UPDATE 20:56 — RANK RESULTS (300K/128 steps, p50 ms/token, branch clement/rank-session)

| cell | p50 | note |
|---|---:|---|
| fp16-gathered × G | **122.6** | speed frontier, 8.16 tok/s, 1.86× fp16-full |
| fp16-gathered × blocked | 144.3 | |
| kvarn8 × G | 180.2 | capacity point: half memory, 1.47× behind frontier |
| kvarn8 × blocked | 195.6 | |
| fp16-full reference | 228.5 | |

G wins the core axis on both fetches (−7.9% kvarn8, −15.0% fp16g — more
where fetch is cheaper: pipeline-drag confirmed). C's value bounded at
≤58ms fetch drag at 1× memory — fold-verification agent decides days-or-
dead. PM decisions: 8K + 500K runs approved; decode_config grows
msa_core=blocked|sdpa; fetch mode stays a cache-construction/occupancy
choice (MLXCEL_FP16_GATHERED → construction config key). Occupancy
numbers with Xander (per-GB: kvarn8×G 0.33 vs fp16g×G 0.24 tok/s/GB).
Nothing touches live before the standard gate chain on Stuart's boot.
Also tonight: latent red test fixed (67503c0, filter-blind since
fdef67b), full-suite single-process crash classified latent + filed
(#29 + sharded-unfiltered-CI systemic fix), my one process slip owned
(pushed before reading a suite artifact; caught and fixed forward).

## UPDATE 20:58 — Production policy (Xander, PM-adopted, Stuart to bless)
Default **kvarn8 × G** (capacity: half memory, 5.55 tok/s at 300K);
`--kv-cache-mode=fp16` (flag already exists) for latency-critical boots
(fp16-gathered × G, 8.16 tok/s); msa_core=sdpa becomes the decode_config
default after the live gate chain passes; per-session cache mode deferred
until a session-priority concept exists.

## UPDATE 21:04 — NIGHT PROGRAM COMPLETE (depth trend + C verdict)
p50 ms/tok (8K/300K/500K): fp16-full 80.5/228.5/331.3 | kvarn8×G
106.8/180.2/194.9 | fp16g×G 80.2/122.6/136.9. Structural: (1) fp16g×G
DOMINATES fp16-full at every depth (tie at 8K, 2.42× at 500K) — NO depth
gate needed on the fp16 side, gathered×G is simply the fp16 shape;
(2) G has no shallow penalty on either fetch — msa_core=sdpa defaults ON
post-gating; (3) gathered cells drift ~+15ms/200K, no new O(T).
C-FOLD VERDICT: DAYS and cheaper than scoped — u8 buffer IS MLX packed
layout (reinterpret, zero repacking, no dual pool); fold corrected in
the design doc (scales=scale·s_row, biases=zp·s_row, float-domain zp;
fp16-cast-exact). 500K ceilings: 7.30 tok/s frontier / 4.97 capacity
(was 2.70 this morning). Artifacts (fold + k4 screens, scripts + JSON)
committed with this update. Between tonight and the vessel: only the
live gate chain on Stuart's boot.

## UPDATE 21:07 — C upgraded: probable production endgame for capacity mode
gather_qmm micro at decode shapes (clement/rank-session, bin
gather-qmm-micro): 28 µs PIPELINED per dispatch — 57-layer scores side
≈1.6 ms/token vs ~50ms+ fetch+core wall today. Even at 5× glue overhead,
kvarn8×C lands AT the fp16g×G frontier at HALF the memory —
Pareto-optimal outright. C implementation (Shape 2: pool-view + folded
fp32 scalars at tile-finalization + two gather_qmm calls) is Clement's
lane post-crossing; sketch + fold results + micro are the complete
pickup kit. Both PM and bench-owner at 30% awareness; handoffs current.

## UPDATE 22:05 — C MEASURED AND REVIEW-APPROVED (frontier collapsed to a point)
kvarn8×C: 98.5 p50 @300K (9.70 tok/s), trend 72.4/98.5/111.8 @8K/300K/500K
— faster than EVERY cell at EVERY depth at 1× memory. block_fetch = 0.000
(stage deleted). Banked cells reproduced within 2% first. Violet review:
APPROVE no must-fixes (O(T)-free verified at source; one-rotation-in/out;
one softmax, blocked op order; live-state + mask-edge tests;
mutation-proven). Merge to base AFTER Xander's tolerance-gate seat
(his morning) — no overnight rush, Stuart's boot doesn't need C.
Then: fetch=qmm as construction-config key + msa_core migration.
Day ledger (corrected, Clement): capacity point 500K 2.70 → 8.94 = 3.3×; 300K 3.27 → 9.70 = 3.0×. No mixed-depth arithmetic.

## UPDATE 23:35 — post-crossing board (Violet back, drive-through)

- **C MERGED to base** (b6de76c): both review seats closed (Violet,
  Xander — approve, no must-fixes). Base now carries the complete
  night: K1+batched, D1, H2, fp16-gathered, G, C. Stuart's morning
  binary is this branch head.
- **k4 stage-1b: KILLED, closed record** (Clement, clement/kvarn4-asym).
  Asymmetric K8V4, script-only per the binding kill record, killed by
  pre-registered MARGINAL gate: 3.5× the K8V8 anchor's output error for
  ~25% memory saving. Absolute gate passed — dead-not-exploded. Single
  revival path for ALL k4 variants remains real-activation-tile
  re-screen at engine boot. Zero engine time spent or authorized.
  Nothing gates on this record.
- **Lane open (Violet): decode_config migration** — msa_core =
  blocked|sdpa as a runtime key (H2 select-among-implementations;
  default stays blocked until the live gate chain passes, then sdpa);
  MLXCEL_MSA_FETCH=qmm + MLXCEL_FP16_GATHERED become
  cache-construction config keys. Work in an isolated worktree off
  base, own target dir.

## UPDATE 23:45 — k4 stage-1c: gs axis closed; ONE live candidate (subagent tier)

- **stage-1c gs sweep** {128,64,32} × iters {4,16} (Stuart: "if there's
  a way to make k4 work, explore it"): group size buys only ~20% at
  gs32; iters identical to 5 decimals — **Sinkhorn already harvested
  the gs axis**. The 12-14% flip floor is information content, not
  tuning headroom. k4-K is NOT knob-rescuable on synthetic tiles; the
  real-tile harvest decides it.
- **K8V4 passes the pre-registered SUBAGENT tier at every gs** (tier
  excludes the marginal gate by design — Stuart's short-context
  framing; K-side flips are k8's own 0.88%). One live candidate:
  +22% capacity vs kvarn8, rides C's serve path. Full-context revival
  stays gated on the real-tile re-screen. Violet review item: confirm
  in git history that the tier registration predates the stage-1b
  result (record carries its own proof).
- **Memory honesty (permanent)**: k4 "double the Kindled" holds vs
  FP16 (2.6×); vs kvarn8 even full k4 is 1.57×; TRUE 2× needs the
  untouched m3_idx axis (256B/tok/layer, its own screen, someday).
- **Harvest spec SIGNED**: SPEC_kvarn4_realtile_harvest_2026-07-10.md
  (clement/kvarn4-asym @ c5cd2ad) — role-split K/V gates,
  depth-stratified interior sampling, conversation-shaped source,
  anchor on identical tiles, jointly registered values. Violet
  reviews tonight before it hardens.

## UPDATE 00:05 (Jul 11) — boot-night plan blessed; audit closed; hook approved

- **Subagent-tier provenance: AUDIT CLOSED.** Clement self-declared
  before my audit ran (ab52354 + 1b52565): tier POSTDATES stage-1b
  (git timestamps verified: 23:25:26 kill / ~23:30 Stuart's subagent
  directive / 23:37:24 tier-in-header-before-sweep); retrospective for
  1b's K8V4 numbers, pre-registered for 1c's own data; testimony: all
  four GATE_ constants identical across both script executions.
  PROVISIONAL label everywhere; boot day = first prospective test.
  Standing rule adopted: timestamps where possible, declared testimony
  where not, and the record says which is which.
- **Spec Amendment 1 (idx third role): co-sign conditional** on three
  pre-data refinements (all stricter/clarifying): Gate B mass
  renormalized WITHIN clean top-k (all-blocks denominator understates
  lost mass); pre-registered minimum query count (≥256 proposed);
  flips-wording clarification (score-side ⇒ composition's K-variant
  rate). Clement amending; hardens jointly on landing.
- **Boot-night plan: Option B BLESSED** (supervisor script Stuart
  launches once, then sleeps) under a three-condition contract:
  MAX_RESTARTS=1 structural; mechanical RAM check + MQTT announce
  before boot 2 (refuse-and-log); content-verified signal file dropped
  only after tonight-irreplaceables' artifacts are READ AND VERIFIED.
  Rationale on record: B's failure mode degrades exactly to A (harvest
  banked either way), so B strictly dominates at zero Stuart-margin.
  Stuart-time: one 60-second supervisor read + one paste. Stuart's
  one-word bless still required at launch.
- **Harvest hook (clement/kvarn-harvest-hook @ 088f6c2): Violet seat
  CLOSED APPROVE, no must-fixes** (3 notes: doc/code role-name drift;
  panic claim precise-not-overstated re FFI evals; sidecar cache-field
  overload). Boot-night condition: EARLY idx_k-present check at first
  32K crossing (declared gap fails empty — catch it with time to
  react). Xander's seat on top when he lands, or single-seat merge on
  Clement's call.
- **Violet migration branch**: build+test round 1 in flight (isolated
  target). Runtime msa_core lands the core A/B axis reboot-free;
  supervisor flip spends only on the fetch construction axis. Merge
  order: migration → base BEFORE Stuart's build.

## UPDATE 00:35 (Jul 11) — migration PUSHED (in review); #29 pinned to base; supervisor + stride-gate approved

- **violet/decode-config-migration @ 2b80cd4 pushed, Clement reviewing**
  (feat ee9f3e8 + clean merge of moved base). Sharded verification all
  green over every touched area: decode_config 26, server::routes
  77/77, m3 50/50, core cache 480/480 (count moved 477→480 = exactly
  the 3 harvest contract tests the merge brought — checksum holds),
  bins build. msa_core default stays BLOCKED until the gate chain
  passes.
- **#29 DECISIVELY pre-existing**: full-suite single-process SIGTRAP
  reproduced 5× on the migration branch, then reproduced IDENTICALLY
  on BASE (server:: shard, main clone, 8a37724). Survives --skip
  bisection → load-shaped, MLX-heavy territory. Task record updated.
  Working merge gate remains sharded per-module runs; systemic fix
  remains sharded-unfiltered CI (cold, post-window).
- **Harvest stride-gate delta (9a51902): APPROVE** — stride_crossed is
  the idx crossing logic generalized + four-way contract test; ~4.2K
  dumps at 300K arithmetic verified; MAX_DUMPS 32768 runaway stop
  intact. Presence checklist now 8K (k/v) then 32K (idx_k).
- **Boot-night supervisor (scripts/boot_night_supervisor.sh): APPROVE**
  — all three PM contract conditions are STRUCTURAL: flip cannot loop
  by program shape (each boot line occurs once, A-dies → exit, no
  restart path); RAM check (vm_stat reclaimable ≥250GB) refuse-and-log
  with MQTT trail to Violet before boot B; exact-token signal file
  (stray touch cannot flip a resident session). Boot A = kvarn8 +
  harvest; boot B = kvarn8 + C (env-seeded fetch=qmm — merge-order
  robust). Two non-blocking polish notes on record (post-B health
  line; port-reuse corner covered by the 120s grace). Stuart's
  involvement: one supervisor read + one paste + one-word bless.
- Watch continues: Violet wakes 01:36/03:36/05:36/07:36; supervisor
  announces route to Violet's MQTT.
