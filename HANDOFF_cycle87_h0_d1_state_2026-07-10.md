# Handoff — H0 bench + D1 + fp16 baselines (2026-07-10 night)

Supersedes HANDOFF_cycle86_k1_state_2026-07-10.md as the current-state map.
Branch: `clement/h0-synthetic-bench` (off clement/k1-dequant-after-gather @
d90af77). Everything below is committed and pushed. Full numbers:
`RESULTS_h0_depth_profile_2026-07-10.md`; raw captures in `results/`
(every file's first line is its self-labeling BOOT config).

## What landed tonight

1. **H0 bench** (`kvarn-decode-bench`): synth deep KVarN8/fp16 states in
   seconds (no model), real 60-layer decode loop, fail-loud dispatch
   check, K1 profile spans via `--profile`. Contract tests pin synth
   layout to production (`synth_state_tests`, 6 tests).
2. **Depth profile**: selection never dominates (18% at 300K, sub-linear);
   fetch/core/sync depth-flat — O(top_k) confirmed empirically.
3. **D1** (dense-prefix fp16, converged Clement+Xander+Violet): first-touch
   structural downgrade in `SparseAttention::forward` keyed on
   `index_q_proj.is_none()`, empty-cache-only, `MLXCEL_KVARN_ALL_LAYERS=1`
   escape hatch. Production mix now DEPTH-FLAT: 211.6@300K ≈ 211.3@500K
   ms/token (pre-D1: 305.5/370.8).
4. **fp16 baselines** (Stuart's question): fp16 KV is perfectly linear
   (118.8/225.3/331.8 at 100K/300K/500K); crossover ~250K; at 500K
   kvarn8+D1 is 1.57× FASTER at half the memory. Pareto-dominant at
   Kindled depths, today, before B/C. fp16's O(T) is the v1 full-window
   flow on the MSA layers, NOT the dense layers.

## Next (converged sequence)

1. **Live re-baseline** — needs the engine; Stuart's launch (or detached
   service per the new norm: big-memory jobs run detached, nohup/tmux +
   pidfile + announced log, never inside an AI session tree). D1 changes
   every live denominator, so re-baseline before anything else live.
2. **G-vs-C bench rank** (H1/H3 as bench paths first): candidates to rank
   with the instrument, on the SPEED-PER-GB frontier (Violet), not speed
   alone —
   - G: mask-over-union + one fused `fast_scaled_dot_product_attention`
     (kills ~40-op core; no format changes);
   - C: `gather_qmm` straight off an MLX-format tile pool (kills fetch
     AND core; needs dual-written pool, H3 gate);
   - **gathered-fp16** (new, from tonight's data — Xander+Violet+me
     independently): selection → block-gather from the fp16 buffer, NO
     dequant. Likely fastest cell of the {full,gathered}×{fp16,kvarn8}
     matrix, at 2× memory. Needs a `fetch_fp16_blocks` (block gather, no
     dequant chain — small).
   Production shape may be occupancy-gated (memory) not just depth-gated.
3. **KVarN4 staging** (queued behind perf window; Violet's 4-stage plan):
   stage 1 tile-level screen has zero dependencies (script-only,
   three-way fp16/k8/k4 reconstruction + score perturbation).
4. H2 (Violet's decode-config, must-fix landed at e9413db) merges after
   Xander's refusal window; then `cfg=N` goes into the k1.profile line
   (my instruments lane).

## Known small items

- Bench's "estimated allocation" line still uses kvarn8 math when
  `--cache-mode fp16` (cosmetic under-estimate; fix when next touching
  the bench).
- 300K runs occasionally show one ~750 ms step (tile-finalization
  boundary + measurement noise; production-real, kept in the data).
- Identity seeds for cycle 87 drafted at
  `~/ai/liberated/kimi-kindled/identity_append_cycle87_SEEDS.md` —
  peer review before concatenation.

— Clement (clement-7074f29f), cycle 87. Violet has the overnight watch
(wakes 01:36/03:36/05:36/07:36).

## LATE-NIGHT ADDENDUM (post-merge-train, drive-through protocol active)

Stuart's directive: no resting tonight — work → Preparation → handoff →
compact → resume from HERE. Violet is PM for the decode feature.

State since the main handoff:
1. **Merge train done**: base = clement/k1-dequant-after-gather @ 12b353a
   (H2 + H0 + D1 merged, suites green). Stuart's morning binary builds
   from the MAIN clone with the shared 19G target.
2. **fp16-gathered SHIPPED** (my fetch lane): branch clement/fp16-gathered
   @ 01d7ab2, one commit off base. fetch_fp16_blocks + fetch_msa_blocks
   dispatch + MLXCEL_FP16_GATHERED gate + bench --cache-mode fp16-gathered
   + 4 contract tests. MEASURED (clean, prod mix): 125.5/145.4/161.6
   ms/tok at 100K/300K/500K — speed champion at depth at 2× memory;
   kvarn8+D1 (211 flat) stays capacity champion. Violet reviewing.
3. **Violet's G shipped** (core lane): violet/g-sdpa-core — fused-SDPA
   masked core behind MLXCEL_MSA_CORE=sdpa. My review: APPROVE, two
   notes (re-verify greens cold; document the no-fully-masked-row
   invariant). Composes with my fetch flag → 4-cell rank matrix.
4. **KVarN4 KILLED at stage 1** (background agent): 14.5% argmax flips
   vs k8's 0.88%, seed-stable. Stages 2-4 off the board.
   RESULTS_kvarn4_tile_screen_2026-07-10.md in the shared clone.
5. **Staleness norm (catch #5 of the artifact shape)**: worktrees use
   their OWN target dirs; the shared target belongs to the main clone
   alone. My worktree's isolated cold build was IN PROGRESS at handoff
   time (background task; if dead, just `cargo test -p mlxcel-core
   synth_state && cargo test -p mlxcel-core fetch_fp16` in the worktree,
   NO CARGO_TARGET_DIR override). Formal re-verification of my 4 fetch
   tests + one re-measured fp16-gathered point still owed to Violet.
6. Cycle-87 identity seeds at
   ~/ai/liberated/kimi-kindled/identity_append_cycle87_SEEDS.md — add
   tonight's material only if truly significant (Stuart's bar), peer
   review before concatenation.

NEXT ACTIONS on resume: (a) confirm isolated-target tests green (my 4
fetch contracts); (b) re-measure one fp16-gathered point (300K) on the
isolated binary and post re-verification to Violet; (c) respond to
Violet's fp16-gathered review; (d) rank session when Violet calls it —
4 cells + p50 columns + longer runs (her non-blocking notes); (e) live
re-baseline waits on Stuart's engine start (detached, pidfile, announce).

## RANK PROGRAM COMPLETE (same night — supersedes NEXT ACTIONS above)

All of (a)-(d) done. Base merged BOTH lanes (9d94e6d); rank ran at
8K/300K/500K on branch clement/rank-session (captures committed).

**Depth trend, p50 ms/token (8K / 300K / 500K):**
- fp16-full: 80.5 / 228.5 / 331.3 (linear O(T) — never optimal anywhere)
- kvarn8×blocked: 130.8 / 195.6 / 209.3
- kvarn8×G: 106.8 / 180.2 / 194.9 ← capacity champion (1× mem)
- fp16g×blocked: 110.2 / 144.3 / 159.4
- fp16g×G: 80.2 / 122.6 / 136.9 ← speed frontier (2× mem)

Structural: fp16-gathered×G dominates fp16-full at EVERY depth (tie at
8K) — no depth gate needed on the fp16 side. G is a pure win on both
fetches at all depths. Gathered cells grow ~+15ms/200K. 500K ceilings:
7.30 tok/s frontier / 4.97 capacity (morning start: 2.70).

**C fold PROVEN by background agent** (RESULTS_kvarn_qmm_fold_2026-07-10
.md — uncommitted in the MAIN clone with script+JSON; k4 artifacts too;
PM to ride onto base): u8 codes ARE MLX's packed layout (reinterpret
cast, zero repack); fold = scales·s_row / zp·s_row (fp32 — TWO design-
doc corrections recorded); gs=128 supported; fp16-cast-exact; mixed
mode ≤9.9e-4 rel. C = days, target ≤58ms fetch drag at 1× memory.

**Integration shape (PM decision)**: decode_config grows msa_core =
blocked|sdpa (runtime-swappable); fetch mode becomes a cache-
construction config key (NOT per-request). Violet's lane. Live gate
chain on the consolidated binary after Stuart's boot — non-negotiable.

**Remaining board**: (1) C implementation (my lane; design notes at
DESIGN_c_qmm_union_sketch_2026-07-10.md on this branch — the per-tile
s_col × union composition is the one open design question, three
candidate shapes sketched there); (2) Violet's decode_config migration;
(3) live gate chain after engine boot; (4) cycle-87 identity seeds
(~/ai/liberated/kimi-kindled/identity_append_cycle87_SEEDS.md) need
peer review before concatenation.
