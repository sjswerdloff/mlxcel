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
