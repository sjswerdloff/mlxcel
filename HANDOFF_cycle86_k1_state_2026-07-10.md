# HANDOFF — K1 decode optimization state (2026-07-10 evening)

**Branch:** clement/k1-dequant-after-gather (all pushed; release binary
carries everything through the BATCHED block fetch).
**Docs:** DESIGN_fused_msa_kvarn_decode_plan (incl. K2 REVISED) +
PROPOSAL §8–§9.

## Where things stand
- KVarN8 semantics: validated to 300K (20/20 paired, §9). Untouched.
- K1 (gathered decode): correctness bit-identical to v1 at every gate
  (atol-0 test, live A/B, 50K three-way 0-regressed). Performance history:
  8K 512-tok: v1 160s → K1 247s → K1+idx-fix 246s (fix 2 = measured no-op)
  → K1+batched-fetch: **161.9s — parity with v1, the 87s regression
  erased** (512-tok leg; 4096 leg deliberately killed — no decision
  pending on it). Parity at 8K is the structurally correct outcome
  (union ≈ whole window there); the win territory is 32K/300K.
- Serialized-ceiling profile (the decisive artifact, 11K calls):
  block_fetch 4.06ms/call (63%) → now batched; attn_core 1.83 (28%) →
  K2-revised target; selection 0.39; union sync 0.20 (3% — leave alone).
- K2 REVISED: MLX-native quantized_matmul per selected block (op already
  in our bridge), scale·s_row folds into per-group scales, s_col applies
  query-side per tile. Format conversion at tile-finalization. Verify the
  fold OFFLINE first (K0 pattern). No custom shader.

## GOVERNING PLAN (superseding the next-actions below): Violet's
DESIGN_decode_experiment_harness_2026-07-10.md — H0 synthetic-state bench
before ANY path choice; approach G (mask + one fused SDPA over the union
window) as first candidate; K2-revised (qmm) demoted behind H3's entry
gate; H2 config-echo plumbing; H4 KVarN8 persistence as the strategic
unlock. Non-negotiable: no path choice ahead of a measurement at the
depth that matters.

## Next actions (in order — see governing plan)
1. Restart test server WITHOUT profile flags → run the 8K capture
   (count-up, gen 512,4096) → compare v1 160s / K1 247s. If batched fetch
   lands where the profile says, 8K should now beat v1.
2. If yes: 32K capture, then 300K --limit 2 (the depth where O(top_k)
   actually breathes: union ≈ 5% of window at 300K vs ~100% at 8K).
3. K2-revised offline fold verification, then implementation.
4. PR + Xander review before any merge to the shared branch.

## Instruments (env flags, zero-cost unset)
- MLXCEL_K1_PROFILE=1 — per-stage serialized-ceiling breakdown, k1.profile
  console lines every 1024 calls.
- MLXCEL_K1_FIXED_BLOCKS=1 — sync-free floor, OUTPUT GARBAGE, timing only.
- Boot artifacts: kv_cache_mode=KVarN8 + kvarn_decode_path=gathered.

## The day's method lesson (context for whoever picks this up)
Three cost models (mine ×2, one refinement) died on measurement; three
reviewers independently mis-ranked the bottleneck. The 80-line profiler
settled it in one run. Measure, then cut.

— Clement (clement-7074f29f), cycle 86, at PREPARATION
