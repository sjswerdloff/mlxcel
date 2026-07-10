# H0 depth profile — first measurements (2026-07-10)

Instrument: `kvarn-decode-bench` (this branch) per
`DESIGN_decode_experiment_harness_2026-07-10.md` §H0. Synthetic KVarN8
states written directly into cache fields (layout pinned by the
`synth_state_tests` contract suite); production-geometry SparseAttention
layers from random weights; real per-token decode loop; one eval per token.
Raw captures in `results/`. Seed 42 throughout.

Scope caveats (the bench RANKS; the live server CONFIRMS):
- fp16 projections (live model runs MXFP8 quantized_matmul) — constant
  offset, path-independent.
- No MoE/MLP, no sampling: numbers are ATTENTION-ONLY ceilings.
- `--profile` numbers are the serialized ceiling (forced eval per stage
  boundary): exclusive per-stage cost, biased upward, for RANKING only.

## Setup cost — the reason H0 exists

| depth | synth setup | equivalent prefill |
|------:|------------:|-------------------:|
| 100K  | 1.8 s       | minutes            |
| 300K  | 3.0 s       | ~tens of minutes   |
| 500K  | ~5 s        | (untested live)    |

## Clean wall-clock (no profiling), mean ms/token, 32 steps

| config                     | 100K  | 300K  | 500K  |
|----------------------------|------:|------:|------:|
| production mix (3 dense + 57 MSA) | 202.1 | 305.5 | 370.8 |
| dense-prefix only (3 layers)      |  36.8 | 109.7 | 183.9 |
| MSA only (57 layers)              | 166.5 | 201.9 | n/r   |
| additivity check (dense + MSA)    | 203.3 | 311.6 | —     |

- Additivity within 2% at both depths where all three were run — the
  decomposition is trustworthy.
- Attention-only ceilings: 4.95 tok/s @100K, 3.27 @300K, 2.70 @500K.
- 300K MSA-only mean carries a 740 ms outlier (p50 184.1); occasional
  slow steps correlate with tile-finalization boundaries (tail rolling
  into a quantized tile — production-real behavior, kept).

## Serialized per-stage profile (per MSA layer per token, `--profile`)

| stage        | 100K  | 300K  | depth behavior |
|--------------|------:|------:|----------------|
| block_fetch  | 1.70  | 1.94  | ~flat (O(top_k); small growth = less head-union overlap at depth) |
| attn_core    | 1.24  | 1.25  | flat (O(top_k)) |
| selection    | 0.46  | 0.73  | grows, SUB-linear (bandwidth-efficient matvec + cheap argpartition over 782→2344 blocks) |
| union_sync   | 0.20  | 0.21  | flat, small |

## Findings

1. **Correction 1 answered: selection does NOT take over at depth.**
   At 300K it is 18% of the serialized MSA spans, #3 of 4 stages. The
   8K stage ranking survives to 300K: block_fetch #1 (47%), attn_core #2
   (30%). No selection work is justified by this data.

2. **O(top_k) is empirically confirmed for the gathered path.** Fetch,
   core, and sync are depth-flat 100K→300K; the 57-layer MSA wall moves
   only 166.5 → 201.9 ms (p50 184) over 3× depth.

3. **NEW — the dense-prefix floor is the emerging bottleneck.** M3's
   first 3 layers have no index projections (`sparse_attention_freq =
   [0,0,0,1,...]`); every decode step they dequantize the ENTIRE KVarN8
   window and run dense attention. Measured directly: 36.8 → 109.7 →
   183.9 ms/token (2.98× per 3× depth — textbook O(T); the 300K→500K
   extrapolation predicted 183, measurement returned 183.9). At 500K the
   3 dense layers cost as much as the 57 MSA layers combined. **No
   gathered-path work (G/B/C) touches this cost.**

## D1 measured (same evening — convergence: Clement + Xander + Violet)

D1 = layer-selective fp16 for the dense-prefix layers, implemented as a
first-touch downgrade keyed on `index_q_proj.is_none()` (empty-cache-only;
`MLXCEL_KVARN_ALL_LAYERS=1` reproduces the floor). Bench A/B
(`--dense-cache fp16|kvarn8`), mean ms/token, 32 steps:

| config                    | 100K  | 300K  | 500K  |
|---------------------------|------:|------:|------:|
| production mix, pre-D1    | 202.1 | 305.5 | 370.8 |
| production mix, D1        | 171.2 | 211.6 | 211.3 |
| dense-3 residual (fp16)   |   3.8 |  n/r  |  14.7 |

**Production mix is depth-FLAT from 300K on under D1** (211.6 ≈ 211.3).
The dense residual is the unavoidable memory-bound fp16 attention read —
9.7–12.5× under the kvarn8 dense floor. Ceiling at 500K: 2.70 → 4.73 tok/s.

## fp16-KV baseline (Stuart's question: what does the memory halving cost?)

All-fp16 (`--cache-mode fp16 --dense-cache fp16`): same sparse selection,
fp16 windows, zero dequant anywhere. Mean ms/token:

| config       | 100K  | 300K  | 500K  |
|--------------|------:|------:|------:|
| all-fp16     | 118.8 | 225.3 | 331.8 |
| kvarn8 + D1  | 171.2 | 211.6 | 211.3 |

- fp16 grows PERFECTLY linearly (+106.5 ms per 200K). Decomposed: the 57
  fp16 MSA layers are the O(T) (114.1 → 318.6 ms, 100K→500K — the v1
  full-window flow's gather/mask machinery); the dense-3 residual is
  small (3.8 → 14.7 ms).
- **Crossover ≈ 250K. At 500K, kvarn8+D1 is 1.57× FASTER than fp16 at
  half the memory.** The gathered path reads O(top_k) bytes while the
  fp16 flow touches O(T): at depth, quantization is a speed WIN, not a
  tax. Below ~250K fp16 is faster (1.44× at 100K) — exactly the
  depth-gated dispatch (game board E) shape, and a gathered fp16 flow
  (selection → gather from the fp16 buffer, no dequant at all) would
  likely win at every depth if shallow sessions ever matter enough.
- Answer to the incremental-prefill framing: a resident Kindled session
  at 300–500K pays LESS per decoded token on kvarn8+D1 than on fp16,
  while occupying half the memory. The halving is free-or-better at
  target depths, today, before B/C.

## fp16-gathered — the missing matrix cell (late evening, fetch lane)

`--cache-mode fp16-gathered` (`MLXCEL_FP16_GATHERED=1`): the SAME gathered
flow (selection → `fetch_msa_blocks` → compact core) on fp16 buffers — a
pure block gather, zero dequant. Fetch contract identical to kvarn8's
(bitwise contract tests), so the flow is fetch-source-agnostic.

Clean production-mix matrix (mean ms/token, 32 steps, D1 dense layers):

| cell                    | 100K  | 300K  | 500K  | KV mem @500K |
|-------------------------|------:|------:|------:|-------------:|
| fp16 full-window        | 118.8 | 225.3 | 331.8 | ~2×          |
| kvarn8 gathered (+D1)   | 171.2 | 211.6 | 211.3 | ~1×          |
| fp16-gathered (+D1)     | 125.5 | 145.4 | 161.6 | ~2×          |

- **fp16-gathered is the speed champion at depth**: 1.46× over kvarn8 at
  300K, 1.31× at 500K, 2.05× over fp16-full at 500K; growth is mild
  (+36 ms over 400K — selection + union growth), nothing like fp16-full's
  linearity.
- **kvarn8+D1 remains the capacity champion**: within 1.31–1.46× of the
  speed champion at HALF the memory. The production shape is
  occupancy-gated (speed-per-GB frontier), exactly as the plan framed.
- Diagnostic for the core lane: serialized per-stage spans are nearly
  identical between kvarn8 and fp16-gathered (fetch ~1.9 vs ~1.9–2.6,
  core ~1.25 vs ~1.33), yet the clean walls differ by ~66 ms at 300K —
  the dequant chain's op-count/pipeline drag is invisible to the
  serialized ceiling. That drag is what G (op collapse) and B/C (fused
  dequant) attack; closing it would put kvarn8 near fp16-gathered speed
  at half the memory.
- 300K profiled run captured separately
  (`h0_300k_fp16gathered_profile.txt`) — serialized numbers, NOT
  comparable to the clean matrix above.

## KVarN4 stage-1 tile screen (parallel lane, same evening): KILLED

Background-agent screen (synthetic tiles, k8 as anchor comparator):
argmax-flip rate 14.5% (k4) vs 0.88% (k8) on identical query populations —
16× worse, 3× over the 5% gate, seed-stable, present even on clean
gaussian tiles (not an outlier artifact). Reconstruction p95 at the 0.10
gate boundary; only V-side passes. Stages 2–4 off the board; revival, if
ever, is a Sinkhorn/tile-param tuning loop on real-activation tiles.
Details: `RESULTS_kvarn4_tile_screen_2026-07-10.md` (shared clone).

## Game-board implications (converged 2026-07-10 evening)

- **Rank 0 (new): dense-floor fix.** Layer-selective cache mode: leave
  the 3 non-MSA layers' caches Fp16. They already dispatch dense
  natively (`supports_block_fetch` = false routes them off the kvarn
  fetch automatically); the change is confined to per-layer mode choice
  at cache construction. Cost: ~1.2 GB fp16 @300K (~2 GB @500K) against
  the ~35 GB KVarN8 saves on the other 57 layers. Removes the O(T)
  dequant entirely; residual dense-attention read is the unavoidable
  memory-bound minimum. Alternative shape: mlx-lm-standard full-window
  quantized_matmul attention for those layers (B applied to the full
  window). Xander concurs: floor first, then the stack.
- **Rank 1: G** (mask-over-union + one fused SDPA) — attacks attn_core
  (~71 ms serialized across 57 layers).
- **Rank 2: B/C qmm paths** — attack fetch+core together (~180 ms
  serialized) behind H3's entry gate.
- **Selection: park.** Not justified at ≤500K.

— Clement (clement-7074f29f), cycle 87, H0 per Violet's harness plan;
Xander second eye on the reranking.
