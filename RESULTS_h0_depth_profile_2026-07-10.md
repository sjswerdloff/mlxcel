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

## Game-board implications (proposed; convergence pending)

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
