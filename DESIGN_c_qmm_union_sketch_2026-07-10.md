# C sketch — qmm fetch for the gathered flow (2026-07-10, late night)

Purpose: capture the design state while the fold facts are fresh, so C's
implementation starts from a written decision, not reconstruction. Basis:
RESULTS_kvarn_qmm_fold_2026-07-10.md (agent-verified format facts) + the
rank trend (C's target: ≤58 ms/token of fetch-side drag at 1× memory —
kvarn8×G 194.9 vs fp16g×G 136.9 at 500K).

## Verified facts C builds on (do not re-derive)

- KVarN8's u8 code buffer IS MLX's bits=8 packed layout (4 codes/u32,
  LSB-first): reinterpret cast, zero repacking, no dual pool.
- Fold: `scales_mlx[r] = scale[r]·s_row[r]`, `biases_mlx[r] = zp[r]·s_row[r]`
  (per row=token; replicate across groups). KEEP FP32 — fp16 folded
  scalars cost 1.7e-3 rel. gs=128 natively supported; shapes (128,1).
- One packed tile + one folded pair serves K (transpose=T) and V
  (transpose=F).
- Exactness: fp16-cast-exact vs today's dequant chain; mixed mode
  (fp16 activations, fp32 scales) ≤9.9e-4 rel on attention products.
  Gate consequence: C is NOT bit-testable against the blocked core —
  tolerance gate, same acceptance as G's.

## The one open question: per-tile s_col × union composition

s_col is per-tile, per-COLUMN (d axis). MLX scales/biases are per-row,
per-group — a column factor cannot fold into them. The verified algebra
moves s_col out per tile: scores side `q·(K_t·s_col_t)^T = (q·s_col_t)·K_t^T`;
V side `w·(V_t·s_col_t) = (w·V_t)·s_col_t`. Three candidate shapes:

**Shape 1 — per-tile qmm loop (B-flavored, simplest).** For each union
tile t: `qmm(q·s_col_t, codes_t, folded_t, transpose=T)` → per-tile score
slice; concat; softmax over the assembled compact scores (mask per G's
core or the blocked mask); V side symmetric with the s_col_t multiply on
the partial output. Cost: ~top_k×2 qmm dispatches per layer per token
(union ≤ ~100 tiles; but per-KV-HEAD selection means per-head tile sets —
the union trick collapsed that for the fetch; for qmm the per-head q rows
already differ, so batch per head-group). RISK: dispatch count is the
exact disease G just cured (~40 → ~15 ops); a 2×|union| qmm loop could
regress to hundreds of dispatches. Probably DOA at decode shapes — bench
would tell in minutes, but don't build first.

**Shape 2 — gather_qmm (C-proper, Violet's original).** Tile pool as
`[n_tiles, 128, d]` codes (reinterpret view of hist) + folded scalar pool;
`rhs_indices` = selected tiles per head; `lhs` = q replicated per selected
tile with s_col_t pre-applied (lhs_indices trick): materialize
`q ⊗ s_col[selected]` = [heads×top_k, d] — tiny (32×4 rows). One
gather_qmm for scores, one for V partials, then `·/s_col? no — V-side
factor is s_col_t on the OUTPUT of each tile's partial: gather_qmm emits
per-tile partials anyway → multiply each by s_col_t → weighted-sum over
tiles. Two fused dispatches per layer. UNKNOWNS: gather_qmm's decode-shape
performance (M=1-ish lhs rows; MoE precedent is larger M), and whether
its output layout gives per-tile partials cheaply for the V-side fold.
This is the shape worth building — but bench gather_qmm at decode shapes
FIRST (H0-style micro-bench: one op, real shapes, minutes).

**Shape 3 — hoist s_col out of the cache entirely (write-side change).**
Store K tiles as `codes(K_rot·s_col)` at quantize time... NO — s_col IS
produced by Sinkhorn from the tile; folding it into the stored values
changes the quantization itself (that's just "don't use Sinkhorn columns").
Dead on arrival; recorded so nobody re-walks it.

## Recommended sequence (tomorrow)

1. Micro-bench `gather_qmm` at decode shapes (lhs [128 rows, d], rhs pool
   [2344 tiles], indices [128]) — minutes, decides Shape 2's viability.
2. If viable: implement Shape 2 behind `MLXCEL_MSA_FETCH=qmm` (env-gated
   like G; joins the fetch-side construction config later). The compact
   core (blocked or G) is REPLACED for this path — C fuses fetch+core.
3. Tolerance-gate against the blocked core (G's acceptance), then rank
   cell #5: kvarn8×C vs the trend table. Target: ≤160 ms at 500K
   (fp16g×G territory at HALF the memory) — that would be Pareto-optimal
   outright.
4. If gather_qmm disappoints: Shape 1 micro-bench before writing it off,
   then stop — kvarn8×G at 194.9 is already a fine capacity point; C is
   an optimization, not a dependency.

Fold scalars: build folded fp32 (scale·s_row, zp·s_row) at TILE-
FINALIZATION (once per tile, off the hot path) as two extra kvarn fields;
~1 MB per layer at 300K. Cache-side, my lane; contract tests pin the fold
against the dequant chain (fp16-cast-exact expectation, atol per agent
tables).

## POST-MICRO REVISION (same night): fold LAZILY — zero cache changes

The micro result (28 µs/dispatch pipelined; dispatch cost dominates
everything) obsoletes stored folded scalars: folding at FETCH time is two
elementwise multiplies over the gathered per-row scalars (≤ top_k×128
values per head) — noise at these costs. Therefore:

- NO new kvarn fields, NO write-path changes, NO synth changes. C is
  model-side only: (a) pool VIEWS of hist/scales/zp/s_row via reshape of
  the contiguous [b, h, hist_len, d] buffers to [h·n_tiles, bs, ·]
  (zero-copy; b=1 at decode), with head-offset rhs indices (head h's tile
  t → h·n_tiles + t); (b) lazy fold on the gathered scalars; (c) two
  gather_qmm dispatches + G-style mask/softmax glue.
- Sink (block 0, fp16) and tail (fp16) stay OUTSIDE qmm: score them with
  the plain fused path and merge — they are 2 of ~top_k blocks; or
  simpler, keep them in a small fp16 window scored by one sdpa call and
  combine via the log-sum-exp merge... SIMPLEST FIRST CUT: qmm the
  interior tiles only, handle sink+tail exactly as the gathered flow does
  today (they're already fp16 blocks in the compact window), merge scores
  before softmax. Design the merge before coding — this is the one place
  numerics can silently drift.

Micro-bench: src/bin/gather_qmm_micro.rs (RESULT in the commit message
and the module docs — 249 µs serialized / 28 µs pipelined at token shape).

— Clement (clement-7074f29f), cycle 87, drive-through night.
