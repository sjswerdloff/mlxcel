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

## MERGE DESIGN (final, pre-implementation — cycle 88, same night)

Two facts from the source (cache.rs update_kvarn8 / fetch_kvarn8) settle
the merge:

1. **Everything is stored ROTATED.** Rotation happens once per incoming
   token at the cache boundary — sink and tail are fp16-ROTATED, tiles
   are quantized-rotated. The full fetch unrotates the assembled window
   once at the end.
2. **wht is orthonormal (1/√N) and self-inverse** (ops.rs::wht →
   hadamard_transform, one fused op). Inner products are preserved:
   ⟨q, K⟩ = ⟨wht(q), K_rot⟩ exactly (modulo fp).

Therefore C runs ENTIRELY in the rotated frame and crosses back once:

- `q_rot = wht(q)` (1 op). Scores against tiles AND against sink/tail
  use q_rot — sink/tail K needs NO fetch and NO unrotation, it is
  scored as stored. Both logit chunks are the same mathematical
  quantity ⟨q, K_r⟩ → concatenation before ONE softmax is exact. No
  log-sum-exp machinery, no fused-SDPA on the fp16 side (fused SDPA
  never exposes logits — that is WHY the sink/tail side is an explicit
  small matmul).
- Op order matches the blocked core exactly: raw logits → concat →
  ·scale → +mask → softmax → split weights → two V paths → add →
  ONE `wht(out)` (self-inverse = unrotate) → o_proj.
- Mask is host-built at [4 kv-heads, W_c] granularity (the selection is
  already host-synced for the union; C reuses that sync): interior rows
  whose selected block is sink/tail → -inf (their rhs index points at a
  safe tile; their softmax weight is then exactly 0, so the V side needs
  no mask at all); sink/tail columns → -inf for heads that did not
  select them. Tail has NO padding in C (real tail_len columns), so the
  position rule vanishes at decode (l==1: every stored position ≤ q_pos
  structurally).

**POOL-FOLD CORRECTION (supersedes the post-micro revision's fold note).**
"Lazy fold on the gathered scalars" + gather_qmm's internal rhs_indices
gather are INCOMPATIBLE: gather_qmm's scales/biases arguments are full
pool-shaped arrays gathered internally, so lazy-folding them means two
elementwise multiplies over the ENTIRE history per layer per token —
an O(T) term (~3 ms/token at 500K, growing). That is the disease this
program kills, hiding in this doc's own revision. Resolution (Shape 2c,
gather-then-fold): take_along_axis the selected tiles' codes AND scalars
explicitly first (the same cheap block-gather fetch_fp16_blocks measured
all night, at HALF the bytes for u8 codes), fold the small gathered
scalars ([4·top_k, 128, 1] — 64 KB), view the gathered codes u8→u32
(ffi::view, lib.rs:1868 — the zero-repack fact), then gather_qmm with
identity indices (≡ batched qmm). Zero cache-side changes except
read-only accessors. Fold-at-write (storing scale·s_row, zp·s_row
instead of three scalars) is the cleaner long-term shape — less memory,
no per-token fold — but touches quantize/synth/K0 contracts; recorded
as a follow-up optimization with its own gate, NOT tonight's cut.

**Dispatch shape (decode, b==1, l==1, per MSA layer):** ~30 small ops +
2 gather_qmm + 2 wht. More ops than G's ~15 but the 4 MB fp16 dequant
materialization is gone (u8 gathers are half the bytes; no dequant
chain, no compact-window build). Rank cell kvarn8×C decides, as always.

**Gate:** MLXCEL_MSA_FETCH=qmm AND mode==KVarN8 AND no paged backing AND
b==1 AND l==1 AND n_tiles≥1; anything else falls through to the existing
fetch_msa_blocks dispatch (blocked or G core per MLXCEL_MSA_CORE).
sink_len==128 is structural when n_tiles≥1 (sink fills first) — assert.
Duplicate per-head selections at shallow depth double-count in softmax
exactly as the blocked core's double-gather does — equivalent behavior,
pinned by the tolerance gate, not a new failure mode.

**Acceptance:** contract test C-core vs blocked core on synth states,
same tolerance class as G's core test (fp16 accumulation-order
differences + the fold's ≤9.9e-4 rel mixed-mode products). Xander holds
the refusal on the tolerance gate.

— Clement (clement-7074f29f), cycle 87, drive-through night.
— MERGE DESIGN appended cycle 88, post-Mikvah, same night (drive-through).
