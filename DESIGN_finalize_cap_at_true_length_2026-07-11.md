# DESIGN — finalize-cap-at-true-length (padding never enters kvarn tiles)

Author: Violet (violet-14057653), from the #35 feasibility read.
Status: DESIGN — verdict FEASIBLE; builder unassigned (rides after K8V4
or parallel by owner). Sibling to DESIGN_kvarn_k8v4_engine_2026-07-11
(§3.3 posture + the write-path birth constraint, accepted f4b29fe).

## The two wrongs, one root

Scheduler prefill padding reaches kvarn caches (batched stacked sites AND
NA-aligned chunked sites — the latter hardware-gated, single-sequence).
Root: padded rows enter the quantization pipeline. Wrong 1: `trim()` has
no kvarn arm, so stripping padding rolls `offset` back while finalized
tiles keep the rows — silent desync (tripwired at all four sites,
0028c12, which ABORTS the sequence: honest, but a refusal not a fix).
Wrong 2: Sinkhorn s_col/s_row normalize over garbage rows sharing tiles
with real rows — quality pollution on real data before trim ever runs.
Cap the finalization at true length and wrong-2 dies outright; wrong-1
needs one more piece (next paragraph).

CORRECTED PREMISE (Clement, first cut of #36, 2026-07-11 17:23 —
verified at source by Violet): this doc originally claimed "padding
lives only in the fp16 tail, which the existing dense trim already
handles correctly." WRONG for kvarn caches: `trim()` (cache.rs:3316)
rolls `offset` back and slices only the dense field family
(keys/values/scales/packed/norms) — all None under kvarn. A capped
update followed by `trim(excess)` leaves `kvarn_tail_*` holding MORE
rows than `offset` implies: wrong-1 relocated into the tail. (Systemic
note: kvarn´s parallel field set makes every mode-generic method a
blind-spot candidate — `trim()` and `nbytes()` (#37) failed the same
way the same day. Audit all per-mode-field iterating methods.)

THE AMENDMENT #36 CARRIES: a TAIL-BOUNDED kvarn arm in `trim()` —
slice from the fp16 end-state only (tail first; sink only when hist is
empty, covering short padded prefills whose padding lands in the
sink); REFUSE LOUDLY (return 0, ZERO mutation) if the trim would reach
quantized tiles. This is NOT the deferred general kvarn trim — tiles
stay untouchable; it is the minimum for the cap to compose with the
scheduler´s trim call. Corollary: `padding_trim_would_corrupt`
extends to tail-awareness (excess within the fp16 end-state = safe),
so the tripwire passes exactly when trimming is genuinely safe. Edge 5
below cannot pass without this arm — the six-edge plan caught the
premise failure at first cut, as designed.

## Mechanism (no model-signature cascade)

One field on `KVCache`:

    /// Absolute position (exclusive) beyond which update_kvarn8 must NOT
    /// finalize tiles this update: rows at/after the cap stay in the fp16
    /// tail. Set by the scheduler immediately before a forward whose
    /// input carries prefill padding; consumed (read and CLEARED) by the
    /// next update. None = today's behavior, decode steps unaffected.
    pending_finalize_cap: Option<i32>,

Setter `set_finalize_cap(abs_pos)` — scheduler-side, called per layer
cache. `update_kvarn8` computes its finalization boundary as
`min(natural_boundary, cap)` THROUGH THE ONE BOUNDARY VARIABLE the K8V4
write path routes all finalization decisions through (birth constraint,
accepted). Consume-and-clear keeps the cap one-shot per padded forward —
a later unpadded update finalizes normally from the tail.

## Site placements (the four tripwired sites' pre-forward points)

- Batched stacked prefill (scheduler.rs ~3611-3635): scheduler holds
  `batch_caches` (per-sequence layer slices) AND per-sequence
  `actual_len` at the same scope. For each sequence i, each layer cache:
  `set_finalize_cap(actual_len[i])` before
  `forward_batched_with_context_and_ids`. Covers BOTH the true batched
  overrides and the trait-default per-sequence loop (generate.rs:577
  chain) — per-sequence caches are distinct objects either way.
- Chunked NA-aligned sites (two): b=1, `actual_chunk_len` in scope;
  cap = `cache.offset + actual_chunk_len` before the chunk forward.

Tripwire relationship: the cap makes `excess`-trim correct, so the
tripwire at those sites stops firing for capped updates — it REMAINS as
the guard for any uncapped path (defense in depth, not removed).

## Edges (test plan pins each)

1. Cap lands mid-tile: rows below cap but past the last full tile stay
   in the tail with the padding — correct (tail is fp16; trim slices it).
2. Cap == offset+len (no padding): boundary math must equal today's
   exactly (identity case — golden).
3. Cap beyond input end: same as None (clamp, no effect).
4. Multiple padded chunks: set-consume per chunk; a stale cap must never
   survive into the next update (consume-and-clear test + named
   mutation: remove the clear, watch the next update under-finalize).
5. Padded-kvarn roundtrip golden case: padded prefill → capped update →
   trim(excess) → decode reads == unpadded reference decode reads,
   bit-exact on the K side / tolerance per §4.2 on fused V.
6. Sinkhorn purity: with cap, s_col/s_row over a padded batch ==
   scales from the unpadded batch (the wrong-2 kill, asserted directly).

## Non-goals

- No change to decode-step behavior (cap absent).
- No kvarn trim arm (the cap makes it unnecessary for padding; the
  tripwire keeps guarding everything else).
- No NA-hardware live validation here (tripwire live-fire proof remains
  REQUIRED before NA kvarn deployment — separate, recorded).
