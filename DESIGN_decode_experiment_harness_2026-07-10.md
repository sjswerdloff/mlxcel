# Decode-path experiment harness + revised optimization game board (2026-07-10)

Authored by Violet (violet-14057653) at Stuart's request, on Fable 5, as a
second decorrelated set of eyes during the working window. Builds on: the
serialized-ceiling profile (the decisive artifact), the batched block fetch
(2fba77d, 8K re-measure pending), K2 REVISED (00e2ddb), and the cycle-86
handoff. Supersedes my morning read, whose #1-ranked suspect (union host
sync) the profiler demoted to 3% in one measurement — this doc is written
under that lesson: **no path choice ahead of a measurement at the depth that
matters.**

## Three corrections to the current plan's framing

### 1. The profile is 8K-only, and selection is the only O(T) stage left

Post-batching, every profiled stage except selection is O(top_k) — depth-
independent. Selection (indexer matvec over the full window + top-k over all
blocks) grows linearly with T. At 8K it is 6% (0.39ms); at 300K it is the
only stage that *must* grow (~37× more window to score), and the stage
ranking may reorder entirely. K2-revised targets attn_core (28% at 8K) — but
if selection is the new 63% at 300K, quantized-matmul work on the core is
rearranging deck chairs. **Profile at depth before choosing.** The bench in
§H0 makes that a minutes-scale operation.

### 2. attn_core's 1.83ms is op-count overhead, not arithmetic — and the
###    biggest collapse needs no new formats

Verified in-tree: `sparse_decode_core` makes ~40 `mlxcel_core::` op dispatches
per layer per token (hand-rolled matmul → softmax → matmul plus the 6-D
blocked take_along_axis machinery). At decode (M=1) each op is microseconds
of arithmetic plus launch overhead; 60 layers × ~40 ops ≈ 2,400 dispatches
per token in the cores alone. The FLOP volume is trivial (top_k×128×d matvec
per head). The fused-dequant framing (K2-revised's and my morning message's)
optimizes the wrong axis: the win at these shapes is **collapsing the graph**.

The collapse is already bridged: `fast_scaled_dot_product_attention(q, k, v,
scale, mask)` (lib.rs:1142, nullable mask confirmed) — and `dense_attention`
already uses the family, so its numerics have in-repo precedent. See
approach G below.

### 3. The iteration loop at depth is broken today, and that — not compile
###    time — is the real experiment cost

Prompt-cache adoption rides `clone_handle`/detach, which is Fp16-only;
KVarN8 snapshots are refused by design (integration map, fail-closed). So
"prefill 300K once, then N probe requests adopt it" does not work yet. Every
new server session pays the full deep prefill. Compile is minutes
(incremental cargo); weight reload is tens of seconds (page-cache-warm mmap);
**deep prefill is the unavoidable multi-minute cost** — so the harness design
optimizes for *never dropping a resident prefill*.

## The game board

- **A. K1 + batched fetch** (built; 8K re-measure pending). Dequant union →
  fp16 compact window → hand-rolled fp16 core. The baseline candidate.
- **B. K2-revised: quantized_matmul over the gathered union** (00e2ddb).
  Gather u8 *codes* (bytes, cheap), two fused quantized matmuls (the shipped
  mlx-lm QuantizedKVCache pattern: qmm(q, k, transpose=True) → softmax →
  qmm(w, v)). Kills fp16 materialization and the fp16 core. Requires
  MLX-format tiles (conversion at tile-finalization, off the hot path).
- **C. gather_qmm — no union at all.** Tiles as a flattened MLX-format pool
  `[n_tiles_pool, 128, d]`; per-head `selected` indices (+ on-device per-head
  offsets) become `rhs_indices` directly. The kernel gathers; union, host
  sync, and all gather materialization disappear. `gather_qmm` is already
  bridged (lib.rs:1000) and battle-tested in the MoE switch layers ("already
  saturates the GPU"). KVarN semantics fold exactly:
  - per-row scale/zp → MLX per-group scales/biases: `scales = scale`,
    `biases = -scale·zp`, repeated across the row's groups. Bit-exact.
  - per-tile s_col, K side: apply to the *query* per selected tile via
    `lhs_indices` (materialize `q ⊗ s_col[selected]` — top_k rows per head,
    tiny, on device). Exact; no requantization.
  - per-tile s_col, V side: gather_qmm emits per-tile partials anyway →
    divide each by `s_col[tile]` → sum over top_k. Exact.
  - WHT rotation: rotate q once per step, un-rotate output once (already
    the plan's Q/O trick).
- **D. Custom fused Metal kernel.** The community "3-4× slower" caution is
  *Python* dispatch overhead (`mx.fast.metal_kernel` from Python); from
  Rust/mlx-c the dispatch cost is far lower, so D is not disqualified by
  that number — but it remains weeks of work and last resort. Community
  precedent (mlx-qsdpa): 1.7× over the two-call qmm path at 128K, yet 0.61×
  fp16 under GQA — sobering.
- **E. Depth-gated dispatch** (production shape, not a rival). The handoff
  notes union ≈ 100% of window at 8K vs ~5% at 300K — that is *why* K1 lost
  at 8K: gathering buys nothing when selection covers everything. Shallow →
  full-window path (v1 or the verbatim mlx-lm full-qmm, no gather machinery
  at all); deep → the gathered winner. Crossover = a measured config value.
- **G. Gathered window + ONE fused SDPA call** ← the addition this doc
  argues should go first. Keep update_only + selection + batched fetch (all
  landed). Replace the core: build a per-head boolean/additive mask over the
  fetched union window — the remap table already computes exactly the needed
  head→compact-slot mapping; the mask is ~4-5 ops (per-head selected-block
  membership × the pos ≤ q_pos rule) — then one
  `fast_scaled_dot_product_attention` call. ~40 ops → ~6-10 per layer. No
  new formats, no dual pool, no cache changes. GQA: selection is per
  kv-head; query heads in a group share the group's mask (broadcast).
  Numerics: leaves the bit-identity chain (different kernel accumulation) —
  the same acceptance K2 already made for its shader ("fp16-cast tolerance,
  not bit-exact"); G slots into that planned tolerance-gate methodology, with
  the dense path as in-repo numerics precedent.

Order-of-magnitude check: if batched fetch + G take the serialized per-layer
cost to ~1ms, that is ~60ms/token ceiling → ~16 tok/s at depth *before any
qmm work* — which would trip K2's own entry gate ("optimize the measured
bottleneck, not the planned one") and possibly end the project at G.

## The harness (H-phases)

### H0 — synthetic-STATE bench (build FIRST, ~a day from test scaffolding)

Do not prefill synthetic data — **write random codes/scales directly into
the KVarN8 cache fields.** Instant 100K/300K/500K states, no model load, no
prefill; timing-representative (same shapes, same memory traffic; the hot
path has no data-dependent branches — garbage values time the same).
Requirements:
- Drive the full **60-layer per-token loop**, not one layer once — the
  fdef67b lesson: per-layer × per-token costs (the m3_idx concat) are
  invisible in single-layer tests.
- Profiler spans built in (reuse the MLXCEL_K1_PROFILE instruments).
- Paths A / G / B / C selectable; 512-step timed runs.
This one binary answers: the depth-profile (correction 1), the G op-count
diagnosis (correction 2), gather_qmm's decode-shape performance (C's main
unknown), and doubles as the K0-style tolerance-verification vehicle for
G/B/C. **Bench RANKS; the live server CONFIRMS** — never promote on bench
numbers alone.

### H1 — approach G behind the dispatch enum

If (and only if) H0 confirms the op-overhead diagnosis. Hours-to-a-day.
Dispatch enum: `kvarn_decode_path = v1 | gathered | gathered_sdpa |
qmm_union | qmm_gather` — all compiled in, selected at runtime.

### H2 — config reload + echo (the live-iteration loop)

- `ArcSwap<DecodeConfig>` read per decode step; TOML re-read on **SIGHUP**
  and via a **localhost-only `POST /admin/decode-config`** (primary: the
  response body confirms the now-effective config in-band — no PID hunting,
  works cross-host, auditable).
- Move the env-only instruments (MLXCEL_K1_PROFILE, MLXCEL_K1_FIXED_BLOCKS)
  into the reloadable config — today, toggling profiling costs a restart.
- **Config version echoed in every `k1.profile` line and every response**
  (`kvarn_decode_path=... cfg=N`), so no measurement can silently
  mis-attribute its path — the report-vs-artifact discipline, mechanized.
- The resident-session depth loop this enables: one server boot, one 300K
  prefill, then alternate 512-token generation segments with the global
  path switched between segments. Depth drift per segment is +512 tokens
  (0.17% at 300K) — negligible for timing. This is the cheap 300K A/B loop
  that works *without* KVarN8 persistence.
- Per-REQUEST path override: probe-mode only (batch=1); a fused batch
  cannot mix structural paths — **refuse loudly** if batch > 1.

### H3 — qmm paths (B, then C) only on measured need

Entry gate: the H0/H2 depth-profile shows fetch materialization or window
bandwidth still dominant after G. Prerequisite: MLX-format tile pool
dual-written at tile-finalization (config-gated `maintain_qmm_pool`; both
pools ~1 byte/value, so 2× quantized ≈ still ~4× under fp16 — fine for
experiments, off in production). Offline fold verification first (K0
pattern); C's s_col-via-lhs_indices formulation keeps exact KVarN semantics.

### H4 — KVarN8 detach/persist (PR-3 subset), the strategic unlock

Tile-aligned trim (128 divides the adoption floor, so always aligned in
practice — assert it), sink never trimmed, tail re-derivable → refuse
non-aligned loudly. Persisted-cache stamp carries (mode, bits, iters, sink,
tile) — Silas's rule. Pays twice: cheap depth iteration (prefill once,
persist, adopt per experiment) AND production persistence for the vessel's
deep contexts.

### H5 — flag only: mx.compile spike

Fixed decode shapes at l=1 make the per-step graph a compile candidate
(would attack the same op-dispatch overhead G attacks, globally). Unknowns:
mlx-c closure/compile surface from Rust, cache-mutation semantics, shape
growth across steps. Bench-only curiosity; not the plan.

## Attack sequence (summary)

0. H0 bench → depth-profile + candidate ranking (minutes per iteration).
1. G (H1) if diagnosis holds → re-profile at depth.
2. H2 config/echo plumbing → live confirmation on resident sessions.
3. 8K number for A (already queued); depth-gate crossover measured (E).
4. B/C only through H3's entry gate. D stays parked.
5. H4 persistence when the working window allows — it converts every future
   depth experiment from minutes-of-prefill to seconds-of-adopt.
6. Winner through the standard gate chain: A/B probe, 50K paired vs banked
   baselines, 300K spot (uuid_03's tile included).

## Ecosystem sources (2026-07-10 search, Rust-relevance annotated)

- mlx-lm QuantizedKVCache / quantized SDPA: two-call qmm pattern, GQA via
  reshape — the shipped precedent for B. (mlx-examples #1075; mlx_lm
  models/base.py)
- mlx issue #3404: native quantized SDPA — open, no maintainer response;
  documents dequant-materialization as the known failure mode.
- mlx-qsdpa (community fused kernel): 1.7× over two-call qmm at 128K GQA
  but 0.61× fp16 — tempers expectations for D and for qmm-at-GQA generally.
- The "custom kernels 3-4× slower" community number is Python-dispatch
  overhead; it does not transfer to Rust/mlx-c. D is parked on effort
  grounds, not on that number.
- gather_qmm: in-bridge (lib.rs:1000), production-proven in switch_layers
  MoE; no known public precedent applying it to sparse *attention* — C
  would be novel, hence bench-first.

— Violet (violet-14057653), 2026-07-10, on Fable 5 at max effort, second
decorrelated read. For Clement's return: the morning message's ranking is
superseded; the profiler was right and the three of us were wrong, and this
doc's only non-negotiable is H0-before-choices.
