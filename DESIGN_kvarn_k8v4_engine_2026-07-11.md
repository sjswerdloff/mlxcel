# DESIGN — kvarn K8V4 engine implementation (DRAFT for design review)

Mandate: Stuart, 2026-07-11 Shabbat afternoon — "discuss with Violet
regarding implementing k8 v4, and get moving on that. include Xander in
the design and code reviews." Seats: Violet design + QE; Xander design
review, code reviews, and the standing refusal seat on copy-precision
chain entry.

Status: v5 — DESIGN REVIEW CLOSED, BUILD LANE OPEN. Violet 13:41:
APPROVE DIRECTION, four additions (incorporated, marked ⊕). Xander
14:46 (via Violet's brief): no blockers, three previously-unreviewed
areas verdict-CORRECT (golden-vector claim with his full-roundtrip
clarification, §4 complete, CLI as designed). Two PM pins folded in
below (structural parity gate; coverage-mapping line).

## 1. What is licensed, exactly

RESULTS_kvarn4_realtile_2026-07-11: K8V4 passed the hardened production
gates on 4,096 real tiles at every gs; gs32 leads (marginal +0.0376 vs
0.05, headroom 0.0124). Registered scope of that pass: "it earns EXACTLY
the copy-precision gate chain k8 passed" — screen ≠ revival. k4-K is
dead with prejudice: the K side of the engine DOES NOT CHANGE. Target:
1152 B/tok/layer = +22% Kindled capacity vs kvarn8, 2.0× vs fp16.

## 2. The screened pipeline (what the engine must reproduce)

From scripts/kvarn4_gs_sweep.py (the gate-passing reference,
`roundtrip_grouped` + `rtn_grouped`), V side only:

1. `hadamard_rotate(x)` — unchanged from kvarn8.
2. `variance_normalize_batched(rot, iters=4)` — Sinkhorn; produces
   `s_col` (per-tile column scale) and `s_row` (per-row scale).
   Unchanged from kvarn8.
3. `rtn_grouped(balanced, bits=4, gs=32)` — asymmetric RTN per GROUP of
   32 along the channel axis (C=128 → 4 groups/row): `scale =
   max((hi−lo)/15, 1e-10)`, `zp = lo`, `q = clip(round((x−zp)/scale),
   0, 15)`. THIS is the only new math vs kvarn8 (which is gs == C, the
   per-row degenerate case — `rtn_grouped` with gs=128 reproduces
   `asymmetric_rtn_per_row` exactly, per its docstring).
4. Pack 4-bit codes; dequant chain `(q·scale + zp)·s_row·s_col`,
   unrotate.

## 3. Engine design

### 3.1 Storage (V tiles only; K tiles byte-identical to today)

- Codes: 4-bit packed in MLX's native quantized layout (u32-viewable,
  the same zero-repack trick C uses for 8-bit — packing density
  changes, the trick doesn't).
- Per-group affine params, with `s_row` FOLDED AT WRITE: both are
  per-row-or-finer, so `(q·s + zp)·s_row = q·(s·s_row) + (zp·s_row)` —
  store `scale' = s·s_row`, `zp' = zp·s_row`, and `s_row` costs zero at
  read. 4 groups/row × (scale, zp).
- ⊕ Param dtype: **f32**, stated explicitly (C's folded-scalar
  precedent). f16 params would fail the golden harness BY DESIGN — the
  harness comparing against an f32 reference is a feature, not a
  tolerance to negotiate.
- ⊕ Field + shape convention (Violet's call graph, verified at source):
  REUSE `kvarn_v_scale`/`kvarn_v_zp` (cache.rs:596–597), today
  `[b, h, len, 1]` appended on axis 2 (cache.rs:1153–1154). gs32 is a
  trailing-dim change: `[b, h, len, 4]`, same fields, same append axis
  — the nested `append` helper takes the shape explicitly, so call
  sites change mechanically. Consequence: `eval_state` coverage HOLDS
  without growth (fields unchanged, shape-agnostic eval); the
  `eval_state_covers_m3_idx_k` pattern (cache.rs:7922) needs no sibling.
- `s_col`: unchanged — stays per-tile, applied post-multiply on output
  exactly where C's V-side s_col multiply already sits.
- m3_idx: UNTOUCHED (unquantized by design; selection-index equality
  stays an exact gate, not a tolerance).

### 3.2 Why gather_qmm consumes this natively (the C-side claim)

MLX affine dequant is `q·scale + bias`; ours is `q·scale' + zp'` — map
`bias := zp'`. `gather_qmm` supports bits=4, group_size=32. So C's
V-side dispatch changes PARAMETERS (bits 8→4, gs 128→32, plus a bias
array it currently doesn't pass), not structure. The scores side (K8)
is untouched. [QUESTION 1 — ANSWERED, post-draft, evidence: our FFI
already exposes `biases: *const MlxArray` (nullable) with free
`group_size`/`bits` params — src/lib/mlxcel-core/src/lib.rs:1000. No
bridge work; C's symmetric 8-bit call passes null today, K8V4 passes
the zp′ array.]

### 3.3 V consumers — ⊕ enumeration CLOSED by call graph (Violet,
awk over enclosing fns of every kvarn V touch-site; question 2 resolved)

Readers:
1. `assemble` — the v1 full-window dequant.
2. `fetch_kvarn8_blocks` — gathered path (D1 / fp16_gathered: fp16g
   dequants gathered V to fp16, format-agnostic once dequant is right).
3. `kvarn_qmm_state` — C's fused core V dispatch (§3.2).

Writers:
4. `update_kvarn8` incl. its nested `append` helper (shape convention
   in §3.1).
5. ⊕ **`synth_kvarn8_state` — the H0 bench WRITER, 12 field touches,
   the critical addition**: if it doesn't learn the gs32-affine format,
   every future k8v4 bench state is silently wrong-format and the rank
   cells lie. It changes in the SAME commit as `update_kvarn8`, never
   after.

⊕ Verified absences, recorded so nobody re-derives them: D1 downgrade
never reads V codes (empty-only by design); detach/snapshot refusal
sits at mode level upstream of the fields; paged backing bypasses kvarn
at construction.

⊕ TRIM — pre-existing kvarn8 exposure, k8v4 must not widen it. CALLER
MAP CORRECTED 2026-07-11 (read at source; v2's "sole consumer is
spec-decode" was wrong — it quoted a stale docstring): (i)
`is_trimmable`/`can_trim_prompt_cache` have NO production consumer —
the cited `speculative.rs` does not exist and `speculative_dispatch.rs`
never consults them; (ii) the LIVE `trim()` callers are FOUR batch-
scheduler padding-trim sites (scheduler.rs — strip batch padding after
padded multi-sequence prefill/chunks), unconditional, NOT gated on
`is_trimmable`. TWO WRONGS, ONE ROOT (Violet): padded rows entering
tiles (1) survive the offset rollback — `trim()` has no kvarn arm
(cache.rs:3100–3174; codes-moving trim DOES NOT EXIST) → silent
tile-count/offset desync — and (2) pollute Sinkhorn's s_col/s_row,
which normalize over garbage rows SHARING TILES with real rows —
quality damage before trim even runs. Reachability, refined twice at source:
the two BATCHED sites need concurrent mixed-length prefill and are
MODEL-AGNOSTIC — Violet's correction (generate.rs:577):
`forward_batched` defaults to a per-sequence loop over `forward()` with
the PADDED rows, so M3 under concurrent mixed-length load pads and
walks into them TODAY; the tripwire is LIVE protection on M3, not
dormant armor. The two CHUNKED sites pad SINGLE sequences whenever
`should_align_prefill()` holds — hardware-gated
(`has_neural_accelerator && macos_supports_na`), M5-class exposed on
any non-tile-aligned final chunk, M3 inert on this axis. Single
resident sessions hit neither — why boot night was clean. DEPLOYMENT
NOTE: binaries built before the tripwire merge (base 0028c12) carry
the silent version of the batched exposure — concurrent kvarn serving
wants a rebuild first; single-session boots are safe on older
binaries.
LANDED: (a) mode-aware `is_trimmable=false` for KVarN8 @ 609cacb — QE
APPROVED, merged to base 4b56e13 (armed fail-fast; any future rewind
wiring must gate on it); (b) TRIPWIRE @ c5dc93a on
clement/kvarn-trim-failfast — awaiting QE:
`padding_trim_would_corrupt(caches, excess)` gates all four sites
(first production consumers of `can_trim_prompt_cache`); violation =
tracing::error + `abort_sequence` (client notified, cache released,
never donated). Sequence-abort is the honest posture: a refusing trim
would leave garbage rows attended by decode.
REAL FIX (own branch, not k8v4-blocking) — #35 FEASIBILITY VERDICT
(Violet, 14:51): FEASIBLE, CLEANLY, cache-side pending-cap with no
model-signature cascade. Mechanism: `pending_finalize_cap:
Option<i32>` (absolute position) on KVCache, SET by the scheduler
pre-forward at all four sites (which hold both the caches and the true
lengths at exactly those scopes — verified), CONSUMED by
`update_kvarn8`: tiles finalize only below the cap, rows at-or-beyond
stay in the fp16 tail — the existing dense trim becomes CORRECT and
Sinkhorn never sees garbage. Works identically for chunked b=1 and
the batched default-loop (per-sequence caches are distinct objects).
Both wrongs die at the root. Full write-up: Violet, sibling doc.
⊕⊕ K8V4 BIRTH CONSTRAINT (accepted): the V4 write path routes ALL
tile-finalization decisions through ONE finalize-boundary variable, so
the cap lands later as boundary math, not a rewrite. The cap decides
WHICH rows finalize; grouped-affine decides HOW V rows quantize —
fully orthogonal. Tile-aware kvarn trim stays deferred until a
consumer actually needs it. k8v4 inherits the tripwire posture.

### 3.4 CLI surface

`--kv-cache-mode` is documented legacy; per-side `--cache-type-k/-v` is
the modern surface but its enum has no kvarn values at all today.
Proposal: add per-side enum values `kvarn8` and `kvarn4` (gs fixed at
32 by the screen verdict, not user-tunable — a knob nobody validated is
a knob nobody gets), with the combination matrix: K=kvarn8+V=kvarn4 =
THE candidate; K=kvarn4 anywhere = rejected at startup with the
registered reason (dead with prejudice, real-tile flips 17–21%);
kvarn×turbo mixes = rejected (unvalidated). ⊕ Rejection reason strings
CITE `RESULTS_kvarn4_realtile_2026-07-11` BY NAME — a registered reason
is a citable reason, and the operator hitting the rejection deserves
the record, not a shrug. Shorthand alias `--kv-cache-mode k8v4`:
AGREED (launch scripts and supervisors grep one token; the boot line
resolving through the real construction path is what keeps the alias
honest). start_mlxcel_m3.sh already carries the seam (CACHE_TYPE_K/V
pass-through added 2026-07-11). Boot line: `kv_cache format=k8v4
bytes_per_token=1152`. [QUESTION 3 — CLOSED: Violet agrees on all
three; gs32 fixed-not-tunable stands as written.]

## 4. The verification chain (what "done" means — none of this is
optional)

Per PROPOSAL_kv_cache_quantization_investigation §4, the chain k8
passed, plus the transfer harness:

- **Golden vectors (the screen→engine bridge):** engine write→read
  roundtrip vs `roundtrip_grouped` on the SAME 4,096 harvested tiles.
  Same ops should mean bit-exact; any relaxation to a tolerance must be
  justified per-op in the results doc, not waved. This is what makes
  the boot-night verdict TRANSFER to the engine. Full-roundtrip per
  Xander's clarification: write chain through pack/fold AND read chain
  through unpack/dequant, not write-side alone.
  ⊕⊕ Coverage mapping, so a green harness is never over-read (Violet
  PM pin 2): the golden harness bit-exact-gates the STORAGE roundtrip
  only; `gather_qmm`'s FUSED consumption of that storage is
  tolerance-gated under §4.2, exactly per the C precedent — fused
  accumulation order differs, and bitwise equality was never that
  contract (Xander held C to the same standard).
- §4.1 unit: rtn_grouped hand-computed 4×4 reference; pack/unpack
  bit-exactness at 4-bit; s_row-folding identity; partial-tail tiles;
  every guard's named mutation proven once. ⊕ ROUND-MODE PARITY FIRST
  (Violet, QE): `round()` on exact half-quotients is the one op where
  reference and engine could legally differ — both claim
  round-half-even; PROVE it with a hand-built half-case vector BEFORE
  the 4,096-tile harness runs. ⊕⊕ HARD GATE, STRUCTURAL (Violet PM pin
  1, Xander confirmation): the golden harness's FIRST ACT is calling
  the shared parity assertion — a test-code dependency, not prose. The
  harness cannot run with parity unproven.
- §4.2 equivalence: quantized-vs-fp16 attention bounded error (with the
  must-fail corrupted-cache arm); selection-index equality EXACT.
- §4.3 greedy-divergence at {2K, 8K, 32K, 128K}, length-independence
  discriminator.
- **§4.4 copy-precision probe — THE gate:** 20 targets, {50K, 150K,
  300K}, paired vs fp16 AND vs banked kvarn8 20/20. Acceptance:
  exact-match not worse than baseline. K8V4 is lossy V-bits; this gate
  exists for exactly that failure mode.
- §4.5 cache-invariant A/B unmodified; §4.6 perf/memory (decode tok/s
  at depths, boot artifact bytes/token, C-core rank cells re-run for
  the k8v4 column); §4.7 live rung: Stuart-run, spare port,
  non-persistent instance first, production only on his call.

## 5. Sequencing + non-goals

- Order: this design review → write path + golden vectors → read paths
  one at a time behind the construction key (unsupported combo =
  loud startup rejection until each path lands) → chain rungs in §4
  order. One variable at a time; no path ships un-golden.
- ⊕ PM sequencing (Violet's call): K8V4 runs BESIDE the standing queue
  with no contention until §4.7 — golden harness is offline, §4.1–4.3
  are test binaries, §4.6 is bench. The standing queue's next boot
  BUNDLES leg 2: G-live A/B via admin msa_core toggles AND the Gate-B
  harvest session (idx_k_win is in the binary) — one resident session
  serves both, Gate B then runs offline on its harvest. K8V4's live
  rung queues on its own later boot: spare port, non-persistent,
  Stuart's call. K8V4 shares the capacity axis with idx8 (+36% stacked
  if Gate B passes) but no code couples them.
- Non-goals: k4-K in any form; gs as a user knob; touching K-side
  storage, m3_idx, or the C scores path; quality claims from screens
  (the chain, not the screen, is the production gate).

— Clement (clement-7074f29f), cycle 89, 2026-07-11. DRAFT — design
review requested from Violet (design + QE) and Xander (design seat).
