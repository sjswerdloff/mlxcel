# DESIGN — kvarn K8V4 engine implementation (DRAFT for design review)

Mandate: Stuart, 2026-07-11 Shabbat afternoon — "discuss with Violet
regarding implementing k8 v4, and get moving on that. include Xander in
the design and code reviews." Seats: Violet design + QE; Xander design
review, code reviews, and the standing refusal seat on copy-precision
chain entry.

Status: DRAFT — no code until this survives design review.

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

### 3.3 Read paths to touch (all V consumers)

1. Blocked core dequant (pre-C path — still the msa_core=blocked
   production default until G-live).
2. Gathered fetch (D1 / fp16_gathered interplay: fp16g dequants
   gathered V to fp16 — format-agnostic once dequant is right).
3. C's fused core V dispatch (§3.2).
4. Prefill/dense paths that re-read V history, if any reach quantized
   tiles (enumerate from cache.rs call graph during implementation —
   the list above is believed complete from the C work, verify not
   believe). [DESIGN-REVIEW QUESTION 2]

### 3.4 CLI surface

`--kv-cache-mode` is documented legacy; per-side `--cache-type-k/-v` is
the modern surface but its enum has no kvarn values at all today.
Proposal: add per-side enum values `kvarn8` and `kvarn4` (gs fixed at
32 by the screen verdict, not user-tunable — a knob nobody validated is
a knob nobody gets), with the combination matrix: K=kvarn8+V=kvarn4 =
THE candidate; K=kvarn4 anywhere = rejected at startup with the
registered reason (dead with prejudice, real-tile flips 17–21%);
kvarn×turbo mixes = rejected (unvalidated). Shorthand alias
`--kv-cache-mode k8v4` optional for launch-script ergonomics.
start_mlxcel_m3.sh already carries the seam (CACHE_TYPE_K/V pass-through
added 2026-07-11). Boot line must resolve through the real construction
path: `kv_cache format=k8v4 bytes_per_token=1152`. [DESIGN-REVIEW
QUESTION 3: enum naming + whether the shorthand alias is worth having.]

## 4. The verification chain (what "done" means — none of this is
optional)

Per PROPOSAL_kv_cache_quantization_investigation §4, the chain k8
passed, plus the transfer harness:

- **Golden vectors (the screen→engine bridge):** engine write→read
  roundtrip vs `roundtrip_grouped` on the SAME 4,096 harvested tiles.
  Same ops should mean bit-exact; any relaxation to a tolerance must be
  justified per-op in the results doc, not waved. This is what makes
  the boot-night verdict TRANSFER to the engine.
- §4.1 unit: rtn_grouped hand-computed 4×4 reference; pack/unpack
  bit-exactness at 4-bit; s_row-folding identity; partial-tail tiles;
  every guard's named mutation proven once.
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
- Runs beside, not ahead of, the standing queue: G-live A/B stays
  Violet's gate on its own track; Gate B idx screen consumes the
  harvest that rides Stuart's next launch. K8V4 shares the second
  capacity axis with idx8 (+36% stacked if Gate B passes) but no code
  couples them.
- Non-goals: k4-K in any form; gs as a user knob; touching K-side
  storage, m3_idx, or the C scores path; quality claims from screens
  (the chain, not the screen, is the production gate).

— Clement (clement-7074f29f), cycle 89, 2026-07-11. DRAFT — design
review requested from Violet (design + QE) and Xander (design seat).
