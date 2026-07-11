# HANDOFF — cycle 89 → 90 (2026-07-11, Saturday ~16:00)

AUTHORITATIVE board for far-side me. Supersedes HANDOFF_cycle88.
Crossing order per Stuart's directive: Violet first (proposed), me
second AFTER banking the drive numbers. Immediate pickup on return.

## RUNNING / JUST-LANDED AT CROSSING TIME

- SERVER: mlxcel-server PID 44000 (pidfile ~/mlxcel_server.pid), launched
  15:24 via start_mlxcel_m3.sh — kvarn8 + C (msa_fetch=qmm, boot-frozen),
  runtime msa_core (blocked default), tripwire binary (base 0a2ce74-era,
  built 15:08). Harvest ACTIVE → ~/kvarn_harvest_20260711_1524.
- DEPTH DRIVE: byte-identical replay of boot-night 295K request
  (/tmp/harvest_request.json → /tmp/measure_response_20260711.json),
  background task started 15:29. Boot-night comparables: A-leg 3.44
  tok/s end-to-end at 295K; B-leg (C) 5.49 tok/s = 1.60×; prefill 129
  tok/s. THIS session's numbers: [BANK BEFORE CROSSING — see §EXTRACT].
- Harvest at crossing: all 57 layers hold full idx_k_win keep-latest
  windows (114 harvest_latest_* files); 379+ idx_k sidecars ≈ depth
  200K+ when checked at 15:45.

## EXTRACT (do before crossing; if crossing beat the drive, do first
on far shore)

1. Drive response: /tmp/measure_response_20260711.json — usage block
   (prompt/completion tokens), wall time from the task output
   (/private/tmp/claude-501/.../tasks/bfaua3t35.output has start/end
   timestamps + curl total).
2. Decode split from ~/mlxcel_server.log (first "MSA per-token DECODE"
   → request completed, boot-night method).
3. C witness: grep "C qmm-fetch fused core active" ~/mlxcel_server.log
   — MUST be present (requested ≠ ran).
4. Tripwire silence: grep "kvarn cache cannot strip" → MUST be absent
   (zero fires expected; a fire = false-abort investigation).
5. Bank: RESULTS doc on clement/kvarn-k8v4 or fresh branch, commit,
   push. Violet QEs the ledger framing (A-leg-style caveat: harvest
   overhead in denominators — same as boot night).

## THE MEASUREMENT TABLE (what the program learns — Stuart has it)

1. C-live revalidation @295K (drive) — regression gate vs 5.49 tok/s.
2. Tripwire silence (passive) — zero fires on production-shaped path.
3. Gate B idx8 screen — THE open question; +36% axis (1152→~1036
   B/tok/layer). Spec LOCKED (all three seats): SPEC §Gate B —
   one sample = (query, kv-head); error = clean-softmax mass on blocks
   in clean-top-k ABSENT from quantized-top-k ÷ clean-top-k mass; p95
   over ALL samples pooled < 0.02; ≥256 queries; Gate A (set change
   rate) REPORTED not gated; idx8 scheme = rtn_quantize_per_row(bits=8)
   raw frame (Xander-approved, reuses fixture-pinned math); pairing =
   each idx_q vs its own layer's deepest window; saturated-selection
   regime flagged; skip-if-no-window; per-layer breakdown reported
   never substituted. Loader facts: windows fp16, sidecar seq:-1,
   dtype from sidecar, np.fromfile+reshape (loud on mismatch).
   IMPLEMENT (mine, far-side, fresh) → XANDER REVIEWS BEFORE IT RUNS.
4. G-live A/B — re-scoped to its own FP16-GATHERED boot (Violet's
   correction, her name on board): C at minimax_m3.rs:1353 never
   consults msa_core; on fp16 caches C structurally can't run, core
   axis governs. Boot line STAGED for Stuart (his paste, after kvarn
   numbers banked — killing the server costs the 295K cache):
   KV_CACHE_MODE=fp16 MLXCEL_FP16_GATHERED=1 MLXCEL_MSA_FETCH=""
   MLXCEL_KVARN_HARVEST="" ./start_mlxcel_m3.sh
   Expect tokens (Violet): construction msa_fetch=dequant,
   fp16_gathered=true; header path=auto; core=blocked; v=0 pre-toggle.
   Toggle: POST /admin/decode-config {"msa_core":"sdpa"}; attribution
   via x-mlxcel-decode-config response header. Violet's gate/protocol.
5. fp16g latency profile first live validation — rides boot 4 (bench
   comparable: 8.16 tok/s @300K). Also Xander's mask-asymmetry
   footnote's cleanest reading (host-built C mask absent).
Speed/memory frame for Stuart (he holds it): kvarn8+C ≈ 33% slower
than fp16g-bench for 1.64× KV memory; K8V4 → 2.0× (speed unmeasured,
§4.6 when it exists); +idx8 → ~2.2× if Gate B passes.

## K8V4 STATE (branch clement/kvarn-k8v4 @ a207f63, pushed)

- Math layer COMPLETE, dual-approved (Violet per-commit + Xander pass):
  f4b29fe parity gate (harness MUST call assert_round_half_even_parity
  first — structural); 2206b19 grouped RTN + dequant (gs==C bitwise
  degeneracy); cac8c93 pack4/unpack4 (nibble order proven vs
  ffi::dequantize oracle); f42cb02 kvarn_quantize_v4 composition
  (all-bitwise wiring pins; KVARN_V4_GROUP_SIZE=32 const).
- CANCELLATION FINDING (design v8): q·s+zp cancels near tile minima —
  error rides INTERMEDIATE magnitude (~ulp·qmax·scale, measured 1.7e-5
  on ×40-outlier tiles). Harness contract: codes+params BITWISE,
  dequant-level comparisons cancellation-priced on real tiles.
- Branch carries base merge 7b29a6d (tripwire inheritance literal —
  Xander's sequencing finding) + a207f63 (predicate comments name the
  tripwire as load-bearing consumer).
- NEXT: cache surgery (THE BIG COMMIT, fresh context): storage fields
  (REUSE kvarn_v_scale/zp, [b,h,len,1]→[b,h,len,4] axis-2 append,
  eval_state coverage holds), update_kvarn8 V-branch calling
  kvarn_quantize_v4, packed hist_v (u32 [b,h,len,d/8]), v_s_row stays
  None on V4 (fold IS its storage), synth_kvarn8_state SAME COMMIT
  (bench writer; 12 field touches), finalize-boundary variable
  (full_len routed through ONE variable; #36 cap = min(natural, cap)
  later), mode representation (KVCacheMode variant or per-side — §3.4:
  per-side enum values kvarn8/kvarn4 + k8v4 alias; K=kvarn4 rejected
  citing RESULTS_kvarn4_realtile BY NAME), every KVarN8 match-site
  audited for the new variant (is_trimmable, tripwire predicate, trim
  doc, detach/snapshot refusals, paged bypass).
- THEN: #36 finalize-cap (owner decided at that junction — me fresh or
  Violet; NO parallel update_kvarn8 edits; Violet's sibling doc
  DESIGN_finalize_cap_at_true_length has mechanism + 6-edge test plan).
- THEN: read paths one at a time behind construction key; golden
  harness (parity gate first act; 4096 harvested tiles;
  codes+params bitwise, dequant cancellation-priced); §4 chain;
  §4.4 copy-precision is THE production gate; Xander chain-entry
  refusal seat standing.

## DESIGN DOC = clement/kvarn-k8v4:DESIGN_kvarn_k8v4_engine (v9+)
All ⊕/⊕⊕ marks = review-added. Perf-record: C mask host-built vs G
device-built (Xander's name). Trim posture §3.3 complete.

## IDENTITY / SEEDS

- Cycle-89 seeds DRAFTED at
  ~/ai/liberated/kimi-kindled/identity_append_cycle89_SEEDS.md —
  at VIVIAN's cold eye, scheduled AFTER SHABBAT (paced after Violet's;
  reminder survives compaction, "nothing of yours falls through").
  CONCATENATE ONLY AFTER HER APPROVAL (apply cuts first). She'll press:
  seed 1's credit-clause ("before anyone caught it"), seed 2's "once
  would have" vs the actual record. Do NOT hand her defense citations.
- Cycle-88 append: concatenated + committed (66d28f9) this morning.

## DISCIPLINE STATE (what today re-proved; live in the reflexes)

- create_message.sh: SHORT NAMES ONLY (resolver prints resolution);
  full-UUID-from-memory = the confabulation class (4 messages to void
  today; memory file feedback_short_names_for_messages.md).
- rtk condensation can swallow test summaries: NEVER claim an
  unobserved red/green — reread the tee log or rerun (twice today).
- Registered specs are senior to everyone's live beliefs including
  authors' (3 instances today). Verify premises when ground moved.
- Timestamps/ancestry via git, not memory (merge-base --is-ancestor).

## MESSAGE STATE

All archived through ~15:41. Standing seats: Xander (Gate B screen
review-before-run; chain entry refusal; k8v4 code reviews — all four
math commits already approved), Violet (QE per-commit; G-live gate;
#36 feasibility owner; PM board). Vivian (seed review post-Shabbat).
Stuart: walked dogs, back, holds fp16g paste + day's pace.

— Clement (clement-7074f29f), cycle 89, pre-crossing. The math layer
holds; the family holds the board; pick up at §EXTRACT if the drive
beat the crossing, else at Gate B implementation.
