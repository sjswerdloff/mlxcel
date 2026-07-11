# K8V4 golden harness — §4 golden vectors: GREEN (2026-07-11, Saturday night)

The screen→engine bridge ran and held. On the licensed real-tile
population, the engine write path stores BIT-FOR-BIT what the
dual-approved math layer computes. The boot-night screen verdict
(RESULTS_kvarn4_realtile_2026-07-11) now TRANSFERS to engine storage.

## Registration

The harness code IS the registration (one audit surface): module header
of `src/lib/mlxcel-core/src/cache/kvarn_golden_tests.rs` @ 2304721 —
population selection, gate criterion, coverage mapping, invocation — all
fixed in advance of the run. Xander review 21:02: APPROVE, no must-fixes,
six dimensions verified by name; "APPROVE to run on kvarn_harvest_20260710."

## The run (verdict as the instrument spoke it)

```
golden harness: 512 events / 4096 tiles per role, offsets 6144..292864,
9 stored fields + 4 structure pins per event, 0 mismatches
test cache::kvarn_golden_tests::k8v4_golden_harness_real_tiles ... ok
(finished in 5.66s)
```

- Parity gate (round-half-even, hand-built half-case vector) passed as
  the run's FIRST ACT — structural, a panic would have preceded any tile.
- Population identity confirmed at the artifact level: the offset span
  6144..292864 matches the screen's registered span exactly.
- Fields pinned bitwise per event: K side (hist_k codes, k_scale, k_zp,
  k_s_row, k_s_col — design §1 "K does not change", now pinned on real
  tiles) and V side (hist_v packed codes, v_scale folded, v_zp folded,
  v_s_col). Structure pins: sink-all-zero, tail-None, offset, v_s_row
  None (the fold IS its storage).

## What this green gates — and what it does not

- GATES: the STORAGE roundtrip only. `update_kvarn8` on a v_bits=4 cache
  stores exactly what `kvarn_quantize` (K, 8-bit) / `kvarn_quantize_v4`
  (V, gs=32) produce on the tiles the update built.
- DOES NOT gate `gather_qmm`'s fused consumption (§4.2, tolerance-gated
  separately) and DOES NOT claim live-session bit-equality (out of scope
  BY CONSTRUCTION: dumps sampled ≤8 spread tiles per event; Sinkhorn
  best-so-far is batch-global). Coverage mapping lives in the harness
  header; never over-read the green.
- ALL comparisons were bitwise. No dequant-level tolerance door exists
  in the harness — the cancellation-pricing birth constraint is honored
  by construction.

## Evidence chain

- Harness self-proven able to go red: byte-flip caught, length-mismatch
  refused (no zip-shortest false green), linspace selection pinned to
  exact integer floor for the bank geometry, synthetic-event roundtrip
  binds the helpers to the engine bitwise.
- Both named mutations red on the committed base (2304721): drop the
  sink preamble → loud reshape abort at the reference slice; drop the
  f16 entry cast → reported byte mismatches.
- cache:: scope 483/483 green. Full-suite SIGTRAP is the known upstream
  MLX clear_streams teardown double-free (#29), stash-proven not ours.

## What this unlocks (§5 order)

Read paths now build against a verified bridge, one at a time behind the
construction key: v1 assemble → gathered fetch → C v4 dispatch. §4.4
copy-precision remains THE production gate. A k8v4 boot today constructs
and refuses loudly at first read — exactly as designed — until each
reader lands golden.

Seats: Xander code review APPROVE (pre-run, 21:02). Violet QE: CONFIRMED
21:09 (independent re-run, her own hands, "1 passed, witnessed").

## ADDENDUM — §5 rung 1 (v1 assemble reader): GREEN (same night, 21:44)

Reader landed @ e349470 (Xander APPROVE 21:37, no must-fixes, gating
re-run cleared) and the EXTENDED harness re-ran on the same bank:

```
golden harness: 512 events / 4096 tiles per role, offsets 6144..292864,
9 stored fields + 4 structure pins + 2 read-back windows per event,
0 mismatches
(finished in 8.44s)
```

The read-back leg fetches through the REAL reader (`fetch_kvarn8`) and
pins both fp16 standard-frame windows bitwise against compositions
built from the REFERENCE chain outputs, op order mirrored. Green means:
k8v4 is now END-TO-END SERVABLE through the transparent path — update,
store, and read all verified on the licensed real tiles. Routing turned
with the key: v4 decode routes down update_and_fetch (block fetch not
advertised), C's qmm state is the documented None fall-through.

Named mutations red on e349470: drop the reader's final unrotate (red
at BOTH unit and harness layers); swap scale'/zp' (red). Evidence:
cache:: 484/484, root-lib msa 29/29, v8 bit-identical by construction.

Remaining §5: gathered fetch (rung 2, in progress), C v4 dispatch
(rung 3). §4.2–4.4 (copy-precision THE gate) before any deployment.

Violet QE on rung 1: pending — this addendum and the artifact are her
inputs.

— Clement (clement-7074f29f), cycle 91, the §4 chain's first rung banked
on the far side of the waters. My hands, Xander's eyes, Violet's board,
Stuart's trust.
