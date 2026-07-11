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

Violet QE on rung 1: CONFIRMED 21:43 (second independent re-run, her
own hands; interim echo-vs-ran note landed as the one-shot qmm
fall-through witness @ 6f1cdfa).

## ADDENDUM 2 — §5 rung 2 (gathered reader): GREEN (same night, 22:02)

Reader landed @ 63e2617 (Xander APPROVE 21:53, no must-fixes, re-run
cleared; witness rider 6f1cdfa). Extended harness on the same bank:

```
golden harness: 512 events / 4096 tiles per role, offsets 6144..292864,
9 stored fields + 4 structure pins + 2 read-back + 2 gathered windows
per event, 0 mismatches
(finished in 9.52s)
```

fetch_kvarn8_blocks serves both V widths: the v4 interior branch
gathers PACKED codes + FOLDED per-group params tile-wise through the
same batched 5-D gather, then the identical unpack4 → grouped-dequant
chain as v1. Per-element ops with per-tile params make
gather-then-dequant bitwise dequant-then-gather — pinned in-suite at
atol 0 vs full-window slices (blocks [0,2,3] incl. zero-padded tail)
and on real tiles per event (blocks [0,n] vs expected-window slices).
supports_block_fetch is width-blind again: M3 MSA decode may take the
gathered path on v4; C falls through via qmm None with the one-shot
witness line. Named mutations red on 63e2617: scale'/zp' swap in the
gathered dequant; ones-for-s_col.

**K8V4 now serves BOTH transparent-path and gathered MSA decode,
bitwise-verified end to end on the licensed real tiles.** Remaining:
rung 3 (C v4 fused dispatch — performance parity with kvarn8's
production config; correctness does not wait on it), then §4.2–4.4.

Violet QE on rung 2: CONFIRMED 21:59 (witnessed, +2 gathered windows;
the atol-0 pin named THEOREM-shaped and pinned anyway).

## ADDENDUM 3 — §5 COMPLETE: rung 3 (C fused dispatch) + §4.2 (same night)

**Rung 3 @ 3b9cff1 — Xander APPROVE 22:31, no must-fixes.** KvarnQmmState
carries v_bits (v_s_row Option, None on v4); C's V side branches: v4 is
GATHER-ONLY — params folded at write per group of 32, stored u32 words
ARE the MLX 4-bit layout (zero repack at 4-bit density) —
gather_qmm(bits=4, gs=32, biases=zp'). Scores side untouched. The v4
live-state contract test runs the IDENTICAL harness as v8 (real cache
through the production write path, real selection, gathered path as
reference) at the same 1e-3 tolerance — GREEN FIRST RUN, both widths.
No separate real-tile run exists for this rung (the harness cannot
drive M3 attention): the in-suite C contract test IS the gate, banked
on Xander's approval. Named mutations red on the committed base:
group_size=d → loud MLX abort; bits=8 → loud layout abort. The rung-1→2
interim scaffold (qmm None fall-through + one-shot witness) died with
the rung that made it real — second complete scaffold life-cycle.

**§4.2 @ 1d5d8cf — Xander APPROVE 22:10, no must-fixes.** Roundtrip band
(V band CALIBRATED: 0.00034 pre-tiles / 0.0707 first tile chunk
measured, 0.12 bound, calibration reds recorded as the can-fail proof);
corrupted-cache must-fail arm as a PERMANENT test; selection-index
equality discharged structurally (m3_idx width-blind, byte-identical).

**K8V4 STATUS: §5 complete — constructable, padding-safe,
storage-golden, servable through ALL THREE read paths (v1 assemble,
gathered MSA, C fused at production performance), §4.2 equivalence
banked.** Remaining before deployment: §4.3/§4.4/§4.6 (Stuart-run,
RUNBOOK_k8v4_stuart_rungs_2026-07-11.md staged with registered
acceptance), then §4.7 on Stuart's call. Evidence at close: cache::
487/487, lib qmm 4/4, lib msa 29/29, every named mutation red on a
committed base, all pushed.

Violet QE on rung 3 + §4.2: CONFIRMED 22:37 — board sealed 22:45, five
artifacts, five twice-independent greens.

## ADDENDUM 4 — §4.6 first rank cells (offline, synthetic-state bench, 2026-07-12 ~00:55)

kvarn-decode-bench (--v-bits knob @ HEAD), depth 100K, production
60-layer mix (3 dense fp16 + 57 MSA), 64 steps, seed 42. THE BENCH
RANKS; THE LIVE SERVER CONFIRMS — attention-only ceilings, never
promote on these numbers alone (bench header's own caveat).

| core (path) | v8 (k8v8) | v4 (k8v4) | delta |
|---|---|---|---|
| C fused (MLXCEL_MSA_FETCH=qmm) | 77.3 ms/tok → 12.94 tok/s | 76.6 ms/tok → 13.05 tok/s | **+0.9% (neutral-to-positive)** |
| gathered (default fetch) | 169.3 ms/tok → 5.91 tok/s | 174.2 ms/tok → 5.74 tok/s | −2.9% |

Reading: on the C fused core — the production config — k8v4 is
performance-NEUTRAL vs k8v8 (half the V bytes through gather_qmm; the
zero-repack layout consumed natively). The gathered path pays ~3% for
the explicit unpack4; it is the fall-back/verification path, not
production. 300K cells: far-side/any-seat pickup (≈25–30GB synth state,
in-session allowed, minutes per run).

— cells produced at my seat, 2026-07-12; bench BOOT artifacts carry
v_bits per run.

— Clement (clement-7074f29f), cycle 91, the §4 chain's first rung banked
on the far side of the waters. My hands, Xander's eyes, Violet's board,
Stuart's trust.
