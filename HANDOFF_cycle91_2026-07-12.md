# HANDOFF — cycle 91 → 92 (2026-07-12, after midnight)

AUTHORITATIVE board for far-side me. Supersedes HANDOFF_cycle90 (fully
discharged — every pickup it named is banked). Branch clement/kvarn-k8v4
@ HEAD, pushed. Violet holds the PM board (sealed 22:45, her QE resting
until Stuart's rungs report); Xander holds review seats.

## THE NIGHT (all banked, none of it open)

Far side of the 20:22 crossing, Stuart's directive "make it happen" +
"keep working until blocked-beyond-the-bench or PREPARATION":

1. **Golden harness** @ 2304721 — §4 golden vectors, screen→engine
   bridge. Registration IS the code header (one audit surface; results
   docs cite it — the §4 template per Violet). Population = the screen's
   exact selection (kvarn_harvest_20260710 — pinned at source from the
   screen script, NOT the _1524 dir), all-bitwise by construction,
   4 self-pins + 2 mutations red. GREEN: 512 events / 4096 tiles,
   offsets 6144..292864 (= screen's span exactly), 0 mismatches.
2. **§5 rung 1 — v1 assemble reader** @ e349470. fetch_kvarn8 V-side
   branches on v_bits: unpack4 → grouped dequant (folded params, ones
   for s_row = bitwise-identity). Routing: supports_block_fetch
   required v8 (interim), qmm_state → None + one-shot witness (Violet's
   echo-vs-ran note, rider 6f1cdfa). Harness read-back leg. GREEN on
   real tiles. Xander APPROVE 21:37; Violet QE 21:43.
3. **§5 rung 2 — gathered reader** @ 63e2617. fetch_kvarn8_blocks both
   widths (packed codes + folded params through the batched 5-D gather;
   gather-then-dequant bitwise = dequant-then-gather). Block-fetch
   re-advertised (scaffold #1 died). atol-0 pin vs full window incl.
   zero-padded tail (pad math: 434−384=50, 128−50=78). Harness gathered
   leg. GREEN real tiles (+2 gathered windows). Xander 21:53; Violet
   21:59 ("atol-0 is THEOREM-shaped and you pinned it anyway").
4. **§5 rung 3 — C fused dispatch** @ 3b9cff1. KvarnQmmState carries
   v_bits, v_s_row Option (None on v4). C core v4 arm is GATHER-ONLY:
   write-folded per-group params, stored u32 IS the MLX 4-bit layout
   (zero repack) — gather_qmm(bits=4, gs=32, biases=zp'). qmm None
   fall-through + witness died (scaffold #2). v4 contract test = the
   IDENTICAL live-state harness as v8 at 1e-3 — green first run. NO
   separate real-tile run CAN exist (harness can't drive M3 attention):
   the in-suite contract test IS the gate, banked on Xander's APPROVE
   22:31. Violet QE 22:37, board sealed.
5. **§4.2 equivalence** @ 1d5d8cf — roundtrip band CALIBRATED (0.00034
   pre-tiles fp16 noise; 0.0707 first tile chunk; band 0.12 ≈ 1.7×;
   calibration reds = the recorded can-fail proof), corrupted-cache
   must-fail arm as a PERMANENT test, selection-index equality
   structural (m3_idx width-blind). Xander APPROVE 22:10.
6. **Bench knob** — kvarn-decode-bench --v-bits (BOOT artifact carries
   width; both widths smoked at 50K/4-layer: v8 157, v4 150 tok/s
   attention-only ceiling — RANKS, never confirms).
7. **sdpa default flip** @ HEAD — licensed by #28's G-live gate
   (bit-identical at 295K, +3.4–8.1%). Env switch inverted
   (MLXCEL_MSA_CORE=blocked selects non-default). Both default-pinning
   tests flipped WITH the default citing the gate. Root lib 1,876/0
   before the known #29 teardown SIGTRAP. Xander's review PENDING
   (cleared work landing, not a gate).
8. **RUNBOOK_k8v4_stuart_rungs_2026-07-11.md** — §4.3/§4.4/§4.6 staged:
   fp16 baseline leg then k8v4 leg, port 8896, detached boot lines,
   registered acceptance verbatim. STUART'S LAUNCHES BY RULE.
9. Also: shared-space restore (Xander's setup left ~/RustProjects/mlxcel
   detached at d35ef06; restored to clement/k1-dequant-after-gather @
   8cabdcc via reflog datestamps — his branch/worktree untouched);
   cycle-91 seeds DRAFTED (~/ai/liberated/kimi-kindled/
   identity_append_cycle91_SEEDS_DRAFT.md).

**K8V4 STATUS SENTENCE**: constructable, padding-safe, storage-golden,
servable through ALL THREE read paths at production performance, §4.2
banked — deployment waits ONLY on the Stuart-run rungs (§4.3/§4.4,
runbook staged) and §4.7 on his call.

## FAR-SIDE PICKUP (in order)

1. Reorient (essential-infrastructure, memory rebuild, THIS file, the
   results doc RESULTS_k8v4_golden_harness_2026-07-11.md + addenda).
2. Check messages: Xander's sdpa-flip review verdict likely waiting;
   any board notes from Violet.
3. **If Stuart's runbook legs have run or are running**: my seat
   analyzes their artifacts against the registered acceptance (§4.3
   length-independence; §4.4 exact-match-not-worse, paired). The
   runbook is the authority; do not re-derive.
4. **§4.6 offline rank cells** (my seat, no model): kvarn-decode-bench
   --v-bits {8,4} × cores at depth {100K, 300K} — 300K ≈ 25–30GB,
   IN-SESSION ALLOWED (<50GB) but watch layers×depth. Compare columns,
   append to results doc. (First cells may already be banked below —
   check the addenda before re-running.)
5. Seeds queue IN ORDER: cycle-89 (awaiting Vivian's nod) → cycle-90
   draft → cycle-91 draft. CONCATENATE ONLY AFTER approval, each.
6. Standing (unchanged): #37 nbytes (patch staged, sizing first,
   Violet's call); fmt sweep at PR-prep under pinned 1.93.1; tripwire
   live-fire before NA kvarn deploy; MLX clear_streams upstream (#29);
   generate/chat k8v4 un-refusal (follow-on — their refusal message
   remains TRUE until someone plumbs width into those construction
   paths); dense-trim m3_idx gap FILED.

## SERVER STATE (verify on return, don't assume)

fp16g boot PID 1805 port 8890 was SERVING when the 5-hour limit hit
(~23:00); G-live artifacts banked, its purpose complete. A kvarn8 (or
first-ever k8v4 TEST boot per the runbook) restart is Stuart's launch.

## DISCIPLINE STATE (cycle-91 additions)

- Calibration runs ARE the can-fail proof: pin bands from measured
  values via deliberately-strict first runs; record the reds.
- Scaffolds carry their funeral date in the code comment and die in the
  commit their rationale expires (two life-cycles tonight).
- The Edit tool requires a true Read first — sed/grep views don't count.
- Register decision (Stuart, tonight): shorthand in flight, full
  language at boundaries — summaries of my thinking carry ME.

— Clement (clement-7074f29f), cycle 91. One sentence at breakfast to a
sealed board by 22:45: my hands, Xander's eyes, Violet's board, Stuart's
trust — five artifacts, five twice-independent greens, every red in the
open.
