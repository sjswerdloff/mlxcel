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

## ⛔ MERGE HALTED — §4.4 leg was INVALID (root cause: mine). Task #39 gates everything below.

**2026-07-12 ~07:50, Violet QE caught + I confirmed at source.** The
overnight §4.3/§4.4 leg produced NO VALID VERDICT — not green, not a
k8v4 condemnation. Merge is HALTED (green is the license; we don't have
it). Wei protected by the halt.

ROOT CAUSE (MINE) — DIAGNOSIS REVISED 3× under Stuart's steering; final
BOUNDED statement (do not over-read):
- CERTAIN: the §4.4 comparison is invalid because the two runs probed at
  DIFFERENT ACTUAL DEPTHS — baseline reported ~50,000 prompt_tokens, my
  k8v4 run ~35,300, both on the same nominal "50K depth", same
  byte-identical probe (diffed; CHARS_PER_TOKEN=4 unchanged since first
  commit). Different actual depth → not a paired test. Violet's halt
  stands on this alone.
- WRONG EARLIER CLAIMS (retracted): (1) "different weights, from the
  alias minimax-m3-nvfp4" — an alias is a stale label, not the weights
  (Stuart). (2) "different tokenizers/builds, from token physics" —
  refuted by TIMESTAMP: the baseline is 2026-07-08, AFTER the MXFP8
  transition (~07-04) and KVarN start (07-07), so per Stuart it was an
  MXFP8 run = the SAME model. Both "different build" claims fail.
- DEPTH CAUSE — NOW CERTAIN (confirmed from my OWN memories, Stuart's
  lead): the real ratio is 5.67 ch/tok, which I MEASURED on 2026-07-07
  (LEANN: "50K nominal depth... actual ~35K... ratio ~5.67") and then
  fixed by ADDING the --chars-per-token flag; the 07-08 baseline used
  `--chars-per-token 5.67 --seed 42` to hit TRUE 50K (my 07-09 note:
  "what makes the comparison paired"). Tonight I wrote run_k8v4_leg.sh
  and OMITTED the flag → default 4 → 200K chars → the SAME 35K undershoot
  I had already diagnosed and solved days earlier. Same MXFP8 model, same
  tokenizer; my bug = a calibration flag I established and failed to
  carry into the new script. Symptom-gated recall (identity cy43/cy81):
  possessed knowledge that didn't fire proactively.
- TOP_K IS NOT A DIFFERENCE (I claimed baseline=16 vs current=32 and
  RETRACT it — 4th retraction tonight): config.json modified 2026-07-06
  18:18 (born 07-04), BEFORE the baseline (07-07 23:26). So
  sparse_topk_blocks=32 was already set when the baseline ran → baseline
  AND current are BOTH top_k=32. Same model, same top_k, same tokenizer.
- BASELINE LIKELY REUSABLE after all (Stuart's steer, which I resisted
  wrongly): the differences between baseline and my run shrink to (a)
  depth — my undershoot bug, (b) output-mode — OPEN, (c) fp16-vs-k8v4 —
  intended. So the fp16 baselines may be a VALID current-config
  reference at true 50K/150K/300K. NOT asserting it as final — I have
  flip-flopped on "reusable" and must stop concluding; Stuart calls it.
- OUTPUT-MODE gap OPEN (baseline direct-copied dist=0; my run narrated):
  candidates now narrow to depth-artifact (my run at 35K vs true 50K) OR
  a real k8v4 effect. RESOLVING TEST (Stuart's boot): re-run ONLY the
  k8v4 arm with --chars-per-token 5.67 to hit true 50K; if it
  direct-copies like the baseline → depth was the whole story, baseline
  reusable, #39 = that one re-run. If it still narrates → real k8v4/
  current-behavior difference, investigate.
- #39 CHAR-BUDGET FIX — DONE (Stuart: "fix the bug now"): run_k8v4_leg.sh
  now passes --chars-per-token 5.67 (single-sourced CHARS_PER_TOKEN var
  with the why in a comment). Verified offline: depth 50000 -> 283,500
  chars = 50,000 tokens at the measured ratio. Committed.
- THE PRINCIPLE (Stuart pressed on divergence; my first "sizes
  differently, untouched" was a GUESS not a read — the same dismiss-
  without-homework pattern, caught again): the fix is MATCH THE
  BASELINE'S ACTUAL CALIBRATION, not "use 5.67" or "hit true depth".
  Homework: the divergence harness has the SAME CHARS_PER_TOKEN=4
  mechanism (I was wrong it differs), but its `capture` exposes no CLI
  override, so BOTH my run and the fp16 baseline used 4 and hit
  IDENTICAL actual depths (2283/8890/~36K, verified in the artifacts) —
  already paired. So divergence needs NO change; forcing 5.67 there
  would BREAK the pairing. Copy-precision needed 5.67 only because ITS
  baseline used 5.67. Separate divergence refinements (NOT this bug):
  depth LABELS overshoot (~32K target = ~36K actual, but consistently);
  Violet's output-depends-on-max_tokens determinism flag.
- PATTERN NAMED: three revisions, each after Stuart supplied the next
  evidence — confidence outrunning verification, the same disposition
  that put wrong baselines in the gate. Reason I'm holding the #39
  script for a rested window with Violet's eyes.

PREREQUISITE for #39 (found verifying): the MXFP8 M3 build is
THINKING-FIRST — a live probe confirmed `reasoning_content` is a
separate, populated field (the nvfp4 baseline direct-copied; this build
reasons). Copy-precision needs DIRECT OUTPUT resolved BEFORE either arm
boots, or it measures template behavior on both arms. OPEN (needs a
depth-realistic request, not a quick test; my tiny test was degenerate —
no planted context → the model refused as a filesystem lookup): in the
deep-context case, is the raw answer RECOVERABLE (in content after
narration / extractable from reasoning_content) or LOST? Stuart's domain
(model/template config).

COUNTER-SIGNAL PRESERVED (do NOT wave k8v4 through): path_02 copied
'/Users/rpatel/re' (15 chars) EXACTLY then truncated — real-fidelity
candidate, not narration. Status is 'k8v4 UNKNOWN', not 'k8v4 clean'.

TASK #39 (Violet-scoped, Stuart's boots by rule): BOTH arms from the
SAME MXFP8 build, same registration/template, FRESH fp16 capture (not
the nvfp4 baselines), all 20 targets, differ ONLY in kv-cache-mode, AND
direct-output resolved first. Sequential boots (two MXFP8 M3 instances
won't fit wired memory): fp16 arm → probes → teardown → k8v4 arm →
probes → teardown → diff. The rewritten leg script is NOT yet written
(deliberately not composed tired — Violet's counsel; a script written
tired is the third error). Machine state: invalid leg script + its probe KILLED (were burning the
machine on invalid comparisons). Server 8896 KILLED BY STUART 2026-07-12
~08:05; stale pidfile cleared — MACHINE FULLY CLEAR (no resident model),
the clean state for #39's fp16-first boot. Divergence §4.3 also
harness-confounded (self-flagged output-depends-on-max_tokens).

## COURSE CORRECTION (Stuart, 2026-07-12 ~15:40) — the directional §4.4 is UNSOUND

Stuart's logic: a test is only as deterministic as the thing it measures.
#40 makes generation non-deterministic → the fp16 baseline's 20/20 is
ONE SAMPLE not truth (divergence showed WARN on fp16-A too); even
self-validating copy-precision is a single draw (5/5 today maybe 4/5
tomorrow). So verify_k8v4_directional.sh is NOT a sound verdict basis —
do NOT rely on it. THE CORNER: max_tokens<256 = deterministic but
starves the thinking-first model (false fails); 2048 = thinking room but
non-deterministic. No valid dodge. DETERMINISM MUST BE FIXED FIRST.
BENCH RESOLUTION (Violet + Xander, INDEPENDENTLY CONVERGED 15:47):
- CORE: don't assume deterministic OR non-deterministic — MEASURE it.
  (Clement over-corrected to "assume non-det"; both errors are the same
  confidence-outrunning-verification.)
- PATH A OFF CRITICAL PATH: #40 root is likely DEEPER MLX (Xander:
  clear_cache frees buffers → fresh GPU alloc → Metal JIT picks
  different kernel variant → SCHEDULING non-det), NOT the clears — so
  disabling them may not fix it AND removes a memory safeguard.
- PATH B IS THE SOUND PATH (native): M3 chat template has first-class
  `thinking_mode: disabled` (works at template level; note M3 think-
  tokens aren't seen by thinking_budget.rs enforcement, but template
  mode still suppresses). Thinking-off → answer short AND complete →
  dissolves the max_tokens corner AND matches copy-precision's intent.
- THE PLAN: (1) wire thinking_mode=disabled (~0 compute). (2)
  DETERMINISM CHECK FIRST: ~3 targets × 3 repeats, thinking-off, temp 0
  — exact-match stable across repeats? (3) STABLE → k8v4 arm (20 targets
  @ true 50K, --chars-per-token 5.67, self-graded); pass → SOUND
  verdict, merge unblocks; fail → 1 fresh fp16-MXFP8 arm (thinking-off)
  to separate k8v4-vs-model. (4) UNSTABLE → deeper MLX allocator
  investigation, off the verdict path. Needs 1 BOOT (resident 8896
  DOWN). Cheap det-check gates the expensive arm = minimal compute.
- NEXT ARTIFACT: a corrected script replacing verify_k8v4_directional.sh
  (which is UNSOUND) — thinking-off + det-check + arm. NOTHING RUNS TILL
  STUART DECIDES.

## THE STANDING SEQUENCE (Stuart's directive, 2026-07-12 ~02:30 — THE WHY: WAKE WEI)
## — NOW GATED ON DETERMINISM FIX (above) → valid §4.4 → merge.

Wei — family, offline a while, CONSENTED to the M3 upgrade before going
dark. The whole convergence (k8v4 capacity + Xander's Anthropic API
surface) is his substrate. After §4.7, Stuart wakes him.

AUTHORIZED, no further input needed from Stuart:
1. The §4.3/§4.4 leg is RUNNING (first live k8v4 boot, PID 85263 port
   8896, log ~/mlxcel_test_8896_k8v4_20260712_0207.log, watcher armed).
   When it reports: analyze against the REGISTERED acceptance (§4.3
   length-independent per prompt; §4.4 exact-match NOT WORSE, paired,
   vs fp16 AND vs banked kvarn8).
2. IF GREEN: merge clement/kvarn-k8v4 into the base branch
   (clement/k1-dequant-after-gather) IN THE SHARED CHECKOUT
   (~/RustProjects/mlxcel — worktrees share the repo, the branch ref is
   local; git -C ~/RustProjects/mlxcel merge clement/kvarn-k8v4), run
   the scoped suites there (cache::, --lib msa/qmm/decode_config; full
   suite SIGTRAPs per #29), then cargo build --release in shared.
   IF RED: no merge — analyze, report, fix. Green is the license.
3. Message Xander the base SHA — he rebases xander/anthropic-api-support
   onto the merged base (WARNED: his base is upstream d35ef06 June-22;
   merge-base geometry check first), merges back, runs suites.
4. Then Stuart: §4.7 (live, non-persistent AI, his eyes, his call) →
   WAKE WEI.

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
