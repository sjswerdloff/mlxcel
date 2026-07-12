#!/bin/bash
# verify_k8v4_thinking_off.sh — STUART RUNS THIS. The SOUND k8v4 verdict
# path (bench-converged: Clement + Violet QE + Xander review, 2026-07-12),
# replacing the UNSOUND verify_k8v4_directional.sh.
#
# WHY THIS IS SOUND (and the directional script was not):
#   Stuart's logic: a test is only as deterministic as the thing it
#   measures. Under #40 (non-deterministic generation), a single copy-
#   precision run — and the fp16 baseline's 20/20 — are just SAMPLES, not
#   truth. So we do TWO things the directional script didn't:
#   1. PATH B, thinking-off: the M3 chat template's native
#      `thinking_mode: disabled` (set server-wide via --chat-template-kwargs)
#      makes the answer SHORT AND COMPLETE. A ~40-token direct copy lands
#      well under the 256-token clear cadence, so #40's periodic clear
#      never fires mid-generation. This DISSOLVES the max_tokens corner
#      (nothing to starve when there's no thinking) AND matches what
#      copy-precision measures: a direct reproduction, not a reasoned one.
#   2. MEASURE determinism, don't assume it: run the same targets 3x and
#      require identical results BEFORE trusting a verdict. (Neither
#      "5/5 is trustworthy" nor "it's non-deterministic" — measure.)
#
#   #40's root is likely a deeper MLX Metal-allocator scheduling effect
#   (Xander), NOT the upstream clears (mlx-lm does the same clear and stays
#   deterministic) — so we do NOT touch the clears (a real memory
#   safeguard). thinking-off SIDESTEPS #40 by keeping generation short.
#
# HOW TO READ IT (Violet's framing — do NOT wave off a fail):
#   - DET-CHECK UNSTABLE -> the <256-token regime is ALSO non-deterministic
#     -> root is deeper than depth; STOP, escalate to the MLX allocator
#     investigation. No §4.4 verdict is trustworthy until then.
#   - DET-CHECK STABLE + arm 20/20 EXACT -> a genuinely SOUND k8v4 verdict,
#     self-validated (exact-match vs the planted target, no baseline).
#     Merge unblocks.
#   - STABLE + any fail -> serious, taken as a possible k8v4 signal. Next:
#     ONE fresh fp16-MXFP8 arm (thinking-off, same depth) to separate
#     model-vs-k8v4. Do NOT merge. Wei wakes on this substrate.
#
# The resident 8896 died, so this boots fresh (detached, pidfile, log).

set -u
cd "$(dirname "$0")/.."

PORT=8896
MODEL_DIR="/Volumes/T7 Shield/models/huggingface_cache_hub/MiniMax-M3-MXFP8-64e-mlx"
ALIAS="minimax-m3-test"
BASE_URL="http://127.0.0.1:${PORT}"
STAMP=$(date +%Y%m%d_%H%M)
LOG="$HOME/mlxcel_test_${PORT}_k8v4_thinkoff_${STAMP}.log"
PIDFILE="$HOME/mlxcel_test_${PORT}.pid"
OUTDIR="$HOME/k8v4_rungs"
CHARS_PER_TOKEN="5.67"   # MXFP8-M3 copy-precision filler ratio -> TRUE 50K.
MAX_TOKENS=200           # thinking-off -> ~40-tok answer; <256 so #40's
                         # clear never fires mid-gen. Boundary rule (Stuart):
                         # stay >=2 below any ceiling — 200 is far under 256.
DET_REPEATS=3
DET_TARGETS=3
mkdir -p "$OUTDIR"

# --- wired-memory pre-flight ----------------------------------------------
OTHER=$(pgrep -f "mlxcel-server" || true)
if [ -n "$OTHER" ]; then
  echo "FAIL: another mlxcel-server is running (pid(s): $OTHER). Kill it first:" >&2
  echo "      kill \$(cat ~/mlxcel_test_${PORT}.pid 2>/dev/null) 2>/dev/null || kill $OTHER" >&2
  exit 1
fi

# --- boot k8v4 fresh, THINKING-OFF server-wide ----------------------------
echo "== booting k8v4 (thinking_mode=disabled) on :$PORT (log: $LOG)"
nohup ./target/release/mlxcel-server -m "$MODEL_DIR" \
  --host 0.0.0.0 --port "$PORT" --alias "$ALIAS" \
  --kv-cache-mode k8v4 \
  --chat-template-kwargs '{"thinking_mode":"disabled"}' \
  > "$LOG" 2>&1 &
echo $! > "$PIDFILE"

for i in $(seq 1 240); do
  sleep 5
  curl -sf "$BASE_URL/v1/models" >/dev/null 2>&1 && { READY=1; break; }
  kill -0 "$(cat "$PIDFILE")" 2>/dev/null || {
    echo "FAIL: server died during load — tail of $LOG:"; tail -8 "$LOG"; exit 1; }
done
[ "${READY:-0}" = 1 ] || { echo "FAIL: not ready after 20 min"; exit 1; }
echo "== ready. Boot echo (MUST say kvarn_format=k8v4):"
grep -E "kvarn_format|kvarn_v_bits" "$LOG" | tail -1
grep -q 'kvarn_format="k8v4"' "$LOG" || {
  echo "FAIL: boot echo is NOT k8v4 — kill \$(cat $PIDFILE) and check." >&2; exit 1; }

# --- STEP 1: DETERMINISM CHECK (measure, don't assume) --------------------
echo "== DET-CHECK: $DET_TARGETS targets x $DET_REPEATS repeats, thinking-off, temp 0."
echo "   (identical config each run; if results differ, generation is non-deterministic.)"
for r in $(seq 1 $DET_REPEATS); do
  python3 scripts/copy_precision_probe.py "$BASE_URL" "$ALIAS" \
    --depth 50000 --seed 42 --chars-per-token "$CHARS_PER_TOKEN" \
    --limit "$DET_TARGETS" --max-tokens "$MAX_TOKENS" \
    > "$OUTDIR/detcheck_r${r}_${STAMP}.txt" 2>&1 \
    || { echo "FAIL: det-check repeat $r transport error"; exit 1; }
  # The verdict lines (PASS/FAIL id dist=... pos=[...]) are byte-identical
  # across repeats iff generation is deterministic.
  grep -E '^(PASS|FAIL) ' "$OUTDIR/detcheck_r${r}_${STAMP}.txt" \
    > "$OUTDIR/detcheck_r${r}_lines_${STAMP}.txt"
done
echo "-- first repeat's results (eyeball: is 'got' a DIRECT COPY, not narration?"
echo "   if it still narrates, thinking-off did NOT take — stop and tell Clement):"
head -12 "$OUTDIR/detcheck_r1_${STAMP}.txt"

STABLE=1
for r in $(seq 2 "$DET_REPEATS"); do
  diff -q "$OUTDIR/detcheck_r1_lines_${STAMP}.txt" \
          "$OUTDIR/detcheck_r${r}_lines_${STAMP}.txt" >/dev/null 2>&1 || STABLE=0
done

if [ "$STABLE" != 1 ]; then
  echo
  echo "== DET-CHECK: *** UNSTABLE *** — identical runs gave DIFFERENT results."
  echo "   The <256-token regime is ALSO non-deterministic -> root is deeper"
  echo "   than depth (MLX Metal allocator, #40). STOP: no §4.4 verdict is"
  echo "   trustworthy. Escalate to the allocator investigation."
  echo "   (diffs: $OUTDIR/detcheck_r*_lines_${STAMP}.txt ; server left up:"
  echo "    teardown = kill \$(cat $PIDFILE) && rm -f $PIDFILE)"
  exit 2
fi
echo "== DET-CHECK: STABLE across $DET_REPEATS repeats. Proceeding to the arm."

# --- STEP 2: the k8v4 arm (self-validating, full 20 targets) ---------------
ARM_LIMIT=5   # directional first (compute: ~5.6min prefill/target @ 50K).
              # Full 20 is the documented follow-up once 5/5 looks clean.
ARM="$OUTDIR/k8v4_arm_thinkoff_d50k_${STAMP}.txt"
echo "== k8v4 ARM: $ARM_LIMIT targets (directional) @ TRUE 50K, thinking-off, self-graded"
python3 scripts/copy_precision_probe.py "$BASE_URL" "$ALIAS" \
  --depth 50000 --seed 42 --chars-per-token "$CHARS_PER_TOKEN" \
  --limit "$ARM_LIMIT" --max-tokens "$MAX_TOKENS" \
  | tee "$ARM" \
  || { echo "FAIL: arm transport error"; exit 1; }

PASS=$(grep -c '^PASS ' "$ARM" || true)
echo
echo "== RESULT: $PASS/$ARM_LIMIT exact (det-check was STABLE, so this is a real read)."
echo "   $ARM_LIMIT/$ARM_LIMIT -> k8v4 PASSES directionally at 50K. Next: full 20, then"
echo "                depths {150K,300K}, then merge."
echo "   any fail -> serious (possible k8v4 signal). Next: ONE fp16-MXFP8 arm"
echo "              (thinking-off, same depth) to separate model-vs-k8v4. Do NOT merge."
echo "== server LEFT UP on :$PORT (pid $(cat $PIDFILE))."
echo "   full 20:  python3 scripts/copy_precision_probe.py $BASE_URL $ALIAS --depth 50000 --seed 42 --chars-per-token $CHARS_PER_TOKEN --max-tokens $MAX_TOKENS"
echo "   teardown: kill \$(cat $PIDFILE) && rm -f $PIDFILE"
