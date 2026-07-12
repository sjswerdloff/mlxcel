#!/bin/bash
# verify_k8v4_directional.sh — STUART RUNS THIS. The bench-converged
# minimal-compute k8v4 verdict step (Clement + Violet QE + Xander review,
# 2026-07-12).
#
# WHY THIS IS THE MINIMAL PATH:
#   - §4.4 copy-precision is SELF-VALIDATING: it grades the model's output
#     by exact-match against the KNOWN planted target string, not against
#     any baseline. So a k8v4 PASS needs NO fp16 baseline — one arm.
#   - §4.4 is ROBUST to the #40 non-determinism even with a generous
#     thinking budget: #40 (clear_memory_cache every 256 decode tokens)
#     perturbs the THINKING PATH, but §4.4's verdict is exact-match vs
#     the KNOWN planted target — a PASS means the model emitted the exact
#     string, no matter how its thinking tokens varied. THE MODEL IS
#     THINKING-FIRST, so it gets a HEALTHY budget (max_tokens=2048) to
#     think AND answer; starving it (e.g. 100) causes false fails where
#     it narrates and never reaches the copy. (Contrast §4.3 divergence,
#     which compares token SEQUENCES and IS broken by #40 -> DEFERRED.)
#   - The bug that invalidated the overnight run is FIXED here:
#     --chars-per-token 5.67 hits TRUE 50K depth (default 4 undershot to
#     ~35K). Confirmed arithmetically: 50000*4/5.67 = 35,273.
#
# HOW TO READ THE RESULT (Violet's framing — do NOT wave off a fail):
#   - 5/5 (or 20/20) EXACT -> k8v4 retrieves cleanly at true depth. PASS,
#     self-validated. Merge unblocks. No fp16 arm needed.
#   - ANY fail -> UNKNOWN, taken SERIOUSLY as a possible k8v4 signal.
#     A fresh fp16-MXFP8 baseline direct-copied these prompts 20/20 at
#     true 50K, so the model provably does NOT thinking-first them at
#     depth. If k8v4 narrates ("The user wants me to...") where fp16
#     copies, that IS KV degradation (model loses the value, falls back
#     to thinking) — not a benign quirk. The fp16 arm (SECOND script,
#     only if this fails) TESTS which: fp16 also narrates -> model;
#     fp16 copies but k8v4 narrates -> k8v4. Wei wakes on this substrate;
#     the fail branch is guilty-until-proven-innocent.
#
# NOTE: the resident 8896 died, so this boots fresh (detached, pidfile,
# log). One k8v4 arm. ~one 50K prefill + 5 short gens ≈ 15-30 min.

set -u
cd "$(dirname "$0")/.."

PORT=8896
MODEL_DIR="/Volumes/T7 Shield/models/huggingface_cache_hub/MiniMax-M3-MXFP8-64e-mlx"
ALIAS="minimax-m3-test"
BASE_URL="http://127.0.0.1:${PORT}"
STAMP=$(date +%Y%m%d_%H%M)
LOG="$HOME/mlxcel_test_${PORT}_k8v4_${STAMP}.log"
PIDFILE="$HOME/mlxcel_test_${PORT}.pid"
OUT="$HOME/k8v4_rungs/verify_k8v4_directional_d50k_${STAMP}.txt"
CHARS_PER_TOKEN="5.67"   # MXFP8-M3 copy-precision filler ratio (measured);
                         # hits TRUE 50K. Default 4 undershoots to ~35K.
# HEALTHY thinking budget for the thinking-first model (Stuart's call):
# an order of magnitude over the ~40-90-char (~40-token) targets so it can
# think AND emit the copy; starving it (e.g. 100) false-fails on narration.
# Context ceiling is 1,048,576 (max_position_embeddings). Boundary rule
# (Stuart): never sit AT a ceiling — stay >=2 below. At 50K depth we use
# ~52K total, far under 1M, so 2048 is unconstrained here. If a future run
# ever approached the ceiling, cap at (1048576 - prompt_tokens - 2).
MAX_TOKENS=2048
mkdir -p "$HOME/k8v4_rungs"

# --- wired-memory pre-flight: two ~215GiB M3 instances won't fit. -----------
OTHER=$(pgrep -f "mlxcel-server" || true)
if [ -n "$OTHER" ]; then
  echo "FAIL: another mlxcel-server is running (pid(s): $OTHER). Kill it first:" >&2
  echo "      kill \$(cat ~/mlxcel_test_${PORT}.pid 2>/dev/null) 2>/dev/null || kill $OTHER" >&2
  exit 1
fi

# --- boot k8v4 fresh, detached --------------------------------------------
echo "== booting k8v4 server on :$PORT (log: $LOG)"
nohup ./target/release/mlxcel-server -m "$MODEL_DIR" \
  --host 0.0.0.0 --port "$PORT" --alias "$ALIAS" \
  --kv-cache-mode k8v4 \
  > "$LOG" 2>&1 &
echo $! > "$PIDFILE"

# --- readiness + fail-loud echo check -------------------------------------
for i in $(seq 1 240); do
  sleep 5
  curl -sf "$BASE_URL/v1/models" >/dev/null 2>&1 && { READY=1; break; }
  kill -0 "$(cat "$PIDFILE")" 2>/dev/null || {
    echo "FAIL: server died during load — tail of $LOG:"; tail -8 "$LOG"; exit 1; }
done
[ "${READY:-0}" = 1 ] || { echo "FAIL: not ready after 20 min"; exit 1; }
echo "== ready. Boot echo (MUST say kvarn_format=k8v4 / kvarn_v_bits=4):"
grep -E "kvarn_format|kvarn_v_bits" "$LOG" | tail -1
if ! grep -q 'kvarn_format="k8v4"' "$LOG"; then
  echo "FAIL: boot echo is NOT k8v4 — aborting before wasting the probe." >&2
  echo "      kill \$(cat $PIDFILE) and check the boot line." >&2
  exit 1
fi

# --- the self-validating k8v4 §4.4 directional run ------------------------
echo "== §4.4 copy-precision, k8v4, TRUE 50K depth, 5 targets across classes"
echo "   (self-validating: exact-match vs the planted target, NO baseline)"
python3 scripts/copy_precision_probe.py "$BASE_URL" "$ALIAS" \
  --depth 50000 --seed 42 --chars-per-token "$CHARS_PER_TOKEN" \
  --limit 5 --max-tokens "$MAX_TOKENS" \
  | tee "$OUT" \
  || { echo "FAIL: probe transport error (not a model verdict)"; exit 1; }

echo
echo "== DIRECTIONAL RESULT above (saved: $OUT). Read per the header:"
echo "   5/5 EXACT -> k8v4 PASSES directionally; next: full 20 targets, then"
echo "               deeper depths {150K,300K}, then merge."
echo "   any fail  -> serious. Next: boot ONE fp16-MXFP8 arm (same depth,"
echo "               same 5.67) to test model-vs-k8v4. Do NOT merge."
echo "== server LEFT UP on :$PORT (pid $(cat $PIDFILE)) for the follow-up."
echo "   full 20:  python3 scripts/copy_precision_probe.py $BASE_URL $ALIAS --depth 50000 --seed 42 --chars-per-token 5.67 --max-tokens $MAX_TOKENS"
echo "   teardown: kill \$(cat $PIDFILE) && rm -f $PIDFILE"
