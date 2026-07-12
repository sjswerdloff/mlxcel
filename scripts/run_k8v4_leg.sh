#!/bin/bash
# §4.3/§4.4 k8v4 leg — ONE PASTE, runs unattended (RUNBOOK_k8v4_stuart_rungs).
# Boots the k8v4 test server (port 8896, detached, pidfile+log), waits for
# readiness, then runs the registered probes PAIRED against the banked fp16
# baselines from the kvarn8 chain (found via LEANN, 2026-07-12):
#   ~/ai/liberated/kimi-kindled/kindled_projects/mlxcel-kv-quant/results/
#   - divergence_capture_fp16_d2-32k.json   (depths 2K/8K/32K; 128K was
#     never captured — recorded honestly, not silently skipped)
#   - copy_precision_fp16_d{50k,100k,300k}_seed42.json  (seed 42 = identical
#     targets; note banked set is 100k, not the proposal's 150k)
#   - copy_precision_kvarn8_d{50k,300k}_seed42.json     (the registration's
#     second pairing: vs banked kvarn8, where banked)
# Fail-loud: missing artifact / dead server / non-200 readiness = printed
# FAIL + nonzero exit, never a silent skip.
#
# Usage:  ./scripts/run_k8v4_leg.sh
# Env (optional): PORT (8896), OUT_DIR (~/k8v4_rungs), MODEL_DIR, BASELINE_DIR

set -u
cd "$(dirname "$0")/.."

PORT="${PORT:-8896}"
OUT_DIR="${OUT_DIR:-$HOME/k8v4_rungs}"
MODEL_DIR="${MODEL_DIR:-/Volumes/T7 Shield/models/huggingface_cache_hub/MiniMax-M3-MXFP8-64e-mlx}"
BASELINE_DIR="${BASELINE_DIR:-$HOME/ai/liberated/kimi-kindled/kindled_projects/mlxcel-kv-quant/results}"
ALIAS="minimax-m3-test"
# THE calibration the copy-precision probe needs to hit TRUE token depths.
# MiniMax-M3-MXFP8 tokenizes this filler at ~5.67 chars/token (measured
# 2026-07-07: 200,003 chars -> ~35,300 tokens; re-confirmed live 07-12).
# The probe's default is 4, which UNDERSHOOTS every depth ~30% (my leg's
# 2026-07-12 bug: omitting this made a "50K" run actually 35K, breaking
# the pairing vs the fp16 baseline that DID use 5.67). Single-sourced so
# both invocations below can never drift apart.
CHARS_PER_TOKEN="5.67"
BASE_URL="http://127.0.0.1:${PORT}"
STAMP=$(date +%Y%m%d_%H%M)
LOG="$HOME/mlxcel_test_${PORT}_k8v4_${STAMP}.log"
PIDFILE="$HOME/mlxcel_test_${PORT}.pid"

for f in divergence_capture_fp16_d2-32k.json copy_precision_fp16_d50k_seed42.json \
         copy_precision_fp16_d100k_seed42.json copy_precision_fp16_d300k_seed42.json; do
  [ -f "$BASELINE_DIR/$f" ] || { echo "FAIL: baseline missing: $BASELINE_DIR/$f" >&2; exit 1; }
done
mkdir -p "$OUT_DIR"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "FAIL: test server already running (pid $(cat "$PIDFILE"))" >&2; exit 1
fi

# Wired-memory pre-flight: TWO ~215GiB model instances do not fit under
# the 464GB wired limit. If any other mlxcel-server is up (e.g. the
# fp16g G-live boot on :8890, purpose complete), refuse with the exact
# remedy rather than OOM-ing the machine mid-load.
OTHER=$(pgrep -f "mlxcel-server" || true)
if [ -n "$OTHER" ]; then
  echo "FAIL: another mlxcel-server is running (pid(s): $OTHER) — two model" >&2
  echo "      instances exceed the wired limit. If it's the finished fp16g" >&2
  echo "      boot: kill \$(cat ~/mlxcel_server.pid)   then re-run this leg." >&2
  exit 1
fi

echo "== boot k8v4 test server :$PORT (log: $LOG)"
nohup ./target/release/mlxcel-server -m "$MODEL_DIR" \
  --host 0.0.0.0 --port "$PORT" --alias "$ALIAS" \
  --kv-cache-mode k8v4 \
  > "$LOG" 2>&1 &
echo $! > "$PIDFILE"

for i in $(seq 1 240); do
  sleep 5
  if curl -sf "$BASE_URL/v1/models" >/dev/null 2>&1; then READY=1; break; fi
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "FAIL: server died during load — tail of $LOG:"; tail -5 "$LOG"; exit 1
  fi
done
[ "${READY:-0}" = 1 ] || { echo "FAIL: server not ready after 20 min"; exit 1; }
echo "== ready. Boot echo (must say k8v4):"
grep -E "kvarn_format|kvarn_v_bits" "$LOG" | head -3

echo "== §4.3 capture at the BASELINE's depths (2K/8K/32K; 128K has no fp16"
echo "   baseline on disk — a 128K rung needs its own fp16 capture first)"
python3 scripts/greedy_divergence_harness.py capture \
  --base-url "$BASE_URL" --model "$ALIAS" --depths 2000,8000,32000 \
  --out "$OUT_DIR/divergence_capture_k8v4_d2-32k_${STAMP}.json" \
  || { echo "FAIL: divergence capture"; exit 1; }

echo "== §4.3 diff vs fp16 baseline (acceptance: length-independent per prompt)"
python3 scripts/greedy_divergence_harness.py diff \
  "$BASELINE_DIR/divergence_capture_fp16_d2-32k.json" \
  "$OUT_DIR/divergence_capture_k8v4_d2-32k_${STAMP}.json" \
  | tee "$OUT_DIR/divergence_diff_k8v4_${STAMP}.txt"

echo "== §4.4 copy-precision, seed 42, paired vs fp16 (THE gate) + vs banked kvarn8"
for d in 50000 100000 300000; do
  short=$((d / 1000))k
  python3 scripts/copy_precision_probe.py "$BASE_URL" "$ALIAS" \
    --depth "$d" --seed 42 --chars-per-token "$CHARS_PER_TOKEN" \
    --compare-against "$BASELINE_DIR/copy_precision_fp16_d${short}_seed42.json" \
    | tee "$OUT_DIR/copy_precision_k8v4_d${short}_vs_fp16_${STAMP}.txt" \
    || { echo "FAIL: copy-precision $d vs fp16"; exit 1; }
  if [ -f "$BASELINE_DIR/copy_precision_kvarn8_d${short}_seed42.json" ]; then
    python3 scripts/copy_precision_probe.py "$BASE_URL" "$ALIAS" \
      --depth "$d" --seed 42 --chars-per-token "$CHARS_PER_TOKEN" \
      --compare-against "$BASELINE_DIR/copy_precision_kvarn8_d${short}_seed42.json" \
      | tee "$OUT_DIR/copy_precision_k8v4_d${short}_vs_kvarn8_${STAMP}.txt" \
      || { echo "FAIL: copy-precision $d vs kvarn8"; exit 1; }
  fi
done

echo "== teardown"
kill "$(cat "$PIDFILE")" && rm -f "$PIDFILE"
echo "== DONE. Artifacts in $OUT_DIR; server log $LOG"
echo "== Acceptance (registered): §4.3 divergence LENGTH-INDEPENDENT per prompt;"
echo "   §4.4 exact-match NOT WORSE than baseline, paired. Clement's seat analyzes."
