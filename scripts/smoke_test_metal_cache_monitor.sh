#!/bin/bash
# Basic E2E for MLX free-buffer cache cap + memory monitor.
#
# Verifies the new MLXCEL_METAL_CACHE_LIMIT + MLXCEL_MEMORY_MONITOR_INTERVAL_SECS
# code path end-to-end:
#   1. Server starts with the env vars set
#   2. Startup log echoes the resolved cap and monitor interval
#      (confirms resolve_metal_cache_limit / resolve_memory_monitor_interval ran)
#   3. After one interval tick the monitor emits an "MLX memory snapshot"
#      INFO line with active_bytes / cache_bytes / peak_bytes / limit_bytes
#      (confirms spawn_memory_monitor's loop is alive)
#   4. cache_bytes in every emitted snapshot is <= configured cap
#      (the actual fix: set_cache_limit took effect)
#   5. A four-turn conversation completes cleanly
#
# This is the partner to smoke_test_msa_chunked.sh — that one exercises
# MSA + chunked prefill; this one exercises the new Metal allocator
# controls. Both share the four-turn convention.
#
# Prereq: release build of mlxcel-server complete in target/release/.
# Run from mlxcel project root.

set -euo pipefail

MODEL_PATH="/Volumes/T7 Shield/models/huggingface_cache_hub/models--sjswerdloff--MiniMax-M3-NVFP4-mlx"
PORT="${PORT:-8891}"
HOST="${HOST:-127.0.0.1}"
ALIAS="${ALIAS:-minimax-m3-nvfp4}"
LOG="${LOG:-/tmp/mlxcel_metal_cache_smoke.log}"
SERVER_PID_FILE="/tmp/mlxcel_metal_cache_smoke.pid"

# Small cap (4 GiB) so the test surfaces fast if the cap doesn't bind.
# Monitor every 10s so we don't wait long for a snapshot.
CACHE_LIMIT="${CACHE_LIMIT:-4GB}"
MONITOR_SECS="${MONITOR_SECS:-10}"
CACHE_LIMIT_BYTES=$((4 * 1024 * 1024 * 1024))

echo "Starting server (logs: $LOG) ..."
echo "  MLXCEL_METAL_CACHE_LIMIT=$CACHE_LIMIT"
echo "  MLXCEL_MEMORY_MONITOR_INTERVAL_SECS=$MONITOR_SECS"
MLXCEL_METAL_CACHE_LIMIT="$CACHE_LIMIT" \
MLXCEL_MEMORY_MONITOR_INTERVAL_SECS="$MONITOR_SECS" \
RUST_LOG="${RUST_LOG:-info}" \
nohup ./target/release/mlxcel-server \
  --model "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --alias "$ALIAS" \
  --temp 0.0 \
  > "$LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$SERVER_PID_FILE"
echo "Server PID: $SERVER_PID"

cleanup() {
  echo "Stopping server (PID $SERVER_PID) ..."
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  rm -f "$SERVER_PID_FILE"
}
trap cleanup EXIT

# Wait for /v1/models. Model load is slow (240 GB weights).
echo "Waiting for /v1/models to respond (model load is slow on 240 GB weights) ..."
until curl -sS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "FAIL: server died during load. See $LOG"
    exit 1
  fi
  sleep 10
done
echo "Server ready."

# Gate A: startup log echoed the resolved cap.
# Asserts resolve_metal_cache_limit ran AND parsed the env var correctly.
echo
echo "=== Gate A: startup log shows resolved cache cap ==="
CAP_LINE=$(grep -E 'MLX free-buffer cache cap: [0-9.]+ GB \(from MLXCEL_METAL_CACHE_LIMIT\)' "$LOG" || true)
if [[ -z "$CAP_LINE" ]]; then
  echo "FAIL: expected startup line"
  echo "  'MLX free-buffer cache cap: <N> GB (from MLXCEL_METAL_CACHE_LIMIT)'"
  echo "  but it was not in $LOG. Either resolve_metal_cache_limit did not"
  echo "  run, the env var was not propagated, or the log format changed."
  exit 1
fi
echo "  found: $CAP_LINE"

# Gate B: startup log shows the monitor was scheduled.
echo
echo "=== Gate B: startup log shows monitor was scheduled ==="
MON_LINE=$(grep -E 'MLX memory monitor: emitting every [0-9]+s' "$LOG" || true)
if [[ -z "$MON_LINE" ]]; then
  echo "FAIL: expected startup line"
  echo "  'MLX memory monitor: emitting every Ns'"
  echo "  but it was not in $LOG. spawn_memory_monitor was not invoked,"
  echo "  or the resolve_memory_monitor_interval path returned None."
  exit 1
fi
echo "  found: $MON_LINE"

# Helper to send a chat completion.
chat() {
  local content="$1"
  curl -sS -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --arg c "$content" --arg m "$ALIAS" \
      '{model: $m, messages: [{role: "user", content: $c}], max_tokens: 32, stream: false}')"
}

# Four turns, light prompts. We're not exercising MSA here (that's the
# other smoke test); we just want enough inference to (a) ensure the
# server is alive AND (b) burn some allocator activity so the monitor
# snapshot has meaningful numbers.
echo
echo "=== Four-turn conversation ==="
for i in 1 2 3 4; do
  case "$i" in
    1) PROMPT="What is the capital of France?" ;;
    2) PROMPT="What is the capital of Germany?" ;;
    3) PROMPT="Which is larger by population?" ;;
    4) PROMPT="Which is older as a settlement?" ;;
  esac
  echo "  turn $i: $PROMPT"
  RESP=$(chat "$PROMPT")
  echo "$RESP" | jq -e '.choices[0].finish_reason' >/dev/null \
    || { echo "FAIL: turn $i did not complete cleanly"; exit 1; }
  CONTENT=$(echo "$RESP" | jq -r '.choices[0].message.content' 2>/dev/null | head -c 120)
  echo "    -> finish=$(echo "$RESP" | jq -r '.choices[0].finish_reason')  content='${CONTENT}'"
done

# Sleep one full monitor interval + slack, so at least one snapshot
# definitely fired AFTER the conversation finished (the first tick is
# skipped post-startup so the very first snapshot lands one interval in).
SLEEP_FOR=$((MONITOR_SECS + 5))
echo
echo "Waiting ${SLEEP_FOR}s for at least one monitor snapshot to fire ..."
sleep "$SLEEP_FOR"

# Gate C: at least one snapshot line emitted with the expected fields.
echo
echo "=== Gate C: monitor emitted at least one snapshot ==="
SNAP_COUNT=$(grep -c 'MLX memory snapshot' "$LOG" || true)
echo "  MLX memory snapshot lines: $SNAP_COUNT (expect >= 1)"
if [[ "$SNAP_COUNT" -lt 1 ]]; then
  echo "FAIL: no 'MLX memory snapshot' line in $LOG. The monitor loop is"
  echo "  not running or the tracing target is filtered out. Check"
  echo "  RUST_LOG includes 'mlxcel::memory_monitor=info' or 'info'."
  exit 1
fi

# Gate D: cache_bytes in every snapshot is <= configured cap.
# This is the load-bearing check: confirms set_cache_limit actually
# took effect, not just that the resolver returned a value.
#
# tracing emits structured fields as `cache_bytes=N`. Extract every
# occurrence and verify the max never exceeds CACHE_LIMIT_BYTES.
echo
echo "=== Gate D: every observed cache_bytes <= cap ($CACHE_LIMIT_BYTES bytes) ==="
MAX_CACHE=$(grep -oE 'cache_bytes=[0-9]+' "$LOG" | sed 's/cache_bytes=//' | sort -n | tail -1)
if [[ -z "$MAX_CACHE" ]]; then
  echo "FAIL: snapshot lines present but no 'cache_bytes=N' field found."
  echo "  Tracing layout may have changed. Inspect $LOG."
  exit 1
fi
echo "  max observed cache_bytes: $MAX_CACHE"
if (( MAX_CACHE > CACHE_LIMIT_BYTES )); then
  echo "FAIL: cache_bytes ($MAX_CACHE) exceeded configured cap ($CACHE_LIMIT_BYTES)."
  echo "  set_cache_limit did not take effect, OR MLX's cap enforcement"
  echo "  is asynchronous and lagged. Investigate."
  exit 1
fi
echo "  cache_bytes stayed within cap."

echo
echo "PASS: cache cap + monitor end-to-end."
echo "  Gate A (startup cap line):        OK"
echo "  Gate B (startup monitor line):    OK"
echo "  Gate C (snapshot emitted):        OK ($SNAP_COUNT lines)"
echo "  Gate D (cache_bytes <= cap):      OK (max $MAX_CACHE bytes)"
echo "  Full server log: $LOG"
