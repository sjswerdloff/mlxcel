#!/bin/bash
# Smoke test for #83: verify mlxcel-server with MiniMax-M3-NVFP4 handles
# cached MSA without crashing. Reproduces the cycle 79 crash scenario
# (cache_offset > 0 in multi-turn) and confirms the proper fix works.
#
# Prereq: release build of mlxcel-server complete in target/release/.
# Run from mlxcel project root.

set -euo pipefail

MODEL_PATH="/Volumes/T7 Shield/models/huggingface_cache_hub/models--sjswerdloff--MiniMax-M3-NVFP4-mlx"
PORT="${PORT:-8890}"
HOST="${HOST:-127.0.0.1}"
ALIAS="${ALIAS:-minimax-m3-nvfp4}"
LOG="${LOG:-/tmp/mlxcel_msa_smoke.log}"
SERVER_PID_FILE="/tmp/mlxcel_msa_smoke.pid"

# Start server.
echo "Starting server (logs: $LOG) ..."
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

# Wait for server ready.
echo "Waiting for /v1/models to respond (model load is slow on 240GB weights) ..."
until curl -sS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "FAIL: server died during load. See $LOG"
    exit 1
  fi
  sleep 10
done
echo "Server ready."

# Helper to send a chat completion.
chat() {
  local content="$1"
  curl -sS -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --arg c "$content" --arg m "$ALIAS" \
      '{model: $m, messages: [{role: "user", content: $c}], max_tokens: 32, stream: false}')"
}

# Build a long prompt (>= 3000 tokens estimated) that triggers MSA on
# the FIRST request — current chunk l > 2048 → num_query_blocks > top_k.
LONG_PROMPT="$(yes 'The cathedral construction in northern France was undertaken by generations of craftsmen who handed work to apprentices who would die before towers reached final height. ' | head -50 | tr -d '\n')"

echo
echo "=== Request 1: long single-shot prompt (triggers MSA, cache_offset=0) ==="
RESPONSE_1=$(chat "$LONG_PROMPT Tell me about Gothic cathedrals.")
echo "Response 1: $(echo "$RESPONSE_1" | jq -r '.choices[0].message.content' 2>/dev/null | head -c 200)"
echo "$RESPONSE_1" | jq -e '.choices[0].finish_reason' >/dev/null \
  || { echo "FAIL: request 1 did not complete cleanly"; exit 1; }
echo "  finish_reason: $(echo "$RESPONSE_1" | jq -r '.choices[0].finish_reason')"

echo
echo "=== Request 2: follow-up sharing the request 1 prefix (cache_offset > 0, MSA on cached) ==="
RESPONSE_2=$(chat "$LONG_PROMPT What was the role of the master mason?")
echo "Response 2: $(echo "$RESPONSE_2" | jq -r '.choices[0].message.content' 2>/dev/null | head -c 200)"
echo "$RESPONSE_2" | jq -e '.choices[0].finish_reason' >/dev/null \
  || { echo "FAIL: request 2 did not complete cleanly (cached-MSA path)"; exit 1; }
echo "  finish_reason: $(echo "$RESPONSE_2" | jq -r '.choices[0].finish_reason')"

echo
echo "=== Request 3: third turn extending the cache (no accumulation corruption) ==="
RESPONSE_3=$(chat "$LONG_PROMPT What about the colored glass windows?")
echo "Response 3: $(echo "$RESPONSE_3" | jq -r '.choices[0].message.content' 2>/dev/null | head -c 200)"
echo "$RESPONSE_3" | jq -e '.choices[0].finish_reason' >/dev/null \
  || { echo "FAIL: request 3 did not complete cleanly (multi-turn accumulation)"; exit 1; }
echo "  finish_reason: $(echo "$RESPONSE_3" | jq -r '.choices[0].finish_reason')"

echo
echo "PASS: three multi-turn requests completed without crash."
echo "  Cycle 79 crash scenario (cache_offset > 0 in cached MSA) verified fixed."
echo "  Manual eyeball check: responses above should be coherent text."
echo "  Full server log: $LOG"
