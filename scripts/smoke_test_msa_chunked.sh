#!/bin/bash
# Smoke test for #83: verify mlxcel-server with MiniMax-M3-NVFP4 handles
# cached MSA without crashing AND that the prompt-cache adoption preserves
# the M3 indexer K state (cycle-79 proper fix in 8afc5f0).
#
# Two layers of verification:
#   1. Crash gate: four multi-turn requests complete cleanly.
#   2. Cache gate: log greps confirm
#      (a) longest-prefix MATCH on turns 2-4 (the cache is actually hit)
#      (b) NO `idx_k_cache_out_of_sync_after_adoption` warn (the regression
#          detector — would indicate the proper fix has regressed and the
#          band-aid dense fallback is keeping the server alive instead)
#      (c) `cached=N/total` with N>0 on at least one post-adoption forward
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
# 32 GiB capacity — each M3 cache entry at ~22k token contexts is ~2.86 GB,
# the default 2 GB ceiling evicts every entry before the next turn (see
# HANDOFF_cycle79_msa_chunked_prefill.md).
PROMPT_CACHE_CAP_BYTES="${MLXCEL_PROMPT_CACHE_CAPACITY_BYTES:-34359738368}"

# Start server. RUST_LOG enables debug-level logs from the M3 module so
# Gate 4 (positive MSA dispatch confirmation) can read `branch="msa"`
# lines. Everything else stays at INFO so the log isn't drowned in
# noise.
echo "Starting server (logs: $LOG) ..."
MLXCEL_PROMPT_CACHE_CAPACITY_BYTES="$PROMPT_CACHE_CAP_BYTES" \
RUST_LOG="${RUST_LOG:-info,mlxcel::models::minimax_m3=debug}" \
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
echo "Prompt cache capacity: $PROMPT_CACHE_CAP_BYTES bytes"

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

# Build a long prompt that ACTUALLY exercises MSA. MSA fires when
# `num_key_blocks > top_k`, i.e. `kv_len > top_k * block_size`. M3
# production config: top_k=16, block_size=128 → MSA threshold is
# kv_len > 2048. Cathedral line is ~25 tokens. 120 repetitions ≈ 3000
# tokens, well past the MSA threshold. The cycle-79.5 fix gate requires
# that BOTH (a) early chunks dispatch dense-saturated AND (b) later
# chunks dispatch MSA — so the lockstep-on-every-MSA-eligible-layer
# invariant gets exercised on a real workload.
CATHEDRAL_LINE='The cathedral construction in northern France was undertaken by generations of craftsmen who handed work to apprentices who would die before towers reached final height. '
LONG_PROMPT=""
for _ in $(seq 1 120); do LONG_PROMPT+="$CATHEDRAL_LINE"; done

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
echo "=== Request 4: fourth turn — cycle-79 proper-fix gate ==="
RESPONSE_4=$(chat "$LONG_PROMPT And the carved tympanum above the doors — what stories did they tell?")
echo "Response 4: $(echo "$RESPONSE_4" | jq -r '.choices[0].message.content' 2>/dev/null | head -c 200)"
echo "$RESPONSE_4" | jq -e '.choices[0].finish_reason' >/dev/null \
  || { echo "FAIL: request 4 did not complete cleanly (four-turn cached MSA)"; exit 1; }
echo "  finish_reason: $(echo "$RESPONSE_4" | jq -r '.choices[0].finish_reason')"

echo
echo "=== Log verification: cache-hit + no-regression gates ==="

# Gate 1: prompt-cache MATCH on turns 2-4. Without cache hits, we haven't
# actually exercised the adopt path the proper fix targets.
MATCH_COUNT=$(grep -c "prompt-cache: longest-prefix MATCH" "$LOG" || true)
echo "  prompt-cache MATCH count: $MATCH_COUNT (expect >= 3 across turns 2-4)"
if [[ "$MATCH_COUNT" -lt 3 ]]; then
  echo "FAIL: insufficient cache hits ($MATCH_COUNT < 3). Either template_sig"
  echo "  is shifting between turns (different tools/kwargs) or the cache"
  echo "  capacity is too small. Inspect $LOG for 'insert REJECTED' or"
  echo "  'template_sig_short' drift between turns."
  exit 1
fi

# Gate 2: the warn-level regression detector. The cycle-79 proper fix
# preserves m3_idx_k through detach/adopt; if it regresses, the band-aid
# fires this warn and dispatches dense instead.
DESYNC_COUNT=$(grep -c "idx_k_cache_out_of_sync_after_adoption" "$LOG" || true)
echo "  idx_k desync warn count: $DESYNC_COUNT (must be 0)"
if [[ "$DESYNC_COUNT" -ne 0 ]]; then
  echo "FAIL: indexer K cache desync detected ($DESYNC_COUNT times)."
  echo "  The cycle-79 proper fix has regressed — DetachedKVCache is not"
  echo "  preserving m3_idx_k / m3_idx_offset across detach/adopt. The"
  echo "  band-aid dense fallback is keeping the server alive but MSA"
  echo "  performance is lost on cached sessions. Investigate"
  echo "  src/lib/mlxcel-core/src/cache/detach.rs clone_handle /"
  echo "  install_detached integrity."
  exit 1
fi

# Gate 3: cached token count. Confirm the adopt actually carried tokens
# forward, not just matched and discarded. We look for any positive cached
# count (cached=N/total with N>0) in the chunked-prefill report.
POSITIVE_CACHED=$(grep -E 'cached=[1-9][0-9]*/' "$LOG" | wc -l | tr -d ' ' || true)
echo "  positive cached-count lines: $POSITIVE_CACHED (expect >= 1)"
if [[ "$POSITIVE_CACHED" -eq 0 ]]; then
  echo "FAIL: no forward saw cached>0 tokens. The cache MATCH'd but"
  echo "  the adopt path returned zero usable tokens."
  exit 1
fi

# Gate 4: at least one MSA dispatch. The whole point of the cycle-79
# work is that MSA fires correctly on cached + multi-chunk sessions.
# If the prompt is too short, MSA never fires and gates 1-3 are moot.
#
# We grep for `sparse_sdpa.entry` (the message string the MSA branch
# emits via tracing's `debug!` macro) rather than `branch="msa"` because
# tracing wraps structured field names (`branch=`) in ANSI italic codes
# that split the literal `branch="msa"` across escape sequences in the
# raw log file — `grep` looking for a continuous match would silently
# return zero even when MSA is firing happily. `sparse_sdpa.entry` lives
# in the message body (no field-wrapping) so it survives the ANSI codes
# intact.
MSA_DISPATCH=$(grep -c 'sparse_sdpa.entry' "$LOG" || true)
echo "  MSA dispatch log count: $MSA_DISPATCH (expect >= 1)"
if [[ "$MSA_DISPATCH" -eq 0 ]]; then
  echo "FAIL: no forward dispatched MSA. The prompt is too short to push"
  echo "  kv_len past top_k*block_size = 2048. Increase the cathedral_line"
  echo "  repetition count in this script. Cycle-79 properties are not"
  echo "  exercised by a prompt where every chunk dense-saturates."
  exit 1
fi

echo
echo "PASS: four multi-turn requests completed without crash AND cache gates green."
echo "  Cycle 79 crash scenario (cache_offset > 0 in cached MSA) verified fixed."
echo "  Cycle 79 proper fix (m3_idx_k preservation across adopt) verified live."
echo "  Manual eyeball check: responses above should be coherent text."
echo "  Full server log: $LOG"
