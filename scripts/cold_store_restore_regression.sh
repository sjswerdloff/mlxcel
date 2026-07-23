#!/bin/bash
# Cold-store persist -> RESTART -> restore end-to-end regression test.
#
# Why a restart: a shared-prefix multi-turn chat against ONE live server is
# served by the HOT in-memory prompt cache and never touches cold-storage
# (disk). Cold-store is only consulted when the hot cache cannot satisfy the
# prefix -- i.e. after a server restart. So this test:
#
#   Phase 1 (PRIME):   start server, send the long-prefix turn, let the
#                      finished sequence DONATE its KV to cold-storage on disk,
#                      stop the server (hot cache is gone).
#   Phase 2 (RESTORE): restart, re-send the SAME prefix. The hot cache is
#                      empty, so a coherent answer with NO "falling back to
#                      cold prefill" in the log == cold-store ADOPT succeeded.
#   Phase 3 (DISCRIMINATION CONTROL): flip one byte in a persisted cold-store
#                      layer file, restart, re-send the prefix. A checksummed
#                      store MUST DECLINE the corrupt entry ("falling back to
#                      cold prefill") -- fail-closed to clean prefill, never
#                      adopt garbage. This is the at-rest twin of the fidelity
#                      gate's META-RED negative control.
#
# REGRESSION SEMANTICS (regression-test-must-regress):
#   * v2 cold-store (no per-layer checksums) is EXPECTED TO FAIL Phase 3: it
#     adopts the corrupted entry silently. That failure is the vulnerability
#     this test exists to catch.
#   * v3-wired cold-store (per-layer SHA-256 + atomic COMMITTED) is EXPECTED
#     TO PASS Phase 3: the flipped byte fails its checksum -> decline. Phase 3
#     going from RED to GREEN is the E2E proof that the integrity fix landed.
#
# HEAVY: this starts a real model server. Run it yourself; do NOT invoke from
# an AI session tree (detached-big-memory-jobs rule). ~64GB+ resident.
#
# Usage:
#   LAUNCHER=~/RustProjects/mlxcel/old_scripts/start_mlxcel_m3.sh \
#     scripts/cold_store_restore_regression.sh
#
# Env:
#   LAUNCHER   command that starts mlxcel-server in the FOREGROUND (required;
#              the script backgrounds it and manages its lifecycle).
#   HOST/PORT  server address (default 127.0.0.1:8890).
#   ALIAS      model alias for the API (default minimax-m3-mxfp8).
#   COLD_DIR   cold-store base dir (default $HOME/.cache/mlxcel/cold-storage).
#   MAX_TOKENS per-turn generation cap (default 32).
#   READY_TIMEOUT  seconds to wait for /v1/models (default 600).

set -uo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8890}"
ALIAS="${ALIAS:-minimax-m3-mxfp8}"
COLD_DIR="${COLD_DIR:-$HOME/.cache/mlxcel/cold-storage}"
MAX_TOKENS="${MAX_TOKENS:-32}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"
LAUNCHER="${LAUNCHER:?set LAUNCHER to a script that starts mlxcel-server in the foreground}"

WORK="$(mktemp -d)"
SERVER_LOG="$WORK/server.log"
SERVER_PID=""
FAILED=0
trap 'stop_server; rm -rf "$WORK"' EXIT

note()  { echo "[cold-store-regression] $*"; }
fail()  { echo "FAIL: $*" >&2; FAILED=1; }

start_server() {
  note "starting server via $LAUNCHER (log: $SERVER_LOG)"
  ( exec "$LAUNCHER" ) >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  local waited=0
  until curl -sSf "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      fail "server process died during startup; tail of log:"; tail -20 "$SERVER_LOG" >&2; return 1
    fi
    sleep 2; waited=$((waited+2))
    if (( waited >= READY_TIMEOUT )); then fail "server not ready after ${READY_TIMEOUT}s"; return 1; fi
  done
  note "server ready after ${waited}s"
}

stop_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  note "stopping server (pid $SERVER_PID)"
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
}

# Marker so each phase greps only the log lines produced in its own window.
mark_log() { echo "===PHASE-MARK: $1===" >>"$SERVER_LOG"; }
log_since_mark() { awk -v m="===PHASE-MARK: $1===" 'f{print} $0 ~ m {f=1}' "$SERVER_LOG"; }

CATHEDRAL_LINE='The cathedral construction in northern France was undertaken by generations of craftsmen who handed work to apprentices who would die before towers reached final height. '
LONG_PROMPT=""
for _ in $(seq 1 120); do LONG_PROMPT+="$CATHEDRAL_LINE"; done   # ~3000 tokens, above the 2048 MSA threshold

chat() {  # $1=user content -> prints coherent response body, exits nonzero on API failure
  curl -sS -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --arg c "$1" --arg m "$ALIAS" --argjson mt "$MAX_TOKENS" \
      '{model:$m, messages:[{role:"user",content:$c}], max_tokens:$mt, stream:false}')"
}

coherent() {  # $1=turn label $2=body -> asserts a finish_reason + non-empty content
  local fr content
  fr=$(echo "$2" | jq -r '.choices[0].finish_reason // empty' 2>/dev/null)
  content=$(echo "$2" | jq -r '.choices[0].message.content // empty' 2>/dev/null)
  if [[ -z "$fr" || -z "$content" ]]; then
    fail "$1: no finish_reason/content (body: $(echo "$2" | head -c 300))"; return 1
  fi
  note "$1: finish=$fr | $(echo "$content" | head -c 120)"
}

# ---------------------------------------------------------------------------
note "Phase 1 - PRIME: donate a long-prefix sequence to cold-storage"
rm -rf "$COLD_DIR"          # clean slate so entry-count assertions are exact
start_server || exit 1
mark_log prime
coherent "turn1-prime" "$(chat "$LONG_PROMPT Tell me about Gothic cathedrals.")" || true
stop_server
if [[ -d "$COLD_DIR" ]] && find "$COLD_DIR" -type f | read -r _; then
  note "cold-store persisted entries on disk: OK"
else
  fail "Phase 1: no cold-store files under $COLD_DIR after a finished sequence (persist did not run)"
fi

# ---------------------------------------------------------------------------
note "Phase 2 - RESTORE: restart (hot cache gone), re-send the same prefix"
start_server || exit 1
mark_log restore
coherent "turn2-restore" "$(chat "$LONG_PROMPT What was the role of the master mason?")" || true
if log_since_mark restore | grep -q "falling back to cold prefill"; then
  fail "Phase 2: server fell back to cold prefill; cold-store ADOPT did not happen after restart"
else
  note "Phase 2: no cold-prefill fallback after restart -> cold-store adopt path exercised: OK"
fi
stop_server

# ---------------------------------------------------------------------------
note "Phase 3 - DISCRIMINATION CONTROL: corrupt one byte, restore must DECLINE"
CORRUPT_TARGET="$(find "$COLD_DIR" -type f -name 'layer_*' | head -1)"
if [[ -z "$CORRUPT_TARGET" ]]; then
  # v3 layout is generation-dir based; fall back to any persisted payload file.
  CORRUPT_TARGET="$(find "$COLD_DIR" -type f ! -name 'COMMITTED' ! -name 'header.bin' | head -1)"
fi
if [[ -z "$CORRUPT_TARGET" ]]; then
  fail "Phase 3: could not find a cold-store payload file to corrupt"
else
  note "flipping one byte in $CORRUPT_TARGET"
  printf '\x01' | dd of="$CORRUPT_TARGET" bs=1 seek=64 count=1 conv=notrunc status=none 2>/dev/null \
    || python3 -c "import sys;p=sys.argv[1];b=bytearray(open(p,'rb').read());b[min(64,len(b)-1)]^=0xFF;open(p,'wb').write(b)" "$CORRUPT_TARGET"
  start_server || exit 1
  mark_log corrupt
  coherent "turn3-corrupt" "$(chat "$LONG_PROMPT What about the colored glass windows?")" || true
  if log_since_mark corrupt | grep -q "falling back to cold prefill"; then
    note "Phase 3: corrupt entry DECLINED -> clean prefill (checksum fail-closed): OK"
  else
    fail "Phase 3: corrupt entry was NOT declined. A checksummed store must reject it. \
(EXPECTED on v2/no-checksums == the vulnerability; must go GREEN once v3 checksums are wired.)"
  fi
  stop_server
fi

# ---------------------------------------------------------------------------
if (( FAILED )); then
  echo "=== COLD-STORE RESTORE REGRESSION: FAIL ==="; exit 1
else
  echo "=== COLD-STORE RESTORE REGRESSION: PASS ==="; exit 0
fi
