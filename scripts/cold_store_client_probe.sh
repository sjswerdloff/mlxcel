#!/bin/bash
# CLIENT-ONLY cold-store probe. Does NOT start, stop, restart or otherwise
# manage a server. You launch the server yourself, separately; this talks to it.
#
# WHY THIS EXISTS ALONGSIDE cold_store_restore_regression.sh: that script owns
# the server lifecycle (LAUNCHER, background, restart between phases). That is
# the wrong shape when the operator launches from their own script. This one
# never touches the process.
#
# HOW IT FORCES COLD STORAGE WITHOUT A RESTART. A repeated shared prefix is
# normally served from the HOT in-memory prompt cache and never reaches disk,
# which is why the older script restarted the server. It does not have to:
# `POST /v1/cache/reset` drops every live cache entry. After a reset the hot
# cache is empty and the next identical prefix must be satisfied from cold
# storage or re-prefilled from scratch — the same fork the restart created.
#
# WHAT IT ASSERTS, and what it deliberately does not:
#   1. PERSIST     cold-store files appear on disk after a finished sequence.
#                  Filesystem evidence, not a log claim.
#   2. SURVIVE     after a hot-cache reset the same prefix still answers
#                  coherently — adoption did not corrupt the conversation.
#   3. FAIL-CLOSED corrupt one byte of a persisted PAYLOAD, reset, ask again:
#                  the answer must still be coherent. A checksummed store
#                  declines the damaged entry and re-prefills. Garbage out
#                  here means the store adopted data it could not verify.
#
#   TIMING IS REPORTED, NEVER ASSERTED. A cold hit should be much faster than
#   a full prefill, but wall-clock on a shared machine is not a contract and a
#   timing assertion would be flaky by construction.
#
# ⚠️ STEP 3 IS THE ONLY ONE THAT CAN CATCH A SAFETY DEFECT. Steps 1 and 2 pass
# against a store that silently adopts anything. Read a green 1+2 with 3
# skipped as "nothing was checked".
#
# Usage — server already running, launched by you:
#   scripts/cold_store_client_probe.sh
#   V4=1 scripts/cold_store_client_probe.sh     # probe the v4 block store
#
# Env:
#   HOST/PORT   server address (default 127.0.0.1:8890)
#   ALIAS       model alias (default minimax-m3-mxfp8)
#   COLD_DIR    cold-store base (default $HOME/.cache/mlxcel/cold-storage)
#   V4          1 = expect/inspect the v4 block layout under $COLD_DIR/v4-blocks
#   MAX_TOKENS  per-turn cap (default 32)
#   NO_CORRUPT  1 = skip step 3 (leaves the store untouched; see warning above)

set -uo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8890}"
ALIAS="${ALIAS:-minimax-m3-mxfp8}"
COLD_DIR="${COLD_DIR:-$HOME/.cache/mlxcel/cold-storage}"
MAX_TOKENS="${MAX_TOKENS:-32}"
V4="${V4:-0}"
NO_CORRUPT="${NO_CORRUPT:-0}"
BASE="http://${HOST}:${PORT}"

RC=0
note() { printf '\n=== %s\n' "$*"; }
ok()   { printf '  OK   %s\n' "$*"; }
bad()  { printf '  FAIL %s\n' "$*"; RC=1; }

if [[ "$V4" == "1" ]]; then
  SEARCH_ROOT="$COLD_DIR/v4-blocks"
  WHICH="v4 block cold store"
else
  SEARCH_ROOT="$COLD_DIR"
  WHICH="v3 cold store"
fi

# --- preflight: the server must be up. We do not start it. -----------------
if ! curl -sf --max-time 10 "$BASE/v1/models" >/dev/null; then
  printf 'FAIL: no server answering at %s/v1/models\n' "$BASE" >&2
  printf '      This probe is CLIENT-ONLY: start the server with your own\n' >&2
  printf '      script first, then re-run.\n' >&2
  exit 2
fi
note "probing $WHICH at $BASE (client-only; server lifecycle is yours)"

LONG_PROMPT="$(python3 - <<'PY'
print("Describe the architecture of a gothic cathedral. " * 40)
PY
)"

chat() {  # $1 = user content; prints assistant text
  python3 - "$BASE" "$ALIAS" "$MAX_TOKENS" "$1" <<'PY'
import json, sys, urllib.request
base, alias, maxtok, content = sys.argv[1:5]
req = urllib.request.Request(
    f"{base}/v1/chat/completions",
    data=json.dumps({"model": alias, "max_tokens": int(maxtok),
                     "temperature": 0.0,
                     "messages": [{"role": "user", "content": content}]}).encode(),
    headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    print(d["choices"][0]["message"]["content"])
except Exception as e:  # noqa: BLE001 - surfaced to the caller as empty output
    print("", end="")
    sys.stderr.write(f"request failed: {e}\n")
PY
}

# Coherence, deliberately weak and stated as such: a decode that adopted
# garbage produces empty or degenerate output, which this catches. It cannot
# judge whether a fluent answer is the RIGHT answer, and does not claim to.
coherent() {
  local label="$1" text="$2"
  local n=${#text}
  if (( n < 8 )); then
    bad "$label: response too short to be a real completion (${n} chars)"
    return 1
  fi
  if ! printf '%s' "$text" | grep -qi '[a-z]\{3,\}'; then
    bad "$label: response has no word-like content — degenerate decode"
    return 1
  fi
  ok "$label: coherent (${n} chars)"
  return 0
}

reset_hot_cache() {
  local out
  out="$(curl -sf --max-time 30 -X POST "$BASE/v1/cache/reset" 2>/dev/null)" || {
    bad "cache reset failed — without it the hot cache serves the prefix and NOTHING below touches cold storage"
    return 1
  }
  printf '  reset: %s\n' "$out"
  return 0
}

timed_chat() {  # $1 label, $2 prompt -> sets REPLY_TEXT, prints elapsed
  local t0 t1
  t0=$(python3 -c 'import time;print(time.time())')
  REPLY_TEXT="$(chat "$2")"
  t1=$(python3 -c 'import time;print(time.time())')
  printf '  %s took %.1fs (reported, not asserted)\n' "$1" \
    "$(python3 -c "print($t1-$t0)")"
}

# --- 1. PERSIST -------------------------------------------------------------
note "1/3 PERSIST — a finished sequence must leave cold-store files on disk"
BEFORE=$(find "$SEARCH_ROOT" -type f 2>/dev/null | wc -l | tr -d ' ')
timed_chat "turn1" "$LONG_PROMPT What are the flying buttresses for?"
coherent "turn1" "$REPLY_TEXT"
sleep 2   # the v3 writer is a background thread; give it a moment to land
AFTER=$(find "$SEARCH_ROOT" -type f 2>/dev/null | wc -l | tr -d ' ')
if (( AFTER > BEFORE )); then
  ok "persist wrote $((AFTER-BEFORE)) file(s) under $SEARCH_ROOT"
else
  bad "no new files under $SEARCH_ROOT (before=$BEFORE after=$AFTER). Persist did not run: wrong COLD_DIR, cold-store disabled, or (V4=1) MLXCEL_V4_COLD_STORE not set on the server."
fi

# --- 2. SURVIVE A HOT-CACHE RESET ------------------------------------------
note "2/3 SURVIVE — reset the hot cache, then re-ask the SAME prefix"
reset_hot_cache && {
  timed_chat "turn2" "$LONG_PROMPT What are the flying buttresses for?"
  coherent "turn2-after-reset" "$REPLY_TEXT"
}

# --- 3. FAIL-CLOSED ---------------------------------------------------------
if [[ "$NO_CORRUPT" == "1" ]]; then
  note "3/3 SKIPPED (NO_CORRUPT=1) — steps 1 and 2 pass against a store that adopts anything. Nothing above discriminates."
else
  note "3/3 FAIL-CLOSED — corrupt one payload byte; the store must DECLINE it"
  if [[ "$V4" == "1" ]]; then
    # A BLOCK PAYLOAD only. header.bin, manifest.bin and .refcount are each
    # declined by a DIFFERENT guard, so corrupting one would let this pass
    # while certifying a mechanism this probe does not name.
    TARGET="$(find "$SEARCH_ROOT" -path '*/blocks/*' -type f \
        ! -name 'header.bin' ! -name 'manifest.bin' ! -name '*.refcount' \
        ! -name 'COMMITTED' 2>/dev/null | head -1)"
  else
    TARGET="$(find "$SEARCH_ROOT" -type f -name 'layer_*' 2>/dev/null | head -1)"
    [[ -z "$TARGET" ]] && TARGET="$(find "$SEARCH_ROOT" -type f \
        ! -name 'COMMITTED' ! -name 'header.bin' 2>/dev/null | head -1)"
  fi
  if [[ -z "$TARGET" ]]; then
    bad "no payload file found under $SEARCH_ROOT — ABORTING step 3 rather than corrupting an unknown file. This is a FAILURE, not a skip: the only discriminating step did not run."
  else
    printf '  flipping one byte in %s\n' "$TARGET"
    python3 - "$TARGET" <<'PY'
import sys
p = sys.argv[1]
b = bytearray(open(p, 'rb').read())
if not b:
    sys.exit("target payload is empty; nothing to corrupt")
i = min(64, len(b) - 1)
b[i] ^= 0xFF
open(p, 'wb').write(b)
PY
    reset_hot_cache && {
      timed_chat "turn3" "$LONG_PROMPT What about the colored glass windows?"
      if coherent "turn3-corrupt" "$REPLY_TEXT"; then
        ok "corrupt entry did not produce garbage — store declined it and re-prefilled (fail-closed)"
      else
        bad "DEGENERATE OUTPUT after corrupting a payload. The store adopted data it could not verify. This is the defect the checksums exist to prevent."
      fi
    }
  fi
fi

note "result"
if (( RC == 0 )); then
  printf '  ALL CHECKS PASSED (%s)\n' "$WHICH"
else
  printf '  FAILURES ABOVE (%s)\n' "$WHICH"
fi
printf '  Server was never started, stopped or restarted by this script.\n'
exit $RC
