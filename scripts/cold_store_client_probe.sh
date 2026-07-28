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
#   MAX_TOKENS  per-turn cap (default 1024)
#   NO_CORRUPT  1 = skip step 3 (leaves the store untouched; see warning above)
#
# ⚠️ COHERENCE IS JUDGED ON THE WHOLE DECODE (reasoning + answer), NOT ON THE
# ANSWER FIELD ALONE — and that is a correctness property, not a convenience.
#
# The original version read `content` only, with MAX_TOKENS=32. Against
# MiniMax-M3 under thinking_mode=adaptive the reasoning block consumed the
# entire budget, every turn returned zero content characters with
# finish_reason=length, and step 3 duly reported "the store adopted data it
# could not verify" — A SAFETY FINDING THAT WAS ENTIRELY THIS SCRIPT'S OWN
# MEASUREMENT ERROR. Measured 2026-07-28:
#   max_tokens=32    -> content 0,    reasoning 146,  finish=length
#   max_tokens=2048  -> content 0,    reasoning 9711, finish=length  (long prompt)
#   max_tokens=2048  -> content 870,  reasoning 433,  finish=stop    (short prompt)
#
# Note the middle row: RAISING THE BUDGET DID NOT FIX IT. Chasing the budget
# was chasing the wrong variable. A corrupted KV prefix produces degenerate
# tokens everywhere — the reasoning block comes out of the same forward pass as
# the answer — so lucid reasoning is exactly as good evidence of a sane decode.
# The instrument was looking at the wrong field.
#
# An empty or truncated ANSWER is therefore not a store finding in either
# direction, and the script says so on stderr rather than scoring it.

set -uo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8890}"
ALIAS="${ALIAS:-minimax-m3-mxfp8}"
COLD_DIR="${COLD_DIR:-$HOME/.cache/mlxcel/cold-storage}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
V4="${V4:-1}"
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

# THE SHARED PREFIX MUST EXCEED ONE WHOLE BLOCK (block_size = 2048 tokens).
#
# `load_prefix` matches by comparing block hashes: it runs
# `compute_block_hashes(tokens, block_size, id)` over the REQUEST's tokens and
# zips against the manifest's, taking the common prefix. A request shorter than
# one block produces a single PARTIAL chunk whose hash can never equal the
# manifest's full 2048-token block 0, so `matched_blocks == 0`, the candidate
# is dropped, and the caller sees NoMatch.
#
# At 40 repetitions the prompt was ~1527 tokens — under one block — so this
# probe COULD NOT MATCH ANYTHING BY CONSTRUCTION, and duly reported
# "both tiers MISS" three times out of three on 2026-07-28. That looked exactly
# like a broken load path. It was a probe that had never been able to test one.
#
# 600 repetitions puts the prompt at 3 blocks (block_count=3, MEASURED on the
# 2026-07-28 run) so turn 2 has two identical whole blocks to match on. 160 was
# still only ~1463 tokens — under one block. Confirm by the manifest log line
# `block_count`, not by counting characters.
LONG_PROMPT="$(python3 - <<'PY'
print("Describe the architecture of a gothic cathedral. " * 600)
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
    choice = d["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""

    # JUDGE THE WHOLE DECODE, not just the answer field.
    #
    # What this probe needs to know is whether adoption corrupted the
    # conversation. A corrupted KV prefix produces degenerate tokens
    # EVERYWHERE — the reasoning block is decoded by the same forward pass as
    # the answer, so 9k characters of lucid reasoning is exactly as good
    # evidence of a sane decode as an answer is.
    #
    # Reading `content` alone made a thinking model look broken whenever its
    # reasoning consumed the budget: on 2026-07-28 that produced a FABRICATED
    # "the store adopted data it could not verify" verdict, and raising
    # max_tokens to 2048 did not fix it (this prompt draws ~9.7k chars of
    # reasoning). The bug was never the budget — it was measuring the wrong
    # field and then inflating the budget to chase it.
    text = (reasoning + "\n" + content) if reasoning else content
    if not content and choice.get("finish_reason") == "length":
        sys.stderr.write(
            f"note: answer truncated by max_tokens={maxtok} "
            f"(reasoning={len(reasoning)} chars). Coherence is judged on the "
            f"full decode, so this is not a store finding either way.\n")
    print(text)
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

# --- adoption witness ---------------------------------------------------
#
# WITHOUT THIS, STEPS 2 AND 3 CERTIFY NOTHING. A coherent answer after a
# hot-cache reset is produced just as happily by a plain re-prefill as by a
# successful cold adoption, and a coherent answer after corrupting a block is
# produced just as happily by never loading the block at all as by a checksum
# declining it. Both greens are compatible with the cold store being entirely
# uninvolved.
#
# That is not hypothetical. On 2026-07-28 this script printed ALL CHECKS PASSED
# while the server log contained ZERO adoption lines across 16 blocks and 13
# manifests written — steps 2 and 3 had exercised nothing.
#
# The server's own log is the only place that distinguishes them:
#   "SSD cold-store has longer match"  -> a cold prefix was ADOPTED
#   "SSD cold-store probe failed"      -> a candidate was DECLINED (fail-closed)
#   "prompt-cache: SSD cold-store probe MISS" -> nothing matched (DEBUG level,
#                                                invisible at default INFO)
#
# Point LOG at the server log to make steps 2 and 3 discriminating. Without it
# they are reported as UNVERIFIED rather than passed, because "I could not
# check" must never render as "it worked".
adoption_count() {  # $1 = pattern; echoes ONE integer, or -1 if no log
  [[ -z "${LOG:-}" || ! -r "${LOG:-}" ]] && { echo -1; return; }
  # `grep -c` PRINTS 0 and EXITS 1 when there are no matches, so a trailing
  # `|| echo 0` emitted a SECOND line and this function returned "0\n0".
  # Every (( )) comparison downstream then died with a syntax error — and
  # bash's `((` failure made the else-branch run, so one step reported OK on
  # a comparison that never happened. A counter that returns two lines is a
  # counter that fabricates verdicts in BOTH directions.
  local n
  n="$(grep -ac "$1" "$LOG" 2>/dev/null)" || true
  [[ "$n" =~ ^[0-9]+$ ]] || n=0
  printf '%s\n' "$n"
}

require_adoption_since() {  # $1 label, $2 baseline count, $3 pattern, $4 meaning
  local label="$1" before="$2" pat="$3" meaning="$4" after
  after="$(adoption_count "$pat")"
  if [[ "$before" == "-1" || "$after" == "-1" ]]; then
    bad "$label: UNVERIFIED — no readable server log (set LOG=/path/to/server.log). Without it a green here is compatible with the cold store never being consulted, so this is NOT reported as a pass."
    return 1
  fi
  if (( after > before )); then
    ok "$label: witnessed in the server log — $meaning ($((after - before)) new)"
    return 0
  fi
  bad "$label: NO log evidence that $meaning. The answer was coherent, but coherence is also what a plain re-prefill produces — the cold store was very likely never consulted, so this step checked nothing."
  return 1
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
# The SUCCESS witness, not the selection one. "SSD cold-store has longer match"
# is emitted BEFORE cache_pool.adopt, so requiring it would certify that a
# candidate was CHOSEN, not that state was installed (Alden, 2026-07-28).
ADOPT_BEFORE="$(adoption_count 'SSD cold-store ADOPTED')"
reset_hot_cache && {
  timed_chat "turn2" "$LONG_PROMPT What are the flying buttresses for?"
  coherent "turn2-after-reset" "$REPLY_TEXT"
  # The coherence check above is necessary and NOT sufficient — see the
  # adoption-witness block. This is the half that makes the step mean anything.
  require_adoption_since "turn2-adoption" "$ADOPT_BEFORE" \
    'SSD cold-store ADOPTED' \
    "a cold prefix was actually ADOPTED and installed, not merely selected"
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
    DECLINE_BEFORE="$(adoption_count 'SSD cold-store probe failed')"
    # Also baselined: the corrupted request must NOT produce an adoption
    # success. Requiring the decline alone would still pass if the store both
    # declined one candidate AND adopted another.
    CORRUPT_ADOPT_BEFORE="$(adoption_count 'SSD cold-store ADOPTED')"
    reset_hot_cache && {
      timed_chat "turn3" "$LONG_PROMPT What about the colored glass windows?"
      if coherent "turn3-corrupt" "$REPLY_TEXT"; then
        # NOT a pass on its own. Coherent output here is produced equally by
        # "the checksum declined the damaged block" and by "the block was
        # never loaded at all" — and on 2026-07-28 it was the latter, with the
        # script reporting fail-closed anyway. The log is what separates them.
        require_adoption_since "turn3-declined" "$DECLINE_BEFORE" \
          'SSD cold-store probe failed' \
          "the store DECLINED the damaged entry (fail-closed) rather than never reading it"
        # AND it must not have adopted anything on this request. A decline plus
        # an adoption would mean it refused one candidate and installed
        # another — fail-closed for the wrong reason, and not what this step
        # claims to have shown.
        CORRUPT_ADOPT_AFTER="$(adoption_count 'SSD cold-store ADOPTED')"
        if [[ "$CORRUPT_ADOPT_BEFORE" == "-1" ]]; then
          : # already reported UNVERIFIED above; do not double-count
        elif (( CORRUPT_ADOPT_AFTER > CORRUPT_ADOPT_BEFORE )); then
          bad "turn3-no-adopt: the store ADOPTED state on the corrupted request ($((CORRUPT_ADOPT_AFTER - CORRUPT_ADOPT_BEFORE)) new). A decline alongside an adoption is not the fail-closed behaviour this step claims."
        else
          ok "turn3-no-adopt: no adoption witness on the corrupted request — the refusal was the outcome, not a detour"
        fi
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
