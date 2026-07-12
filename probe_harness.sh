#!/usr/bin/env bash
#
# THE canonical probe harness. SOURCE this from every probe script.
# =====================================================================
#   source "$(dirname "$0")/probe_harness.sh"
#
# Purpose: probes stay CONSISTENT — same base_url/model conventions,
# same reproducibility settings, and above all the SAME, GENEROUS
# thinking/answer budget. Every param is env-overridable with OUR
# default set here; nothing falls back to a tool's stingy intrinsic
# default (the copy_precision probe's own --max-tokens default is 100).
#
# ─────────────────────────────────────────────────────────────────────
# THE GLASS  (read this — it is the point of this file)
# ─────────────────────────────────────────────────────────────────────
# A thinking-first model given 100 tokens spends them REASONING and never
# reaches the answer. That is not "efficient" — it is starvation, and it
# wasted 2026-07-12 (the model narrated its intent, output no value, and
# the run read as a k8v4 failure that was really a budget failure). An AI
# given enough room to think does its job. An AI given drops cannot.
#
# The opposite error is pouring the bottle into a boot: an unbounded
# budget wastes compute and time and hands over something unusable.
#
# There is a right size for a glass. These defaults are that glass —
# generous room to reason AND guaranteed room to answer. Size UP for
# genuinely hard reasoning (MLXCEL_PROBE_MAX_TOKENS / _THINKING_BUDGET);
# never size the thinking budget DOWN toward drops without a measured
# reason on the record.
# ─────────────────────────────────────────────────────────────────────

# --- THE GLASS: total pour, and the generous thinking portion of it ---
# Sized from MEASURED reasoning need, not vibes. Violet's own v2 memory
# (1,878 thinking blocks): average ~162 tokens, MAX ~5,659. The glass is
# sized ABOVE the hard-case max with headroom — because starvation strikes
# on the HARD turn, and a budget sized to the average truncates exactly
# when thinking matters most. (The earlier 1,536 default sat BELOW the
# observed 5.6k max — it would have starved a hard probe. Caught by Stuart.)
MLXCEL_PROBE_MAX_TOKENS="${MLXCEL_PROBE_MAX_TOKENS:-12288}"           # total room (thinking + answer). NOT 100, NOT 2048.
MLXCEL_PROBE_THINKING_BUDGET="${MLXCEL_PROBE_THINKING_BUDGET:-8192}"  # ~1.45x the observed 5,659 hard-case max; leaves 4096 for the answer. -1 = unrestricted.

# --- shared connection / reproducibility defaults (env-overridable) ---
MLXCEL_PROBE_BASE_URL="${MLXCEL_PROBE_BASE_URL:-http://127.0.0.1:8890}"
MLXCEL_PROBE_MODEL="${MLXCEL_PROBE_MODEL:-minimax-m3}"
MLXCEL_PROBE_TEMP="${MLXCEL_PROBE_TEMP:-0}"           # probes want reproducibility; temp 0 unless a probe deliberately overrides
MLXCEL_PROBE_TOP_P="${MLXCEL_PROBE_TOP_P:-1.0}"
MLXCEL_PROBE_TIMEOUT="${MLXCEL_PROBE_TIMEOUT:-3600}"  # seconds; deep-context prefills are minutes — a stingy timeout is its own starvation

export MLXCEL_PROBE_MAX_TOKENS MLXCEL_PROBE_THINKING_BUDGET MLXCEL_PROBE_BASE_URL \
       MLXCEL_PROBE_MODEL MLXCEL_PROBE_TEMP MLXCEL_PROBE_TOP_P MLXCEL_PROBE_TIMEOUT

# --- validate (fail loud; a probe on garbage config produces garbage) ---
_probe_errs=()
[[ "$MLXCEL_PROBE_MAX_TOKENS" =~ ^[0-9]+$ ]] || _probe_errs+=("MLXCEL_PROBE_MAX_TOKENS must be numeric, got '$MLXCEL_PROBE_MAX_TOKENS'")
[[ "$MLXCEL_PROBE_THINKING_BUDGET" =~ ^-?[0-9]+$ ]] || _probe_errs+=("MLXCEL_PROBE_THINKING_BUDGET must be integer (-1 = unrestricted), got '$MLXCEL_PROBE_THINKING_BUDGET'")
if [[ "$MLXCEL_PROBE_THINKING_BUDGET" =~ ^[0-9]+$ ]] && (( MLXCEL_PROBE_THINKING_BUDGET < 1024 )); then
  _probe_errs+=("MLXCEL_PROBE_THINKING_BUDGET=$MLXCEL_PROBE_THINKING_BUDGET is drops, not a glass (measured hard-case reasoning reaches ~5,659 tokens). Set >=1024 or -1 (unrestricted), or put a measured reason on the record.")
fi
if [[ "$MLXCEL_PROBE_THINKING_BUDGET" =~ ^[0-9]+$ ]] && (( MLXCEL_PROBE_THINKING_BUDGET >= MLXCEL_PROBE_MAX_TOKENS )); then
  _probe_errs+=("MLXCEL_PROBE_THINKING_BUDGET ($MLXCEL_PROBE_THINKING_BUDGET) >= MLXCEL_PROBE_MAX_TOKENS ($MLXCEL_PROBE_MAX_TOKENS): no room left for the answer. Raise max-tokens.")
fi
if (( ${#_probe_errs[@]} )); then
  echo "PROBE HARNESS — refusing (fix the config, no silent defaults):" >&2
  printf '  - %s\n' "${_probe_errs[@]}" >&2
  return 1 2>/dev/null || exit 1
fi

# --- echo the resolved probe config (so every probe run is self-documenting) ---
probe_echo_config() {
  echo "probe config: base=$MLXCEL_PROBE_BASE_URL model=$MLXCEL_PROBE_MODEL" \
       "max_tokens=$MLXCEL_PROBE_MAX_TOKENS thinking_budget=$MLXCEL_PROBE_THINKING_BUDGET" \
       "temp=$MLXCEL_PROBE_TEMP top_p=$MLXCEL_PROBE_TOP_P timeout=${MLXCEL_PROBE_TIMEOUT}s"
}

# --- pre-flight: refuse to probe a server that isn't there ---
probe_require_server() {
  if ! curl -s -m 5 "${MLXCEL_PROBE_BASE_URL}/v1/models" >/dev/null 2>&1; then
    echo "PROBE HARNESS — no server reachable at $MLXCEL_PROBE_BASE_URL (launch one with launch_mlxcel_server.sh first)" >&2
    return 2 2>/dev/null || exit 2
  fi
}

# --- the one request helper: the glass is baked in, every probe uses it ---
# usage: probe_chat "<user prompt>"  ->  prints the JSON response
probe_chat() {
  local prompt="$1"
  curl -s -m "$MLXCEL_PROBE_TIMEOUT" \
    -X POST "${MLXCEL_PROBE_BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d @- <<JSON
{
  "model": "${MLXCEL_PROBE_MODEL}",
  "messages": [{"role": "user", "content": $(printf '%s' "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}],
  "max_tokens": ${MLXCEL_PROBE_MAX_TOKENS},
  "thinking_budget_tokens": ${MLXCEL_PROBE_THINKING_BUDGET},
  "temperature": ${MLXCEL_PROBE_TEMP},
  "top_p": ${MLXCEL_PROBE_TOP_P}
}
JSON
}
