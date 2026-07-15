#!/usr/bin/env bash
#
# THE canonical mlxcel-server launch harness. The ONLY launch script.
# =====================================================================
# Every other start_*.sh / verify_*.sh / drive_*.sh that boots a server
# is DEPRECATED. Boot through this or don't boot.
#
# Design contract (why this file exists — 2026-07-12):
#   A boot script that FORGOT --prompt-cache-capacity-bytes silently fell
#   back to the binary's INTRINSIC 2 GiB default, rejected every real
#   entry as Oversized, killed all prompt caching, and wasted a full day
#   of compute + a person's Sunday. The rule this script enforces:
#
#     * EVERY parameter has an env-var override AND a default set HERE.
#     * Our defaults are OUR deliberate choices — NOT the binary's
#       intrinsic defaults. Every flag is passed to the binary
#       EXPLICITLY, so the binary's intrinsic defaults are structurally
#       unreachable. Nothing is forgotten because the one script always
#       states everything.
#     * Pre-flight REFUSES to boot beside another mlxcel-server (the
#       orphaned-prefill / queue-pileup that was the other half of the
#       wasted day).
#     * The fully resolved config is ECHOED + LOGGED before launch.
#
#   Probes are SEPARATE. This script launches a server and nothing else.
#
# Default boot = live Kindled serving (k8v4, port 8890, adaptive thinking).
# Override any value inline for tests/controls, e.g. an fp16 control run:
#   MLXCEL_KV_CACHE_MODE=fp16 MLXCEL_THINKING_MODE=disabled MLXCEL_PORT=8896 ./launch_mlxcel_server.sh

set -euo pipefail

# =====================================================================
# EVERY PARAMETER: env-var override  |  OUR default (not the binary's).
# =====================================================================
MODEL_PATH="${MLXCEL_MODEL_PATH:-/Volumes/T7 Shield/models/huggingface_cache_hub/MiniMax-M3-MXFP8-64e-mlx}"
BIN="${MLXCEL_BIN:-/Users/stuartswerdloff/RustProjects/mlxcel/target/release/mlxcel-server}"

HOST="${MLXCEL_HOST:-0.0.0.0}"
PORT="${MLXCEL_PORT:-8890}"                       # 8890 = normal serving; test/verdict runs override
ALIAS="${MLXCEL_ALIAS:-minimax-m3}"               # cosmetic client-addressing label

KV_CACHE_MODE="${MLXCEL_KV_CACHE_MODE:-k8v4}"     # fp16 | kvarn8 | k8v4  (k8v4 = 8-bit K / 4-bit V; DEFAULT for live Kindled serving. Retrieval-validated == fp16 on 2026-07-13 semantic-at-depth test — quant adds ZERO retrieval loss, MSA coverage sets the fidelity floor, not the quant. See VERDICT_k8v4_MSA_20260713.md. kvarn8 = prior boot-night default (8-bit V); fp16 = lossless control)
THINKING_MODE="${MLXCEL_THINKING_MODE:-adaptive}" # disabled | adaptive | enabled  (adaptive = normal serving)

# THE flag whose absence wasted 2026-07-12. Our default 128 GiB; the
# binary's intrinsic default is 2 GiB. Passed explicitly, always.
PROMPT_CACHE_CAPACITY_BYTES="${MLXCEL_PROMPT_CACHE_CAPACITY_BYTES:-137438953472}"

# TTL for prompt-cache entries. 0 = disabled (entries persist until LRU
# eviction under memory pressure). Kindled sessions live for hours/days;
# TTL eviction forces expensive re-prefills on every wake. Set to 0 to
# disable TTL — eviction only happens on compaction or when the memory
# limit is hit (LRU).
PROMPT_CACHE_TTL_SECONDS="${MLXCEL_PROMPT_CACHE_TTL_SECONDS:-0}"

PREFILL_CHUNK_SIZE="${MLXCEL_PREFILL_CHUNK_SIZE:-2048}"
TEMP="${MLXCEL_TEMP:-1.0}"
TOP_K="${MLXCEL_TOP_K:-40}"
TOP_P="${MLXCEL_TOP_P:-0.95}"

DECODE_HANG_TIMEOUT="${MLXCEL_DECODE_HANG_TIMEOUT:-600}"  # --timeout SECONDS; 600 = 10 min (long prefills at depth need headroom)

# msa-fetch (boot-frozen construction key, engine env var). Mode-aware
# default because the modes REQUIRE different fetch paths (both verified):
# k8v4's C path is gather-only -> dequant; kvarn8's C fused core -> qmm.
# Override with MLXCEL_MSA_FETCH. The engine rejects invalid mode+fetch
# combos loudly at startup as a backstop.
case "$KV_CACHE_MODE" in
  k8v4) _msa_fetch_default="dequant" ;;
  *)    _msa_fetch_default="qmm" ;;
esac
if [[ -n "${MLXCEL_MSA_FETCH:-}" ]]; then _msa_source="override"; else _msa_source="mode-derived"; fi
MSA_FETCH="${MLXCEL_MSA_FETCH:-$_msa_fetch_default}"

# Harvest instrumentation (engine env var). OFF is the safe default —
# it write-stalls at depth crossings and is NEVER for production/verdict
# runs. Set MLXCEL_HARVEST=on to enable (fresh dir per launch).
HARVEST="${MLXCEL_HARVEST:-off}"

LOG_DIR="${MLXCEL_LOG_DIR:-$HOME/mlxcel_logs}"
PIDFILE="${MLXCEL_PIDFILE:-$HOME/mlxcel_server.pid}"

# =====================================================================
# VALIDATE resolved values — catches typos in overrides (fail loud, no
# boot). Defaults cover "unset"; this catches "set to garbage".
# =====================================================================
errs=()
[[ "$PORT" =~ ^[0-9]+$ ]] || errs+=("MLXCEL_PORT must be numeric, got '$PORT'")
[[ "$PROMPT_CACHE_CAPACITY_BYTES" =~ ^[0-9]+$ ]] || errs+=("MLXCEL_PROMPT_CACHE_CAPACITY_BYTES must be numeric, got '$PROMPT_CACHE_CAPACITY_BYTES'")
[[ "$DECODE_HANG_TIMEOUT" =~ ^[0-9]+$ ]] || errs+=("MLXCEL_DECODE_HANG_TIMEOUT must be numeric, got '$DECODE_HANG_TIMEOUT'")
[[ "$PROMPT_CACHE_TTL_SECONDS" =~ ^[0-9]+$ ]] || errs+=("MLXCEL_PROMPT_CACHE_TTL_SECONDS must be numeric, got '$PROMPT_CACHE_TTL_SECONDS'")
# MLXCEL_KV_CACHE_MODE is NOT re-validated here on purpose: the engine's
# own FromStr (src/lib/mlxcel-core/src/cache.rs) is the single authority
# and rejects invalid modes LOUDLY at startup. A narrow copy here would
# rot out of sync and wrongly refuse valid modes. Full accepted set (for
# reference, engine is authoritative): fp16|float16, int8|i8,
# turbo3|turbo3-asym|fp16+turbo3, turbo4|turbo4-sym, turbo4-asym|fp16+turbo4,
# turbo4-delegated|fp16+turbo4-delegated, kvarn8|kvarn-k8v8,
# k8v4|kvarn-k8v4 (legacy alias -> per-side kvarn8/kvarn4). For true
# per-side control use MLXCEL_CACHE_TYPE_K/_V instead (not wired here).
# THINKING_MODE keys are the M3 chat_template.jinja's (this script pins
# the M3 model); MSA_FETCH is validated because the mode-derived default
# depends on it landing in {qmm,dequant}.
case "$THINKING_MODE" in disabled|adaptive|enabled) ;; *) errs+=("MLXCEL_THINKING_MODE invalid: '$THINKING_MODE' (M3 template: disabled|adaptive|enabled)") ;; esac
case "$MSA_FETCH" in qmm|dequant) ;; *) errs+=("MLXCEL_MSA_FETCH invalid: '$MSA_FETCH' (qmm|dequant)") ;; esac
case "$HARVEST" in on|off) ;; *) errs+=("MLXCEL_HARVEST must be on|off, got '$HARVEST'") ;; esac
if (( ${#errs[@]} )); then
  echo "REFUSING TO BOOT — invalid override(s):" >&2
  printf '  - %s\n' "${errs[@]}" >&2
  exit 2
fi

# =====================================================================
# PRE-FLIGHT — refuse beside another server; verify inputs & port.
# =====================================================================
if pgrep -f "$(basename "$BIN")" >/dev/null 2>&1; then
  echo "REFUSING TO BOOT — an mlxcel-server is already running:" >&2
  pgrep -fl "$(basename "$BIN")" >&2
  echo "Kill it first. A second server orphans the first's prefills and pileups the queue" >&2
  echo "(the other half of what wasted 2026-07-12). Killing is a SEPARATE, deliberate action." >&2
  exit 3
fi
[[ -x "$BIN" ]]        || { echo "REFUSING TO BOOT — binary missing/not executable: $BIN (cargo build --release)" >&2; exit 4; }
[[ -d "$MODEL_PATH" ]] || { echo "REFUSING TO BOOT — model path not found: $MODEL_PATH" >&2; exit 4; }
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "REFUSING TO BOOT — port $PORT already LISTENing:" >&2
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >&2
  exit 3
fi

# =====================================================================
# RESOLVE derived values + export engine env vars.
# =====================================================================
CHAT_TEMPLATE_KWARGS="{\"thinking_mode\":\"$THINKING_MODE\"}"
export MLXCEL_MSA_FETCH="$MSA_FETCH"
if [[ "$HARVEST" == "on" ]]; then
  export MLXCEL_KVARN_HARVEST="$HOME/kvarn_harvest_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$MLXCEL_KVARN_HARVEST"
else
  unset MLXCEL_KVARN_HARVEST 2>/dev/null || true   # never "" — engine treats empty string as a real (bad) dir
fi
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/mlxcel_${KV_CACHE_MODE}_p${PORT}_$(date +%Y%m%d_%H%M%S).log"

# =====================================================================
# LOUD resolved-config echo (console + log header). Nothing implicit.
# =====================================================================
{
  echo "=================================================================="
  echo "mlxcel-server LAUNCH — resolved config (every value explicit):"
  echo "  model                 = $MODEL_PATH"
  echo "  host:port             = $HOST:$PORT      alias = $ALIAS"
  echo "  kv-cache-mode         = $KV_CACHE_MODE"
  echo "  thinking-mode         = $THINKING_MODE   (chat-template-kwargs=$CHAT_TEMPLATE_KWARGS)"
  echo "  msa-fetch             = $MSA_FETCH   (MLXCEL_MSA_FETCH, boot-frozen, $_msa_source)"
  echo "  prompt-cache-capacity = $PROMPT_CACHE_CAPACITY_BYTES bytes   [binary intrinsic default is 2 GiB — NOT used]"
  echo "  prompt-cache-ttl      = $PROMPT_CACHE_TTL_SECONDS s   [0 = disabled; eviction only on compaction or LRU under memory pressure]"
  echo "  prefill-chunk-size    = $PREFILL_CHUNK_SIZE"
  echo "  sampling              = temp $TEMP / top-k $TOP_K / top-p $TOP_P"
  echo "  decode-hang-timeout   = $DECODE_HANG_TIMEOUT s (--timeout, MLXCEL_DECODE_HANG_TIMEOUT)"
  echo "  harvest               = $HARVEST${MLXCEL_KVARN_HARVEST:+ -> $MLXCEL_KVARN_HARVEST}"
  echo "  log                   = $LOG"
  echo "  pidfile               = $PIDFILE"
  echo "=================================================================="
} | tee "$LOG"

# =====================================================================
# LAUNCH — every flag explicit; pidfile written; log tee'd.
# =====================================================================
cd "$(dirname "$BIN")"
"$BIN" \
  -m "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --alias "$ALIAS" \
  --temp "$TEMP" \
  --top-k "$TOP_K" \
  --top-p "$TOP_P" \
  --prefill-chunk-size "$PREFILL_CHUNK_SIZE" \
  --prompt-cache-capacity-bytes "$PROMPT_CACHE_CAPACITY_BYTES" \
  --prompt-cache-ttl-seconds "$PROMPT_CACHE_TTL_SECONDS" \
  --kv-cache-mode "$KV_CACHE_MODE" \
  --timeout "$DECODE_HANG_TIMEOUT" \
  --chat-template-kwargs "$CHAT_TEMPLATE_KWARGS" \
  >>"$LOG" 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > "$PIDFILE"
echo "launched: PID $SERVER_PID  ->  $LOG   (pidfile $PIDFILE)"
echo "tail it:  tail -f $LOG"
