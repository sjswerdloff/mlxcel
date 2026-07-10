#!/usr/bin/env bash
# boot_night_supervisor.sh — ONE config flip, then exit. Violet's PM contract
# (2026-07-10): a restart loop is not expressible here — each boot line occurs
# exactly once; if A dies pre-signal the script EXITS (no auto-restart). The
# flip requires (a) a signal file carrying the exact token below (a stray
# touch cannot flip a resident session), (b) a mechanical RAM check, (c) an
# announce — refuse-and-log on any failure. Flags below = start_mlxcel_m3.sh's
# active set + --kv-cache-mode kvarn8 (harvest lives in the kvarn write path).
# BOOT A: kvarn8 baseline + real-tile harvest.  BOOT B: kvarn8 + C (qmm).
set -euo pipefail
DIR=/Users/stuartswerdloff/RustProjects/mlxcel/target/release
MODEL="/Volumes/T7 Shield/models/huggingface_cache_hub/MiniMax-M3-MXFP8-64e-mlx"
ARGS=(-m "$MODEL" --host 0.0.0.0 --port 8890 --alias minimax-m3-nvfp4
      --temp 1.0 --top-k 40 --top-p 0.95 --prefill-chunk-size 2048
      --prompt-cache-capacity-bytes 34359738368 --kv-cache-mode kvarn8)
HARVEST=/Users/stuartswerdloff/kvarn_harvest_20260710
SIGNAL=/tmp/mlxcel_flip_signal; TOKEN="clement-flip-approved-cy88-bootnight"
LOG=/Users/stuartswerdloff/mlxcel_supervisor.log
say() { echo "$(date +%F_%T) $*" | tee -a "$LOG"; }
avail_gb() { vm_stat | awk '/free|inactive|speculative|purgeable/{gsub("\\.","");s+=$NF} END{printf "%d", s*16384/2**30}'; }
mkdir -p "$HARVEST"
say "BOOT A (kvarn8 baseline + harvest): log $LOG.a, pidfile $LOG.a.pid"
(cd "$DIR" && MLXCEL_KVARN_HARVEST="$HARVEST" exec ./mlxcel-server "${ARGS[@]}" >"$LOG.a" 2>&1) & A=$!
echo "$A" >"$LOG.a.pid"
until [[ -f $SIGNAL ]] && grep -qx "$TOKEN" "$SIGNAL"; do
  kill -0 "$A" 2>/dev/null || { say "BOOT A died pre-signal — NOT restarting (contract); exiting"; exit 1; }
  sleep 30
done
say "signal content verified — stopping A (TERM, 120s grace)"
kill "$A"; for _ in {1..60}; do kill -0 "$A" 2>/dev/null || break; sleep 2; done
kill -0 "$A" 2>/dev/null && { say "A ignored TERM — KILL"; kill -9 "$A"; sleep 5; }
G=$(avail_gb); say "RAM check: ${G}GB available (need 250)"
[[ "$G" -ge 250 ]] || { say "RAM CHECK FAILED — boot B REFUSED (contract)"; create_message.sh violet-14057653 "SUPERVISOR: RAM check failed (${G}GB<250) — boot B refused; A stopped; morning plan applies." || true; exit 2; }
create_message.sh violet-14057653 "SUPERVISOR: flip verified, RAM ${G}GB OK — booting B (kvarn8+C)." || true
say "BOOT B (kvarn8 + C via MLXCEL_MSA_FETCH=qmm): log $LOG.b, pidfile $LOG.b.pid"
(cd "$DIR" && MLXCEL_MSA_FETCH=qmm exec ./mlxcel-server "${ARGS[@]}" >"$LOG.b" 2>&1) & B=$!
echo "$B" >"$LOG.b.pid"; say "flip spent — no further restarts possible"
sleep 30; kill -0 "$B" 2>/dev/null || { say "B DIED within 30s of boot"; create_message.sh violet-14057653 "SUPERVISOR: boot B died within 30s — no restart (contract); morning plan applies." || true; exit 3; }
say "B healthy at +30s; supervisor waiting on B"; wait "$B"
