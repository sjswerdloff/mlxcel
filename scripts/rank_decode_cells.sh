#!/bin/zsh
# Four-cell decode rank runner (harness plan: fetch ∈ {kvarn8, fp16-gathered}
# × core ∈ {blocked, sdpa}) + the all-fp16 v1 reference. One command per rank
# session; every run's config is in its own RESULT line (self-labeling).
#
# Usage: rank_decode_cells.sh <bench-binary> <results-dir> [depth] [steps]
#   depth defaults 300000; steps defaults 128 (Violet's rank note: longer
#   runs + p50 emphasis — read p50 from the RESULT lines, means carry
#   ~7-9% tile-finalization outlier inflation at 32 steps).
#
# RAM note: fp16 cells at 500K are ~70-100 GB — honor the RAM protocol
# (announce) before a 500K invocation. Engine must be STOPPED or the
# announce must cover the overlap.
set -euo pipefail

BENCH=${1:?bench binary path}
OUT=${2:?results dir}
DEPTH=${3:-300000}
STEPS=${4:-128}
WARMUP=8
mkdir -p "$OUT"

run() {
  local label=$1; shift
  local envs=$1; shift
  echo "== cell: $label (depth=$DEPTH steps=$STEPS) =="
  env $envs "$BENCH" --depth "$DEPTH" --steps "$STEPS" --warmup "$WARMUP" "$@" \
    2>&1 | tee "$OUT/rank_${label}_d${DEPTH}.txt" | grep -E "BOOT|dispatch|RESULT"
  # Mis-attribution guard: an sdpa-labeled cell on a binary WITHOUT the G
  # core would silently run the blocked core (env no-op). G's helper logs
  # an INFO line when active — require it, or the capture lies.
  if [[ $envs == *"MSA_CORE=sdpa"* ]] && ! grep -q "MLXCEL_MSA_CORE=sdpa" "$OUT/rank_${label}_d${DEPTH}.txt"; then
    echo "FATAL: cell $label requested the sdpa core but the binary never announced it — wrong binary for this cell" >&2
    exit 1
  fi
}

# Cell 1: kvarn8 fetch × blocked core (today's production shape, D1 dense)
run kvarn8_blocked "" --cache-mode kvarn8
# Cell 2: kvarn8 fetch × sdpa core (G)
run kvarn8_sdpa "MLXCEL_MSA_CORE=sdpa" --cache-mode kvarn8
# Cell 3: fp16-gathered fetch × blocked core
run fp16g_blocked "" --cache-mode fp16-gathered
# Cell 4: fp16-gathered fetch × sdpa core (G)
run fp16g_sdpa "MLXCEL_MSA_CORE=sdpa" --cache-mode fp16-gathered
# Cell 5: kvarn8 × C (qmm fused fetch+core — replaces BOTH stages; the
# MSA_CORE env is irrelevant where C dispatches). Fail-loud: the env
# announce says C was REQUESTED; only the model's first-dispatch line
# proves C actually RAN — require it, or the capture lies (a gate
# fall-through would silently run the blocked core under a qmm label).
run kvarn8_qmm "MLXCEL_MSA_FETCH=qmm" --cache-mode kvarn8
if ! grep -q "C qmm-fetch fused core active" "$OUT/rank_kvarn8_qmm_d${DEPTH}.txt"; then
  echo "FATAL: cell kvarn8_qmm requested the qmm core but the model never dispatched it — wrong binary or gate fall-through" >&2
  exit 1
fi
# Reference: all-fp16 full-window flow (v1 shape)
run fp16_full "" --cache-mode fp16

echo "== rank table (mean & p50 from RESULT lines) =="
grep -h "RESULT" "$OUT"/rank_*_d${DEPTH}.txt
