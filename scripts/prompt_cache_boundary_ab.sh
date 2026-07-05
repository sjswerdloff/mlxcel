#!/usr/bin/env bash
# Top-rung prompt-cache A/B correctness probe (assumes a running server).
#
# Exercises the prompt-prefix cache adopt path against the live model at
# temperature 0 and asserts the reuse invariants that engine bugs in the
# wild have violated (vMLX 36ae3d0 poisoned-prefix aliasing, mlx-lm #975
# one-stale-token trim, SGLang #22819 block-boundary divergence):
#
#   A: cold request  X + suffix1            -> O_A   (warms prefix X)
#   B: request       X + suffix2            -> O_B   (partial-prefix hit on X)
#   C: repeat        X + suffix2            -> O_C   (full-prefix hit on B's donation)
#   D: repeat        X + suffix1            -> O_D   (full-prefix hit on A's entry)
#
#   HARD assertions:
#     O_B == O_C   adoption is deterministic and B's donated entry replays
#     O_A == O_D   B's adoption did not mutate/poison A's stored entry
#
# The shared prefix X is regenerated per run with a random tag, so the run
# is cold on any server without needing a restart. Prefix length is swept
# across several sizes to move the raw match point across KV block
# boundaries (block size 128 on the paged path; the dense path adopts at
# exact matched_len, which this sweep also exercises).
#
# The tasks are verbatim-echo prompts so greedy decoding is strongly
# determined: a mismatch means cache state, not a benign logit tie.
# Note the cold-vs-hit pair (A vs D) crosses two prefill code paths;
# a rare divergence at an exact logit tie is theoretically possible, so
# treat a lone A!=D with B==C as "investigate logits" rather than proof.
#
# Run requests sequentially on an otherwise idle server: concurrent
# batching changes kernel shapes and can perturb near-tie greedy picks.
#
# Usage: prompt_cache_boundary_ab.sh [BASE_URL] [MODEL]
#   BASE_URL default: http://127.0.0.1:8890
#   MODEL    default: minimax-m3-nvfp4

set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8890}"
MODEL="${2:-minimax-m3-nvfp4}"
MAX_TOKENS=48

command -v jq >/dev/null || { echo "jq is required" >&2; exit 2; }

TAG="cachetest-$RANDOM$RANDOM"

# One request at temperature 0. Prints "content<TAB>prompt_tokens<TAB>cached_tokens".
ask() {
  local user_content="$1"
  local body response
  body=$(jq -n \
    --arg model "$MODEL" \
    --arg content "$user_content" \
    --argjson max_tokens "$MAX_TOKENS" \
    '{model: $model, temperature: 0.0, max_tokens: $max_tokens, stream: false,
      messages: [{role: "user", content: $content}]}')
  response=$(curl -sS -X POST "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$body")
  jq -r '[(.choices[0].message.content // "NULL"),
          (.usage.prompt_tokens // -1),
          (.usage.prompt_tokens_details.cached_tokens // .usage.cached_tokens // -1)]
         | @tsv' <<<"$response"
}

# Build a shared prefix of N filler sentences, each unique via the tag so
# no previous server state can match it.
build_prefix() {
  local n="$1" i out=""
  for ((i = 1; i <= n; i++)); do
    out+="Context line $i of run $TAG holds marker value $((i * 7)). "
  done
  printf '%s' "$out"
}

FAILURES=0
check() {
  local label="$1" left="$2" right="$3"
  if [[ "$left" == "$right" ]]; then
    echo "  PASS  $label"
  else
    echo "  FAIL  $label"
    echo "    first : ${left:0:160}"
    echo "    second: ${right:0:160}"
    FAILURES=$((FAILURES + 1))
  fi
}

# Sweep prefix sizes. Sentence token counts vary with the tokenizer, so
# exact block alignment cannot be forced from bash — the sweep instead
# walks the raw match point across a >2-block span and reports the
# observed (prompt_tokens, cached_tokens) pairs for each request so
# block-boundary behavior is visible in the output.
#
# The MSA-discriminating regime needs DEPTH: M3 selects sparse_topk_blocks
# (16) blocks of sparse_block_size (128) tokens, so below 16*128 = 2048
# tokens of context every block is selected and MSA is mathematically
# dense — a shifted pooling grid cannot change a selection that keeps
# everything. Only the deep sweeps (160/280 sentences, ~2.5-5k tokens)
# put the top-k selection, the query-pooling grid, and multi-chunk
# resumed prefill (chunk 2048) in play. Override with SWEEP_SENTENCES.
#
# ALIGN (default 128) asserts every reported cached_tokens is a multiple
# of the model's prefill alignment — the client-visible proof of the
# adoption-flooring fix. Set ALIGN=0 for models without an alignment
# quantum.
ALIGN="${ALIGN:-128}"
check_aligned() {
  local label="$1" cached="$2"
  [[ "$ALIGN" -le 1 || "$cached" -le 0 ]] && return 0
  if ((cached % ALIGN == 0)); then
    echo "  PASS  $label cached=$cached is ${ALIGN}-aligned"
  else
    echo "  FAIL  $label cached=$cached NOT a multiple of $ALIGN (misaligned adoption)"
    FAILURES=$((FAILURES + 1))
  fi
}

for sentences in ${SWEEP_SENTENCES:-12 20 33 54 160 280}; do
  PREFIX="$(build_prefix "$sentences")"
  S1="After the context, repeat exactly this sentence once and stop: The quick auditor checks block $TAG-alpha."
  S2="After the context, repeat exactly this sentence once and stop: A careful engine replays prefix $TAG-beta."

  echo "== prefix sweep: $sentences sentences (tag $TAG) =="

  IFS=$'\t' read -r O_A PT_A CT_A < <(ask "$PREFIX$S1")
  echo "  A cold        prompt_tokens=$PT_A cached=$CT_A"
  IFS=$'\t' read -r O_B PT_B CT_B < <(ask "$PREFIX$S2")
  echo "  B partial-hit prompt_tokens=$PT_B cached=$CT_B"
  IFS=$'\t' read -r O_C PT_C CT_C < <(ask "$PREFIX$S2")
  echo "  C full-hit    prompt_tokens=$PT_C cached=$CT_C"
  IFS=$'\t' read -r O_D PT_D CT_D < <(ask "$PREFIX$S1")
  echo "  D full-hit    prompt_tokens=$PT_D cached=$CT_D"

  check "B == C (adopted-prefix generation replays deterministically)" "$O_B" "$O_C"
  check "A == D (stored entry survives B's adoption un-poisoned)" "$O_A" "$O_D"
  check_aligned "B" "$CT_B"
  check_aligned "C" "$CT_C"
  check_aligned "D" "$CT_D"

  if [[ "$CT_B" -le "$CT_A" && "$CT_A" -ge 0 && "$CT_B" -gt 0 ]]; then
    echo "  WARN  B's cached_tokens ($CT_B) did not exceed A's ($CT_A); prefix may not have matched as intended"
  fi
done

echo
if [[ "$FAILURES" -eq 0 ]]; then
  echo "RESULT: all prompt-cache reuse invariants held"
else
  echo "RESULT: $FAILURES invariant violation(s) — capture the server log and diff cached vs cold logits"
  exit 1
fi
