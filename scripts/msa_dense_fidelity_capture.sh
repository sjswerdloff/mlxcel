#!/usr/bin/env bash
# Capture one greedy (temp 0) generation with per-token logprobs from a
# running mlxcel server, for the MSA-vs-dense fidelity regression.
#
# Protocol (server starts are the operator's):
#   1. m3_toggle_sparse.sh <MODEL_DIR> off && restart server
#      msa_dense_fidelity_capture.sh http://127.0.0.1:8896 <model> dense.json
#   2. m3_toggle_sparse.sh <MODEL_DIR> on  && restart server
#      msa_dense_fidelity_capture.sh http://127.0.0.1:8896 <model> msa.json
#   3. msa_dense_fidelity_diff.py msa.json dense.json
#
# The prompt is FIXED (no randomness) so both captures condition on
# identical tokens. The server restart between captures also clears the
# in-process prompt cache, so neither run adopts the other's KV.
#
# Historical baseline (2026-06-24, cycle 64): MSA vs dense on a fixed
# prompt produced 49/50 identical greedy tokens; the single divergence
# (position 24) was a literal tie — both candidates at logprob -1.0
# exactly in both modes. MSA drops K-blocks by design once selection is
# active, so identity is NOT the contract; the diff script classifies
# divergences by logprob gap instead (tie = benign, gap = investigate).
#
# Usage: msa_dense_fidelity_capture.sh [BASE_URL] [MODEL] [OUTFILE] [MAX_TOKENS]

set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8890}"
MODEL="${2:-minimax-m3-nvfp4}"
OUTFILE="${3:-fidelity_capture.json}"
MAX_TOKENS="${4:-50}"

command -v jq >/dev/null || { echo "jq is required" >&2; exit 2; }

# Fixed diagnostic prompt: open-ended enough that generation exercises
# real next-token uncertainty (ties can surface), specific enough to be
# deterministic at temp 0.
PROMPT="Explain, in three short paragraphs, why a key-value cache makes \
autoregressive decoding faster, and what could go wrong if the cache \
returned values for the wrong positions."

BODY=$(jq -n \
  --arg model "$MODEL" \
  --arg content "$PROMPT" \
  --argjson max_tokens "$MAX_TOKENS" \
  '{model: $model, temperature: 0.0, max_tokens: $max_tokens, stream: false,
    logprobs: true, top_logprobs: 20,
    messages: [{role: "user", content: $content}]}')

curl -sS -X POST "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' -d "$BODY" > "$OUTFILE"

TOKENS=$(jq '[.choices[0].logprobs.content[]?] | length' "$OUTFILE")
if [[ "$TOKENS" == "0" || "$TOKENS" == "null" ]]; then
  echo "WARN: no logprobs in response — check the server supports logprobs" >&2
  jq -r '.error.message? // empty' "$OUTFILE" >&2
  exit 1
fi
echo "captured $TOKENS tokens with top-20 logprobs -> $OUTFILE"
