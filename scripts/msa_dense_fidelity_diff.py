#!/usr/bin/env python3
"""Diff two msa_dense_fidelity_capture.sh outputs and classify divergence.

Usage: msa_dense_fidelity_diff.py MSA.json DENSE.json [--tie-eps 0.02]

Token-by-token comparison of two greedy captures (same fixed prompt, one
under MSA dispatch, one under dense-forced dispatch). After the first
divergent position the sequences condition on different histories, so
comparison stops there and the divergence itself is classified:

  TIE   — at the divergent position, the two chosen candidates sit within
          --tie-eps of each other in BOTH captures' top-logprob tables.
          Greedy argmax between (near-)equals is a coin flip; benign.
          (Historical baseline 2026-06-24: divergence at position 24,
          ' it' vs ' one', both at logprob -1.0 exactly in both modes.)
  GAP   — the candidates are separated by more than --tie-eps in either
          capture: the two attention paths genuinely rank tokens
          differently at that position. At depths where MSA's top-k block
          selection is inactive (context <= topk_blocks*block_size, 2048
          for production M3) this is a correctness signal worth chasing;
          at active-selection depths some divergence is by design.

Also reports position-0 distribution drift (max |delta logprob| over the
tokens present in both top-20 tables) — high-confidence tokens should
match almost exactly; extreme-tail wobble of ~1 unit is the historical
norm.

Exit codes: 0 identical or TIE; 3 GAP; 2 input problems.
"""

import argparse
import json
import sys


def load_tokens(path):
    with open(path) as f:
        data = json.load(f)
    try:
        content = data["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError):
        sys.exit(f"{path}: no choices[0].logprobs.content — captured with logprobs on?")
    if not content:
        sys.exit(f"{path}: empty logprobs content")
    return content


def top_table(entry):
    """token -> logprob from one position's top_logprobs list."""
    return {t["token"]: t["logprob"] for t in entry.get("top_logprobs", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("msa_file")
    ap.add_argument("dense_file")
    ap.add_argument(
        "--tie-eps",
        type=float,
        default=0.02,
        help="max |logprob gap| between the two divergent candidates that "
        "still counts as a tie (default 0.02; the historical tie was 0.0)",
    )
    args = ap.parse_args()

    msa = load_tokens(args.msa_file)
    dense = load_tokens(args.dense_file)
    n = min(len(msa), len(dense))
    print(f"comparing {n} positions (msa={len(msa)}, dense={len(dense)} tokens)")

    # Position-0 distribution drift over shared top-20 tokens.
    t0_msa, t0_dense = top_table(msa[0]), top_table(dense[0])
    shared = sorted(set(t0_msa) & set(t0_dense), key=lambda t: -t0_msa[t])
    if shared:
        drift = max(abs(t0_msa[t] - t0_dense[t]) for t in shared)
        print(f"position-0 distribution: {len(shared)} shared top-20 tokens, "
              f"max |delta logprob| = {drift:.4f}")
        for t in shared[:3]:
            print(f"  {t!r}: msa={t0_msa[t]:.4f} dense={t0_dense[t]:.4f}")

    for i in range(n):
        if msa[i]["token"] == dense[i]["token"]:
            continue

        m_tok, d_tok = msa[i]["token"], dense[i]["token"]
        print(f"\nDIVERGENCE at position {i}: msa={m_tok!r} dense={d_tok!r} "
              f"({i}/{n} matched before it)")

        gaps = []
        for label, table in (("msa", top_table(msa[i])), ("dense", top_table(dense[i]))):
            if m_tok in table and d_tok in table:
                gap = abs(table[m_tok] - table[d_tok])
                print(f"  {label}: {m_tok!r}={table[m_tok]:.4f} "
                      f"{d_tok!r}={table[d_tok]:.4f} |gap|={gap:.4f}")
                gaps.append(gap)
            else:
                print(f"  {label}: one candidate missing from top-20 "
                      f"(gap > table depth) — treating as GAP")
                gaps.append(float("inf"))

        if all(g <= args.tie_eps for g in gaps):
            print(f"\nVERDICT: TIE (both captures rank the candidates within "
                  f"{args.tie_eps}) — benign argmax coin flip.")
            sys.exit(0)
        print(f"\nVERDICT: GAP — the paths genuinely disagree at position {i}. "
              f"If context here is under the MSA selection threshold "
              f"(topk_blocks*block_size), investigate before trusting MSA.")
        sys.exit(3)

    print(f"\nVERDICT: IDENTICAL — {n}/{n} greedy tokens match.")
    sys.exit(0)


if __name__ == "__main__":
    main()
