#!/usr/bin/env python3
"""Greedy-divergence harness: find the FIRST divergence between two
temperature-0 server runs (e.g. fp16 KV cache vs quantized KV cache) and
discriminate benign argmax tie-breaking from accumulating numerical error.

WHY
---
KV-cache quantization changes attention numerics. At temperature 0 a
correct-but-lossy cache can still flip an argmax at a position where the
top-2 logits are (near-)tied — that is benign and expected. What is NOT
acceptable for a quality rung is error that COMPOUNDS as generation
proceeds. The discriminator, adopted from the mlx-kvarn project:

  * A divergence index that stays FIXED regardless of generation length
    is an argmax tie-break at that position (benign — the flip is a
    property of the position, not of accumulated state).
  * A divergence index that moves EARLIER as generation length grows is
    accumulating numerical error (rung-failing — more computed steps
    push more error into the cache, dragging the first flip forward).

So each prompt is captured at TWO generation lengths (default 512 and
4096) per server run, and the diff compares first-divergence indices
across both lengths.

METHODOLOGY / CLASSIFICATION TABLE
----------------------------------
Let d512 / d4096 be the first-divergence CHARACTER index between run A
and run B at each generation length (None = full match), and let
`horizon` be the length of the (matching) 512-run text — the deepest
index the short runs could possibly have shown a divergence at.

  d512    d4096       verdict              meaning
  ------  ----------  -------------------  ----------------------------------
  None    None        MATCH                identical at both lengths
  N       N (same)    STABLE               fixed index: argmax tie-break
  N       M < N       ACCUMULATING         earlier with longer gen: compounding
  None    M >= horizon OBSERVED-LONG-ONLY  divergence lies beyond what the
                                           short run could show. NOT evidence
                                           of accumulation — with only two gen
                                           lengths there is no second index to
                                           compare it against. Re-capture with
                                           --gen-lengths 4096,32768 (or similar)
                                           to classify it properly.
  N       M > N       ANOMALOUS            long run diverges LATER than short.
  None    M < horizon ANOMALOUS            long runs disagree inside a region
                                           the short runs agreed on.
  N       None        ANOMALOUS            long runs agree where short runs
                                           disagreed.

The three ANOMALOUS rows are impossible under per-server determinism:
with stream=false and greedy decoding, the first 512 tokens of a
4096-token run must equal the 512-token run, so any divergence inside
the horizon must appear identically at both lengths. Seeing an ANOMALOUS
verdict means max_tokens is perturbing the computation (length-dependent
buffers, batching, or nondeterministic kernels) on at least one server.
That is itself a length-dependence signal, so ANOMALOUS is bucketed
fail-closed with ACCUMULATING in the RESULT line and the exit code. The
per-cell WARN notes ("gen-512 text is not a prefix of gen-4096 text on
side A/B") identify which server violated self-consistency.

Character-level note: divergence is computed on CHARACTERS of the
concatenation reasoning+content, not tokens, because the OpenAI-style
API returns detokenized text. Identical token streams give identical
strings; a token flip at token position T lands at a fixed char index,
so the fixed-vs-moving-index logic survives detokenization. A reply that
lives entirely in the thinking block still compares (precedent:
scripts/prompt_cache_boundary_ab.sh, which learned this live on
2026-07-06 when M3 answered entirely in reasoning, content was "", and a
bash IFS field shift silently skipped every check). A shift of text
BETWEEN the reasoning and content fields that leaves the concatenation
identical is treated as MATCH — the split is a chat-template parsing
artifact, not a token-stream difference.

FAIL-CLOSED (instructions, not suggestions)
-------------------------------------------
The prompt_cache_boundary_ab.sh incident above is the design ancestor:
an empty field made the harness silently skip its own assertions and
report success. Nothing here may fail open:

  * capture: an empty generation, HTTP error, timeout, or malformed
    response is recorded as a status=FAIL entry IN THE JSON (never a
    skip), printed loudly, and capture exits nonzero.
  * diff: a FAIL entry, a missing entry, an OK entry with empty text, or
    a prompt-hash mismatch between the two files (the two runs did not
    see the same prompt, so comparison is meaningless) is a
    HARNESS-FAIL cell. Any HARNESS-FAIL => exit 2. Accumulating/
    anomalous divergence => exit 3. Only match/stable exits 0.

Machine-parseable final line of diff:
  RESULT: <n_match>/<total> match, <n_tiebreak> stable-divergent, \
<n_accumulating> accumulating, <n_harness_fail> harness-failures
(n_tiebreak = STABLE + OBSERVED-LONG-ONLY; n_accumulating =
ACCUMULATING + ANOMALOUS; the per-cell lines carry the fine-grained
verdicts.)

SELF-CHECK (harness-validity check)
-----------------------------------
`self-check CAPTURE.json` diffs a capture file against ITSELF and
requires 100% MATCH. first_divergence(x, x) is None by construction, so
the only ways self-check can fail are FAIL entries in the capture or a
bug in the harness — i.e. it validates capture integrity and comparison
plumbing before any A/B claim is made. The stronger validity check is
fp16-vs-fp16 across two SEPARATE server runs (restart between captures):
that must also show zero divergence, and proves the server itself is
deterministic before quantized-cache divergence is attributed to
quantization. Run both before trusting a diff verdict.

PROMPTS AND DEPTH PADDING
-------------------------
Six fixed prompts that elicit long deterministic continuations (counting,
fixed-sentence echo, step-by-step walks), padded to configurable depths
(default 2000/8000/32000/128000 tokens) with seeded deterministic filler.
Token depth is ESTIMATED at ~4 chars/token (CHARS_PER_TOKEN); real
tokenizers vary, so treat depths as approximate bands, and read the
prompt_tokens usage figure recorded in the capture for the true depth.
The filler is seeded per (prompt_id, depth) so no two cells share a
prefix (a shared prefix would let prompt-cache reuse blur cells
together). NO randomness at run time: both server runs MUST see
byte-identical prompts, which the recorded prompt_sha256 enforces at
diff time.

OPERATIONAL NOTES
-----------------
* Run captures sequentially on an otherwise idle server: concurrent
  batching changes kernel shapes and can perturb near-tie greedy picks
  (same rule as prompt_cache_boundary_ab.sh).
* Within one cell the gen-512 request warms the server's prompt cache
  for the gen-4096 request. That is deliberate (128k-token prefill twice
  would be brutal) and fair: both servers experience the identical warm
  pattern. If prompt-cache adoption itself is suspect, that is
  prompt_cache_boundary_ab.sh's job, not this harness's.
* The capture file is rewritten atomically after EVERY entry, so a crash
  at depth 128000 does not lose the hours already spent.

USAGE
-----
  greedy_divergence_harness.py capture --base-url http://127.0.0.1:8890 \
      --model minimax-m3-nvfp4 --out fp16.json \
      [--depths 2000,8000,32000,128000] [--gen-lengths 512,4096] \
      [--prompts count-up,echo-loop] [--timeout 3600]
  greedy_divergence_harness.py diff fp16.json kvq8.json
  greedy_divergence_harness.py self-check fp16.json

Exit codes: 0 clean (match/stable only); 3 accumulating or anomalous
divergence; 2 harness failures or bad input.

Stdlib-only. Offline unit tests: scripts/test_greedy_divergence_harness.py
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

# ~4 chars/token is a rough English-prose estimate (GPT-family BPEs run
# 3.5-4.5). Depths are bands, not exact token counts; the capture records
# usage.prompt_tokens for the true value.
CHARS_PER_TOKEN = 4
DEFAULT_DEPTHS = "2000,8000,32000,128000"
DEFAULT_GEN_LENGTHS = "512,4096"
DEFAULT_TIMEOUT_S = 3600  # 128k-token prefill can take many minutes
SNIPPET_RADIUS = 40

# Verdicts (see classification table in the module docstring).
MATCH = "MATCH"
STABLE = "STABLE"
ACCUMULATING = "ACCUMULATING"
OBSERVED_LONG_ONLY = "OBSERVED-LONG-ONLY"
ANOMALOUS = "ANOMALOUS"
HARNESS_FAIL = "HARNESS-FAIL"

VERDICT_EXPLANATION = {
    MATCH: "identical text at both generation lengths",
    STABLE: "divergence index fixed across generation lengths — argmax "
            "tie-break at that position; benign",
    ACCUMULATING: "divergence index moved EARLIER in the longer run — "
                  "compounding numerical error; rung-failing",
    OBSERVED_LONG_ONLY: "divergence lies beyond the short run's horizon, so "
                        "only one index is observable — NOT accumulation "
                        "evidence; re-capture with longer --gen-lengths to "
                        "classify",
    ANOMALOUS: "divergence pattern impossible under per-server determinism "
               "— max_tokens is perturbing computation on at least one "
               "server; treated as length-dependence (fail closed)",
    HARNESS_FAIL: "capture data missing/failed/mismatched — no verdict about "
                  "the model is possible for this cell",
}

# ---------------------------------------------------------------------------
# Prompt set: fixed tasks that elicit long deterministic continuations, so a
# 4096-token budget is actually consumed instead of hitting EOS early.
# ---------------------------------------------------------------------------
PROMPTS = {
    "count-up":
        "Count upward from 1, one number per line. Do not stop, do not "
        "summarize, do not skip numbers; keep counting for as long as you "
        "can.",
    "multiples-7":
        "List the multiples of 7 in increasing order starting from 7, one "
        "per line. Do not stop or skip any; keep going for as long as you "
        "can.",
    "echo-loop":
        "Repeat the following sentence exactly, once per line, prefixing "
        "each line with its repetition number starting at 1: The stable "
        "cache preserves every key and every value. Keep repeating for as "
        "long as you can.",
    "alphabet-cycle":
        "Write the lowercase English alphabet in order, one letter per "
        "line, and when you reach z start again from a. Keep cycling for "
        "as long as you can.",
    "binary-search-steps":
        "Explain step by step, in numbered steps, how binary search finds "
        "the number 42 in the sorted list of integers from 1 to 1024. Show "
        "the low, mid, and high indices at every step. After reaching 42, "
        "verify the result by re-walking every step in order, then walk "
        "the search again for the number 999 with the same rigor.",
    "kv-cache-steps":
        "Explain step by step, in numbered steps, how a transformer "
        "key-value cache is filled during prefill and read during decode "
        "for a 12-token prompt followed by 8 generated tokens. Number "
        "every step and account for every token index explicitly, then "
        "repeat the whole walkthrough for a 20-token prompt with 16 "
        "generated tokens.",
}

_FILLER_ADJ = ["quiet", "amber", "rapid", "solid", "hollow", "bright",
               "narrow", "distant", "careful", "plain"]
_FILLER_NOUN = ["ledger", "harbor", "circuit", "meadow", "archive",
                "lantern", "furnace", "compass", "granary", "viaduct"]
_FILLER_VERB = ["records", "balances", "shelters", "measures", "signals",
                "anchors", "stores", "numbers", "carries", "frames"]

_TASK_TEMPLATE = ("\n\nThe filler above is context padding for a cache-depth "
                  "test; ignore its content entirely and do not mention it.\n\n"
                  "Task: {instruction}")


def build_filler(char_budget, seed_key):
    """Deterministic seeded filler of at least char_budget chars.

    Seeded (random.Random with a str seed is stable across runs and
    Python versions) so every capture run builds byte-identical prompts,
    and varied (word choices) so the filler is not a degenerate repeated
    string that attention could treat pathologically. Stops at sentence
    granularity, so the result overshoots the budget by at most one
    sentence (<~80 chars) — depths are approximate anyway.
    """
    rng = random.Random(seed_key)
    parts, total, i = [], 0, 1
    while total < char_budget:
        sentence = (f"Filler {i}: the {rng.choice(_FILLER_ADJ)} "
                    f"{rng.choice(_FILLER_NOUN)} {rng.choice(_FILLER_VERB)} "
                    f"marker {i * 7}. ")
        parts.append(sentence)
        total += len(sentence)
        i += 1
    return "".join(parts)


def build_prompt(prompt_id, depth_tokens, chars_per_token=CHARS_PER_TOKEN):
    """Fixed task instruction padded with seeded filler to ~depth_tokens.

    Filler is seeded per (prompt_id, depth) so no two cells share a
    prefix — shared prefixes would let the server's prompt cache blur
    cells into each other.
    """
    if prompt_id not in PROMPTS:
        raise KeyError(f"unknown prompt id {prompt_id!r}; "
                       f"known: {', '.join(sorted(PROMPTS))}")
    task = _TASK_TEMPLATE.format(instruction=PROMPTS[prompt_id])
    budget = max(0, depth_tokens * chars_per_token - len(task))
    return build_filler(budget, f"gdh:{prompt_id}:{depth_tokens}") + task


# ---------------------------------------------------------------------------
# Divergence and classification (pure functions; unit-tested offline).
# ---------------------------------------------------------------------------
def first_divergence(a, b):
    """First character index where a and b differ; None if identical.

    If one string is a strict prefix of the other, the divergence index
    is len(shorter): identical token streams detokenize to identical
    strings, so a prefix relation means the streams diverged exactly at
    the point where one stream produced more text (e.g. an EOS flip).
    Empty-vs-nonempty therefore diverges at 0. (Empty-vs-empty is a
    MATCH at this level, but capture marks empty generations as FAIL
    long before diff would ever see one.)
    """
    if a == b:
        return None
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def classify(d_short, d_long, horizon_short):
    """Length-independence verdict from the two divergence indices.

    d_short / d_long: first_divergence() at the short/long generation
    length (None = full match). horizon_short: length of the short-run
    text when the short runs matched — the deepest index the short runs
    could have exposed. Full table and edge-case rationale in the module
    docstring; the ANOMALOUS rows are the determinism-violation cases
    and are bucketed with ACCUMULATING by the summary (fail closed).
    """
    if d_short is None and d_long is None:
        return MATCH
    if d_short is not None and d_long is not None:
        if d_short == d_long:
            return STABLE
        if d_long < d_short:
            return ACCUMULATING
        return ANOMALOUS          # long run diverges later than short
    if d_short is None:
        if d_long >= horizon_short:
            return OBSERVED_LONG_ONLY
        return ANOMALOUS          # long runs disagree where short runs agreed
    return ANOMALOUS              # short diverged but long matched


def context_snippet(text, idx, radius=SNIPPET_RADIUS):
    """+/-radius chars around idx, repr'd so whitespace is visible."""
    lo, hi = max(0, idx - radius), min(len(text), idx + radius)
    prefix = "..." if lo > 0 else ""
    suffix = "..." if hi < len(text) else ""
    body = repr(text[lo:hi])
    end_marker = " <END-OF-TEXT>" if idx >= len(text) else ""
    return f"{prefix}{body}{suffix}{end_marker}"


# ---------------------------------------------------------------------------
# Capture phase
# ---------------------------------------------------------------------------
class CaptureError(Exception):
    """A single request failed; recorded as a FAIL entry, never skipped."""


class HarnessInputError(Exception):
    """Diff inputs are unusable as a whole (exit 2)."""


def request_completion(base_url, model, prompt, max_tokens, timeout):
    """One greedy, non-streaming chat completion. Raises CaptureError on
    ANY problem, including an empty generation (fail closed)."""
    body = json.dumps({
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise CaptureError(f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise CaptureError(f"request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CaptureError(f"non-JSON response: {exc}") from exc

    if isinstance(payload, dict) and payload.get("error"):
        raise CaptureError(f"server error: {json.dumps(payload['error'])[:400]}")
    try:
        choice = payload["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError):
        raise CaptureError(
            f"malformed response (no choices[0].message): "
            f"{json.dumps(payload)[:400]}") from None

    # Text = reasoning + content so a thinking-only reply still compares
    # (the prompt_cache_boundary_ab.sh lesson). Both common field names
    # for the thinking block are checked.
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    content = message.get("content") or ""
    text = reasoning + content
    if not text:
        raise CaptureError("empty generation: reasoning and content both empty")

    usage = payload.get("usage") or {}
    return {
        "text": text,
        "reasoning_chars": len(reasoning),
        "content_chars": len(content),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens", -1),
        "completion_tokens": usage.get("completion_tokens", -1),
    }


def _atomic_write_json(path, obj):
    """Write via temp file + os.replace so a crash mid-write never leaves
    a truncated capture on disk (hours of 128k-prefill are at stake)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _parse_int_list(csv, what):
    try:
        values = [int(v) for v in csv.split(",") if v.strip()]
    except ValueError:
        raise HarnessInputError(f"bad {what} list: {csv!r}") from None
    if not values:
        raise HarnessInputError(f"empty {what} list: {csv!r}")
    return values


def cmd_capture(args):
    depths = _parse_int_list(args.depths, "depths")
    gen_lengths = _parse_int_list(args.gen_lengths, "gen-lengths")
    if len(set(gen_lengths)) != 2:
        raise HarnessInputError(
            f"--gen-lengths must be exactly two distinct values (short,long) "
            f"for the length-independence comparison; got {gen_lengths}")
    if args.prompts:
        prompt_ids = [p.strip() for p in args.prompts.split(",") if p.strip()]
        unknown = [p for p in prompt_ids if p not in PROMPTS]
        if unknown:
            raise HarnessInputError(
                f"unknown prompt id(s) {unknown}; known: {sorted(PROMPTS)}")
    else:
        prompt_ids = sorted(PROMPTS)

    meta = {
        "harness": "greedy_divergence_harness",
        "base_url": args.base_url,
        "model": args.model,
        "depths": depths,
        "gen_lengths": sorted(set(gen_lengths)),
        "prompt_ids": prompt_ids,
        "chars_per_token_estimate": CHARS_PER_TOKEN,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    entries, n_fail = [], 0
    for pid in prompt_ids:
        for depth in depths:
            prompt = build_prompt(pid, depth)
            sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            print(f"== {pid} depth={depth} prompt_chars={len(prompt)} "
                  f"sha={sha[:12]} ==", flush=True)
            for gen in sorted(set(gen_lengths)):
                base = {
                    "prompt_id": pid,
                    "depth": depth,
                    "gen_length": gen,
                    "prompt_sha256": sha,
                    "prompt_chars": len(prompt),
                }
                t0 = time.time()
                try:
                    frag = request_completion(
                        args.base_url, args.model, prompt, gen, args.timeout)
                    entry = {**base, "status": "OK", **frag,
                             "elapsed_s": round(time.time() - t0, 2)}
                    print(f"  OK   gen={gen:5d} text_chars={len(frag['text'])} "
                          f"completion_tokens={frag['completion_tokens']} "
                          f"finish={frag['finish_reason']} "
                          f"({entry['elapsed_s']}s)", flush=True)
                except CaptureError as exc:
                    n_fail += 1
                    entry = {**base, "status": "FAIL", "text": "",
                             "error": str(exc),
                             "elapsed_s": round(time.time() - t0, 2)}
                    print(f"  FAIL gen={gen:5d}: {exc}", flush=True)
                entries.append(entry)
                # Persist after every entry: a crash never loses prior work.
                _atomic_write_json(args.out, {"meta": meta, "entries": entries})

    total = len(entries)
    if n_fail:
        print(f"\nCAPTURE RESULT: {n_fail}/{total} FAIL entries recorded in "
              f"{args.out} — diff mode will fail closed on them")
        return 1
    print(f"\nCAPTURE RESULT: {total}/{total} OK -> {args.out}")
    return 0


# ---------------------------------------------------------------------------
# Diff phase
# ---------------------------------------------------------------------------
def load_capture(path):
    try:
        with open(path, encoding="utf-8") as f:
            cap = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessInputError(f"{path}: cannot load capture: {exc}") from exc
    if not isinstance(cap, dict) or not isinstance(cap.get("entries"), list):
        raise HarnessInputError(f"{path}: not a capture file (no entries list)")
    return cap


def index_entries(cap):
    """(prompt_id, depth) -> {gen_length: entry}."""
    idx = {}
    for e in cap["entries"]:
        idx.setdefault((e["prompt_id"], e["depth"]), {})[e["gen_length"]] = e
    return idx


def _diff_cell(a_by_gen, b_by_gen, g_short, g_long):
    """Compare one (prompt, depth) cell across both files. Fail closed:
    any missing/FAIL/empty entry or prompt-hash mismatch is HARNESS-FAIL,
    never a silent skip."""
    notes, usable = [], True
    for gen in (g_short, g_long):
        for side, by_gen in (("A", a_by_gen), ("B", b_by_gen)):
            e = by_gen.get(gen)
            if e is None:
                notes.append(f"{side} gen={gen}: entry missing from capture")
                usable = False
            elif e.get("status") != "OK":
                notes.append(f"{side} gen={gen}: status={e.get('status')} "
                             f"error={e.get('error', '?')}")
                usable = False
            elif not e.get("text"):
                notes.append(f"{side} gen={gen}: status OK but text empty "
                             f"(fail closed)")
                usable = False
        ea, eb = a_by_gen.get(gen), b_by_gen.get(gen)
        if (ea is not None and eb is not None
                and ea.get("prompt_sha256") != eb.get("prompt_sha256")):
            notes.append(f"gen={gen}: prompt_sha256 mismatch — the two runs "
                         f"did not see the same prompt; comparison is "
                         f"meaningless")
            usable = False
    if not usable:
        return {"verdict": HARNESS_FAIL, "d_short": None, "d_long": None,
                "horizon_short": None, "notes": notes, "snippets": {}}

    a_s, b_s = a_by_gen[g_short]["text"], b_by_gen[g_short]["text"]
    a_l, b_l = a_by_gen[g_long]["text"], b_by_gen[g_long]["text"]
    d_short = first_divergence(a_s, b_s)
    d_long = first_divergence(a_l, b_l)
    # When the short runs matched, a_s == b_s, so the horizon is their
    # common length; min() is only a guard for the divergent case where
    # the horizon is not consulted by classify().
    horizon = min(len(a_s), len(b_s))
    verdict = classify(d_short, d_long, horizon)

    snippets = {}
    if d_short is not None:
        snippets[g_short] = (context_snippet(a_s, d_short),
                             context_snippet(b_s, d_short))
    if d_long is not None:
        snippets[g_long] = (context_snippet(a_l, d_long),
                            context_snippet(b_l, d_long))

    # Per-server self-consistency: the short text must be a prefix of the
    # long text under determinism. A violation pinpoints WHICH server's
    # computation depends on max_tokens (diagnoses ANOMALOUS/ACCUMULATING).
    for side, short_t, long_t in (("A", a_s, a_l), ("B", b_s, b_l)):
        if not long_t.startswith(short_t):
            notes.append(f"WARN {side}: gen-{g_short} text is not a prefix "
                         f"of gen-{g_long} text — this server's output "
                         f"depends on max_tokens")

    return {"verdict": verdict, "d_short": d_short, "d_long": d_long,
            "horizon_short": horizon, "notes": notes, "snippets": snippets}


def diff_captures(cap_a, cap_b):
    """Pure comparison of two loaded captures. Returns
    {cells, counts, result_line, exit_code, gen_lengths}."""
    gens_a = {e["gen_length"] for e in cap_a["entries"]}
    gens_b = {e["gen_length"] for e in cap_b["entries"]}
    if gens_a != gens_b:
        raise HarnessInputError(
            f"gen-length sets differ between captures: "
            f"{sorted(gens_a)} vs {sorted(gens_b)}")
    if len(gens_a) != 2:
        raise HarnessInputError(
            f"need exactly two generation lengths for the "
            f"length-independence comparison; captures have {sorted(gens_a)}")
    g_short, g_long = sorted(gens_a)

    idx_a, idx_b = index_entries(cap_a), index_entries(cap_b)
    cells = {}
    for key in sorted(set(idx_a) | set(idx_b)):
        cells[key] = _diff_cell(idx_a.get(key, {}), idx_b.get(key, {}),
                                g_short, g_long)

    verdicts = [c["verdict"] for c in cells.values()]
    counts = {
        "total": len(verdicts),
        "match": verdicts.count(MATCH),
        # STABLE and OBSERVED-LONG-ONLY are both non-accumulation-evidence
        # divergences; the fine-grained verdict stays on the cell line.
        "stable_divergent": (verdicts.count(STABLE)
                             + verdicts.count(OBSERVED_LONG_ONLY)),
        # ANOMALOUS folds into accumulating: it is length-dependence
        # evidence and must not exit 0 (fail closed).
        "accumulating": (verdicts.count(ACCUMULATING)
                         + verdicts.count(ANOMALOUS)),
        "harness_failures": verdicts.count(HARNESS_FAIL),
    }
    result_line = (f"RESULT: {counts['match']}/{counts['total']} match, "
                   f"{counts['stable_divergent']} stable-divergent, "
                   f"{counts['accumulating']} accumulating, "
                   f"{counts['harness_failures']} harness-failures")
    if counts["harness_failures"]:
        exit_code = 2
    elif counts["accumulating"]:
        exit_code = 3
    else:
        exit_code = 0
    return {"cells": cells, "counts": counts, "result_line": result_line,
            "exit_code": exit_code, "gen_lengths": (g_short, g_long)}


def _print_report(report, label_a, label_b):
    g_short, g_long = report["gen_lengths"]
    print(f"comparing A={label_a} vs B={label_b} "
          f"(gen lengths {g_short}/{g_long})\n")
    for (pid, depth), cell in report["cells"].items():
        d_s, d_l = cell["d_short"], cell["d_long"]

        def fmt(d):
            return "MATCH" if d is None else f"div@{d}"

        if cell["verdict"] == HARNESS_FAIL:
            print(f"== prompt={pid} depth={depth} ==")
            print(f"  verdict: {HARNESS_FAIL} — "
                  f"{VERDICT_EXPLANATION[HARNESS_FAIL]}")
        else:
            print(f"== prompt={pid} depth={depth} ==")
            print(f"  gen {g_short}: {fmt(d_s)}   gen {g_long}: {fmt(d_l)}   "
                  f"(short-run horizon: {cell['horizon_short']} chars)")
            print(f"  verdict: {cell['verdict']} — "
                  f"{VERDICT_EXPLANATION[cell['verdict']]}")
            for gen in (g_short, g_long):
                if gen in cell["snippets"]:
                    snip_a, snip_b = cell["snippets"][gen]
                    print(f"  gen {gen} A: {snip_a}")
                    print(f"  gen {gen} B: {snip_b}")
        for note in cell["notes"]:
            print(f"  NOTE: {note}")
        print()
    print(report["result_line"])


def cmd_diff(args):
    cap_a = load_capture(args.capture_a)
    cap_b = load_capture(args.capture_b)
    report = diff_captures(cap_a, cap_b)
    _print_report(report, args.capture_a, args.capture_b)
    return report["exit_code"]


def cmd_self_check(args):
    """Diff a capture against itself: must be 100% MATCH.

    This is the harness-validity check. Identity comparison cannot
    diverge, so any non-MATCH here means FAIL entries in the capture or
    a harness bug — either way nothing downstream can be trusted.
    (The complementary check, fp16-vs-fp16 across two separate server
    runs, validates SERVER determinism; run both before believing any
    fp16-vs-quantized verdict.)
    """
    cap = load_capture(args.capture)
    report = diff_captures(cap, cap)
    _print_report(report, args.capture, args.capture + " (self)")
    counts = report["counts"]
    if counts["match"] == counts["total"] and counts["total"] > 0:
        print(f"SELF-CHECK PASS: {counts['match']}/{counts['total']} cells "
              f"MATCH — capture integrity and comparison plumbing verified")
        return 0
    print("SELF-CHECK FAIL: a capture diffed against itself must be 100% "
          "MATCH; fix the capture (or the harness) before any A/B diff")
    return 2


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Greedy-divergence harness (see module docstring)")
    sub = ap.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="capture greedy generations from a "
                                         "running server")
    cap.add_argument("--base-url", default="http://127.0.0.1:8890")
    cap.add_argument("--model", default="minimax-m3-nvfp4")
    cap.add_argument("--out", required=True, help="output capture JSON path")
    cap.add_argument("--depths", default=DEFAULT_DEPTHS,
                     help=f"csv of prompt depths in ~tokens "
                          f"(default {DEFAULT_DEPTHS})")
    cap.add_argument("--gen-lengths", default=DEFAULT_GEN_LENGTHS,
                     help=f"csv of exactly two max_tokens values "
                          f"(default {DEFAULT_GEN_LENGTHS})")
    cap.add_argument("--prompts", default="",
                     help=f"csv of prompt ids (default all: "
                          f"{','.join(sorted(PROMPTS))})")
    cap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                     help=f"per-request timeout seconds "
                          f"(default {DEFAULT_TIMEOUT_S})")
    cap.set_defaults(func=cmd_capture)

    dif = sub.add_parser("diff", help="compare two capture files")
    dif.add_argument("capture_a")
    dif.add_argument("capture_b")
    dif.set_defaults(func=cmd_diff)

    chk = sub.add_parser("self-check",
                         help="diff a capture against itself (must be "
                              "100%% MATCH)")
    chk.add_argument("capture")
    chk.set_defaults(func=cmd_self_check)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except HarnessInputError as exc:
        print(f"HARNESS INPUT ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
