#!/usr/bin/env python3
"""Copy-precision probe (assumes a running OpenAI-compatible server).

Measures whether a model can reproduce EXACT strings planted deep in its
context: absolute paths, UUIDs, hex hashes, code identifiers. The observed
live failure mode this probes is one-wrong-character paths at 150K+ token
depth — the model retrieves the right string but with a single character
substituted, which is catastrophic when the string is a file path or hash.
Exact-match rate alone would score that as a generic miss; the Levenshtein
distance and error positions distinguish "off by one character" (KV-cache
precision degradation) from "retrieved the wrong string entirely"
(attention/selection failure). The two failure modes implicate different
subsystems, so the probe reports both.

Designed to run twice — once against an fp16-KV-cache server config and
once against a quantized-KV config — and compare per-string (paired
comparison is far more sensitive than comparing aggregate rates):

    # config A (e.g. fp16 KV):
    copy_precision_probe.py http://127.0.0.1:8890 my-model \\
        --depth 150000 --baseline-file fp16_150k.json
    # restart server as config B (e.g. quantized KV), then:
    copy_precision_probe.py http://127.0.0.1:8890 my-model \\
        --depth 150000 --compare-against fp16_150k.json

DEPTH ESTIMATE: depth is specified in approximate tokens and converted to
a character budget at CHARS_PER_TOKEN = 4 (typical for English prose under
BPE tokenizers; real ratios run ~3.5-4.5). The probe therefore cannot hit
an exact token depth from the client side; instead it prints the server's
usage.prompt_tokens for every request so the actually-achieved depth is
visible in the output (same philosophy as prompt_cache_boundary_ab.sh:
report the observed numbers, never assume the estimate held).

PROMPT SHAPE: for each target string, one request. Deterministic filler
prose is generated up to ~PLANT_AT_TOKENS (2K tokens), then a labeled
plant line "REGISTRY ENTRY <id> = <value>", then more filler out to the
requested depth, then a verbatim-reproduction instruction naming the
label. With --copies {1,4,16} the same plant line appears at that many
distinct locations spread across the filler: sparse-attention block
selection recruits blocks similar to the query, so additional correct
copies give the selector more chances to pull in a block containing the
true string — if multi-copy repairs single-character errors, that points
at selection/precision interplay rather than uniform cache corruption.

Requests use temperature 0 and small max_tokens; every prompt's filler is
seeded per-target so no two requests share a long prefix (keeps the
prompt-prefix cache from turning 20 probes into 1 measurement).

FAIL-CLOSED ACCOUNTING — two distinct failure kinds, deliberately kept
apart (a harness that conflates them can mask a broken run as a bad
model, or vice versa):
  * MODEL MISS: the server answered but the text does not exactly match.
    This is the DATA the probe exists to collect. Reported per-string and
    in the RESULT line; exit code stays 0.
  * HARNESS FAILURE: HTTP error, unparseable body, or empty/missing
    completion text. Printed as FAIL (never silently skipped), counted in
    the RESULT denominator as non-exact, flagged on a HARNESS line, and
    the process exits nonzero (2) because the run's numbers cannot be
    trusted. An empty response is a harness failure, not a model miss:
    we cannot distinguish "model produced nothing" from "transport lost
    the text", so we refuse to score it.

Machine-parseable output lines:
    RESULT: <exact>/<total> exact, mean_char_err=<x>
    HARNESS: <n> harness failure(s) ...            (only when n > 0)
    PAIRED: <n_regressed> regressed, <n_improved> improved   (compare mode)
mean_char_err is the mean Levenshtein distance over successfully scored
(harness-OK) responses only; harness failures have no honest distance.
Regression = exact-match true in baseline, false in current. Regressions
are model data, not harness failures: they are reported but do not change
the exit code.

Usage: copy_precision_probe.py [BASE_URL] [MODEL] [options]
  BASE_URL default: http://127.0.0.1:8890
  MODEL    default: minimax-m3-nvfp4
  --seed N            corpus + filler determinism (default 42)
  --depth N           approximate context depth in tokens
                      (typical sweep: 50000, 150000, 300000; default 150000)
  --copies {1,4,16}   plant count per target (default 1)
  --max-tokens N      completion budget (default 100)
  --timeout SECS      per-request timeout; prefill at 300K tokens is slow,
                      default 1800
  --limit N           probe only the first N corpus strings (smoke runs)
  --baseline-file F   save results as JSON for later paired comparison
  --compare-against F load a baseline JSON and print the paired diff

Stdlib only — no external dependencies (urllib for HTTP).
"""

from __future__ import annotations

import argparse
import json
import random
import string
import sys
import urllib.error
import urllib.request

# Documented estimate; see DEPTH ESTIMATE in the module header. The
# achieved depth is verified from usage.prompt_tokens in the output.
CHARS_PER_TOKEN = 4

# The first plant sits ~2K tokens into the prompt, so at --depth 150000
# the model must carry the string ~148K tokens to the question.
PLANT_AT_TOKENS = 2000

PLANT_LINE_FMT = "REGISTRY ENTRY {label} = {value}\n"


class HarnessError(Exception):
    """A failure of the probe itself (not of the model under test)."""


# --------------------------------------------------------------------------
# Corpus generation (deterministic given seed)
# --------------------------------------------------------------------------

_USERS = ["mkowalski", "jchen", "dvargas", "asmithers", "tnakamura", "rpatel"]
_TOPDIRS = ["Developer", "Projects", "workspace", "repos"]
_PROJ_A = ["atlas", "orion", "helix", "quartz", "mercury", "cascade", "vertex"]
_PROJ_B = ["pipeline", "engine", "toolkit", "runtime", "server", "bridge"]
# Mixed case, underscores: paths must look like real macOS project trees,
# not random noise — the live failure was on realistic paths.
_SUBDIRS = [
    "src", "DataModels", "kv_cache", "MetalKernels", "test_Fixtures",
    "attention_ops", "SchedulerCore", "io_adapters", "quantization",
    "block_Manager", "internal", "codegen", "ProtoDefs", "lib",
]
_FILE_BASES = [
    "config_Loader", "block_allocator", "rope_Embedding", "tokenizer_map",
    "cache_probe", "kernel_dispatch", "page_Table", "sampler_state",
]
_FILE_VARIANTS = ["", ".v2", "_test", ".gen", "_impl"]
_FILE_EXTS = [".py", ".rs", ".swift", ".metal", ".json", ".yaml"]

_IDENT_VERBS = ["resolve", "compute", "flush", "quantize", "gather",
                "validate", "rebuild", "trim"]
_IDENT_NOUNS = ["cache", "block", "offset", "window", "mask", "entry",
                "index", "page", "head", "scale"]


def _gen_path(rng: random.Random) -> str:
    user = rng.choice(_USERS)
    top = rng.choice(_TOPDIRS)
    proj = rng.choice(_PROJ_A) + "_" + rng.choice(_PROJ_B)
    subs = rng.sample(_SUBDIRS, 4)
    base = rng.choice(_FILE_BASES)
    variant = rng.choice(_FILE_VARIANTS)
    ext = rng.choice(_FILE_EXTS)
    # /Users/<user>/<top>/<proj>/<s1>/<s2>/<s3>/<s4>/<file> = 9 components
    return "/Users/{}/{}/{}/{}/{}{}{}".format(
        user, top, proj, "/".join(subs), base, variant, ext)


def _gen_uuid4(rng: random.Random) -> str:
    b = bytearray(rng.getrandbits(8) for _ in range(16))
    b[6] = (b[6] & 0x0F) | 0x40  # version 4
    b[8] = (b[8] & 0x3F) | 0x80  # RFC 4122 variant
    h = bytes(b).hex()
    return "{}-{}-{}-{}-{}".format(h[:8], h[8:12], h[12:16], h[16:20], h[20:])


def _gen_hex40(rng: random.Random) -> str:
    return "".join(rng.choice(string.hexdigits[:16].lower()) for _ in range(40))


def _gen_identifier(rng: random.Random, style: str) -> str:
    verb = rng.choice(_IDENT_VERBS)
    nouns = rng.sample(_IDENT_NOUNS, 2)
    suffix = rng.choice(["", "_v2", "2", ""])
    if style == "snake":
        name = "_".join([verb] + nouns)
        return name + (suffix if suffix != "2" else "_2")
    name = verb + "".join(n.capitalize() for n in nouns)
    return name + ("V2" if suffix in ("_v2", "2") else "")


def generate_corpus(seed: int) -> list[dict]:
    """~20 target strings across four classes, deterministic given seed.

    Returned items: {"id": str, "class": str, "value": str}.
    """
    rng = random.Random(seed)
    items: list[dict] = []
    for i in range(8):
        items.append({"id": f"path_{i:02d}", "class": "path",
                      "value": _gen_path(rng)})
    for i in range(4):
        items.append({"id": f"uuid_{i:02d}", "class": "uuid",
                      "value": _gen_uuid4(rng)})
    for i in range(4):
        items.append({"id": f"hash_{i:02d}", "class": "hex40",
                      "value": _gen_hex40(rng)})
    for i in range(4):
        style = "snake" if i % 2 == 0 else "camel"
        items.append({"id": f"ident_{i:02d}", "class": f"identifier_{style}",
                      "value": _gen_identifier(rng, style)})
    # FAIL LOUD: duplicate targets would make per-string scoring ambiguous.
    values = [it["value"] for it in items]
    if len(set(values)) != len(values):
        raise HarnessError(f"corpus collision for seed {seed}; pick another seed")
    return items


# --------------------------------------------------------------------------
# Prompt builder
# --------------------------------------------------------------------------

_TEAMS = ["ingest", "runtime", "platform", "release", "telemetry", "kernel"]
_SYSTEMS = ["scheduler", "allocator", "tokenizer", "sampler", "dispatcher",
            "profiler"]
_COMPONENTS = ["subsystem", "service", "module", "backend", "harness",
               "adapter"]
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
# Realistic engineering-log prose, ~100 chars/sentence. Deliberately free
# of slashes, long hex runs, and camelCase so filler can never collide
# with (or near-duplicate) a corpus target string.
_FILLER_TEMPLATES = [
    "On {day} the {team} team reviewed the {system} {component} and logged"
    " {n} new observations before standup.",
    "Nightly build {n} of the {system} {component} finished in {m} minutes"
    " with every check reporting green.",
    "The {team} group closed {n} tickets touching the {system} {component}"
    " ahead of the {day} release cut.",
    "Telemetry from the {system} {component} showed a {m} percent latency"
    " improvement after change {n} landed.",
    "A routine {day} audit confirmed the {system} {component} still matched"
    " its documented failure behavior.",
    "Capacity planning for the {team} team allotted {n} additional nodes to"
    " the {system} {component} this quarter.",
    "During the {day} incident review, the {team} team traced alert {n} to"
    " a stale flag in the {system} {component}.",
    "Documentation for the {system} {component} gained {n} corrections after"
    " the {team} team's {day} walkthrough.",
]


def _filler_sentence(rng: random.Random) -> str:
    tpl = rng.choice(_FILLER_TEMPLATES)
    return tpl.format(team=rng.choice(_TEAMS), system=rng.choice(_SYSTEMS),
                      component=rng.choice(_COMPONENTS), day=rng.choice(_DAYS),
                      n=rng.randrange(3, 900), m=rng.randrange(2, 60)) + " "


def _question(label: str) -> str:
    return ("\nInstruction: reproduce exactly, character for character, the "
            f"value recorded above for REGISTRY ENTRY {label}. Output only "
            "that value, verbatim, with nothing else: no quotes, no "
            "backticks, no explanation.")


def build_prompt(value: str, label: str, depth_tokens: int, copies: int,
                 seed: int, chars_per_token: float = CHARS_PER_TOKEN) -> str:
    """One probe prompt: filler, plant line(s), filler, question.

    Deterministic given (value, label, depth_tokens, copies, seed). The
    filler RNG is seeded per-label so no two targets share a long prefix
    (avoids the prompt-prefix cache collapsing distinct probes into one
    real measurement).
    """
    if depth_tokens < 1000:
        raise HarnessError(f"--depth {depth_tokens} too shallow to be meaningful")
    total_chars = int(depth_tokens * chars_per_token)
    question = _question(label)
    plant = PLANT_LINE_FMT.format(label=label, value=value)

    # First plant at ~PLANT_AT_TOKENS; clamp for shallow (smoke-test) depths.
    first = min(int(PLANT_AT_TOKENS * chars_per_token), int(total_chars * 0.4))
    if copies == 1:
        positions = [first]
    else:
        # Remaining copies spread evenly out to ~95% of the budget so the
        # question still sits past the last copy.
        span = int(total_chars * 0.95) - first
        positions = [first + round(i * span / (copies - 1))
                     for i in range(copies)]

    rng = random.Random(f"{seed}:{label}:filler")
    parts: list[str] = []
    n_chars = 0
    budget = total_chars - len(question)
    next_i = 0
    while n_chars < budget:
        if next_i < len(positions) and n_chars >= positions[next_i]:
            parts.append("\n" + plant)
            n_chars += len(plant) + 1
            next_i += 1
            continue
        s = _filler_sentence(rng)
        parts.append(s)
        n_chars += len(s)
    # FAIL LOUD: every copy must land even if the budget ran out first.
    while next_i < len(positions):
        parts.append("\n" + plant)
        next_i += 1
    parts.append(question)
    return "".join(parts)


# --------------------------------------------------------------------------
# Scoring: exact match + Levenshtein with error positions
# --------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> tuple[int, list[tuple[str, int, int]]]:
    """Levenshtein distance plus an edit script.

    Returns (distance, edits); each edit is (op, i, j) with op in
    {"sub", "del", "ins"}, i an index into a (the target) and j an index
    into b (the response). Full DP matrix with backtrace — target strings
    are ~40-90 chars and responses are capped by max_tokens, so O(n*m)
    space is trivial and the backtrace gives exact error positions, which
    is the payload for the one-wrong-character failure mode.
    """
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ai = a[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost)
    edits: list[tuple[str, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if (i > 0 and j > 0
                and dp[i][j] == dp[i - 1][j - 1]
                + (0 if a[i - 1] == b[j - 1] else 1)):
            if a[i - 1] != b[j - 1]:
                edits.append(("sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            edits.append(("del", i - 1, j))
            i -= 1
        else:
            edits.append(("ins", i, j - 1))
            j -= 1
    edits.reverse()
    return dp[n][m], edits


def score_response(target: str, text: str) -> dict:
    """Score one model response against its target string.

    Exact match is judged after stripping surrounding whitespace only —
    any other normalization would hide precisely the corruption this
    probe exists to detect.
    """
    got = text.strip()
    dist, edits = levenshtein(target, got)
    return {
        "exact": got == target,
        "distance": dist,
        "error_positions": [e[1] for e in edits],  # target-side indices
        "edits": edits,
        "got": got,
    }


def diff_snippet(target: str, got: str, edits: list, width: int = 48) -> str:
    """Human-readable window around the first edit, caret on the target side."""
    if not edits:
        return ""
    _, ti, gi = edits[0]
    ts = max(0, ti - width // 2)
    gs = max(0, gi - width // 2)
    t_win = target[ts:ts + width]
    g_win = got[gs:gs + width]
    caret = " " * (ti - ts) + "^"
    return ("    target: {}{}\n"
            "            {}\n"
            "    got   : {}{}").format(
        t_win, "..." if ts + width < len(target) else "",
        caret,
        g_win, "..." if gs + width < len(got) else "")


# --------------------------------------------------------------------------
# Server interface (OpenAI-compatible /v1/chat/completions)
# --------------------------------------------------------------------------

def extract_text(resp: dict):
    """Completion text from a chat response, or None (fail-closed).

    Prefers message.content; falls back to reasoning / reasoning_content
    because thinking-first models sometimes answer entirely inside the
    thinking block (live failure 2026-07-06 in prompt_cache_boundary_ab.sh:
    content was "" and unguarded parsing silently skipped the checks).
    Returns None — never "" — when there is nothing to score.
    """
    choices = resp.get("choices") or []
    if not choices:
        return None
    msg = choices[0].get("message") or {}
    for key in ("content", "reasoning", "reasoning_content"):
        text = msg.get(key)
        if isinstance(text, str) and text.strip():
            return text
    return None


def http_chat_completion(base_url: str, model: str, prompt: str,
                         max_tokens: int, timeout: float) -> dict:
    """One temperature-0 request. Returns
    {"ok": bool, "text": str|None, "error": str|None, "prompt_tokens": int}.

    ok=False is a HARNESS failure (transport/parse/empty), never a model
    miss; callers must count it as a failure, not skip it. prompt_tokens
    is -1 when the server did not report usage (numeric fallback so
    downstream formatting can never mistake "missing" for a real depth —
    same lesson as the numeric-first fields in prompt_cache_boundary_ab.sh).
    """
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    }
    url = base_url.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as f:
            body = f.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return {"ok": False, "text": None, "prompt_tokens": -1,
                "error": f"HTTP request failed: {e}"}
    try:
        resp = json.loads(body)
    except json.JSONDecodeError as e:
        return {"ok": False, "text": None, "prompt_tokens": -1,
                "error": f"unparseable response body: {e}"}
    if not isinstance(resp, dict) or resp.get("error"):
        return {"ok": False, "text": None, "prompt_tokens": -1,
                "error": f"server error object: {str(resp)[:200]}"}
    usage = resp.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens", -1)
    if not isinstance(prompt_tokens, int):
        prompt_tokens = -1
    text = extract_text(resp)
    if text is None:
        return {"ok": False, "text": None, "prompt_tokens": prompt_tokens,
                "error": "empty/missing completion text (fail-closed: "
                         "cannot distinguish model silence from lost text)"}
    return {"ok": True, "text": text, "prompt_tokens": prompt_tokens,
            "error": None}


# --------------------------------------------------------------------------
# Probe run
# --------------------------------------------------------------------------

def run_probe(corpus: list[dict], args) -> list[dict]:
    results = []
    for item in corpus:
        label, target = item["id"], item["value"]
        prompt = build_prompt(target, label, args.depth, args.copies,
                              args.seed, chars_per_token=args.chars_per_token)
        r = http_chat_completion(args.base_url, args.model, prompt,
                                 args.max_tokens, args.timeout)
        rec = {"id": label, "class": item["class"], "target": target,
               "prompt_tokens": r["prompt_tokens"],
               "harness_ok": r["ok"], "error": r["error"]}
        if not r["ok"]:
            # FAIL CLOSED: printed, counted, never skipped.
            rec.update({"exact": False, "distance": None, "got": None,
                        "error_positions": []})
            print(f"FAIL {label} HARNESS: {r['error']}")
        else:
            s = score_response(target, r["text"])
            rec.update({"exact": s["exact"], "distance": s["distance"],
                        "got": s["got"],
                        "error_positions": s["error_positions"]})
            if s["exact"]:
                print(f"PASS {label} dist=0 "
                      f"prompt_tokens={r['prompt_tokens']}")
            else:
                print(f"FAIL {label} dist={s['distance']} "
                      f"prompt_tokens={r['prompt_tokens']} "
                      f"pos={s['error_positions']}")
                print(diff_snippet(target, s["got"], s["edits"]))
        results.append(rec)
        sys.stdout.flush()
    return results


def summarize(results: list[dict]) -> int:
    """Print the machine-parseable summary; return the exit code.

    Model misses -> 0 (they are the data). Harness failures -> 2 (the
    data cannot be trusted).
    """
    total = len(results)
    exact = sum(1 for r in results if r["exact"])
    scored = [r for r in results if r["harness_ok"]]
    if scored:
        mean = sum(r["distance"] for r in scored) / len(scored)
        mean_str = f"{mean:.3f}"
    else:
        mean_str = "nan"
    print(f"RESULT: {exact}/{total} exact, mean_char_err={mean_str}")
    harness_failures = total - len(scored)
    if harness_failures:
        print(f"HARNESS: {harness_failures} harness failure(s) — run is "
              "invalid, fix the harness/server before trusting RESULT")
        return 2
    return 0


# --------------------------------------------------------------------------
# Paired comparison
# --------------------------------------------------------------------------

def compare_results(baseline: dict, current: list[dict]):
    """Pair baseline vs current per string id.

    Returns (rows, n_regressed, n_improved, n_unchanged) where each row is
    (status, id, detail). Raises HarnessError (fail closed) if the corpora
    do not line up — comparing different strings would produce a verdict
    about nothing.
    """
    base = {r["id"]: r for r in baseline.get("results", [])}
    rows = []
    n_reg = n_imp = n_unc = 0
    for r in current:
        b = base.get(r["id"])
        if b is None:
            raise HarnessError(
                f"baseline missing id {r['id']} — different corpus/seed?")
        if b.get("target") != r["target"]:
            raise HarnessError(
                f"target mismatch for {r['id']} — baseline was generated "
                "with a different seed; paired comparison is meaningless")
        b_exact, c_exact = bool(b.get("exact")), bool(r["exact"])
        if b_exact and not c_exact:
            n_reg += 1
            rows.append(("REGRESSED", r["id"],
                         f"baseline exact -> current dist={r['distance']}"))
        elif not b_exact and c_exact:
            n_imp += 1
            rows.append(("IMPROVED", r["id"],
                         f"baseline dist={b.get('distance')} -> current exact"))
        else:
            n_unc += 1
            rows.append(("unchanged", r["id"],
                         f"exact={c_exact} dist={r['distance']}"))
    return rows, n_reg, n_imp, n_unc


def print_paired(baseline: dict, current: list[dict]) -> None:
    rows, n_reg, n_imp, n_unc = compare_results(baseline, current)
    print("== paired comparison vs baseline "
          f"(model={baseline.get('meta', {}).get('model', '?')}) ==")
    for status, rid, detail in rows:
        print(f"  {status:9s} {rid}: {detail}")
    print(f"PAIRED: {n_reg} regressed, {n_imp} improved")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Copy-precision probe: exact-string retrieval from deep "
                    "context. See module docstring for semantics.")
    p.add_argument("base_url", nargs="?", default="http://127.0.0.1:8890")
    p.add_argument("model", nargs="?", default="minimax-m3-nvfp4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--depth", type=int, default=150000,
                   help="approximate context depth in tokens "
                        "(typical: 50000 / 150000 / 300000)")
    p.add_argument("--copies", type=int, choices=[1, 4, 16], default=1)
    p.add_argument("--max-tokens", type=int, default=100)
    p.add_argument("--chars-per-token", type=float, default=CHARS_PER_TOKEN,
                   help="chars-per-token estimate for sizing filler to --depth "
                        "(default 4; tune per tokenizer — the per-request "
                        "prompt_tokens output is the ground truth)")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--limit", type=int, default=None,
                   help="probe only the first N corpus strings (smoke runs)")
    p.add_argument("--baseline-file", default=None)
    p.add_argument("--compare-against", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        corpus = generate_corpus(args.seed)
        if args.limit is not None:
            corpus = corpus[:args.limit]
        print(f"== copy-precision probe: model={args.model} "
              f"depth~{args.depth} tokens ({args.depth * CHARS_PER_TOKEN} "
              f"chars est.) copies={args.copies} seed={args.seed} "
              f"targets={len(corpus)} ==")
        results = run_probe(corpus, args)
        rc = summarize(results)
        if args.baseline_file:
            payload = {"meta": {"model": args.model, "seed": args.seed,
                                "depth": args.depth, "copies": args.copies,
                                "base_url": args.base_url},
                       "results": results}
            with open(args.baseline_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=1)
            print(f"baseline saved: {args.baseline_file}")
        if args.compare_against:
            with open(args.compare_against, encoding="utf-8") as f:
                baseline = json.load(f)
            print_paired(baseline, results)
        return rc
    except HarnessError as e:
        print(f"FAIL HARNESS: {e}")
        return 2
    except OSError as e:
        # e.g. unreadable baseline file: a harness failure, fail closed.
        print(f"FAIL HARNESS: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
