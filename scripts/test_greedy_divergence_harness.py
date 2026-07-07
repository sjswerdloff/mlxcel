#!/usr/bin/env python3
"""Offline unit tests for greedy_divergence_harness.py (no server needed).

Covers, per the harness contract:
  * first_divergence(): match, early divergence, divergence at the
    prefix boundary, empty-vs-nonempty, index 0.
  * classify(): every row of the classification table, including all
    three ANOMALOUS (determinism-violation) rows and both sides of the
    OBSERVED-LONG-ONLY horizon boundary.
  * diff_captures() on canned capture pairs: verdicts, count folding,
    the machine-parseable RESULT line, and exit codes.
  * Fail-closed: FAIL entries, missing entries, empty-text OK entries,
    and prompt-hash mismatches all yield HARNESS-FAIL + exit 2, never a
    silent pass.
  * Self-diff identity: a capture diffed against itself is 100% MATCH
    (the harness-validity check) — unless it contains FAIL entries, in
    which case even the self-diff fails closed.
  * Prompt construction: determinism (both server runs MUST see
    byte-identical prompts), depth sizing, per-cell distinct prefixes.

Every test can fail; mutation notes on selected tests record the exact
harness change that turns them red (a regression test must regress).

Run:  pytest scripts/test_greedy_divergence_harness.py
  or: python3 scripts/test_greedy_divergence_harness.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import greedy_divergence_harness as gdh  # noqa: E402

# A deterministic base text long enough to slice divergence scenarios
# from. Non-repeating within a 10-char window so single-char mutations
# are always genuine first divergences.
BASE = "".join(f"line {i:04d} of the reference stream;\n" for i in range(200))
assert len(BASE) > 2000


def mutate(text, idx, ch="@"):
    """Flip one char at idx; the '@' never appears in BASE."""
    assert text[idx] != ch
    return text[:idx] + ch + text[idx + 1:]


def entry(pid, depth, gen, text, status="OK", sha="sha-fixed", error=None):
    e = {"prompt_id": pid, "depth": depth, "gen_length": gen,
         "status": status, "text": text, "prompt_sha256": sha}
    if error is not None:
        e["error"] = error
    return e


def cap(*entries):
    return {"meta": {}, "entries": list(entries)}


def cell_pair(pid, depth, a512, a4096, b512, b4096, sha_b="sha-fixed"):
    """Build (cap_a, cap_b) holding one (pid, depth) cell."""
    cap_a = cap(entry(pid, depth, 512, a512),
                entry(pid, depth, 4096, a4096))
    cap_b = cap(entry(pid, depth, 512, b512, sha=sha_b),
                entry(pid, depth, 4096, b4096, sha=sha_b))
    return cap_a, cap_b


# ---------------------------------------------------------------------------
# first_divergence
# ---------------------------------------------------------------------------
def test_first_divergence_identical_is_none():
    assert gdh.first_divergence(BASE[:500], BASE[:500]) is None


def test_first_divergence_both_empty_is_none():
    assert gdh.first_divergence("", "") is None


def test_first_divergence_early():
    # Mutation that turns this red: making first_divergence return the
    # LAST differing index (e.g. scanning from the end) instead of the
    # first — it would report an index > 37 because everything after the
    # mutation still matches only up to the different suffix lengths.
    assert gdh.first_divergence(BASE[:500], mutate(BASE[:500], 37)) == 37


def test_first_divergence_at_index_zero():
    assert gdh.first_divergence(BASE[:100], mutate(BASE[:100], 0)) == 0


def test_first_divergence_prefix_boundary():
    # One stream stopped (EOS flip), the other continued: divergence is
    # exactly at len(shorter), not a match.
    # Mutation that turns this red: `if a == b or a in b: return None`
    # (treating a prefix as a match) — the classic fail-open shortcut.
    assert gdh.first_divergence(BASE[:300], BASE[:800]) == 300
    assert gdh.first_divergence(BASE[:800], BASE[:300]) == 300


def test_first_divergence_empty_vs_nonempty():
    assert gdh.first_divergence("", BASE[:10]) == 0
    assert gdh.first_divergence(BASE[:10], "") == 0


# ---------------------------------------------------------------------------
# classify: the full table
# ---------------------------------------------------------------------------
def test_classify_match():
    assert gdh.classify(None, None, 2000) == gdh.MATCH


def test_classify_stable_same_index():
    assert gdh.classify(150, 150, 2000) == gdh.STABLE


def test_classify_stable_at_index_zero():
    # Immediate divergence at 0 in both runs is still length-independent.
    assert gdh.classify(0, 0, 2000) == gdh.STABLE


def test_classify_accumulating():
    # Divergence moved EARLIER with longer generation: compounding error.
    # Mutation that turns this red: swapping the comparison in classify()
    # from `d_long < d_short` to `d_long > d_short` (or dropping the
    # branch so everything unequal becomes ANOMALOUS/STABLE).
    assert gdh.classify(800, 300, 2000) == gdh.ACCUMULATING


def test_classify_observed_long_only_at_horizon():
    # Divergence exactly at the first char the short run never produced.
    assert gdh.classify(None, 2000, 2000) == gdh.OBSERVED_LONG_ONLY


def test_classify_observed_long_only_beyond_horizon():
    assert gdh.classify(None, 3500, 2000) == gdh.OBSERVED_LONG_ONLY


def test_classify_anomalous_long_diverges_later():
    # Long run diverging LATER than short violates prefix determinism.
    assert gdh.classify(300, 800, 2000) == gdh.ANOMALOUS


def test_classify_anomalous_inside_short_horizon():
    # Short runs matched through 2000 chars but long runs diverge at 500:
    # NOT accumulation evidence per se — determinism violation.
    assert gdh.classify(None, 500, 2000) == gdh.ANOMALOUS


def test_classify_anomalous_short_only():
    # Short runs diverged, long runs fully match: impossible under
    # determinism; must not be treated as recovered/benign.
    assert gdh.classify(400, None, 2000) == gdh.ANOMALOUS


# ---------------------------------------------------------------------------
# diff_captures on canned capture pairs
# ---------------------------------------------------------------------------
def test_diff_all_match():
    a, b = cell_pair("count-up", 2000,
                     BASE[:400], BASE[:2000], BASE[:400], BASE[:2000])
    rep = gdh.diff_captures(a, b)
    assert rep["cells"][("count-up", 2000)]["verdict"] == gdh.MATCH
    assert rep["exit_code"] == 0
    assert rep["result_line"] == (
        "RESULT: 1/1 match, 0 stable-divergent, 0 accumulating, "
        "0 harness-failures")


def test_diff_stable_tiebreak():
    a, b = cell_pair("count-up", 2000,
                     BASE[:400], BASE[:2000],
                     mutate(BASE[:400], 90), mutate(BASE[:2000], 90))
    rep = gdh.diff_captures(a, b)
    cell = rep["cells"][("count-up", 2000)]
    assert cell["verdict"] == gdh.STABLE
    assert cell["d_short"] == 90 and cell["d_long"] == 90
    assert rep["counts"]["stable_divergent"] == 1
    assert rep["exit_code"] == 0  # benign tie-break is not a failure
    # Divergence context snippets are reported for both gen lengths.
    assert 512 in cell["snippets"] and 4096 in cell["snippets"]


def test_diff_accumulating_exits_3():
    a, b = cell_pair("count-up", 8000,
                     BASE[:400], BASE[:2000],
                     mutate(BASE[:400], 350), mutate(BASE[:2000], 120))
    rep = gdh.diff_captures(a, b)
    cell = rep["cells"][("count-up", 8000)]
    assert cell["verdict"] == gdh.ACCUMULATING
    assert cell["d_short"] == 350 and cell["d_long"] == 120
    assert rep["counts"]["accumulating"] == 1
    # Mutation that turns this red: exit_code computed as 0 whenever
    # harness_failures == 0 (i.e. forgetting the accumulating branch) —
    # the rung-failing case would then silently pass in CI.
    assert rep["exit_code"] == 3
    assert rep["result_line"] == (
        "RESULT: 0/1 match, 0 stable-divergent, 1 accumulating, "
        "0 harness-failures")


def test_diff_observed_long_only_counts_as_stable_divergent():
    # 512 runs match fully (horizon 400); 4096 runs diverge at 900 which
    # the short runs could never have shown -> OBSERVED-LONG-ONLY, folded
    # into the stable-divergent bucket (NOT accumulation evidence).
    a, b = cell_pair("echo-loop", 32000,
                     BASE[:400], BASE[:2000],
                     BASE[:400], mutate(BASE[:2000], 900))
    rep = gdh.diff_captures(a, b)
    cell = rep["cells"][("echo-loop", 32000)]
    assert cell["verdict"] == gdh.OBSERVED_LONG_ONLY
    assert cell["horizon_short"] == 400
    assert rep["counts"]["stable_divergent"] == 1
    assert rep["counts"]["accumulating"] == 0
    assert rep["exit_code"] == 0


def test_diff_anomalous_folds_into_accumulating_bucket():
    # 512 runs match through 400 chars, but 4096 runs diverge at 100:
    # determinism violation -> ANOMALOUS, bucketed fail-closed with
    # accumulating so it cannot exit 0.
    a, b = cell_pair("echo-loop", 8000,
                     BASE[:400], BASE[:2000],
                     BASE[:400], mutate(BASE[:2000], 100))
    rep = gdh.diff_captures(a, b)
    cell = rep["cells"][("echo-loop", 8000)]
    assert cell["verdict"] == gdh.ANOMALOUS
    assert rep["counts"]["accumulating"] == 1
    assert rep["exit_code"] == 3
    # The prefix-consistency note should finger side B.
    assert any("WARN B" in n for n in cell["notes"])


def test_diff_fail_entry_is_harness_failure_exit_2():
    # A FAIL capture entry must surface as HARNESS-FAIL and exit 2 —
    # never a skip, never a pass (the prompt_cache_boundary_ab.sh
    # empty-field incident is the ancestor of this rule).
    cap_a = cap(entry("count-up", 2000, 512, BASE[:400]),
                entry("count-up", 2000, 4096, BASE[:2000]))
    cap_b = cap(entry("count-up", 2000, 512, "", status="FAIL",
                      error="empty generation"),
                entry("count-up", 2000, 4096, BASE[:2000]))
    rep = gdh.diff_captures(cap_a, cap_b)
    assert rep["cells"][("count-up", 2000)]["verdict"] == gdh.HARNESS_FAIL
    assert rep["counts"]["harness_failures"] == 1
    assert rep["exit_code"] == 2
    assert rep["result_line"].endswith("1 harness-failures")


def test_diff_ok_but_empty_text_fails_closed():
    # Defense in depth: even if capture mislabeled an empty generation
    # as OK, diff must not compare it as if empty == empty were a MATCH.
    cap_a = cap(entry("count-up", 2000, 512, ""),
                entry("count-up", 2000, 4096, BASE[:2000]))
    cap_b = cap(entry("count-up", 2000, 512, ""),
                entry("count-up", 2000, 4096, BASE[:2000]))
    rep = gdh.diff_captures(cap_a, cap_b)
    assert rep["cells"][("count-up", 2000)]["verdict"] == gdh.HARNESS_FAIL
    assert rep["exit_code"] == 2


def test_diff_missing_cell_entry_is_harness_failure():
    # B is missing the gen-4096 entry for one cell (while another cell
    # keeps the gen-length sets equal): that cell fails closed.
    cap_a = cap(entry("count-up", 2000, 512, BASE[:400]),
                entry("count-up", 2000, 4096, BASE[:2000]),
                entry("echo-loop", 2000, 512, BASE[:400]),
                entry("echo-loop", 2000, 4096, BASE[:2000]))
    cap_b = cap(entry("count-up", 2000, 512, BASE[:400]),
                entry("count-up", 2000, 4096, BASE[:2000]),
                entry("echo-loop", 2000, 512, BASE[:400]))
    rep = gdh.diff_captures(cap_a, cap_b)
    assert rep["cells"][("echo-loop", 2000)]["verdict"] == gdh.HARNESS_FAIL
    assert rep["cells"][("count-up", 2000)]["verdict"] == gdh.MATCH
    assert rep["exit_code"] == 2


def test_diff_prompt_sha_mismatch_is_harness_failure():
    # Same texts, different prompts recorded: the runs are not comparable.
    a, b = cell_pair("count-up", 2000,
                     BASE[:400], BASE[:2000], BASE[:400], BASE[:2000],
                     sha_b="sha-OTHER")
    rep = gdh.diff_captures(a, b)
    assert rep["cells"][("count-up", 2000)]["verdict"] == gdh.HARNESS_FAIL
    assert rep["exit_code"] == 2


def test_diff_gen_length_set_mismatch_raises():
    cap_a = cap(entry("count-up", 2000, 512, BASE[:400]),
                entry("count-up", 2000, 4096, BASE[:2000]))
    cap_b = cap(entry("count-up", 2000, 512, BASE[:400]))
    try:
        gdh.diff_captures(cap_a, cap_b)
    except gdh.HarnessInputError:
        pass
    else:
        raise AssertionError("gen-length set mismatch must raise "
                             "HarnessInputError (exit 2), not compare")


def test_diff_single_gen_length_raises():
    cap_a = cap(entry("count-up", 2000, 512, BASE[:400]))
    cap_b = cap(entry("count-up", 2000, 512, BASE[:400]))
    try:
        gdh.diff_captures(cap_a, cap_b)
    except gdh.HarnessInputError:
        pass
    else:
        raise AssertionError("a single gen length cannot support the "
                             "length-independence verdict; must raise")


def test_diff_multiple_cells_counts_and_result_line():
    a1, b1 = cell_pair("count-up", 2000,
                       BASE[:400], BASE[:2000], BASE[:400], BASE[:2000])
    a2, b2 = cell_pair("echo-loop", 8000,
                       BASE[:400], BASE[:2000],
                       mutate(BASE[:400], 50), mutate(BASE[:2000], 50))
    a3, b3 = cell_pair("multiples-7", 32000,
                       BASE[:400], BASE[:2000],
                       mutate(BASE[:400], 300), mutate(BASE[:2000], 30))
    cap_a = cap(*(a1["entries"] + a2["entries"] + a3["entries"]))
    cap_b = cap(*(b1["entries"] + b2["entries"] + b3["entries"]))
    rep = gdh.diff_captures(cap_a, cap_b)
    assert rep["counts"] == {"total": 3, "match": 1, "stable_divergent": 1,
                             "accumulating": 1, "harness_failures": 0}
    assert rep["result_line"] == (
        "RESULT: 1/3 match, 1 stable-divergent, 1 accumulating, "
        "0 harness-failures")
    assert rep["exit_code"] == 3


# ---------------------------------------------------------------------------
# Self-diff identity (the harness-validity check)
# ---------------------------------------------------------------------------
def test_self_diff_is_all_match():
    cap_a = cap(entry("count-up", 2000, 512, BASE[:400]),
                entry("count-up", 2000, 4096, BASE[:2000]),
                entry("echo-loop", 128000, 512, BASE[:333]),
                entry("echo-loop", 128000, 4096, BASE[:1777]))
    rep = gdh.diff_captures(cap_a, cap_a)
    assert all(c["verdict"] == gdh.MATCH for c in rep["cells"].values())
    assert rep["counts"]["match"] == rep["counts"]["total"] == 2
    assert rep["exit_code"] == 0


def test_self_diff_with_fail_entry_still_fails_closed():
    # Self-check must NOT pass a capture containing FAIL entries: 100%
    # MATCH means all cells usable AND identical, not "identical where
    # convenient".
    cap_a = cap(entry("count-up", 2000, 512, "", status="FAIL",
                      error="timeout"),
                entry("count-up", 2000, 4096, BASE[:2000]))
    rep = gdh.diff_captures(cap_a, cap_a)
    assert rep["counts"]["harness_failures"] == 1
    assert rep["counts"]["match"] < rep["counts"]["total"]
    assert rep["exit_code"] == 2


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def test_build_prompt_is_deterministic():
    # Both server runs MUST see byte-identical prompts; any run-time
    # randomness (like prompt_cache_boundary_ab.sh's $RANDOM tag, correct
    # THERE, wrong HERE) would make every diff meaningless.
    # Mutation that turns this red: seeding build_filler from
    # time/os.urandom instead of the fixed (prompt_id, depth) key.
    assert (gdh.build_prompt("count-up", 2000)
            == gdh.build_prompt("count-up", 2000))
    assert (gdh.build_prompt("kv-cache-steps", 128000)
            == gdh.build_prompt("kv-cache-steps", 128000))


def test_build_prompt_depth_sizing():
    for depth in (2000, 8000, 32000, 128000):
        p = gdh.build_prompt("count-up", depth)
        target = depth * gdh.CHARS_PER_TOKEN
        # Sentence-granular overshoot only (< ~80 chars + slack).
        assert target <= len(p) <= target + 120, (depth, len(p))


def test_build_prompt_cells_do_not_share_prefixes():
    # Distinct filler per (prompt_id, depth) so the server prompt cache
    # cannot blur cells together via a common prefix.
    p_a = gdh.build_prompt("count-up", 2000)
    p_b = gdh.build_prompt("echo-loop", 2000)
    p_c = gdh.build_prompt("count-up", 8000)
    assert p_a[:200] != p_b[:200]
    assert p_a[:200] != p_c[:200]


def test_build_prompt_contains_task_instruction():
    p = gdh.build_prompt("count-up", 2000)
    assert gdh.PROMPTS["count-up"] in p
    assert p.endswith(gdh.PROMPTS["count-up"])


def test_build_prompt_unknown_id_raises():
    try:
        gdh.build_prompt("no-such-prompt", 2000)
    except KeyError:
        pass
    else:
        raise AssertionError("unknown prompt id must raise, not pad garbage")


def test_prompt_set_has_six_prompts():
    assert len(gdh.PROMPTS) == 6


# ---------------------------------------------------------------------------
# Plain-python runner fallback (pytest preferred but not required)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    failures = 0
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    sys.exit(1 if failures else 0)
