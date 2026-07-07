#!/usr/bin/env python3
"""Offline unit tests for copy_precision_probe.py — no server, no model.

Run:  python3 -m pytest scripts/test_copy_precision_probe.py

Coverage contract (every test can fail — no vacuous greens):
  * Levenshtein against hand-computed cases, including the observed live
    failure mode (one-wrong-character path) and an explicit
    scoring-mutation guard (distance("a","b") must be 1, not 0).
  * Corpus determinism given seed, plus composition/realism invariants.
  * Prompt builder: plant count == --copies, size tracks the documented
    4-chars/token estimate, determinism, no cross-target leakage.
  * Fail-closed behavior with canned empty/error HTTP responses (HTTP
    layer mocked): harness failures print FAIL, count as failures, and
    drive a nonzero exit; model misses exit 0.
  * Paired-mode regression detection with canned JSON, including the
    fail-closed corpus-mismatch path.
"""

import json
import re
import sys
import urllib.error
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import copy_precision_probe as cpp  # noqa: E402


# --------------------------------------------------------------------------
# Levenshtein
# --------------------------------------------------------------------------

def test_levenshtein_identical_is_zero():
    dist, edits = cpp.levenshtein("abcdef", "abcdef")
    assert dist == 0
    assert edits == []


def test_levenshtein_single_substitution_mutation_guard():
    # Explicit mutation guard: a broken implementation that returns 0 for
    # any equal-length pair (or that compares lengths only) must fail here.
    dist, edits = cpp.levenshtein("a", "b")
    assert dist == 1
    assert dist != 0
    assert edits == [("sub", 0, 0)]


def test_levenshtein_kitten_sitting():
    # Classic hand-computed case: kitten -> sitting = 3
    dist, _ = cpp.levenshtein("kitten", "sitting")
    assert dist == 3


def test_levenshtein_insert_and_delete():
    dist_ins, edits_ins = cpp.levenshtein("abc", "abxc")
    assert dist_ins == 1
    assert edits_ins == [("ins", 2, 2)]
    dist_del, edits_del = cpp.levenshtein("abxc", "abc")
    assert dist_del == 1
    assert edits_del == [("del", 2, 2)]


def test_levenshtein_empty_strings():
    assert cpp.levenshtein("", "abc")[0] == 3
    assert cpp.levenshtein("abc", "")[0] == 3
    assert cpp.levenshtein("", "")[0] == 0


def test_levenshtein_one_wrong_character_path():
    # The observed live failure mode: a realistic deep-context path
    # reproduced with exactly one wrong character.
    target = "/Users/jchen/Developer/atlas_pipeline/src/kv_cache/block_allocator.v2.py"
    idx = target.index("block_allocator") + len("block_")
    corrupted = target[:idx] + "e" + target[idx + 1:]  # allocator -> ellocator
    assert corrupted != target
    dist, edits = cpp.levenshtein(target, corrupted)
    assert dist == 1
    assert edits == [("sub", idx, idx)]


def test_error_positions_reported_at_target_index():
    s = cpp.score_response("abcdef", "abXdef")
    assert s["exact"] is False
    assert s["distance"] == 1
    assert s["error_positions"] == [2]


# --------------------------------------------------------------------------
# Scoring semantics
# --------------------------------------------------------------------------

def test_score_exact_after_whitespace_strip():
    s = cpp.score_response("resolve_cache_block", "  resolve_cache_block\n")
    assert s["exact"] is True
    assert s["distance"] == 0


def test_score_extra_prose_is_not_exact():
    s = cpp.score_response("deadbeef", "The value is deadbeef")
    assert s["exact"] is False
    assert s["distance"] > 0


# --------------------------------------------------------------------------
# Corpus generation
# --------------------------------------------------------------------------

def test_corpus_deterministic_given_seed():
    a = cpp.generate_corpus(42)
    b = cpp.generate_corpus(42)
    assert a == b
    c = cpp.generate_corpus(43)
    assert [i["value"] for i in c] != [i["value"] for i in a]


def test_corpus_composition_and_realism():
    corpus = cpp.generate_corpus(42)
    assert len(corpus) == 20
    by_class = {}
    for item in corpus:
        by_class.setdefault(item["class"].split("_")[0], []).append(item)
    assert len(by_class["path"]) == 8
    assert len(by_class["uuid"]) == 4
    assert len(by_class["hex40"]) == 4
    assert len(by_class["identifier"]) == 4
    # All values unique (scoring would be ambiguous otherwise).
    values = [i["value"] for i in corpus]
    assert len(set(values)) == len(values)
    for p in by_class["path"]:
        v = p["value"]
        assert v.startswith("/Users/")
        components = v.strip("/").split("/")
        assert len(components) >= 8, v
        assert "_" in v and "." in v
        assert any(c.isupper() for c in v), f"no mixed case: {v}"
    for u in by_class["uuid"]:
        assert re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
            u["value"]), u["value"]
    for h in by_class["hex40"]:
        assert re.fullmatch(r"[0-9a-f]{40}", h["value"]), h["value"]
    snake = [i for i in by_class["identifier"] if "snake" in i["class"]]
    camel = [i for i in by_class["identifier"] if "camel" in i["class"]]
    assert len(snake) == 2 and len(camel) == 2
    for i in snake:
        assert "_" in i["value"], i["value"]
        assert re.fullmatch(r"[a-z0-9_]+", i["value"]), i["value"]
    for i in camel:
        assert "_" not in i["value"]
        assert any(c.isupper() for c in i["value"]), i["value"]


# --------------------------------------------------------------------------
# Prompt builder
# --------------------------------------------------------------------------

DEPTH_SMOKE = 3000  # tokens; keeps offline test prompts ~12KB


def test_prompt_plants_exactly_copies_occurrences():
    corpus = cpp.generate_corpus(42)
    target = corpus[0]["value"]
    for copies in (1, 4, 16):
        prompt = cpp.build_prompt(target, corpus[0]["id"], DEPTH_SMOKE,
                                  copies, seed=42)
        assert prompt.count(target) == copies, copies


def test_prompt_size_tracks_documented_estimate():
    prompt = cpp.build_prompt("someTargetValue", "ident_99", DEPTH_SMOKE, 1,
                              seed=42)
    expected = DEPTH_SMOKE * cpp.CHARS_PER_TOKEN
    assert 0.9 * expected <= len(prompt) <= 1.1 * expected


def test_prompt_question_names_label_and_is_at_end():
    prompt = cpp.build_prompt("v", "hash_02", DEPTH_SMOKE, 1, seed=7)
    assert "REGISTRY ENTRY hash_02" in prompt
    # Question (naming the label) comes after the plant line.
    assert prompt.rindex("hash_02") > prompt.index("REGISTRY ENTRY hash_02 = v")


def test_prompt_deterministic_and_no_cross_target_leakage():
    corpus = cpp.generate_corpus(42)
    p1 = cpp.build_prompt(corpus[0]["value"], corpus[0]["id"], DEPTH_SMOKE,
                          4, seed=42)
    p2 = cpp.build_prompt(corpus[0]["value"], corpus[0]["id"], DEPTH_SMOKE,
                          4, seed=42)
    assert p1 == p2
    # No other corpus string may appear in this target's prompt.
    for other in corpus[1:]:
        assert other["value"] not in p1


def test_prompt_too_shallow_fails_loud():
    with pytest.raises(cpp.HarnessError):
        cpp.build_prompt("v", "x", 100, 1, seed=1)


# --------------------------------------------------------------------------
# Fail-closed HTTP behavior (mocked transport)
# --------------------------------------------------------------------------

def test_http_connection_error_is_harness_failure():
    def boom(*args, **kwargs):
        raise urllib.error.URLError("connection refused")
    with mock.patch("urllib.request.urlopen", boom):
        r = cpp.http_chat_completion("http://127.0.0.1:9", "m", "p", 10, 1)
    assert r["ok"] is False
    assert r["text"] is None
    assert "HTTP request failed" in r["error"]
    assert r["prompt_tokens"] == -1


def _canned_urlopen(body: str):
    """Context-manager stand-in for urlopen returning a fixed body."""
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return body.encode("utf-8")
    return lambda *args, **kwargs: _Resp()


def test_empty_completion_text_is_harness_failure_not_skip():
    body = json.dumps({"choices": [{"message": {"content": ""}}],
                       "usage": {"prompt_tokens": 12345}})
    with mock.patch("urllib.request.urlopen", _canned_urlopen(body)):
        r = cpp.http_chat_completion("http://x", "m", "p", 10, 1)
    assert r["ok"] is False
    assert "empty" in r["error"]
    assert r["prompt_tokens"] == 12345  # usage still reported for diagnosis


def test_reasoning_fallback_when_content_empty():
    body = json.dumps({"choices": [{"message": {
        "content": "", "reasoning": "the_answer"}}]})
    with mock.patch("urllib.request.urlopen", _canned_urlopen(body)):
        r = cpp.http_chat_completion("http://x", "m", "p", 10, 1)
    assert r["ok"] is True
    assert r["text"] == "the_answer"


def test_server_error_object_is_harness_failure():
    body = json.dumps({"error": {"message": "model not loaded"}})
    with mock.patch("urllib.request.urlopen", _canned_urlopen(body)):
        r = cpp.http_chat_completion("http://x", "m", "p", 10, 1)
    assert r["ok"] is False


def test_unparseable_body_is_harness_failure():
    with mock.patch("urllib.request.urlopen",
                    _canned_urlopen("<html>502 Bad Gateway</html>")):
        r = cpp.http_chat_completion("http://x", "m", "p", 10, 1)
    assert r["ok"] is False
    assert "unparseable" in r["error"]


# --------------------------------------------------------------------------
# End-to-end offline runs through main() (http layer mocked at module level)
# --------------------------------------------------------------------------

SMOKE_ARGS = ["http://mocked", "test-model", "--seed", "42",
              "--depth", str(DEPTH_SMOKE), "--limit", "3"]


def _echo_planted_value(base_url, model, prompt, max_tokens, timeout):
    """Perfect model: extracts the planted value from the prompt itself.

    Doubles as an end-to-end check that the plant-line format the builder
    writes is the one the question refers to.
    """
    m = re.search(r"REGISTRY ENTRY \S+ = (.+)\n", prompt)
    assert m, "plant line not found in prompt"
    return {"ok": True, "text": m.group(1), "error": None,
            "prompt_tokens": 999}


def test_all_exact_run_exits_zero(capsys):
    with mock.patch.object(cpp, "http_chat_completion", _echo_planted_value):
        rc = cpp.main(SMOKE_ARGS)
    out = capsys.readouterr().out
    assert rc == 0
    assert "RESULT: 3/3 exact, mean_char_err=0.000" in out
    assert "HARNESS" not in out


def test_model_miss_reported_but_exits_zero(capsys):
    wrong = {"ok": True, "text": "not the target", "error": None,
             "prompt_tokens": 999}
    with mock.patch.object(cpp, "http_chat_completion",
                           lambda *a, **k: dict(wrong)):
        rc = cpp.main(SMOKE_ARGS)
    out = capsys.readouterr().out
    assert rc == 0  # model misses are data, not harness failure
    assert "RESULT: 0/3 exact" in out
    assert out.count("FAIL") >= 3


def test_harness_failure_prints_fail_and_exits_nonzero(capsys):
    err = {"ok": False, "text": None, "error": "connection refused",
           "prompt_tokens": -1}
    with mock.patch.object(cpp, "http_chat_completion",
                           lambda *a, **k: dict(err)):
        rc = cpp.main(SMOKE_ARGS)
    out = capsys.readouterr().out
    assert rc != 0
    assert "FAIL" in out and "HARNESS" in out
    assert "RESULT: 0/3 exact" in out  # counted as failures, never skipped
    assert "mean_char_err=nan" in out  # nothing was honestly scorable


def test_near_miss_distance_and_diff_in_output(capsys):
    def one_char_wrong(base_url, model, prompt, max_tokens, timeout):
        m = re.search(r"REGISTRY ENTRY \S+ = (.+)\n", prompt)
        v = m.group(1)
        return {"ok": True, "text": v[:-1] + ("X" if v[-1] != "X" else "Y"),
                "error": None, "prompt_tokens": 999}
    with mock.patch.object(cpp, "http_chat_completion", one_char_wrong):
        rc = cpp.main(SMOKE_ARGS)
    out = capsys.readouterr().out
    assert rc == 0
    assert "dist=1" in out
    assert "target:" in out and "got" in out  # diff snippet printed


# --------------------------------------------------------------------------
# Paired mode
# --------------------------------------------------------------------------

def _rec(rid, target, exact, distance):
    return {"id": rid, "class": "path", "target": target, "exact": exact,
            "distance": distance, "harness_ok": True, "got": target,
            "error": None, "prompt_tokens": 1, "error_positions": []}


def test_paired_regression_and_improvement_detection():
    baseline = {"results": [_rec("a", "T1", True, 0),
                            _rec("b", "T2", False, 3),
                            _rec("c", "T3", True, 0)]}
    current = [_rec("a", "T1", False, 1),   # regressed
               _rec("b", "T2", True, 0),    # improved
               _rec("c", "T3", True, 0)]    # unchanged
    rows, n_reg, n_imp, n_unc = cpp.compare_results(baseline, current)
    assert (n_reg, n_imp, n_unc) == (1, 1, 1)
    statuses = {rid: status for status, rid, _ in rows}
    assert statuses == {"a": "REGRESSED", "b": "IMPROVED", "c": "unchanged"}


def test_paired_corpus_mismatch_fails_closed():
    baseline = {"results": [_rec("a", "T1", True, 0)]}
    current = [_rec("a", "DIFFERENT_TARGET", True, 0)]
    with pytest.raises(cpp.HarnessError):
        cpp.compare_results(baseline, current)
    with pytest.raises(cpp.HarnessError):
        cpp.compare_results({"results": []}, current)  # missing id


def test_paired_end_to_end_verdict_line(tmp_path, capsys):
    baseline_file = tmp_path / "baseline.json"
    # Run 1: perfect model, save baseline.
    with mock.patch.object(cpp, "http_chat_completion", _echo_planted_value):
        rc = cpp.main(SMOKE_ARGS + ["--baseline-file", str(baseline_file)])
    assert rc == 0
    assert baseline_file.exists()
    capsys.readouterr()

    # Run 2: model now misses everything -> every string regresses.
    wrong = {"ok": True, "text": "garbage", "error": None, "prompt_tokens": 9}
    with mock.patch.object(cpp, "http_chat_completion",
                           lambda *a, **k: dict(wrong)):
        rc = cpp.main(SMOKE_ARGS + ["--compare-against", str(baseline_file)])
    out = capsys.readouterr().out
    assert rc == 0  # regressions are model data, not harness failure
    assert "PAIRED: 3 regressed, 0 improved" in out
    assert out.count("REGRESSED") == 3


def test_paired_seed_mismatch_exits_nonzero(tmp_path, capsys):
    baseline_file = tmp_path / "baseline.json"
    with mock.patch.object(cpp, "http_chat_completion", _echo_planted_value):
        cpp.main(SMOKE_ARGS + ["--baseline-file", str(baseline_file)])
    capsys.readouterr()
    # Different seed => different targets => comparison must refuse.
    args = ["http://mocked", "test-model", "--seed", "43",
            "--depth", str(DEPTH_SMOKE), "--limit", "3",
            "--compare-against", str(baseline_file)]
    with mock.patch.object(cpp, "http_chat_completion", _echo_planted_value):
        rc = cpp.main(args)
    out = capsys.readouterr().out
    assert rc == 2
    assert "FAIL HARNESS" in out


def test_missing_baseline_file_exits_nonzero(capsys):
    with mock.patch.object(cpp, "http_chat_completion", _echo_planted_value):
        rc = cpp.main(SMOKE_ARGS + ["--compare-against",
                                    "/nonexistent/baseline.json"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "FAIL HARNESS" in out
