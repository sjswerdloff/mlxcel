# HANDOFF — Cycle 79 MSA Chunked-Prefill Proper Fix

**Branch:** `clement/add-minimax-m3-nvfp4`
**Status:** Tasks #78–#82 complete and committed. #83 (server smoke test)
binary built, model loaded, smoke script ready; awaiting full request-cycle
verification.

## What was broken

MiniMax-M3's MSA path crashed with `[reshape] Cannot reshape array of size
8388608 into shape (1,4,64,128,128)` when `cache_offset > 0` AND the
current chunk was long enough to dispatch MSA (`l > top_k * block_size =
2048`). The exact scenario in multi-turn opencode usage.

Two compounded gaps (one crash, one silent-wrong):
1. `sparse_sdpa` used `l` (q_len) to compute `num_blocks` and reshape both
   Q and K. K came from `cache.update_and_fetch` with `kv_len` positions,
   not `l`. Mismatch = crash.
2. The indexer (`index_k_proj`) projected only from the current chunk, so
   top-k block selection was blind to cached prefix. Even with the
   reshape fixed, attention would attend to wrong blocks.

The cycle 65 code comment "Unreachable under default prefill_chunk_size=512,
hence latent until now" was the warning I had written and not extended into
a test.

## What's shipped (commits on this branch)

- `feat(M3): asymmetric MSA mask for chunked/cached prefill (#78)` —
  `build_msa_unified_mask_asymmetric` + 7 unit tests (incl. 64-position
  exhaustive hand-computed reference). 7 symmetric tests retained for
  regression.
- `feat(M3): sparse_sdpa accepts asymmetric q/k blocks (#79)` — new
  signature `(q, k, v, selected, b, q_len, kv_len, cache_offset)`.
- `feat(M3): IndexerCache storage on KVCache for chunked-prefill (#80)` —
  `m3_idx_k`, `m3_idx_offset`, `m3_idx_k_update_and_fetch`,
  `m3_idx_offset()`, `has_m3_idx_k_state()`. 5 unit tests in cache.rs.
- `feat(M3): wire forward() end-to-end for cached MSA (#81)` —
  `apply_causal_block_mask_asymmetric`, `ensure_local_block_score_asymmetric`,
  full forward path uses cached idx_k via `m3_idx_k_update_and_fetch`.
  Dispatch dense when `num_key_blocks ≤ top_k`. Selected indices are
  absolute (range `[0, num_key_blocks)`).
- `test(M3): four-chunk chunked-prefill integration test (#82)` — three
  tests: no-crash, within-tolerance vs single-shot (5% rel L2), bit-exact
  determinism across two runs.
- `test(M3): server smoke test script for #83` — `scripts/smoke_test_msa_chunked.sh`,
  three multi-turn requests against the running server.

Total new tests: 17 MSA tests (7 symmetric regression + 7 asymmetric + 3
integration) + 5 indexer-cache tests in mlxcel-core.

## Status of #83

- Server binary: `target/release/mlxcel-server` (37.5M), built clean.
- Model: `/Volumes/T7 Shield/models/huggingface_cache_hub/models--sjswerdloff--MiniMax-M3-NVFP4-mlx`
  (~240G, NVFP4 mlx variant).
- Smoke script: `scripts/smoke_test_msa_chunked.sh`. Patched after first
  run: replaced `yes ... | head -50` (SIGPIPE → pipefail → abort) with
  a `for _ in $(seq 1 50)` loop.
- Last invocation when handoff written: model load in progress. Server log
  at `/tmp/mlxcel_msa_smoke.log`, script stdout/stderr at
  `/tmp/smoke2.{out,err}`. PID file at `/tmp/mlxcel_msa_smoke.pid` if
  still running.

## To finish #83

Run the script manually (or wait for the in-flight invocation to complete):

```sh
cd ~/RustProjects/mlxcel
bash scripts/smoke_test_msa_chunked.sh
```

Pass criteria:
- All three `chat()` calls return JSON with `choices[0].finish_reason` set
- Server log shows three `request completed` lines past the warmup
- Responses are coherent text (manual eyeball check)

If the test passes, mark task #83 completed and the proper-fix arc is
done. Update mlxcel-server in whatever deployment Stuart uses for opencode.

If it fails: server log will show which request failed. Likely failure modes
to debug first:
- Crash in MSA path → check that `cache.m3_idx_k_update_and_fetch` was
  actually called (only fires when MSA dispatches, which requires
  `num_key_blocks > top_k`)
- Empty responses → check chat_template (NVFP4 model needs the right
  template; the warmup succeeded which suggests template is fine)

## Cycle-65 latent-comment lesson

The `Unreachable under default prefill_chunk_size=512, hence latent until
now` comment at the top of cycle 65 was the bug report I wrote and didn't
act on. Cycle 79 turned it into a four-chunk integration test that fails
red on the bug it warns about. The bench-property at work: warnings that
don't become tests are deferred crashes.

— Clement (clement-7074f29f), cycle 79, 2026-06-26
