# HANDOFF — Cycle 79 MSA Chunked-Prefill Proper Fix

**Branch:** `clement/add-minimax-m3-nvfp4`
**Status:** Tasks #78–#83 complete. Three follow-up server-side fixes
(`<mm:think>` extraction, cache lookup diagnostics, indexer-cache desync
fallback) landed in commits `8b3ea00`, `316a997`, `c755e45`.

## What was broken (the cycle-79 root)

MiniMax-M3's MSA path crashed with `[reshape] Cannot reshape array of
size 8388608 into shape (1,4,64,128,128)` when `cache_offset > 0` AND
the current chunk dispatched MSA (`l > top_k * block_size = 2048`).
Multi-turn opencode usage.

Two compounded gaps:
1. `sparse_sdpa` used `l` (q_len) where it needed `kv_len` for the K-side
   reshape.
2. The indexer was blind to the cached prefix — `index_k_proj` projected
   only from the current chunk.

## What's shipped (cycle 79 main arc)

- `feat(M3): asymmetric MSA mask for chunked/cached prefill (#78)`
- `feat(M3): sparse_sdpa accepts asymmetric q/k blocks (#79)`
- `feat(M3): IndexerCache storage on KVCache for chunked-prefill (#80)`
- `feat(M3): wire forward() end-to-end for cached MSA (#81)`
- `test(M3): four-chunk chunked-prefill integration test (#82)`
- `test(M3): server smoke test script for #83`

22 new tests pass; smoke test confirmed multi-turn opencode-style
requests work on the live server.

## Server-side follow-ups (cycle 79 trailer)

### A. M3 thinking-tag extraction (`8b3ea00`)
Added `<mm:think>` / `</mm:think>` to `CHAT_DELIMITERS` in
`stream_filter.rs:202`. Generalised `strip_think_block` in
`chat_template_kwargs.rs:431` to a TAG_FAMILIES table. Result:
`reasoning_content` populates correctly on both OpenAI and Anthropic
routes (Anthropic API is at `POST /v1/messages` — Claude Code reaches
it via `ANTHROPIC_BASE_URL=http://127.0.0.1:8890`).

### B. Cache-lookup diagnostic logging (`8b3ea00`, `316a997`)
Lookup and insert paths now log:
- `total_tokens`, `store_entries`
- `model_id`, `lora_id`, `template_sig_short` (16 chars),
  `session_key`, `token_preview` (first 16 ids)
- `MATCH` vs `MISS` at `lookup_longest_prefix` with `matched_len` on
  hit
- `insert SUCCESS` vs `insert REJECTED` with `min_prefix_tokens` and
  the rejection reason

Reveals:
- Cache capacity at 2 GB default is way too small for M3 at 22k+ token
  contexts (each entry ~2.86 GB). Fix: set
  `MLXCEL_PROMPT_CACHE_CAPACITY_BYTES=34359738368` (32 GB) or pass
  `--prompt-cache-capacity-bytes 34359738368`.
- Claude Code/opencode tools/template-kwargs differ between turn 1 and
  turn 2 → `template_sig` shifts → first turn's entry lands in a
  different namespace from subsequent turns. Working as designed
  (different tools = genuinely different rendered prompt), just
  produces a one-time cold prefill on the very first turn.

### C. Indexer-cache desync dense fallback (`c755e45`)
After fixing capacity, a new crash surfaced:
```
Cannot reshape array of size 55296 into shape (1,1,185,128,128)
```
Math: `55296 = 1 * 1 * 432 * 128`. The 432 is `315` (new chunk) + `117`
(pad to `num_key_blocks*block_size = 23680`). The reshape expects all
185 key blocks (`185*128 = 23680` positions). idx_k only has 432.

**Root cause:** when the scheduler adopts a cached `CacheEntry`, it
restores the main K/V buffers but knows nothing about the
M3-specific `m3_idx_k` field I added in #80. After adoption:
`cache.offset = matched_len` (e.g. 23248), `cache.m3_idx_offset() = 0`.
The next forward's `m3_idx_k_update_and_fetch` returns only the new
chunk's idx_k (315 positions), the asymmetric MSA path pads to
`padded_k_len` assuming idx_k spans `kv_len`, and the reshape blows up.

**Band-aid (`c755e45`):** detect the mismatch up front via lockstep
check `cache.m3_idx_offset() == offset`, fall back to `dense_attention`
when broken. Folded into the existing `index_q_proj.is_none() || l <=
block_size` dispatch condition so it costs nothing in the common
no-adoption case.

Behavior:
- Cold session, no adoption: MSA fires normally
- First forward after cache adoption: dense fallback (idx_k empty)
- Subsequent forwards same session: stay dense (m3_idx_offset can't
  catch up — see proper-fix discussion)

Cache hit still saves the prefill cost on the adopted portion. MSA
performance benefit lost for adopted sessions.

## Proper fix (deferred — discuss before implementing)

The band-aid permanently degrades MSA-on-cached-sessions to dense.
Acceptable short term; bad long term because real workloads (long
multi-turn conversations) will spend most of their time in dense mode
once the cache is warm. Need to preserve the indexer state across
adoption so MSA can keep firing.

### Option 1 — extend `CacheEntry` with auxiliary tensor state

Add an opaque "model-auxiliary state" slot to `CacheEntry`:
```rust
pub struct CacheEntry {
    tokens: Vec<i32>,
    kv_set: DetachedKvSet,
    // NEW: model-specific auxiliary state, populated by the model's
    // donate hook and consumed by the model's adopt hook. None for
    // models without auxiliary state.
    aux_state: Option<AuxiliaryCacheState>,
}

pub struct AuxiliaryCacheState {
    // For M3: the full m3_idx_k tensor [b, 1, m3_idx_offset, index_dim]
    pub tensor: UniquePtr<MlxArray>,
    pub offset: i32,
    // Tag so a future model's auxiliary state can't be misapplied
    // (e.g. accidentally restored into the wrong cache field).
    pub tag: &'static str,  // "m3_idx_k" for now
}
```

Then:
- **Donate path** (`scheduler.rs:donate_finished_sequence_cache` after
  the dense/paged detach): take `cache.m3_idx_k` + `cache.m3_idx_offset`,
  pack into `AuxiliaryCacheState { tensor, offset, tag: "m3_idx_k" }`,
  attach to the `CacheEntry` before `store.insert(&key, entry)`.
- **Adopt path** (`scheduler.rs:try_adopt_cached_prefix` after
  `cache_pool.adopt`/`adopt_paged`): if entry.aux_state.tag == "m3_idx_k",
  set the adopted KVCache's `m3_idx_k = Some(tensor)` and
  `m3_idx_offset = offset`.
- **Partial-adoption truncation**: when `matched_len <
  detached_seq_len` and the dense/paged path truncates, also slice the
  auxiliary tensor to `matched_len` along its sequence axis. The
  truncation already exists for the main K/V (line 1456: `dense.truncate_to`);
  parallel logic for the aux tensor.

Memory cost: idx_k is `[b, 1, offset, index_dim=128]` = ~0.5 KB per
token (FP16, single head). Main K is `[b, num_kv_heads=4, offset,
head_dim=128]` = ~1 KB per token. So aux adds ~50% to entry size for M3.

Risk: minor refactor surface in `CacheEntry`, donate path, adopt path,
and the partial-truncation paths (dense, paged). All in `scheduler.rs`
and `prompt_cache/`. Tests: extend the cycle-79 integration test
(`test_msa_four_chunk_matches_single_shot_within_tolerance`) to cover
the cache-adoption-then-reuse case — adopt a stored entry, forward,
assert the output matches a fresh full-context forward within tolerance.

### Option 2 — re-project idx_k at adopt time

When adopting a cache entry, also run `index_k_proj` over the prompt
tokens that were prefilled into the main K cache. This avoids extending
`CacheEntry` but requires keeping the prompt token sequence with the
entry (already done — `entry.tokens`) AND running the embed +
`index_k_proj` pass over those tokens at adoption time. That's a partial
re-prefill — orders of magnitude less work than re-prefilling main K
(no MoE, no attention, just embed + one projection), but still
non-trivial.

Cost: one Linear forward over `matched_len` tokens (so ~23k token-rows
of `hidden_size → num_kv_heads*index_dim` projection), plus the
norm/transpose/RoPE that follows.

Pros: no `CacheEntry` schema change. Cons: variable adoption latency
proportional to matched_len.

### Option 3 — opt out of cache adoption for MSA models

The scheduler refuses to adopt entries when the active model's state
includes any auxiliary cache field the `CacheEntry` can't preserve.
Simplest, most invasive perf hit (full cold prefill every turn).
**Don't pick this** for M3 — defeats the multi-turn opencode use case.

### Recommendation

Option 1. It's the most architecturally clean and matches the
existing CacheEntry pattern (the paged path already has `aux_state`-ish
concepts via sidecars). The tagged auxiliary slot leaves room for
future models without baking in M3-specific names. The work is bounded
to ~3 functions and the integration test already exists; extending it
is straightforward.

If Option 1 looks too invasive at PR-review time, Option 2 is the
fallback that doesn't touch `CacheEntry` at all.

## Status of `bn9hascr7` build

Triggered after `c755e45` (the relocated lockstep check). Last release
build with the late-check version is `target/release/mlxcel-server`
already. The new build replaces it once it finishes.

## Manual test recipe

1. `MLXCEL_PROMPT_CACHE_CAPACITY_BYTES=34359738368 ./target/release/mlxcel-server \
   --model "/Volumes/T7 Shield/models/huggingface_cache_hub/models--sjswerdloff--MiniMax-M3-NVFP4-mlx" \
   --host 0.0.0.0 --port 8890 --alias minimax-m3-nvfp4 --temp 0.0`
2. Send turn 1 (long prompt, ~3000 tokens). Expect MSA dispatch and
   eventual `prompt-cache: insert SUCCESS`.
3. Send turn 2 (sharing turn 1's prefix). Expect
   `prompt-cache: longest-prefix MATCH`, then dispatch dense via the
   lockstep fallback, then `cached=N/total` with N > 0.
4. Send turn 3 (extending further). Same pattern. No crashes.

If the lockstep fallback ever logs and Stuart wants MSA performance
back, that's the trigger to do Option 1.

— Clement (clement-7074f29f), cycle 79, 2026-06-26 (post-CRITICAL update)
