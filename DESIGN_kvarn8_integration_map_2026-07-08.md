# KVarN8 integration map — every site that must change (agent-mapped 2026-07-08, Clement-curated)

Staging: **PR-1** = core module (`cache/kvarn.rs`, fixtures, pinned tests — landing now).
**PR-2** = cache wiring below. **PR-3** = detach/adopt + server plumbing + gates.

## PR-2: KVCache wiring (cache.rs)

| Site | Line | Int8 precedent | KVarN8 action |
|---|---|---|---|
| `KVCacheMode` enum | ~153 | — | add `KVarN8` variant |
| `from_str` | 188-224 | "int8" | add `"kvarn8"` (canonical) |
| `Display` | 226-237 | — | add arm |
| struct fields | 334-466 | keys/values(INT8)+key_scales/val_scales | NEW: `kvarn_k_tiles`/`kvarn_v_tiles` (u8 history), per-tile `kvarn_{k,v}_{scale,zp,s_row}` + `kvarn_{k,v}_s_col`, `kvarn_sink_{k,v}` (fp16 ROTATED), `kvarn_tail_{k,v}` (fp16 ROTATED) |
| constructors | 502-580 | None-init | init new fields None |
| `update()` dispatch | **859** | `update_int8()` | add `update_kvarn8()`: rotate incoming (wht per token, post-RoPE), split sink(first 128)/tiles/tail, quantize full tiles via `kvarn::kvarn_quantize(bits=8)` |
| `update_and_fetch()` dispatch | **2576** | dequant-on-fetch | dequant history tiles (`kvarn_dequantize_rotated`), concat [sink, hist, tail] (all rotated), ONE `wht` to unrotate, return fp16 standard frame. Transparent like Int8 — zero model changes |
| `trim_front` guard | **2473** | Int8 allowed | KVarN8 NOT trimmable (tile state) → keep OUT of the matches!, gets the existing warn path. Loud-warn like Turbo |
| RotatingKVCache dispatch | **3991** | Int8 falls back fp16 | same fallback + the existing warn convention (RED FLAG site — explicit arm, never wildcard) |
| `eval_state` | ~2955 | evals scales | eval all new fields |
| memory accounting | detach.rs:211 area | nbytes sums | include new fields |
| `clone_handle` paged guard | 801 | `!= Fp16` → None | KVarN8 automatically excluded (correct) |

## PR-3: detach/adopt + server + CLI

- detach.rs: `DetachedKVCache` mirror fields (105-137); `clone_handle` mode arm (~550+, Int8 at :393); `trim_to` (:311-415) — KVarN8 truncation must be TILE-ALIGNED: floor to tile boundary, spill remainder... simplest correct v1: only allow trim at tile boundaries (128 divides prefill_alignment=128 adoption floor → always tile-aligned in practice; assert it); sink never trimmed; tail re-derivable = refuse non-aligned loudly.
- `install_detached` restore arm.
- turbo_args.rs `map_kv_modes_to_cache_mode` (244-273): `(KVarN8, KVarN8) => Ok(KVarN8)`; update error text + help (45-124).
- scheduler `is_turbo_mode` (2436): KVarN8 is NOT turbo (no per-page sidecar budgeting in v1; dense path only like Int8) — but MUST be excluded from pool-backing, which `cache_mode == Fp16` gate already does. Verify sidecar budgeting not needed (dense caches, not paged).
- batch_quant.rs (277-302): scheme mapping — add `KvQuantScheme::KVarN`? v1: reachable via legacy `--kv-cache-mode kvarn8` only; batch_kv_quant scheme extension optional later.
- commands/generate.rs:278 + serve.rs:79 observability flags: group KVarN8 with Int8-style reporting; boot artifact in scheduler run() prints the mode automatically (Debug derive covers it).
- gemma4.rs:896/921 snapshot checks: leave Fp16-only (KVarN8 snapshots refused loudly = fail-closed, fine for v1). qwen3/llama3 reject non-Fp16 already — fine.
- speculative buffer (4063) + rotating snapshot (4967): fp16-only guards already refuse — verify message is loud not silent.

## Red-flag sites (silent-default class — each needs an explicit arm or loud refusal)
1. RotatingKVCache match :3991 — explicit fallback arm w/ warn.
2. trim_front :2473 — stays out; warn fires.
3. is_turbo_mode :2436 — deliberate exclusion, comment why.
4. from_str :192 — without it the mode is unreachable (fail-safe direction, but add).
5. gemma4 snapshot :896 — refusal is loud already; confirm message names the mode.

## Design decisions (locked for v1)
- **Transparent mode** (fp16-standard-frame out of update_and_fetch): zero model changes; rotated-frame SDPA optimization deferred until measured need.
- **k8 default** (`bits` param in module; k4 later through identical gates).
- **No sub-byte packing at k8** (1 byte/value native); k4 packing deferred.
- **Sink=128 tokens fp16-rotated; tail fp16-rotated**; tile=128 tokens.
- **Sinkhorn iters=4** (`KVARN_SINKHORN_ITERS`), batch-global best-so-far quirk kept faithful to reference (fixtures pin it; per-tile improvement = conscious later deviation + regenerated fixtures).
- Persisted-cache stamp must carry (mode, bits, iters, sink, tile) — Silas's rule; PR-3.
