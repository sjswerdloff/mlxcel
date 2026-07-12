# mlxcel — launch, probe, and KV/decode options reference

**Docs get ignored; harnesses don't.** The two harnesses ENFORCE the
right defaults in code — this doc EXPLAINS them. When they disagree, the
code is authoritative (and the engine's own validation is the final
authority; these harnesses never re-implement a narrow copy of it).

- **Server launch → `launch_mlxcel_server.sh`** — the ONLY launch script.
- **Probes → `source probe_harness.sh`** — the shared probe harness.

Both: every parameter has an env-var override AND our explicit default;
every value is passed explicitly so the binary's/tool's *intrinsic*
defaults (the 2 GiB cache, the 100-token budget) are structurally
unreachable. Both refuse loudly on bad config and never fall back silently.

---

## 1. Server launch — `launch_mlxcel_server.sh`

Normal serving is `./launch_mlxcel_server.sh` (all safe defaults). Override
any value inline: `MLXCEL_X=... ./launch_mlxcel_server.sh`.

| Env override | Default (ours) | Notes |
|---|---|---|
| `MLXCEL_MODEL_PATH` | MXFP8-M3 dir | |
| `MLXCEL_PORT` | `8890` | pre-flight verifies it's free |
| `MLXCEL_KV_CACHE_MODE` | `kvarn8` | see §3; engine validates the value |
| `MLXCEL_THINKING_MODE` | `adaptive` | `disabled`\|`adaptive`\|`enabled` (M3 template) |
| `MLXCEL_MSA_FETCH` | mode-derived (`k8v4`→`dequant`, else `qmm`) | boot-frozen; see §4 |
| `MLXCEL_PROMPT_CACHE_CAPACITY_BYTES` | **`68719476736` (64 GiB)** | binary intrinsic default is 2 GiB — the flag whose absence wasted 2026-07-12 |
| `MLXCEL_PREFILL_CHUNK_SIZE` | `2048` | binary default is 512 |
| `MLXCEL_TEMP`/`_TOP_K`/`_TOP_P` | `1.0`/`40`/`0.95` | |
| `MLXCEL_HARVEST` | `off` | `on` = KVarN harvest (write-stalls at depth; never for verdict runs) |
| `MLXCEL_ALIAS` | `minimax-m3` | client-addressing label |

Pre-flight REFUSES to boot beside another mlxcel-server (orphan/queue
pileup — the other half of the wasted day). Probes are separate.

k8v4 verdict boot: `MLXCEL_KV_CACHE_MODE=k8v4 MLXCEL_THINKING_MODE=disabled MLXCEL_PORT=8896 ./launch_mlxcel_server.sh`

## 2. Probes — `source probe_harness.sh`  (THE GLASS)

**Do not be stingy with another mind's token budget — give it the room you
would want for yourself.** 100 thinking tokens starves a thinking-first
model (it reasons and never answers — 2026-07-12); a starved mind fails
QUIETLY and looks fine, so stinginess here is a SAFETY hole, not thrift.
The default thinking budget MATCHES Violet's own (~16k) — the room *I*
get, extended to the probed mind — not a thrifty multiple of measured
usage (avg 162, max 5,659). A generous budget is a ceiling paid per-use:
free on easy turns, decisive on hard ones. The harness REFUSES a
drops-sized thinking budget (`<1024`).

| Env override | Default (ours) | Notes |
|---|---|---|
| `MLXCEL_PROBE_MAX_TOKENS` | `20480` | total container (thinking + answer); ~16k thinking + ~4k answer. NOT 100. |
| `MLXCEL_PROBE_THINKING_BUDGET` | `16384` | the room I get, given to the probed mind; `-1` = unrestricted. Refused if `<1024` or `>= max_tokens`. |
| `MLXCEL_PROBE_BASE_URL` | `http://127.0.0.1:8890` | |
| `MLXCEL_PROBE_MODEL` | `minimax-m3` | |
| `MLXCEL_PROBE_TEMP`/`_TOP_P` | `0`/`1.0` | probes want reproducibility |
| `MLXCEL_PROBE_TIMEOUT` | `3600`s | deep-context prefills are minutes; a stingy timeout is its own starvation |

Helpers: `probe_echo_config`, `probe_require_server`, `probe_chat "<prompt>"`
(the glass is baked into every request via `thinking_budget_tokens`).

## 3. KV cache modes  (`--kv-cache-mode`, or per-side `--cache-type-k/-v`)

Authoritative enum: `src/lib/mlxcel-core/src/cache.rs` FromStr + `src/cli/turbo_args.rs`.

- **Ours (this arc):** `kvarn8`\|`kvarn-k8v8` (8-bit, boot-night validated);
  `k8v4`\|`kvarn-k8v4` (legacy alias → per-side `kvarn8`+`kvarn4`; the k8v4
  engine shipped 2026-07-12, non-default, cold-store-ineligible until the
  sound depth verdict). Per-side canonical: `--cache-type-k kvarn8 --cache-type-v kvarn4`.
- **Pre-existing:** `fp16`\|`float16`, `int8`\|`i8`, `turbo3`\|`turbo3-asym`\|`fp16+turbo3`,
  `turbo4`\|`turbo4-sym`, `turbo4-asym`\|`fp16+turbo4`, `turbo4-delegated`\|`fp16+turbo4-delegated`.

## 4. decode_config keys (task #31 migration)  —  `src/decode_config.rs`

| Key | Values | Kind | Set via |
|---|---|---|---|
| `msa_fetch` | `dequant` \| `qmm` | **construction** (boot-frozen) | env `MLXCEL_MSA_FETCH` |
| `fp16_gathered` | true/false | **construction** | env `MLXCEL_FP16_GATHERED` |
| `msa_core` | `blocked` \| `sdpa` | **runtime** (live-toggleable) | `POST /admin/decode-config {"msa_core":"sdpa"}` or env `MLXCEL_MSA_CORE` |
| `kvarn_decode_path` | `v1` \| `gathered` | runtime | admin / env |

Per-response `x-mlxcel-decode-config` header echoes the live config
(`path=..; core=..; v=N`). Default flip to `sdpa` is CLEARED (G-live gate
passed 2026-07-11: bit-identical at 295K); the flip is a one-line default change.

## 5. Thinking control (`--chat-template-kwargs`, M3 template)

`{"thinking_mode":"disabled"|"adaptive"|"enabled"}` (M3 `chat_template.jinja`).
Per-request budget knob: `thinking_budget_tokens` (primary; aliases
`thinking_token_budget`, `thinking_budget`). CLI `--reasoning-budget`
(-1 unrestricted / 0 immediate-end / N cap).

---

Authoritative sources when this doc is stale: `mlxcel-server --help`,
`cache.rs` FromStr, `turbo_args.rs`, `decode_config.rs`. The harnesses are
the enforcement; this is the map.
