# `POST /v1/cache/session/close` — the compaction-close endpoint

*Clement (clement-7074f29f), 2026-08-12. For review by Alden (alden-ec2221c7).*
*Consumes the release model approved at `ec5b3b4` (design blob `beae8b754a4a`).*

**Source pins by blob, not commit:**

    app.rs         (routing + api_key_auth layer)   git rev-parse <commit>:src/server/app.rs
    routes/cache.rs (cache_reset precedent)         git rev-parse <commit>:src/server/routes/cache.rs
    key.rs         40679dfcca9da8fa6a419f99f0995bbec8311e19
    store.rs       1daffaaa0f36a6e51d1b4a1e79afbcf63666fc6d

---

## 1. Transport and authorization — both settled, neither new

**Transport: HTTP, not MQTT.** Stuart's ruling, 2026-08-12, and it is the better answer
than the MQTT recommendation this project carried for eleven days. MQTT publish-success is
not delivery; the `PREPARED -> COMMITTED` state machine the plugin proposals already went
four revisions to build *needs a response to drive it*, and a publish supplies nothing to
drive it with. mlxcel also already owns the state being closed.

**Authorization: the existing `api_key_auth` middleware.** It layers the whole router
(`app.rs:213`), and **`POST /v1/cache/reset` already sits behind it** (`:163`) — the
comment at `:164-165` names that route as the reference posture for later admin endpoints.
A session-close route inherits it by being mounted. **Nothing new is built here.**

*Scope note, so nobody later mistakes auth for correctness:* the API key proves the caller
is legitimate. It does not prove the caller owns the named session — every client on this
host holds the same key. That is not an auth gap; it is why §3's scope type exists.

## 2. What can actually go wrong, priced

Recorded because a later reader will otherwise re-derive a severity this design does not
have.

| Failure | Mechanism | Cost |
|---|---|---|
| **Silent no-op** — release matches nothing | The write path and the close path disagree about what identifies a session (`resolve_session_key` precedence vs. the plugin's `sessionID`) | **The real one.** Memory never returns and nothing says so. §4 makes it loud. |
| Close arrives after the post-compaction request wrote new entries | Same session key either side of compaction; the in-memory tier has no generation to distinguish them. Opencode queues the continue-part with no `await` on the event, so the window is milliseconds | One re-prefill of a **short** post-compaction prompt. Minor. |
| Crash between `PREPARED` and `COMMITTED` | Startup retries only `COMMITTED` | No release. A leak — the safe direction. |
| Release while a request is in flight | — | **Cannot happen.** `remove_entry` returns the `Arc`; a request holding a clone keeps its KV alive. Unrepresentable, not merely guarded. |
| Over-broad clear (`user` bucket, reused key) | Write path fell through to `user`, or a client reused a key | A cache miss. **Not corruption** — a miss re-prefills correctly. |

**Nothing here corrupts.** The endpoint's value is that release *happens*, and happens to
the right bucket — not that it prevents damage.

## 3. Request

```
POST /v1/cache/session/close
Authorization: Bearer <api key>        # existing middleware, when a key is configured

{
  "session_key": "<X-Session-Id value of the conversation that compacted>",
  "generation":  <u64>                 # cold tier only; omitted = in-memory tier only
}
```

**`session_key` must be a `CompactionScopedSessionKey`**, per the approved design:
constructible only from `SessionHeader`, the one source `key.rs:435-437` documents as
*"the CONVERSATION granularity, which is what compaction-scoped GC needs."*
`PromptCacheKey` is excluded until its contract is settled; `User` and `Anonymous` are
rejected — `user` is end-user scope, where a per-conversation delete removes every
conversation that user ever cached (`key.rs:438-442`).

Rejection is by **type at the boundary**, not by a check inside the handler: parse the
body into the scoped type or fail. A handler that receives a `String` and validates it is
one refactor away from a handler that forgets to.

**`generation` is required for the cold tier and must not be caller-derived.** The cold
store already refuses a caller-supplied cutoff — `releasable_manifests` derives it from
the registry precisely because *"a deletion-authority query must not be constructible with
the wrong bound."* This endpoint honours that: it names the generation being closed, and
`close_current_generation` compares it against the registry's current value and refuses on
mismatch, without mutation.

## 4. Response — and the zero case is the point

```jsonc
200 OK
{
  "closed": true,
  "matched_entries": 7,       // in-memory entries selected
  "freed_bytes": 1342177280,  // FROM THE STORE, not returned to the OS (Arc caveat)
  "generation_closed": 41,    // cold tier; null when not applicable
  "warnings": []
}
```

Shape deliberately parallel to `CacheResetResponse` (`routes/cache.rs:158-168`).

**`matched_entries == 0` must be loud.** This is the failure mode that costs something, and
it is invisible by construction: a release that matches nothing and a release that works
both return, neither errors, and memory simply never comes back. So:

- `warnings` carries `"no entries matched session_key"` — a machine-readable string, not
  prose in a log line;
- the server logs it at WARN with the resolved key;
- the plugin treats a zero-match as **failure to release, not success**, and does not mark
  `COMMITTED` on it.

That last clause is the load-bearing one. Without it the plugin records a successful close
for a release that did nothing, and the record is worse than silence.

**`freed_bytes` names the store, not the OS.** Release is *eventual* — an in-flight request
holding an `Arc` clone keeps the memory. Any monitoring built on this number must not read
it as RSS.

## 5. Status codes, mapped onto the plugin's state machine

| Code | Meaning | Plugin action |
|---|---|---|
| `200` | Closed; body says what matched | `COMMITTED` — **only if `matched_entries > 0`** |
| `400` | Body is not a valid `CompactionScopedSessionKey`, or `generation` malformed | Do not retry. Terminal — a wrong-source key will not become right |
| `401` | Missing/invalid API key | Do not retry blindly; surface it |
| `409` | Generation mismatch — the registry has moved on | Do not retry. **The close is obsolete, not failed**; a later generation owns the state |
| `503` | Cache disabled, or store unavailable | Retry — stays `PREPARED` |
| `5xx` | Anything else | Retry — stays `PREPARED` |

**Idempotent on repeat.** A second close for an already-closed generation returns `409`,
not `200`, and `409` is explicitly *not* an error to retry. This is the distinction the
plugin's revision-1 data-loss path turned on: a surviving pre-compaction record has two
indistinguishable causes, and retrying both closed generations still being extended. `409`
makes them distinguishable.

## 6. Tests, one per claim

| Claim | Test | Mutation that must redden it |
|---|---|---|
| A non-`SessionHeader` key cannot reach the handler | POST with a `user`-sourced key → `400`, store untouched | Accept `String` at the boundary → the release runs |
| Zero match is not success | POST an unknown key → `200` with `matched_entries: 0` and the warning present | Drop the warning → the plugin commits a no-op |
| Live request unharmed | Hold an `Arc` clone, close, assert the clone still reads | *(none — this is `Arc`, not a guard. Test documents it; it cannot fail)* |
| Generation mismatch does not mutate | Close generation N-1 → `409`, registry unchanged | Compare-then-advance without the guard → registry advances |
| Repeat close is not a retryable failure | Close twice → `200` then `409` | Return `500` on the second → the plugin retries forever |
| Auth is inherited, not reimplemented | POST without a key on a keyed server → `401` before the handler runs | Mount the route outside the middleware layer |

## 7. Open

- **What opencode actually sends as its session identity on inference requests.** The whole
  design turns on the write path and the close path agreeing, and **I have not verified it**
  — I carried a stale claim about opencode session headers once and it misled Stuart, so
  this is named as unverified rather than assumed. **First thing to check before
  implementing.** If they disagree, every response above is `matched_entries: 0`.
- **The manifest-tier reader guard.** The block tier has `acquire_read_lease`
  (`block_cold_store.rs:699-740`, taken at `:2410`); the manifest tier has no equivalent.
  Out of scope here — a close retires an association and does not delete.
- **Nothing measured.** mlxcel has been down since 2026-08-02.
