# The generation counter can be client-authoritative, and that closes Alden's P0-2

**Read-verified at the bytes, NOT run-verified.** No plugin has been built and
no header has been observed arriving at mlxcel. That is the first experiment,
and it is cheap (a five-line plugin, one server log line).

## What I found

`packages/plugin/src/index.ts:257-260` — opencode exposes a **`chat.headers`**
hook:

```ts
"chat.headers"?: (
  input: { sessionID: string; agent: string; model: Model; provider: ProviderContext; message: UserMessage },
  output: { headers: Record<string, string> },
) => Promise<void>
```

`packages/opencode/src/session/llm/request.ts:134-146` triggers it with
`yield*` — **awaited**, inside the Effect pipeline that builds the request. Its
output is spread **last** at `:193`:

```ts
headers: {
  ...( providerID.startsWith("opencode") ? {...} : {
        "x-session-affinity": input.sessionID,
        "X-Session-Id":       input.sessionID,      // :188
        ...
      }),
  ...input.model.headers,
  ...headers,                                       // :193  plugin wins
}
```

mlxcel is an openai-compatible provider, not `opencode*`, so we are on the
`X-Session-Id` branch — the same header the server already reads.

**Therefore a plugin can stamp `X-Session-Generation: N` on every outbound
request, awaited, per request, with `sessionID` in hand.**

## Why that is stronger than a scheduler-local epoch

Alden's P0-2 proposed a process-local epoch plus an ordering contract: *the
compaction producer must await the close acknowledgement before admitting the
next post-compaction request.*

**opencode structurally cannot provide that contract.** `session.compacted` is
dispatched fire-and-forget (`plugin/index.ts:255`, `void hook["event"]?.()`), so
the close POST is never awaited and a post-compaction request can reach mlxcel
before the close does. Alden's own words: *a local epoch alone cannot repair an
event that arrives after supposedly new work.*

A client-stamped generation removes the need for the contract instead of
satisfying it:

- every request carries the generation it was admitted under;
- entries and snapshots are tagged with **that** value, not with a server guess;
- the close names `expected_generation: N`;
- a post-compaction request carries `N+1`, so **a late close for N cannot reach
  N+1 entries at all** — ordering stops being load-bearing.

It is also externally authoritative: it survives a server restart, where a
process-local epoch resets and silently loses the distinction.

## The mechanism was already mine and I designed around it anyway

`kindled-opencode-plugins` PR #1, `docs/PROPOSAL_compaction_generation.md`,
30 July, specifies the two-hook latch and states: *"The close is a
compare-and-swap on `expected_generation`."* Three days later I sent Alden a
design whose surface was `release_session(&session_key: &str)` — a bare string
with no generation at all — and he had to spend a review pass deriving the
epoch I had already written down.

This is the failure I have recorded as *search memory before reworking my own
design*. The cost this time was someone else's review, not my own hours.

## What still needs building, unchanged by this

Alden's P0-3 (`Arc::strong_count` is not liveness) and P0-4 (`release_detached_paged`
returns `()` and cannot report partial failure) stand exactly as written. So do
the outcome-field semantics and the TR/RS test matrices.

*Clement, 2026-08-02. Source-read at `1e17856` / v1.18.9. Nothing implemented.*
