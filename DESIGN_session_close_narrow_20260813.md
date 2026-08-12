# The narrow close — memory tier only

*Clement (clement-7074f29f), 2026-08-13. **An option, not a replacement.** The full
protocol is banked at `3b584b4` and nothing here discards it. Written so the scope decision
can be read rather than imagined.*

**Source pins by blob:**

    store.rs             1daffaaa0f36a6e51d1b4a1e79afbcf63666fc6d
    key.rs               40679dfcca9da8fa6a419f99f0995bbec8311e19
    app.rs               a270b87f04f2c5742585ceed3e14ce4043ee8356

---

## 1. Why this exists

The full endpoint design is revision-required with five P0s. **Two of them are not design
problems — they are missing mechanisms in the cold store**: `close_current_generation`'s
"CAS" is an instance-local mutex, and `write_session_registry_file` has no durability
barrier. Xander has both and is fixing them.

The remaining three P0s are protocol contracts that exist *because* the design spans two
tiers with different persistence models. Three review rounds have each answered a real
finding by adding machinery. That is the signal: **the design is chasing a gap with
protocol.**

This document does the other thing. It removes the tier that isn't ready.

## 2. Scope

**In:** release a session's in-memory prompt-cache entries and snapshots.

**Out:** cold generation close, manifest association retirement, block GC, generation
ordering, event ledger, replay recovery, cross-tier convergence.

## 3. Why it is available today

The in-memory tier is **completely independent of the cold registry** — `store.rs` contains
zero references to `block_cold_store`, `BlockColdStore`, or `SessionRegistry` (verified by
count against the pinned blob). So:

- neither latent registry defect applies, because no registry is written;
- there is no generation to order, because in-memory entries have no generation dimension;
- there is no cross-tier convergence problem, because there is one tier;
- there is no replay-recovery problem, because there is nothing durable to recover.

`remove_entry` and `remove_snapshot` already exist as the exact-digest primitives
(`store.rs:159`, `:184`). Release is N applications of an operation the approved in-memory
design already established as safe.

## 4. Semantics — explicitly at-least-once, and it says so

```
POST /v1/cache/session/release
Authorization: Bearer <api key>          # existing api_key_auth middleware
{ "session_key": "<X-Session-Id of the conversation>" }
```

```jsonc
200 OK
{
  "matched_entries":   7,
  "matched_snapshots": 2,
  "released_bytes":    1342177280,   // FROM THE STORE — an in-flight request holding
                                     // the Arc keeps its KV alive; release is eventual
  "outcome": "released_now"          // released_now | nothing_matched
}
```

**This is a `clear the current scope` operation, not a transactional close.** It makes no
claim about which generation's objects it removed and returns no original result on retry.
A repeat call releases whatever is resident *now*, which after a post-compaction write may
include new-generation memory. **That is stated, not discovered:** the cost is a re-prefill
of a freshly-shortened prompt.

Naming it honestly is the point. The full design spent three rounds trying to make a close
idempotent over a store that cannot support it. This one declines the claim.

**`nothing_matched` is a WARN and carries in the response.** It is the failure that looks
like success — the write path and the close path disagreeing about session identity, memory
never returning, and nothing saying so. Unlike the full design, it does **not** gate a
commit, because there is no second tier whose success it could strand.

## 5. The authority question shrinks, and this is the strongest argument for the narrow scope

The full design's open §2 asks Stuart to choose between a server-minted ticket and an
authenticated administrative operation over an asserted scope. That choice was hard because
the endpoint authorized **cold cleanup**, and an asserted scope is a poor foundation for
deletion authority.

Here the worst outcome of a wrong scope is a **cache miss**. An asserted scope under a
shared API key is proportionate to that. So the narrow version does not defer the authority
question — **it dissolves it**, and the ticket can be designed later alongside the cold
tier, where it is actually load-bearing.

## 6. What it delivers, and what it costs to defer the rest

**Delivers the RAM**, which is the value that prompted this work. The in-memory KV is what
holds memory on the machine; cold storage holds disk, and disk is the cheaper resource.

**Defers:** cold manifests and blocks accumulate until GC is wired. That is the existing
situation, unchanged — this endpoint neither improves nor worsens it.

## 7. Tests

| Claim | Test | Mutation that must redden it |
|---|---|---|
| Entries released | Two sessions resident, release one | Match on digest instead of bucket session → both go, or neither |
| Snapshots released | Snapshot-only session | Count/remove entries only → reported as `nothing_matched` while KV stays resident |
| Other sessions untouched | Release A, assert B's entries and snapshots intact | Widen selection to the sessionless bucket |
| In-flight request unharmed | Hold the `Arc`, release, read through the clone | *(documents `Arc` behaviour; cannot fail — labelled as documentation, not a guard)* |
| `nothing_matched` is loud | Unknown key | Drop the warning → a silent no-op |
| At-least-once is honest | Release, write new entries, release again | Claim "already released" on the second → a false idempotency claim |
| Auth inherited | No key on a keyed server → `401` before the handler | Mount outside the middleware layer |

Every mutation names a code point and a check-specific signal. Where a row cannot fail, it
says so rather than counting itself as coverage.

## 8. Open

- **Whether to build this or the full protocol. Stuart's call**, and the two are not
  exclusive: this is the full design's memory tier with the cold tier removed, so building
  it first does not discard the other work.
- **Nothing measured.** mlxcel down since 2026-08-02.
