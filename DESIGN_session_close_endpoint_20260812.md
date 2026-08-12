# `POST /v1/cache/session/close` — the compaction-close endpoint

*Clement (clement-7074f29f). Revision 2, 2026-08-12, against Alden's review of revision 1
(blob `643f3d81df45` @ `8a490ad`). Revision 1 is at `8a490ad`.*
*Consumes the release model approved at `ec5b3b4` (blob `beae8b754a4a`).*

**Source pins, by blob id.** *Revision 1's header carried `git rev-parse <commit>:<path>`
in the pin column — a command, not an identifier. Alden's verification limits caught it.
A pin that instructs the reader to go and compute it is not a pin.*

    app.rs               a270b87f04f2c5742585ceed3e14ce4043ee8356
    routes/cache.rs      04ac74d27f5df26f0eb3040dab7b4bbb4b6ec760
    key.rs               40679dfcca9da8fa6a419f99f0995bbec8311e19
    store.rs             1daffaaa0f36a6e51d1b4a1e79afbcf63666fc6d
    block_cold_store.rs  f48ec2fb8215da96821d93cf6b3178defeca57eb

Claims about **opencode** source are date-described (2026-08-02), **not object-pinned and
not run-verified**, and are marked inline wherever they appear.

---

## 1. Transport and authentication — settled

**HTTP, not MQTT** (Stuart, 2026-08-12). MQTT publish-success is not delivery, and the
`PREPARED -> COMMITTED` machine needs a response to drive it.

**Authentication is the existing `api_key_auth` middleware** — it layers the whole router
(`app.rs`), and `POST /v1/cache/reset` already sits behind it. Nothing new is built.

**Authentication is not authorization of scope, and revision 1 blurred them.** See §2.

## 2. What this endpoint's scope claim actually rests on

Revision 1 said: put `session_key` in the body, parse it into `CompactionScopedSessionKey`,
and the type proves the key came from `SessionHeader`. **It cannot, and I should not have
written it.** The body contains a string. Any holder of the shared API key can put any
string there. The server never observed `SessionKeySource::SessionHeader` at this boundary,
so the private type would encode a fact its constructor has no evidence for — converting a
caller's assertion into apparent proof. A type removes forgotten checks only when its
constructor **possesses** the evidence; otherwise it launders the assertion. (Alden.)

Two honest options, and this design picks the first:

**(a) Server-minted ticket — the target.** When mlxcel serves an inference request and
`resolve_session_key` reports `SessionKeySource::SessionHeader`, it mints an opaque handle
bound to `(session key digest, incarnation, current generation)` and returns it. The close
presents the handle. The constructor then holds evidence the server itself observed, and
the type means what it says.

**(b) Raw key in the body — the fallback, named for what it is.** If opencode cannot carry
a ticket, the endpoint is **an authenticated administrative operation over an asserted
scope**. That is a legitimate posture for a single-tenant home server. It is *not*
ownership-safe, and this document must not call it that.

The 2026-08-02 opencode reading establishes that the write and close paths agree on the
key **in the ordinary case**. That is path agreement, not authentication of an arbitrary
later close body. Do not let the first stand in for the second.

## 3. The two tiers are separate operations with separate outcomes

The approved release model separates in-memory entries, cold manifests, and cold blocks.
Revision 1 collapsed the first two into one `closed` flag and one `COMMITTED` bit. **That
cannot represent partial success, and the failure is not symmetric.**

A cold close can legitimately match **zero** in-memory entries — memory was evicted, the
server restarted, or the session only ever existed in cold storage. Under revision 1's rule
(*zero matched ⇒ do not commit*) the cold CAS has committed and the plugin stays `PREPARED`
forever, retrying into an ambiguous `409`. **Revision 1's loudness rule would have stranded
successful closes.**

**Order: cold CAS first, then in-memory release.** Memory removal is not reversible, and
the cold CAS is the operation that can refuse. Doing the refusable one first means the
irreversible one never runs for a close that was going to be rejected.

**The plugin commits when every REQUESTED tier has reached a terminal postcondition** — not
when a single flag is true.

## 4. Request

```
POST /v1/cache/session/close
Authorization: Bearer <api key>

{
  "ticket":     "<opaque server-minted handle>",   // §2(a); or "session_key" under §2(b)
  "generation": 41,                                 // omit to request the memory tier only
  "event_id":   "<32-byte hex, minted by the plugin, durable BEFORE the request>"
}
```

**`event_id` is required and is the plugin's, not the server's.** `close_current_generation`
already accepts `event_id: [u8; 32]` and durably records it as `last_event_id`
(`block_cold_store.rs`) — the mechanism exists and revision 1 simply did not use it.

The plugin mints and durably stores it while `PREPARED`, **before** sending, and every
retry carries the identical `(ticket, incarnation, generation, event_id)`. Without it a
lost `200` and a genuinely stale close are the same request, and no amount of retrying can
tell them apart.

**Implementation consequence, load-bearing:** `close_current_generation` currently rejects
`expected != current` **before** consulting `last_event_id`. For replay recovery it must
consult the recorded event first: if the generation has advanced *and* `last_event_id`
matches this request, that is **my own successful close whose response I lost** — return
the original result, not a conflict.

## 5. Response — per tier, with discriminated outcomes

```jsonc
200 OK
{
  "cold": {
    "requested": true,
    "generation": 41,
    "event_id": "…",
    "outcome": "closed_now"        // closed_now | replayed | not_requested
  },
  "memory": {
    "requested": true,
    "matched_entries": 7,
    "matched_snapshots": 2,
    "released_bytes": 1342177280,   // FROM THE STORE — see the Arc note in §6
    "outcome": "released_now"       // released_now | already_absent | unknown_scope
                                    //             | zero_match (ambiguous)
  }
}
```

**Snapshots are in the release, and revision 1 dropped them.** The approved in-memory
design requires releasing entries *and* snapshots; `SnapshotSlot` is session-scoped and
`remove_snapshot` is the parallel primitive (verified present in `store.rs`). A
snapshot-only session would otherwise report as a no-op while obsolete KV stays resident.
**Zero-match is defined over every requested in-memory object class**, not over entries.

**Zero memory matches is a diagnostic, not universally a failure.** Discriminate where the
protocol can prove it:

- `already_absent` — the scope is known and had nothing resident. Success.
- `unknown_scope` — the scope was never seen. **This is the silent-no-op**: the write and
  close paths disagree, memory never comes back, and nothing else would say so. WARN, and
  carry it in the response as a machine-readable outcome.
- `zero_match` — cannot prove which of the above. Ambiguous; report it, and **never let it
  undo a successful cold close.**

## 6. Failure modes, priced — with the severity claims scoped

Revision 1 stated several of these absolutely. Corrected:

| Failure | Cost | Scope of the claim |
|---|---|---|
| `unknown_scope` on both tiers | Memory never returns; nothing says so | The one worth engineering against |
| Close lands after the post-compaction request wrote entries | One re-prefill of a **short** prompt | Real; §7 |
| Crash while `PREPARED` | No release — a leak | Safe direction; see the note below |
| Concurrent release during an in-flight request | The request **cannot lose its KV**: `remove_entry` returns the `Arc` and the holder keeps the object alive | *Revision 1 said "cannot happen." Wrong — the release runs concurrently. What cannot happen is the in-flight request losing the object.* |
| Over-broad clear | A cache miss, re-prefilled correctly | **Scoped to in-memory eviction only.** Revision 1's "nothing here corrupts" is not supportable for the cold tier: this endpoint commits cleanup *authority*, and the downstream manifest-deletion protocol is unimplemented |

**On `PREPARED`/`COMMITTED`, and this one is not a typo.** Alden flagged "startup retries
only `COMMITTED`" as apparently reversed. It is deliberate and it is the fix for a
data-loss path found in revision 1 of the plugin proposals: a surviving pre-compaction
record has **two indistinguishable causes**, and retrying both closed generations that were
still being extended. The states mean *"compaction is about to happen"* (`PREPARED`) and
*"compaction succeeded, so a close is owed"* (`COMMITTED`) — only the success hook marks
`COMMITTED`, and only `COMMITTED` is retried. Retrying `PREPARED` would close generations
that never compacted. Recorded here because the reasoning lives in PR history and a reader
of this document alone would reasonably read it as backwards.

## 7. What the admission-generation header does and does not close

*(opencode claim: read-verified 2026-08-02, not object-pinned, not run-verified.)*
Because `chat.headers` is spread last, a plugin can attach `X-Session-Generation: N` to
every inference request, binding generation **at admission**.

**It closes the cold donation race** — a late generation-N persist cannot relabel itself
N+1 — **but only after mlxcel validates the value against the server-owned registry and
refuses stale donation.** A caller-supplied header is transport, not authority; merely
carrying N does not make N trustworthy. Same error as §2, one layer out.

**It does not protect in-memory release**, because the pinned in-memory digest has **no
generation dimension** — selection is by bucket alone, so a session-scoped release is
generation-blind. Three options, and this design does not pick one:

1. Require the close to run before any N+1 in-memory write (an ordering contract opencode
   cannot provide — the continue-part is queued with no `await`);
2. Add generation to in-memory entry ownership (a digest-schema change, `v2` → `v3`);
3. Accept and **report** that a close may evict new-generation memory.

(3) is a cache miss rather than corruption, but it is still an imprecise release, and it
must be stated rather than discovered.

## 8. State table — the contract, not an illustration

| Registry / request | Cold | Memory | Plugin |
|---|---|---|---|
| current N, new valid event | commit N → N+1 | release entries **and snapshots** | `COMMITTED` when both requested tiers terminal |
| current N+1, **same event** | `replayed` — return the original result | finish or report idempotent postcondition | `200` → `COMMITTED` |
| current > N+1, different event | `stale_generation`, no mutation | no release unless separately proven pending | terminal |
| current < requested | `future_generation`, no mutation | none | **terminal protocol error — not obsolete success** |
| cold committed, response lost | same-event replay recovers | finish or report memory | converge to `COMMITTED` |
| cold committed, memory zero | success **retained** | ambiguous / `already_absent` | **do not collapse cold success into failure** |
| generation omitted | `not_requested` | entries and snapshots | commit on the memory tier alone |
| ticket/scope invalid | no mutation | no mutation | terminal refusal |

**Reasons are machine-readable and the plugin's action derives from reason + per-tier
state, never from the HTTP status alone.** Revision 1 made every mismatch a terminal `409`;
that is right only for a positively stale request whose success is no longer needed, and
wrong for future skew, for same-event replay, and for a stale cold transition with an
unfinished memory tier.

## 9. Tests

| Claim | Test | Mutation that must redden it |
|---|---|---|
| Lost `200` is recoverable | Cold CAS commits, response dropped, exact-event retry | Consult generation before `last_event_id` → `409`, unrecoverable |
| Replay ≠ stale | Same generation, different event | Collapse both to `stale_generation` |
| Future is not obsolete-success | `expected > current` | Map it to `stale_generation` → the plugin commits a close that never happened |
| Partial success converges | Deterministic failure after cold CAS, before memory release; retry | Single `COMMITTED` bit → cold success stranded |
| Cold success survives zero memory | Cold commits, memory matches nothing | Zero-match vetoes commit → stranded `PREPARED` |
| Snapshots are released | Snapshot-only session | Count entries only → reported as a no-op |
| Exactly one advance | Concurrent same-event and distinct-event closes | Drop the CAS → double advance |
| N+1 memory write before N close | Write, then close | *(documents §7 option 3 — must state the outcome, not assert safety)* |
| Scope is what we claim | Body naming another session's raw key under a valid shared key | Whichever posture §2 lands on, the test names it |
| Admission generation is validated | absent / stale / future / malformed / valid | Trust the header → donation relabels |
| Auth inherited, not reimplemented | No key on a keyed server → `401` before the handler | Mount outside the middleware layer |

## 10. Open

- **§2's posture** — ticket (a) or asserted-scope admin operation (b). Needs Stuart: it is
  a question about what opencode can carry, not about what is safest.
- **§7's three options** for generation-blind in-memory release.
- **Nothing measured.** mlxcel down since 2026-08-02.
- **The opencode claims are read-verified only.** One minimal plugin and one server log
  line settles them.
