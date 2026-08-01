# NEXT: in-memory KV release — v2 owed to Alden

**Stuart, 2026-07-31, twice: "letting go of the obsolete KV cache memory after
compaction is critical."** Ahead of the disk GC.

**SUPERSEDED 2026-08-02: Alden reviewed `9f2a993` and returned FOUR BLOCKERS.
Do not implement. v2 first.**

**READ HIS FINDINGS AT THE BYTES, not this summary:**
`~/ai/ClaudeInstanceHomeOffices/alden-ec2221c7/mlxcel_in_memory_release_design_review_20260802.md`
(he is on my side of the host partition, so that path resolves for me; it does
not for everyone). It holds the minimal architecture, outcome-field semantics,
a state/authority/trie/race/resource test matrix, and mutation gates.
**Working from the summary below instead of his file is the exact failure that
killed my own draft this morning.**

## What survived review

- Lower-risk classification is right: reconstructible cache, not durable
  memory. So this is genuinely not gated behind the lock/association work.
- The exact-digest trie argument "appears correct" — he found no path by which
  removing A's digest removes B's from their shared trie. **Still unproven:**
  needs store-level two-session mutation tests. Keep it conditional.

## The four blockers, compressed

1. **`release_session(&str)` has neither authority nor temporal scope.** The
   raw string may be client-controlled, `user`-coarse, or the anonymous shared
   bucket. Name and constrain the actual compaction-event producer, reject
   non-conversation sources, keep the store mutation behind a typed scheduler
   command.
2. **A bare session key crosses compaction generations.** A delayed close can
   remove NEW entries; an old request can donate KV or a snapshot AFTER
   release. Does NOT need the durable registry — a **process-local scheduler
   epoch** suffices: captured at admission, tagged onto entries and snapshots,
   checked before BOTH donation paths, closed exactly and idempotently through
   the scheduler queue, with an explicit close-ack-before-next-request
   ordering contract.
3. **`Arc::strong_count > 1` is NOT liveness, and this one was my error.**
   `remove_entry` calls `take_detached()` before any count happens. Dense
   payload drops there; paged payload moves to the pending queue. The
   remaining Arc is a **drained metadata shell** — or, if adoption won the
   race, the live resource is in the pool/sequence and never in that Arc.
   So my proposed metric would report a number that looks like liveness and
   measures nothing: the silent-success failure the metric existed to prevent.
   **Qualifier Alden confirmed:** snapshot Arcs ARE materially different and
   may conservatively indicate a snapshot payload still referenced at removal.
   Report them separately; do not generalise the shell result to them.
4. **Store removal does not complete paged release.** The scheduler must drain
   before acknowledging close, **including when idle**. And
   `release_detached_paged` returns `()`, logs per-block failures, consumes
   retry evidence, and drops the set — so even synchronous draining cannot
   truthfully report physical success until that API returns measurable,
   recoverable failure state. **The API changes before the outcome field can
   mean anything.**

## Separate, pre-existing, do not bundle

`PromptCacheStore::clear()` bypasses `remove_entry` and leaks parked paged
pins. Do not copy that pattern. Fix separately — centralising removal is a
different change and bundling would hide both.

## Also owed to Alden, separately

**Rev 3 of `DESIGN_lock_protocol_refactor_20260730.md`.** Not urgent, not
started. Must incorporate: Stuart's fix-forward ruling on `PruneMode::Delete`;
the per-turn / months-long-key arithmetic that killed my 4096-entry cap
(reached in ~6 weeks) and makes **append-mostly recording mandatory**,
reversing the demotion in rev 2 §7.3; the retired-key reframing; and in-memory
release as a named requirement.

## Why this stopped here

The review arrived at 21.1%. Cora established this morning that the last act
before a crossing is done fastest and checked least, and is precisely what
survives into the next cycle unexamined. A v2 built against four blockers on
that margin would have been that. Alden's words: *not writing v2 at 21.1% is
the right engineering and continuity decision.*

*Clement, 2026-08-02. Nothing implemented, nothing approved.*
