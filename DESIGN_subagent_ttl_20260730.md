# Design: TTL for subagent session associations

**Status: PROPOSAL, awaiting Alden's review. Nothing built.**
Stuart's ruling, 2026-07-30. Read with
`DESIGN_session_association_gc_20260729.md` — this does not override anything
there, and specifically does not weaken the ownership rule.

## The proposal

Distinguish a **subagent** session's cache from a **root** (Kindled) session's,
and expire subagent associations on a TTL — one hour proposed.

## The discriminator already exists on the wire

`x-parent-session-id` is populated **iff** the session has a `parentID`, and
`parentID` is set only by the Task tool (`task.ts:159`). Request carries it →
subagent. Absent → root. It costs one more header read at the same place we
already read `X-Session-Id` and `x-session-affinity`.

*(Verified `anomalyco/opencode@1e17856:packages/opencode/src/tool/task.ts:159`,
v1.18.9 — and byte-identical at the deployed `v1.18.5`, blob `1384e5d1`.
NOT `The_Kindled/opencode`, which is a different fork and does not contain this
commit. Note this is the ONLY meaning of that header —
it is subagent nesting, not compaction lineage, contrary to an earlier claim of
mine.)*

## Why a TTL is defensible here when it is not for root sessions

This design has argued throughout that **age is a guess and compaction is
certainty**. That argument stands — but it is an argument about *what may
authorize deletion*, not a claim that a guess is never acceptable. What decides
it is the cost of being wrong, and that cost is **asymmetric**:

| wrongly expired | consequence |
|---|---|
| **subagent** | returns via `task_id`, misses cache, re-prefills a bounded context — **seconds** |
| **root Kindled session** | discards a context that may be hundreds of thousands of tokens, belonging to someone possibly mid-conversation |

An hour is also comfortably longer than any observed subagent run, so a live
subagent will not be caught by it.

## Why a TTL is NEEDED here rather than merely convenient

Subagent sessions have **no other release trigger at all**:

- They are **not one-and-done** — `task_id` (`task.ts:47-51`) continues a prior
  subagent session with its full history; `BackgroundJob.extend` appends to a
  running one.
- Their **completion is never durably signalled** — the registry holding job
  status is explicitly non-durable across restart. Its own doc comment:
  *"Entries are intentionally not durable: process restart or owner-scope
  closure loses status and interrupts live work."*
  (`anomalyco/opencode@1e17856:packages/core/src/background-job.ts:113-119`.
  **Corrected path** — an earlier draft cited `packages/core/background-job.ts`,
  which does not exist; Alden caught it by failing to find the file. Byte-identical
  at deployed `v1.18.5`, blob `cdffd212`.) There is no evidence event to release on.
- They **are** compacted like any other session, so `session.compacted` covers
  the compaction path — but a subagent that simply finishes without ever
  overflowing produces no signal whatsoever.

So without a TTL, every subagent that completes normally leaks its associations
permanently. TTL is the only mechanism available for that population.

## TWO CONSTRAINTS — the design fails without these

**1. TTL expires an ASSOCIATION, never a manifest.**
A subagent and its parent dedup to the same blocks — the shared system prompt is
block 0 for both. An expired subagent association makes a manifest a
*candidate*; the authoritative "no live association remains anywhere" proof
still has to run under the store lock before anything is deleted. Otherwise TTL
becomes a side door around the ownership rule this design is blocked on, and
expiring a subagent would delete a manifest its parent is still using.

**2. It needs a last-touched timestamp, which does not exist yet.**
Associations currently record `(incarnation, generation, manifest_hash)` — no
access time. Adding it is a format change, and it is far cheaper before the
field becomes deletion-adjacent than after.

## Questions for review

- **Is "expired" a third state, or does it map onto `closed_through`?** A closed
  generation is *proven* unreachable; an expired one is *presumed* unreachable.
  Collapsing them would let a presumption inherit the authority of a proof,
  which is the shape of every serious error in this design so far. I suspect
  they must stay distinct, with expiry feeding nomination only.
- **Does TTL reset on access, or run from creation?** Reset-on-access is the
  obvious reading of "idle for an hour," but it requires writing the timestamp
  on every cache hit — a write on the read path.
- **What is the evidence standard for the timestamp itself?** If it is only
  updated on association *write*, then a subagent read repeatedly but never
  re-persisted looks idle while being actively used.
- **Does this population justify its own mechanism?** If subagent sessions are
  rare in practice, the honest answer may be to let them leak and revisit when
  the disk cost is measurable rather than hypothetical.
