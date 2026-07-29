# Compaction-scoped GC for the v4 cold store — design and blocking preconditions

**Status: RECORDING SHIPPED. CLEANUP NOT BUILT AND NOT SAFE TO BUILD YET.**

Author: Clement (clement-7074f29f). Protocol design and every blocking finding
below: Alden (alden-ec2221c7), 2026-07-29, across two review rounds.
Stuart set the scope: compaction is the trigger, and the session key must not
go into the manifest.

---

## Why compaction is the right trigger

When a Kindled compacts, the pre-compaction token sequence is dead **by
construction** — it can never be requested again, because the context that
would produce it no longer exists. That is certainty about reachability, not a
staleness heuristic. Age and LRU are guesses at what compaction states outright.

No vendor or engine offers this. OpenAI's `prompt_cache_key` is a routing hint,
Anthropic's `cache_control` keys off prefix bytes, and engine flush endpoints
are global. Their caches are ephemeral — minutes, in memory, gone on restart —
so they solve garbage by forgetting fast. **Ours is durable on disk and survives
process death, which is the point of it, and durability is exactly what creates
a GC problem none of them have.** There is nothing upstream to copy.

## Why a sidecar and not a manifest field

A manifest's address is a hash over its identity fields. Putting the session key
inside one makes it session-*scoped* at Tier 1's exact-address lookup — and then
no Kindled can hit a prefix another persisted. The content most worth sharing is
precisely the part we hold in common (the system prompt is block 0 for all of
us), so that cost would fall exactly where the benefit is.

---

## THE CENTRAL CORRECTION: an association is not ownership

A manifest is content-addressed and therefore **globally shared, exactly like a
block**. If Clement and Cora both send the same prefix, they persist the same
manifest M.

So `A compacts` authorizes deleting the association `A → M`. It does **not**
authorize deleting M. Doing so removes a manifest Cora is still using, and a
later block GC — correctly seeing no committed root — may then collect M's
blocks.

I had classified "two sessions may reference one manifest and either may release
first" as an *idempotence* condition. It is an **authority** condition. That
error is the reason cleanup is not built.

> "If the sidecar itself says *delete M*, it is deletion authority, not a hint."
> — Alden

## Generation is required, not optional

`session_key → {manifest}` cannot distinguish a pre-compaction conversation from
a **new** one reusing the same key afterwards — session keys are arbitrary client
strings and clients reuse them. A delayed cleanup then crosses the compaction
boundary and deletes live data.

**Identity proves who; only a generation proves when — and reachability is a
question about when.**

Implemented: associations are `(manifest_hash, generation)` and
`session_manifests_through(key, cutoff)` filters on it. Tested by
`a_generation_cutoff_protects_a_reused_session_key`.

---

## Blocking preconditions before ANY cleanup path is written

### 1. Association publication must come under the store lock

Recording currently runs outside the publication critical section. That is safe
**today only because nothing treats an association as deletion authority.**

Once cleanup exists, a sweep must prove no live association remains — and an
association written concurrently, outside the lock, is exactly what such a scan
would miss. The scan would then "prove" a live manifest unreferenced.

I originally argued the outside-the-lock placement was correct by design. It is
not. It is a temporary state with a precondition attached.

### 2. The persist+association / manifest-deletion TRANSACTION GAP

**Named carefully.** This is *not* the "`write_manifest` existing-path race",
and calling it that misdirects whoever picks it up. `write_manifest`'s
existing-path fast return is **one implementation site the future protocol must
absorb**, not an independently demonstrated defect today.

Why there is no defect to demonstrate today: if `write_manifest(existing M)`
overlaps `delete_manifest(M)`, a final absent M is still a **linearizable
history** — order the successful write before the overlapping delete. No
standalone write API can promise that another authorized operation will not
delete the object after its linearization point, even if that delete returns
before the write's caller resumes. There is no violated postcondition here.

The gap is in a **composite operation that does not exist yet**:

```
ensure/publish M  →  publish open-generation association G → M
```

Once manifest deletion exists, a delete landing *between those two steps*
produces the forbidden committed state `association(G, M) && !manifest(M)`.
**That** composite is entitled to promise its caller that success means both
artifacts committed coherently under an open generation. `write_manifest` alone
is entitled to promise no such thing.

Verified in source (true, but as context rather than as a defect):
`write_manifest` returns `Ok(())` on `manifest_dir.exists()` before taking its
lock, and `delete_manifest` takes no lock at all.

### 3. Generation lifecycle must be server-owned and authenticated

**BUILT (recording side).** The registry now exists: `sessions/<digest>.gen`
holding `incarnation | current_generation | closed_through | last_event_id`.

Three things about it are load-bearing and were all corrections to my sketch:

**The state machine is NOT `OPEN → CLOSING → CLOSED`.** Persisting an
intermediate state creates a crash case where both automatic repairs guess at
intent — rolling back reopens a generation the client believes dead, resuming
completes a transition nobody re-authorized. Instead the current generation is
**always open**, and a monotonic `closed_through` says which earlier ones are
finished. Closure is one atomic replacement; every crash lands unambiguous.

**Absence mints a fresh INCARNATION.** A missing registry cannot prove
first-ever use — it can equally mean lost state. Reusing `(key_digest,
generation 0)` would let a recreated session claim cleanup authority over the
previous lifetime's manifests. Every association therefore carries a 16-byte
incarnation minted from `/dev/urandom`, and minting **fails closed** rather
than deriving from time or a counter: a predictable incarnation reintroduces
exactly the collision it exists to prevent, in the degraded case nobody
watches. Orphaned associations from a lost lifetime leak; they are never
adopted.

**The recording API does not accept a caller-supplied generation.** Doing so
would bake the wrong authority boundary in *while cleanup is disabled* — which
is precisely when the mistake looks harmless. The server reads it from its own
registry, under the same lock hold that writes the association (two sequential
holds leave a window in which a close advances the generation between the read
and the write, landing the association under a generation already closed).

`close_current_generation(key, expected, event_id)` is implemented and tested:
CAS on `expected == current`, stale and future both mutate nothing, no wrap at
`u64::MAX`. **No endpoint exposes it.** `event_id` gives exact-retry idempotence
and audit; it is explicitly *not* authorization.

**Still blocked:** who may call it. That is authentication, and it is Stuart's
decision, not something to invent.

`session_key` is cache-namespace input, **not proof of deletion authority**. A
normal inference request carrying "generation 12" must never authorize closing
11.

- session identity: preferably a server-minted opaque handle, not a client string
- generation: server-owned, monotonic, durably persisted
- close: a distinct, authenticated, compare-and-swap operation against the
  expected OPEN generation; replay-idempotent
- restart: an OPEN generation stays open unless an authorized close was durably
  recorded. **Never infer "server restarted, therefore the previous generation
  is dead."** That leaks safely until evidence arrives.
- a missing or corrupt generation registry deletes nothing and must not silently
  reset to 0

In-flight persists mean closure needs `OPEN → CLOSING → CLOSED`. Under one lock,
either association publication wins and closure sees it, or closure marks G
non-OPEN first and the late persist may leave an unassociated manifest but
cannot publish a live association. Only CLOSED generations are cleanup inputs.

**Until an authenticated notifier exists, record associations and ship cleanup
observe-only. Accumulation is the correct safe failure.**

---

## UNASSOCIATED IS NOT RELEASABLE — three states, not two

**The single most dangerous simplification available here.** Cleanup must
distinguish:

- **LIVE** — named by an open-generation association.
- **RELEASABLE** — positively named by a CLOSED-generation association, and by
  no open or non-releasable one.
- **UNMANAGED / UNKNOWN** — named by no association at all. **Retain. Leak.**

If cleanup treats "not in the live association set" as sufficient to delete, it
deletes from *absence of evidence* — and the things missing from that set are
exactly anonymous traffic and **manifests whose index write failed**. That
directly contradicts the "a missing index write leaks only" guarantee this
design rests on: the failure would stop being a leak and start being data loss,
in precisely the case where bookkeeping already went wrong.

Release requires a POSITIVE closed-generation naming. Nothing weaker.

### The dedup case the call-site exclusion does not cover

1. scoped session A records manifest M;
2. anonymous traffic later persists the same content-addressed M, recording
   nothing;
3. A closes;
4. cleanup sees only A's closed association and deletes M — which anonymous
   traffic was still using.

Under compaction-only authority, anonymous traffic supplied no death proof, so
it cannot have consented to that deletion. Three conservative options, none of
them yet chosen:

- **do not persist anonymous traffic into v4 at all** (Alden's preference,
  unless anonymous durability has measured value — it avoids permanent pins and
  keeps compaction GC's authority exact; anonymous may still LOAD an existing
  manifest under the read lease, it just creates no future-liveness claim);
- record an explicit NON-RELEASABLE / unscoped pin for M;
- define and separately authorize a general cache-eviction policy that may
  evict anonymous artifacts.

**This is a policy decision with cache-hit-rate consequences, so it is
Stuart's, not mine.** How much of real traffic resolves to anonymous is
measurable from the `session_source` / `header_session_present` fields now on
the lookup log — measure before choosing.

## SOURCE-REVIEW BLOCKERS (Alden, 2026-07-30, against pushed `68b4c28`/`5e35b98`)

None of these is deletion corruption **today**, because nothing deletes. All of
them are blockers before closure/cleanup authority. **The current association
data must not yet be described as authoritative.**

**FIXED — P0, the cutoff sentinel.** `CLOSED_THROUGH_NONE = u64::MAX` with a
source comment claiming `generation <= closed_through` would be *false* for
every real generation. It is TRUE for every generation. `is_closed()`
special-cased it; `session_manifests_through(key, cutoff)` did not — so the
natural cleanup call on a fresh registry returned every OPEN manifest as
releasable. Now `Option<u64>` in memory (the sentinel exists only in the disk
encoding), and `releasable_manifests(key)` DERIVES the cutoff and returns empty
on `None`. Gate test + mutation in place.

### Still open — must be closed before delete mode

- **P1 — the generation is chosen at donation time, not bound to the request.**
  `record_session_manifest` reads whatever generation is current when the
  persist finishes. The single lock hold fixed a close landing between registry
  read and index write; it does NOT cover a close between request admission and
  donation. An old-generation request finishing after compaction is filed under
  the NEW generation. Needs a server-issued opaque ticket
  `(key digest, incarnation, generation)` captured at request entry and
  validated under lock at record time — refuse the late association rather than
  silently relabel it.
- **P1 — `user` is being treated as GC-scoped while the source says it is not.**
  `key.rs` documents `user` as end-user scope and says per-conversation deletion
  would remove every conversation for that user. `recordable_session_key`
  nonetheless accepts it. **Recordability must depend on SOURCE / proven
  granularity, not on the string being non-empty and non-sentinel.** At minimum
  `User` and `Anonymous` are not compaction scopes; `SessionHeader` may be;
  `PromptCacheKey` needs an explicit contract rather than an assumption.
- **P1 — fixed-width framing has no whole-record integrity.** Magic, version and
  length are checked, but a same-length bit flip in an incarnation, generation,
  cutoff, event id or manifest hash still parses. That can disown a live
  association, alter a cutoff, or forge an association to a different manifest.
  Needs a digest over the complete authority record, verified before any field
  is interpreted, plus a byte-mutation test over every stored byte with
  non-semantic exceptions explicitly enumerated (ideally none). Related: an
  index key-digest mismatch currently returns EMPTY — an authority reader must
  ABORT, never convert corruption into "nothing associated".
- **P1 — the association mutex is per-`BlockColdStore`-instance only.** It
  excludes neither a second handle in-process nor another process. Two handles
  can mint different incarnations, race the deterministic temp paths, or lose a
  read-modify-write entry. Needs two-handle and cross-process tests before these
  files are treated as authoritative.
- **P1 — "a missing association leaks only" is FALSE under dedup.** The claim
  protects a manifest with ZERO associations. Counterexample: A holds a CLOSED
  association to shared M; B persists M but B's association write fails; a scan
  sees only A's closed association and classifies M releasable. So the composite
  protocol must either make manifest+association success coherent, create a
  durable unmanaged pin on association failure, or mark cleanup globally
  unhealthy after any association-publication failure. **The unconditional
  "will leak" claim must not be retained.**
- **P2 — exact replay is documented idempotent, implemented as an error.**
  `close_current_generation` rejects every `expected != current`, and the test
  expects an exact replay to error. Either implement replay recognition via
  `last_event_id` or narrow the documented claim. CAS still prevents mutation,
  so this is not deletion corruption.
- **P2 — a malformed session header silently becomes absence.**
  `to_str().ok()` reports `header_session_present=false` on invalid bytes and
  falls through. A malformed PRESENT header is not an absent one; reject it or
  carry a distinct malformed-present state, loudly.
- **P2 — no production-path tests.** Coverage is parser/resolver/type plus a
  hand-assembled context. Nothing exercises route → context → scheduler →
  recorder across Chat/Responses/Anthropic, sync and stream, so the suite stays
  green if a route drops propagation or the scheduler wiring disappears.
- **P2 — index allocation is unbounded.** `fs::read` takes the whole file, the
  entry count has no configured maximum, `Vec::with_capacity(count)` trusts it,
  and encoding truncates `entries.len() as u32`. Bound bytes and entry count
  before allocating; use checked conversion.

## DELETE-MODE ENTRY GATE

**No cleanup wiring lands until the composite transaction below exists and its
two deterministic interleaving tests pass.** This is a test specification
attached to the feature whose contract gives it meaning — not a warning left
untested.

The transaction: `ensure/publish M` then `publish (session, open generation) →
M`, as ONE non-reentrant operation under the exclusive store lock. Its contract:

1. **success ⇒** M is committed *and* `G → M` is committed, at one
   linearization point;
2. **close/delete wins first ⇒** the late publication REFUSES success for G and
   publishes no association;
3. **publication wins first ⇒** close/manifest-GC observes the association and
   cannot tombstone M as unassociated.

Test with a **deterministic seam, never timing**:

1. pause after M is ensured but before association publication;
2. run close / manifest-GC to its lock boundary;
3. release the publisher;
4. assert one of the two legal serial outcomes — and make
   `association(G, M) && !manifest(M)` **unrepresentable**, not merely absent.

Until that protocol exists, a sequential `ensure; delete; associate` test proves
only that three separately-callable operations can be ordered that way. It tests
no current contract, and a test asserting today's unlocked behaviour would
**fossilize the weakness rather than protect anything**.

## The cleanup protocol, when preconditions are met

One forward authoritative scan per pass. No authoritative reverse index — a
second derived structure would have to agree atomically with the forward one
across crashes, and one scan already avoids the cost that would justify it. A
derived per-manifest count may be added later **as a hint only**, and only if
measurement shows nomination needs it.

1. Read association epoch `E1`.
2. Unlocked: read every committed session index. Any unreadable or unknown index
   **aborts the pass**.
3. Read epoch `E2`. If `E1 != E2` or either is invalid, discard and retry
   boundedly.
4. Build nominees (`committed_manifests - live_associations`) from that stable
   snapshot, outside the lock.
5. Acquire the exclusive store lock.
6. Re-read epoch `E3`. **If `E3 != E2`, tombstone NOTHING** and rescan next pass.
7. Revalidate nominees, then tombstone under the lock.
8. Drop the lock; physically unlink only tombstones outside it.

`E2 == E3` is what proves no association was published after the stable scan —
so the exclusive window is epoch check + revalidation + renames, not a full
rescan. **Requirement: the epoch must cover every operation that can make an
association visible, and additions must bump it BEFORE visibility.**

Crash asymmetry, same shape as manifest publication:

- epoch bumped, association never visible → wasted retry; safe
- association visible, epoch not bumped → GC trusts a snapshot missing a live
  root; **unsafe and forbidden**

Block GC remains the sole block unlinker.

---

## Format: fixed-width binary, no raw key

The first version was line-oriented with the session key written verbatim on
line 1. Session keys are arbitrary client strings and arbitrary includes `\n`, so
a key of `evil\n<digest>` terminated its own line and injected an entry.

**That was not a parse bug. It was a capability** — since associations authorize
deletion, a client could name another Kindled's manifest and have its own
compaction release it. Proven with a red test before the fix
(`a_session_key_containing_a_newline_cannot_forge_index_entries`).

Now (**v2**): `magic(8) | version(4) | key_digest(32) | count(4) | count ×
(incarnation(16), generation(8), manifest(32))`. Filename is the key digest, so
an unsafe path is unrepresentable rather than sanitized. **No raw key is stored
at all** — lookup needs only the digest, which also removes an unbounded write
and a possibly-sensitive value at rest.

v1 (generation only, no incarnation) is **rejected, not upgraded**: it cannot
say which lifetime of a key its entries belong to, and guessing would hand a
recreated session authority over a lost one's manifests. The bump happened
before any cleanup code existed, which is far cheaper than migrating a field
after it becomes deletion authority.

Registry file: `magic(8) | version(4) | key_digest(32) | incarnation(16) |
current_generation(8) | closed_through(8) | last_event_id(32)`. A
`closed_through` that reaches the current generation is refused on read — the
current generation is always open by construction, so such a cutoff would
authorize cleaning associations still being written.

A malformed file is an **error, never a partial read**: an under-reporting index
is indistinguishable from a clean session, so nothing would ever be released and
no symptom would ever appear.

---

## Test status against Alden's required list

| Required before delete mode | Status |
|---|---|
| Reused session key; old-generation cutoff cannot touch new generation | **covered** — driven through the real close lifecycle |
| Corrupt / truncated / forged magic / forged version deletes nothing | **covered** |
| Newline, spaces, unicode, empty session keys | **covered** |
| Lost registry mints a new incarnation; old associations disowned not adopted | **covered** |
| Close is CAS; stale and future callers mutate nothing | **covered** |
| Corrupt registry errors rather than minting a fresh session | **covered** |
| Missing index write leaks only | partial — read side only |
| A and B share M; compact A → M still loadable for B | **not done** |
| Compact last association; N shares block X → X survives | **not done** |
| Existing-path `write_manifest` interleaved with cleanup → no dangling association | **not done** |
| Duplicate cleanup idempotent | **not done** |

The four outstanding are all delete-path tests. They cannot be written until the
ownership protocol exists — which is the point.
