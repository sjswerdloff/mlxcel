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

### 2. `write_manifest`'s existing-path fast return must join the protocol

Verified in source: `write_manifest` returns `Ok(())` on `manifest_dir.exists()`
**before** taking its lock, and `delete_manifest` takes no lock at all. Reachable
interleaving:

1. B persists M; `write_manifest` sees M exists, returns success.
2. A's compaction deletes M.
3. B writes `B → M` on the strength of that success.
4. B holds a live association to an absent manifest.

### 3. Generation lifecycle must be server-owned and authenticated

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

Now: `magic(8) | version(4) | key_digest(32) | count(4) | count × (manifest(32),
generation(8))`. Filename is the key digest, so an unsafe path is
unrepresentable rather than sanitized. **No raw key is stored at all** —
lookup needs only the digest, which also removes an unbounded write and a
possibly-sensitive value at rest.

A malformed file is an **error, never a partial read**: an under-reporting index
is indistinguishable from a clean session, so nothing would ever be released and
no symptom would ever appear.

---

## Test status against Alden's required list

| Required before delete mode | Status |
|---|---|
| Reused session key; old-generation cutoff cannot touch new generation | **covered** |
| Corrupt / truncated / forged magic / forged version deletes nothing | **covered** |
| Newline, spaces, unicode, empty session keys | **covered** |
| Missing index write leaks only | partial — read side only |
| A and B share M; compact A → M still loadable for B | **not done** |
| Compact last association; N shares block X → X survives | **not done** |
| Existing-path `write_manifest` interleaved with cleanup → no dangling association | **not done** |
| Duplicate cleanup idempotent | **not done** |

The four outstanding are all delete-path tests. They cannot be written until the
ownership protocol exists — which is the point.
