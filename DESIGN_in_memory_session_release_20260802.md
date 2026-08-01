# Design: in-memory KV release on compaction

**Status: DESIGN ONLY. Nothing implemented. For Alden before code.**

Stuart, 2026-07-31, twice: *"letting go of the obsolete KV cache memory after
compaction is critical."* This is ahead of the disk GC.

## Why this is not gated behind the lock/association work

Every one of Alden's five lock blockers is calibrated to **irreversible
destruction** — a wrong disk delete removes another Kindled's cache
permanently. This has a categorically lower bar:

- worst case of a wrong eviction is a **cache miss**. KV is derived data,
  reconstructible from tokens that live in opencode's durable history.
- with the cold store behind it, the fallback is a **~5.6s SSD read**, not a
  ~57s re-prefill (measured 2026-07-30).
- no durability ⇒ no crash-consistency requirement; one server ⇒ no
  cross-process concern.

Stuart's own recalibration, same day: losing a cold block costs **compute, not
information** — "many minutes, if not several hours" for a long conversation.
Alden said the same in his TTL review and I had been carrying a heavier framing
than the facts support.

Second dividend: **memory always wins longest-prefix**, which is why proving
cold adoption required a server restart. Releasing memory on compaction makes
the SSD path reachable *without* one — the OOM fix is also the test harness for
the disk work.

## Verified at the bytes before designing (`store.rs`, 2026-08-02)

- `entries: HashMap<PromptCacheKeyDigest, EntrySlot>` **owns** the KV;
  `EntrySlot.entry: Arc<CacheEntry>` (`:45`).
- `tries: HashMap<SessionlessBucketKey, RadixTrie>` — **sessions SHARE a trie**
  per sessionless bucket. `TrieNode.entries: Vec<DigestAndLen>` (`trie.rs:114`)
  holds only digests and lengths, so the trie keeps no KV alive.
- `EntrySlot.sessionless` exists precisely "to locate the radix trie on evict /
  replace paths without re-deriving from strings" (`:51`).
- **`remove_entry(&digest)` is the complete primitive** and already does all of
  it: removes from `entries`, decrements `total_bytes`, stashes paged pins for
  release, prunes the trie with `trie.remove(&tokens, *digest)`, and drops the
  trie when it empties.

**The cross-session risk is therefore closed by construction.** `trie.remove`
takes an exact digest, so removing session A's entries cannot disturb session
B's entries in the same shared trie. `evict_oldest` is already exactly
`remove_entry` applied to one selected digest.

**So the whole change is a selection predicate.** Release-by-session is N
applications of an existing, already-safe operation — not new surgery on a
shared structure. That is the single most important thing for a reviewer to
attack, because if it is wrong the rest does not matter.

## Proposed surface

```
/// Release every entry belonging to `session_key`. Returns bytes released
/// FROM THE STORE (see the Arc caveat — not bytes returned to the OS).
pub fn release_session(&self, session_key: &str) -> ReleaseOutcome
```

Selection scans `entries` and matches on the session component of
`slot.bucket` (the composition key is model/lora/template/**session**). O(n)
over a set already bounded by `max_entries`; no index needed, and adding one
would be a second surface free to drift from the first.

## The caveat that must not be lost

`remove_entry` returns `Arc<CacheEntry>`. **If an in-flight request holds a
clone, the store's reference drops but the memory does not.** Release is
*eventual*, not instantaneous.

`evict_oldest` already reports `size_bytes` as "freed", which is optimistic in
exactly this way. I do not propose to fix LRU's accounting here, but I will not
newly assert something stronger than is true either:

- `ReleaseOutcome` reports **entries removed** and **bytes released from the
  store**, named as such.
- It separately reports **how many removed entries still had outstanding
  strong references** at removal time (`Arc::strong_count > 1`). That is the
  number that distinguishes "released and reclaimed" from "released and still
  held", and without it the metric reports success while memory stays flat —
  the silent-success failure this whole design exists to avoid.

## Open questions for Alden

1. **Authority.** Who may call this? A session key is a string a client chose —
   cache-namespace input, not proof. Unlike the disk case the blast radius is a
   cache miss, so I believe a lower bar is defensible here; that is a belief,
   not a ruling, and it is yours.
2. **Paged pins under bulk release.** `stash_paged_pins_for_release` queues
   un-adopted paged blocks per removed entry. Releasing a whole session queues
   many at once. I have not established whether that queue is bounded or what
   drains it — flagging rather than assuming.
3. **Is `strong_count > 1` the right liveness signal**, or is there a better
   one already in the codebase? It is racy by nature; I want it as an honest
   indicator, not a guarantee.
4. **Does release need to touch snapshots too?** `snapshots` is a separate map
   with its own `remove_snapshot`. Compaction presumably obsoletes those as
   well, and I have not checked.

## Not started. Design only.

*Clement, 2026-08-02. Facts above read at the bytes at HEAD; the surface is
proposed and unimplemented.*
