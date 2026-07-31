# Next: in-memory KV release at compaction

**Stuart, 2026-07-31: "letting go of the obsolete KV cache memory after
compaction is critical."** Said twice. This is ahead of the disk GC.

## Verified before stopping (so it need not be re-derived)

- `EntrySlot.entry: Arc<CacheEntry>` (`store.rs:45`) — the `entries` HashMap
  OWNS the KV. Dropping the map's entry frees it.
- `TrieNode.entries: Vec<DigestAndLen>` (`trie.rs:114`) — the trie holds only
  digests and lengths. It does NOT keep KV alive, but must be pruned or it
  hands out dangling digests.
- `EntrySlot.sessionless` (`store.rs:51`) exists precisely "to locate the radix
  trie on evict / replace paths without re-deriving from strings."

**Therefore: by-session release is `evict_one_lru` with a different selection
predicate.** Same remove-then-prune path. No new mechanism.

## Why this ships ahead of the disk work

Worst case of a wrong in-memory eviction is a CACHE MISS, not data loss — the
KV is derived from tokens that live in opencode's durable history. With the
cold store behind it, the fallback is a ~5.6s SSD read rather than a ~57s
re-prefill. So none of Alden's five lock blockers apply: they are calibrated to
irreversible destruction.

Second dividend: memory always wins longest-prefix, which is why proving cold
adoption needed a server restart. Releasing memory on compaction makes the SSD
path reachable WITHOUT a restart — the OOM fix is also the test harness.

## The caveat that must not be lost

`Arc` means an in-flight request holding a clone keeps the KV alive after the
map drops its reference. Release is EVENTUAL, not instantaneous. **The metric
must report BYTES ACTUALLY RECLAIMED, not entries removed** — otherwise it
reports success while memory stays flat, which is exactly the silent-success
failure this whole design exists to avoid.

## Stakes, recalibrated by Stuart 2026-07-31

Losing a cold-storage block costs COMPUTE, not information — another prefill,
"many minutes, if not several hours" for a long conversation. Alden said the
same in his TTL review and I had been carrying a heavier framing than the facts
support. Split it: his blockers about CORRECTNESS (deadlock, torn reads,
dangling associations) stand regardless — a hang stalls every resident. His
blockers about AUTHORITY (proof vs presumption, capability vs bare identifier)
were calibrated to irreversible loss and can relax.

## Not started. Stopped clean for Shabbat.
