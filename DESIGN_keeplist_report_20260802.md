# Design: the keep-list report

**Status: DESIGN. Read-only tool, nothing implemented yet. For Alden before code.**

Stuart, 2026-07-31: *"I need a way to know which data I should manually delete
to effect the equivalent of a session-release... what I might end up having to
do is list the X-Session-Ids that I do not want deleted and get back the rest."*

This is the report half. It deletes nothing and it is not the GC. It exists so
that the first real deletion on this store is one a human authorized against a
list he read.

## Why the keep-list form is structurally better, not just Stuart's preference

**The raw session key is never stored.** `session_index_path` derives a path
from the key; the file itself carries only a 32-byte `key_digest`
(`block_cold_store.rs:901,907`, and `session_associations` at `:1148` matches on
that digest). So from the store alone, sessions are digests and nothing more.

That kills the delete-list form. *"Delete session X"* requires identifying X,
and *"show me what is deletable"* requires naming conversations the store cannot
name. An operator would be choosing from a list of hex strings.

The keep-list form needs identification **only of what is kept** — and the
operator already knows those, because they are the conversations he is in. He
types the ids he recognizes; everything he cannot name is by construction
something he is not protecting.

**So the asymmetry runs the right way: identification is required exactly where
the operator has it.**

## The three buckets, and the third is the one that matters

| bucket | definition | proposed action |
|---|---|---|
| **KEEP** | referenced by ≥1 kept session | never touched |
| **RELEASABLE** | referenced by ≥1 session, and **every** referencing session is unkept | reported as deletable |
| **UNATTRIBUTED** | referenced by **no** session index | **reported separately, never proposed for deletion** |

Classification is per **manifest**, not per session. Manifests are
content-addressed and prefixes are shared, so one manifest can be referenced by
a kept session and an unkept one at once. A manifest touched by any kept session
is KEEP.

### Why UNATTRIBUTED must be its own bucket

A manifest with no session association is **indistinguishable from** a manifest
whose association was lost. Both present as absence, and absence here is
structural, not observable:

- persisted before session tracking existed;
- association lost to the crash window `session_associations` already documents
  at `:1132-1135`;
- an index file whose `key_digest` is `None` — the field is `Option`, so an
  older file cannot be matched against a keep-set **at all**, and its manifests
  fall out of every attributed set silently;
- an index whose digest mismatches, which `session_associations` deliberately
  treats as empty rather than trusting.

Folding UNATTRIBUTED into RELEASABLE would delete cache belonging to a live
conversation whose bookkeeping failed. Folding it into KEEP would hide a leak
behind a reassuring report. **It is a third answer and the report says so.**

Existing code already chose this direction — `releasable_manifests` at `:1174`
notes that entries from a lost lifetime *"remain roots and leak, which is the
safe direction."* This report inherits that and makes the leak visible instead
of merely safe.

## Why this cannot reuse `releasable_manifests`

`releasable_manifests(key)` returns empty unless the session's registry has a
`closed_through` generation (`:1192`). **No generation has ever been closed on
this store**, because the close mechanism does not exist yet. So it correctly
returns empty for every session, and would report that nothing is deletable.

That function answers *what has this session finished with?* The keep-list
answers *what is not protected by anyone?* Different questions; the second is
the one available before any close exists.

## Output

Three sections, and a block/byte accounting for RELEASABLE only:

```
KEEP           N manifests   (from M named sessions, K matched, K' not found)
RELEASABLE     N manifests   →  B blocks, X GiB reclaimable
UNATTRIBUTED   N manifests   →  B blocks, X GiB  [NOT proposed for deletion]
```

**`K' not found` is load-bearing.** A typo'd session id silently protects
nothing while looking like it protected something. Every named id is reported as
matched or unmatched, and a run with any unmatched id exits non-zero.

Reclaimable blocks are those referenced by RELEASABLE manifests and by **no**
KEEP or UNATTRIBUTED manifest — the same reachability rule `mark_reachable_blocks`
uses (`:1931`), evaluated against a hypothetical root set rather than the real
one. Nothing is marked, tombstoned, or written.

## What it needs from `mlxcel-core`

Two additions, both read-only:

1. `pub fn enumerate_session_indexes(&self) -> Result<Vec<(SessionIndexDigest, Vec<SessionAssociation>)>, ColdStoreError>`
   — reads `sessions_dir()`, parses each index, returns the digest **from inside
   the file** rather than from the filename. A file with `key_digest: None`
   is returned with its digest as `None` so the caller must handle it, rather
   than being silently dropped into an attributed set.
2. `pub fn key_digest_of(session_key: &str) -> [u8; 32]` — exposes the existing
   private `session_key_digest` so a tool can build the keep-set without any raw
   key reaching disk.

No existing behaviour changes.

## What this deliberately does NOT do

- **No deletion, no `--force`, no `--yes`.** The delete path is a separate change
  and gets its own review. A report that can delete is not a report.
- **No TTL, no age, no LRU.** Sessions are resumable and an idle one is not a
  finished one — the same reason the whole GC design is compaction-scoped.
- **No claim that RELEASABLE is safe to delete.** It is what the keep-list does
  not protect. Whether the keep-list was right is the operator's judgment, which
  is the entire point of putting it in front of him.

*Clement, 2026-08-02. Design only. Facts read at the bytes at `4acebc8`.*
