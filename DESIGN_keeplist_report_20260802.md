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
- an index whose digest mismatches, which `session_associations` deliberately
  treats as empty rather than trusting.

> **RETRACTED, 2026-08-02, before implementation.** A fourth cause appeared in
> the first draft of this document: *an index file whose `key_digest` is `None`,
> so an older file cannot be matched against a keep-set at all.* **It does not
> exist.** `SessionIndexFile.key_digest` is `Option` because `None` means *the
> file is absent* — `read_session_index_file` returns it from the
> `read_bounded` miss at `:3619-3623`. Every file that parses carries a digest
> at bytes `12..44` of a fixed-width header (`:3638-3639`), so there is no
> legacy shape without one.
>
> I inferred a legacy-file case from the presence of an `Option` and shipped it
> to a reviewer as the most interesting of the four. It is a claim about what a
> mechanism **reaches**, which is the class I already know I under-check.

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

---

# REVISION 2 — against Alden's review of `84edcb4`

He returned four P0s and the verdict *do not use this report to select data for
manual deletion.* All four are accepted. What changed:

**RELEASABLE is renamed UNPROTECTED, and no byte figure is called
"reclaimable".** The bucket only ever meant *nothing in THIS keep list
references it*. An unnamed session may be active, resumable, unknown to the
operator, or forgotten. The design said as much and the artifact said the
opposite — the finding lived where I last wrote rather than where a reader
lands.

**An empty keep list is refused** unless `--keep-none` is passed. The previous
version had a *test asserting* that an empty list reports every attributed
manifest as unprotected and exits 0, so an unset shell variable would have
produced a whole-store report that read as authoritative. The test certified
the hazard.

**Quiescence is required and cannot be proved.** `record_session_manifest`
serialises on a **process-local `Mutex`**, not the cross-process `flock` on
`store.lock` — verified at the bytes. So a separate binary cannot exclude
session index writes and cannot take a coherent snapshot. `--store-is-quiescent`
is now mandatory, recorded in the artifact as *operator-asserted*, and
contradicted by a `pgrep` probe that **fails closed**: a probe that could not
run is refused rather than treated as a probe that found nothing.

*The real fix is to extend the cross-process lock to cover session index
writes.* That is a serving-path change and not this tool's to make. It would
also close an undocumented single-writer-process assumption: today, mutual
exclusion for session indexes rests on a shell script's pre-flight check.

**Path/header agreement is checked, not trusted.** `session_associations(key)`
compares a file's header digest against an *expected* one; an enumeration has no
expectation, so the filename stem is the only independent witness. A file placed
or symlinked under one session's name while claiming another would otherwise
expose the first session's manifests as unprotected. A mismatch fails the whole
report and is never converted to absence. **My design's earlier claim that
"digest mismatch is read as empty" applies here was wrong** — that behaviour
belongs to the keyed lookup, not the enumeration.

**An `--artifact` emits the exact sorted manifest and block hashes** with store
path, code commit, quiescence basis, keep-list labels and completeness status.
Aggregate counts cannot authorize specific objects, and a later tool recomputing
the set would not inherit this review. The artifact is evidence for a separate
deletion decision; it is not a grant.

**Session ids stay out of the report body.** Unmatched entries are identified by
input line and a 12-hex label. `--keep-file` is preferred over `--keep` because
argv is world-readable. Duplicates are refused rather than silently skewing the
named-versus-matched counts.

## Two bugs the revision introduced, both caught by RUNNING it

**`--store` level was inverted.** I validated that `--store` names
`cold-storage-v4`. `BlockColdStore` appends that root itself, so the validated
form produced `cold-storage-v4/cold-storage-v4`, enumerated nothing, and printed
a clean zero report — *exactly the P1 the validation was written to prevent*.
The contract was in code I had already read this same session.

**The quiescence probe failed open.** `.unwrap_or(false)` turned "could not
check" into "no server found". Now a three-way result: match, clean no-match,
or could-not-run — and the third refuses.

## Still open, and not claimed as done

- **Filesystem and CLI integration tests.** The eight unit tests cover the pure
  classifier and the label; the guards were exercised by hand against the live
  store and their exit codes confirmed, which is not a regression barrier.
- **The happy path has never run.** Every index on the only real store is
  pre-seal (v2), so the tool has never produced a complete report against real
  data. It fails loud, which is right, and it is not evidence the report works.
- **The quiescence refusal has no positive control.** The server was down for
  every run, so `ServerRunning` has never been observed firing.
