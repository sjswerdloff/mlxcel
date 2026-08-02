# Design: the keep-list report

**Status: REVISION 4. Read-only DIAGNOSTIC tool. Not evidence, not deletion
authority.** Implemented at `src/bin/cold_store_keeplist.rs`; 14 tests.

Stuart, 2026-07-31: *"I need a way to know which data I should manually delete
to effect the equivalent of a session-release... what I might end up having to
do is list the X-Session-Ids that I do not want deleted and get back the rest."*

> **The superseded revision-1 contract — `RELEASABLE`, "reported as deletable",
> "GiB reclaimable" — is quarantined at the bottom of this file under
> HISTORICAL. It must not be read as a specification.** Revision 2 appended its
> corrections and left the rejected contract as the first thing a reader met,
> which is the failure mode this whole document is about. (Alden, 2026-08-02.)

## What the report answers, and what it does not

**One question: which manifests were referenced only by sessions the operator
did not name in this run?** That is an observation about *this invocation's keep
list*. It is not a claim that anything is safe to delete — an unnamed session may
be active, resumable, unknown to the operator, or simply forgotten.

Hence the bucket is **UNPROTECTED**, no byte figure is called *reclaimable*, and
the program calls its own output a diagnostic report rather than evidence.

## Why the keep-list form is structurally forced

**The raw session key is never stored.** The index filename is the key's digest
and the file's header carries only that digest. From the store alone, sessions
are digests.

That kills the delete-list form: *"show me what is deletable"* would offer an
operator hex strings to choose between. A keep-list requires identification
**only of what is kept** — the conversations the operator is in. Identification
is required exactly where the operator has it.

## The three buckets

Per **manifest**, never per session. Manifests are content-addressed and
prefixes are shared, so one manifest can be referenced by a kept session and an
unkept one at once.

| bucket | definition | what happens to it |
|---|---|---|
| **KEEP** | referenced by ≥1 named session | protected; its blocks are roots |
| **UNPROTECTED** | referenced by ≥1 session, and **every** referencing session was unnamed | listed, with its blocks; **not** called deletable |
| **UNATTRIBUTED** | referenced by **no** session index | protected; its blocks are roots; never proposed |

**Kept wins.** A manifest touched by any named session is KEEP. This is the
safety property and it is mutation-verified: swapping the two arms reddens
`shared_manifest_is_protected_by_any_keeper` and only that test.

### Why UNATTRIBUTED is its own answer

A manifest with no association is **indistinguishable from** one whose
association was lost — persisted before session tracking, lost to the crash
window `session_associations` documents, or belonging to an index whose digest
did not match. Folding it into UNPROTECTED would expose a live conversation's
cache; folding it into KEEP would hide a leak. Its blocks join the **protected**
root set.

## Quiescence: required, asserted, never proved

`record_session_manifest` serialises on a **process-local `Mutex`**, not the
cross-process `flock` on `store.lock`. **A separate binary therefore cannot
exclude session index writes and cannot take a coherent snapshot.** Alden's
interleaving: the report reads unkept session U referencing manifest M; kept
session K then adds a reference to M; the report enumerates M and classifies it
from the stale read; M is reported as unprotected while a named conversation
depends on it.

So `--store-is-quiescent` is **mandatory**, printed with the report as
`OPERATOR-ASSERTED`, and never described as verified. A `pgrep` contradiction
probe refuses on a positive, and **fails closed** on any outcome that is not a
clean no-match: a check that could not run is not a check that found nothing.

**The report is stale the moment anything writes to the store.** A later
deletion boundary must revalidate these exact roots under authoritative
cross-process exclusion.

*The real fix is to extend the cross-process lock to cover session index
writes.* That is a serving-path change, not this tool's to make, and it would
also close an undocumented single-writer-process assumption: mutual exclusion
for session indexes currently rests on a shell script's pre-flight check.

## Identity checks

**Path/header agreement, on the path actually enumerated.** `enumerate_session_indexes`
returns the file's own path alongside the digest its header claims. The stem
must be 64 hex characters and must equal that digest. Revision 2 instead derived
the filename that *should* hold the digest and checked whether such a file
existed — a different question, which `a.idx` claiming digest `b` passes
whenever `b.idx` also exists. Mutation-verified.

**Symlinks are rejected at enumeration**, where `DirEntry::file_type` can still
see them. Once opened, a link to another session's index is indistinguishable
from that session's own file.

**A mismatch fails the whole report** and is never converted to absence.

## No durable evidence artifact — WITHDRAWN in revision 4

Revision 3 wrote a sealed `--artifact`. **It is withdrawn, not repaired.**

Alden's writer review found that it contained
`let _ = std::fs::remove_file(&tmp)` on a predictable sibling path with the
error discarded — **an unconditional delete of a file the invocation may not
have created, inside a tool whose entire contract is that it deletes nothing.**
It also claimed no-clobber semantics it did not have: it checked the
destination was absent and then used `rename`, which replaces.

Both were mine, and the first is the sharpest thing anyone caught: I built this
report so that the first deletion on the store would be one a human authorized,
and put an unauthorized delete in it.

**Why withdrawn rather than fixed.** The artifact was built ahead of its own
precondition. Alden's original requirement was an immutable artifact *after the
snapshot problem was solved*; quiescence here is operator-asserted and cannot
be proved, so a sealed artifact would attest to a snapshot nobody can
establish. No deletion tool consumes it. Three of the last four defects came
from durable-evidence machinery with no consumer.

The report prints to stdout, including the exact manifest and block hashes —
counts alone would leave Stuart's manual-selection need looking satisfied while
it was silently deferred. **Capturing that output is deliberately not
recommended here:** ordinary `>` truncates its destination before this process
starts and can be pointed inside the store, which moves the destructive step
into the shell and out of review.

**The invariant this leaves, and how it is enforced:** the production path
performs no destructive filesystem call. That is held by review, not by a test
— a search of this source for `remove_file` cannot tell a call from a mention,
and fires on the withdrawal notice that quotes the offending line. Absence is a
claim about structure; a string search cannot establish it. Revision 4 briefly
shipped exactly that green-but-vacuous test.

## Failure behaviour

| condition | result |
|---|---|
| any manifest unreadable | **all** block and byte figures withheld; exit 3 |
| any block unsizeable, or the total overflows | byte total withheld; exit 3 |
| index vanished mid-scan | exit 3 — the store was not quiescent |
| stem/header disagreement, symlink, bad store level | exit 2 |
| a named session matched nothing | counts printed, **object identities withheld**, exit 1 |

### The emission decision table

Which outcome may emit what is **one decision made once**, in `decide`. Revision
4 put the object identities inside a renderer that `Unmatched` also called, so a
typoed keep id classified with less protection than the operator intended,
printed the complete candidate list, and then announced that identities were
withheld — revision 2's artifact-ordering failure in a new medium. Fixing an
ordering in one place and reintroducing it by adding a feature to a shared
renderer is the shape to watch for.

| outcome | `bytes` | counts | object identities | exit |
|---|---|:---:|:---:|:---:|
| Complete, all matched, all sized | `Some` | yes | yes | 0 |
| Complete, all matched, per-block sizing failed | `None`, unsized > 0 | yes | yes, total withheld | 3 |
| Complete, all matched, **aggregate overflow** | `None`, unsized = **0** | yes | yes, total withheld | 3 |
| Unmatched keep id, fully sized | `Some` | yes | **no** | 1 |
| Unmatched keep id, sizing failed | `None` | yes | **no** | **3** |
| Incomplete scan | — | no actionable classification or byte figures | no | 3 |
| Refused validation or quiescence | — | no | no | 2 |

**Exit status keys on `bytes.is_none()`, not on `unsized_blocks > 0`.** Aggregate
overflow also yields no total while leaving that count at zero, so keying on the
count printed `SIZE WITHHELD` and exited **0** — contradicting both the usage
text and this table. `unsized_blocks` stays the honest count it is rather than
being falsified to drive control flow.

**Unmatched *and* size-incomplete exits 3, not 1.** Both conditions are real and
the more conservative machine-readable status dominates; a caller keying on 1
would treat a size-incomplete run as merely mis-typed. Identities stay withheld
either way, because the keep input did not match.

**The sizing-failure row prints identities deliberately.** A block that cannot
be sized does not invalidate manifest or block *reachability* — the
classification is complete and the identities are coherent. Only the byte total
is withheld, and exit 3 says the run is not clean.

Pinned by `emission_decision_table`, which asserts on emitted **text** rather
than on the returned enum: the regression lived in the renderer, where an
enum-level test could not see it.

**Its first version covered four of seven rows, and the three it omitted were
the sizing rows — which is exactly why the exit-0 overflow bug survived the
barrier this test was written to be.** A test named for a decision table asserts
coverage of that table by its name. The sizing rows are now synthesised from a
constructed `Report` rather than provoked, since an 18-EiB filesystem is not
needed to pin what `decide` does with `bytes: None`. The Refused and Incomplete
exits are asserted exactly; the first version accepted 2 *or* 3 for either, so
swapping them would have stayed green.

Mutation-verified twice: calling `render_identities` from the `Unmatched` arm
reddens the identity assertion, and keying the exit back on `unsized_blocks`
reddens the overflow row while every other row stays green.

An unreadable manifest cannot prove what it references, and an unknown root can
overlap any candidate — so no figure is approximated. A qualified number still
gets read as a number.

## What it needs from `mlxcel-core`

Three read-only additions, no existing behaviour changed: `key_digest_of`,
`enumerate_session_indexes` (returning path + claimed digest + associations, and
rejecting symlinks), and `enumerate_manifest_hashes`.

## What this deliberately does NOT do

- **No deletion, no `--force`, no `--yes`.** A report that can delete is not a
  report.
- **No TTL, age or LRU.** Sessions are resumable; an idle one is not finished.
- **No claim that UNPROTECTED is safe to delete.**

## Known gaps, stated rather than implied

- **The happy path now runs against a real fixture**, built with the store's own
  `write_manifest` / `record_session_manifest`. It has still never produced a
  complete report against the *production* store, because every index there is
  pre-seal — see `FINDING_session_authority_version_skew_20260802.md`.
- **The quiescence refusal has no live positive control.** `ServerRunning` is
  unit-tested against a synthetic `pgrep` result; it has never been observed
  firing against an actual running server.
- **Fixture block directories carry a payload file but no real KV data.** That
  is sufficient for a report that only enumerates and sizes, and insufficient
  for any test of block I/O.
- The legacy diagnostic repair **landed** at `e777097`: magic and version are
  checked at fixed offsets before the seal, and a pre-seal file now reports an
  unsupported/unsealed version rather than tampering. What remains pending on
  Stuart is the **migration policy** — what should happen to v1/v2 files.
- **Correction to a claim made at `43e1ba8`, now measured rather than
  attributed.** I wrote that disabling the seal comparison reddens *exactly* the
  new control. Asserted from a FILTERED run that could only see my own two
  tests. Alden identified two existing exhaustive corruption tests that catch
  the same mutation; the unfiltered run confirms him and gives the real number:

  > `FAILED. 1232 passed; 3 failed` — `every_stored_byte_of_the_index_is_covered`,
  > `every_stored_byte_of_the_registry_is_covered`, and my control.

  **Three of 1235, not one.** And the two I did not know about are the stronger
  pair: they enumerate *every* byte offset in the record and prove each is
  covered, where mine flips one offset. Under the mutation they report 158 of
  192 index bytes and 89 of 140 registry bytes as freely alterable.

  *Mutation proves necessity, not exclusivity — and a denominator asserted from
  a filtered run is not a denominator.* I had that written down.

*Clement, 2026-08-02, revision 4 against Alden's review of `d6d0ca4`.*

---

# HISTORICAL — superseded revision 1 and 2 text

**Everything below is retained for provenance and is NOT a specification.** The
revision-1 body used `RELEASABLE`, described that bucket as *"reported as
deletable"*, and printed *"GiB reclaimable"* — all three rejected in review.

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
