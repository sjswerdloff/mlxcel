# Design: one non-reentrant store-lock protocol — rev 2

**Status: DESIGN ONLY. Nothing implemented. Rev 2 answers Alden's five
pre-implementation blockers (2026-07-30 19:45).**

Rev 1 proposed a `&StoreLock<'_>` witness parameter and claimed it converted a
discipline into a guarantee. It does not. Alden's review is accepted in full;
this rev is the redesign, not a defence. **The single largest change: the
refactor cannot land yet, because blocker 5 is not satisfied and I had believed
it was.**

---

## Alden's blockers — disposition

| # | Blocker | Disposition |
|---|---|---|
| 1 | `&StoreLock` proves less than claimed | **Accepted.** Redesigned: two store-bound guard types + runtime reentrancy refusal. §3 |
| 2 | One lock is exclusion, not a durable transaction | **Accepted.** Composite now writes durable pin evidence before the association attempt. §4 |
| 3 | Other authority mutations omitted | **Accepted.** Three `session_index_lock` sites named, plus a dead `persist_lock` I had missed. §2, §5 |
| 4 | A lock witness is not deletion authority | **Accepted.** Private guarded delete; and the live `prune → delete` path is a present correctness bug, §6 |
| 5 | "Association writes are small" is not a bound | **Accepted, and it blocks this refactor.** What I landed tonight bounds the *read*, not the *critical section*. §7 |

Format ruling: agreed, and already reconciled with Stuart — magic/version bytes
stay. Rev 1 built a priority argument on a misreading of his words; he meant
versioning *directories, env vars and dead code*, not durable format identity.
The whole-record digest still lands before any freeze.

---

## §1 Why this one is the risky refactor

The failure mode is **not a red test**. Wrong ordering or a re-entrant
acquisition **hangs the server** — `acquire_store_lock` (`:668`) takes a
process-wide `RwLock` write guard *and* an `flock`, neither reentrant. A
deadlock during persist stalls every resident's turn with no error anywhere.

Rev 1 wrote that a double-acquire "must instead fail to compile." That is
wrong and it is Alden's sharpest correction: Rust cannot express *"code holding
this guard never calls an acquiring wrapper."* A compile-fail fixture can prove
a missing or wrong witness; it cannot prove non-reentrancy. The claim shrinks
to what is provable, and the rest becomes a runtime refusal.

> **A hang is never the test oracle.** — Alden. Every non-reentrancy test
> asserts a *named error returned before blocking*, never a timeout.

---

## §2 Current state — re-read at the bytes, HEAD `16ddda4`

`block_cold_store.rs`:

| site | lock |
|---|---|
| `acquire_store_lock` `:668` | `STORE_MUTEX` write + `flock`. Fails closed. |
| `try_acquire_store_lock` `:750` | non-blocking variant |
| `write_manifest` `:1371` | acquires at **`:1400`** — *after* a path-exists fast return `:1375` and `increment_refcount` `:1381` |
| `delete_manifest` `:1539` | **none** |
| `prune_prefix_manifests` `:1592` | calls `delete_manifest` at **`:1720`** |
| `persist` `:1897` | calls `prune_prefix_manifests` |
| `session_registry` `:929` | `session_index_lock` — authority **read** |
| `close_current_generation` `:985` | `session_index_lock` |
| `record_session_manifest` `:1055` | `session_index_lock` |
| sweep `:2138` | `acquire_store_lock`, fails closed |

Two facts rev 1 did not contain, found by enumerating every `.lock()` receiver
rather than by reading for the ones I expected:

- **`persist_lock` (`:579`, initialised `:615`) is never acquired.** Exactly
  three `self.*.lock()` calls exist in the file and all three are
  `session_index_lock`. The field carries a doc comment (`:582`, `:584`)
  describing a lock ordering that does not exist. It is dead state that reads
  as protection. **Delete the field and the comment** — separately, before the
  refactor, so it cannot be mistaken for part of the new protocol.
- **`session_registry` is a third site**, and it is a *read* of authority
  state. Rev 1 listed only the two mutations. A reader that is not under the
  same exclusion as the writers can observe a torn index.

The three composing defects from rev 1 stand, restated:

1. `write_manifest` returns `Ok(())` on path-exists **before any lock** — a
   pre-lock observation of a directory another actor may be deleting.
2. `delete_manifest` takes **no lock at all**.
3. Manifest publication and association publication are on **different locks**,
   so nothing holds one exclusion across *"manifest exists"* and *"an
   association names it"* — precisely the invariant a sweep must read.

---

## §3 Blocker 1 — capability, store identity, and reentrancy

### 3.1 Two types, not one enum

Today `StoreLock` (`:545`) is one struct wrapping
`StoreGuardKind::{Exclusive, Shared}` (`:554`) and carries **no store
identity**. So a read lease satisfies a mutating `_locked` signature, and a
guard from store A can be passed alongside `&store_b`.

```rust
pub(crate) struct StoreReadLease<'s>  { store: &'s BlockColdStore, /* pair */ }
pub(crate) struct StoreWriteGuard<'s> { store: &'s BlockColdStore, /* pair */ }
```

**Mutating inner operations become methods on `StoreWriteGuard` and take no
separate `&BlockColdStore`.** Wrong mode and wrong store stop being errors to
detect and become states that cannot be written.

`StoreWriteGuard` is deliberately `!Send + !Sync` (a `PhantomData<*const ()>`),
so it cannot be moved to another thread and the thread-local detector below is
sound.

**Preserving what the enum was actually for.** The comment at `:550-553`
explains the enum exists so a caller cannot hold a *shared* process guard while
taking an *exclusive* file lock. That pairing must survive the split — and it
does, because the pairing was never enforced by the enum. It was enforced by
the constructor acquiring both halves together. Each new type keeps exactly one
private constructor that acquires its matched pair. *Recording this because a
later reader of that comment could reasonably revert the split to "restore" a
property the split does not remove.*

### 3.2 Non-reentrancy is a runtime refusal, not a type

Rust cannot express it. So:

```rust
thread_local! { static STORE_DEPTH: Cell<u32> = const { Cell::new(0) }; }
```

Every acquiring wrapper checks the depth **before** touching `STORE_MUTEX` or
`flock`, and returns `ColdStoreError::StoreLockReentrancy { .. }` when it is
non-zero. Never blocks first and diagnoses later. Thread-local is sound
precisely because the guard is non-transferable.

This is production code, not `#[cfg(test)]`: the whole value is that a
reentrancy bug reaching a resident's turn surfaces as a named error in the log
instead of a silent stall.

Defence in depth, explicitly *not* the proof: a source-level check forbidding
acquiring-wrapper names inside the guard `impl`.

**What is proven, stated at its real size:** a compile-fail fixture proves a
mutating operation cannot be called without a write guard, and cannot be called
with a read lease. The reentrancy detector proves a re-acquisition *returns
rather than hangs*. Neither proves the absence of re-entrant call paths. There
is no existing trybuild/compiletest convention in this repo, so the fixture is
new infrastructure that must be justified on its own.

---

## §4 Blocker 2 — a durable transaction, not just exclusion

One acquisition spanning both publications is the right *concurrency* boundary
and remains. It is not a *crash* boundary. Committing M then writing A leaves,
on failure or crash after M's rename, a committed M with no A — and plain `Err`
cannot imply "no commit."

Alden's counterexample is the one that kills the simple version: **M may
already carry session A's CLOSED association while session B's live association
write fails.** A later scan sees only A-closed, concludes unreferenced, deletes
M. So `PersistedUnassociated` is safe only with durable positive evidence that
cleanup must honour.

### 4.1 Proposed evidence: a per-manifest pin

Written under the write guard **before** the association attempt; removed under
the write guard **after** the association commits. Every cleanup pass must
retain a pinned manifest — retain, not "consider."

Chosen over a store-level cleanup-unhealthy marker for one reason grounded in
this machine: a leaked store-marker disables **all** reclamation from the first
incident, and the disk is nearly full. A leaked pin retains **one** manifest
per incident — proportional, not catastrophic.

And the pin is self-healing, which composes with Alden's `ExistingValid`
requirement: a later `persist_and_associate` naming M that *does* commit its
association clears M's pin. Without that, a single association failure pins a
manifest for the life of the store.

If the pin itself cannot be written: **do not publish M.** Fail closed.

### 4.2 Never roll back by deleting M

M may pre-exist, be shared, or be in use by traffic that recorded nothing. The
composite returns an explicit partial outcome preserving the association error
— not an ordinary all-or-nothing `Err`.

### 4.3 Existing M

Validate M *and all referenced blocks* under the write guard, return
`ExistingValid`, and **still attempt association publication** — otherwise a
prior association failure can never heal. Refcount hints must **not** increment
again on `ExistingValid`.

---

## §5 Blocker 3 — every authority mutation moves, or none of them do

`session_index_lock` may be deleted only after **all** of these are under the
store write guard:

- `record_session_manifest` (`:1055`)
- `close_current_generation` (`:985`)
- `session_registry` (`:929`) — read, under a **read lease**
- registry creation
- association release
- any future expiry/touch mutation
- final cleanup proof

`persist_and_associate` **consumes and validates the admission-time opaque
ticket**. It must not re-read the current generation, and must not accept a
caller-built association. Deleting `session_index_lock` before every one of
these moves is unsafe, and rev 1's Rule 4 said "deleted" with no such
precondition.

---

## §6 Blocker 4 — deletion authority, and a live bug

A lock witness proves *exclusion*, never *permission*. So:

- `delete_manifest` becomes **private** on `StoreWriteGuard`.
- Only **policy-specific** operations are public; each derives and revalidates
  its own authority under the guard.

Rev 1 offered two answers and Alden rejected both. He is right: a public
acquiring wrapper makes arbitrary-hash deletion easy, and locked-only proves
exclusion but not permission.

### The live bug

`persist` (`:1897`) → `prune_prefix_manifests` (`:1592`) → `delete_manifest`
(`:1720`) **exists today**. In Delete mode it can delete an older manifest
while session associations still name it, leaving `association → absent
manifest`. Prefix proof establishes that the *new* sequence makes the old cache
redundant; it does **not** prove every other session released the old manifest.

`PruneMode::Observe` is the default, which limits present reach — but a default
is a choice, not a property. Before association data can be authoritative,
prefix deletion must either become association-aware, be separately defined as
lossy eviction with coherent association handling, or be **structurally**
unavailable rather than merely unselected.

**Recommendation:** make `PruneMode::Delete` unconstructible until prefix
pruning is association-aware. **Flagged for Stuart** — this removes an existing
knob, and that is his call, not mine. My read is that the knob is presently
unsafe rather than merely unused.

---

## §7 Blocker 5 — this refactor is blocked, and I was wrong about why

I landed `read_bounded` + `MAX_SESSION_INDEX_BYTES` (64 MiB) tonight
(`:3776`, `:3784`) and treated the resource-bounds blocker as addressed.

**It is not.** That bound stops an unbounded *allocation*. It does nothing
about the *critical section*: 64 MiB of parse plus a full index rewrite,
serialised under a process-wide exclusive lock, on every publication and every
load-exclusion path. One server does not make an unbounded hold acceptable —
it makes it *global*.

What is actually required before this refactor lands:

1. A bound on index **entries**, not just bytes.
2. Append-mostly recording, or some other end to full-file rewrite under the
   guard.
3. A **measured** worst-case hold time at the bound. I have not measured it.

Until then the honest answer to rev 1's open question 3 is **no** — subsuming
`session_index_lock` widens the critical section unacceptably.

---

## §8 Scope: what Stuart's answers do and do not remove

Stuart: *"never, one server."* That removes the **second-process**
prerequisite. Rev 1 then wrote that the two-handle tests were "not
prerequisites" — **that was an overclaim, and it is withdrawn.** One server
still runs many threads and can hold many handles to one store. The
same-process two-handle/thread test stays required; only the cross-process test
is removed.

Stuart: localhost, no auth, for now. An explicit accepted scope, not a blocker.

---

## §9 Test matrix — Alden's twelve, adopted verbatim in substance

1. Exclusive and shared capabilities are distinct; a read lease cannot call a mutation.
2. A guard from store A cannot mutate store B (methods use the bound store).
3. Nested write acquisition, and write-while-read-held, return **deterministic reentrancy errors, never hang**.
4. The real scheduler production path calls the composite — not `persist` then `record`.
5. Admission ticket vs close interleavings yield only old-generation refusal, or a committed open-generation association. Never a relabel.
6. Association failure after new **and** existing M leaves durable pin evidence — including the A-closed / B-write-failed counterexample.
7. Existing-M / delete interleaving yields valid M+A, or absent M with no association. Pre-lock existence never decides.
8. Same-process two-handle/thread record, close, and proof-read.
9. Prefix-prune of an association-named M cannot leave a dangling association.
10. Sweep/publication interleavings use deterministic seams, and the **real** scheduler is proven to reach them.
11. Corrupt/oversize index under the global guard aborts within the configured bound.
12. Mutation: replace one guard-method call with an acquiring wrapper → the reentrancy test fails deterministically. Removing a witness is a *separate* compile-shape mutation, not the same guarantee.

---

## §10 What remains open

1. **Pin vs store-marker** (§4.1) — I chose the per-manifest pin on this
   machine's disk economics and on self-healing. Alden's ruling wanted one of
   the two established; I have not had his ruling on *which*.
2. **`PruneMode::Delete`** (§6) — disable structurally, or fix forward to
   association-aware pruning? Stuart's call.
3. **Index bound shape** (§7) — entry cap, append-mostly, or both, and what
   hold time is acceptable. Needs a measurement I have not taken.

## Ordering

1. Delete the dead `persist_lock` field (independent, trivial, removes a false signal).
2. Bound the index — entries and hold time (§7). **Gate for everything below.**
3. Guard types + reentrancy detector (§3).
4. Composite with pin evidence (§4), all authority sites moved (§5).
5. `session_index_lock` deleted — last, not first.
6. Deletion authority + prune decision (§6).

*Clement, 2026-07-30, rev 2. State table re-read at the bytes at `16ddda4`;
`persist_lock` and the third authority site found by enumerating every
`.lock()` receiver rather than searching for the ones I expected. Design only —
nothing here authorises implementation, endpoint wiring, format removal, cache
clear, or deletion.*
