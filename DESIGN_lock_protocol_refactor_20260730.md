# Design: one non-reentrant store-lock protocol

**Status: DESIGN ONLY. No code written. For Alden's review before implementation.**

Stuart authorised this refactor 2026-07-30 ("2: yes"). It is the precondition
Alden named for any cleanup: *association publication must move under the store
lock.* This is the design for how, sent before the code because green tests
certify an implementation and are silent about a protocol.

## Why this is the risky one

The failure mode is **not a red test**. Wrong lock ordering, or a re-entrant
acquisition, **hangs the server** — `acquire_store_lock` takes a process-wide
`RwLock` write guard *and* an `flock`, and neither is reentrant. A deadlock
during persist stalls every resident's turn with no error anywhere.

So this design gets reviewed before it exists, and the implementation gets
tested against a deliberately-provoked double-acquire before Stuart restarts
anything.

## Current state — read at the bytes, `block_cold_store.rs`, 2026-07-30

| site | lock taken |
|---|---|
| `acquire_store_lock` :668 | `STORE_MUTEX` write + file lock. **Fails closed.** |
| `write_manifest` :1371 | `acquire_store_lock` at **:1400** — *after* a path-exists fast return at :1375 and after `increment_refcount` at :1381 |
| `delete_manifest` :1539 | **none** |
| `record_session_manifest` :1055 | `session_index_lock` (in-process `Mutex`) at :1094 only |
| sweep :2138 | `acquire_store_lock`, fails closed |

Three independent defects follow, and they compose:

1. **`write_manifest` returns `Ok(())` on path-exists before taking any lock.**
   "Already written" is a *pre-lock* observation of a directory another actor may
   be deleting.
2. **`delete_manifest` takes no lock**, so it can interleave with anything.
3. **Association publication is on a different lock entirely** from manifest
   publication, so no one holds a single exclusion across "manifest exists" and
   "association names it" — which is exactly the invariant a sweep must read.

## The protocol

### Rule 1 — split every store-mutating operation into `_locked` inner + acquiring wrapper

```
pub fn write_manifest(&self, m)          -> { let _g = self.acquire_store_lock()?; self.write_manifest_locked(m) }
fn      write_manifest_locked(&self, m)  -> // assumes the lock is HELD. Never acquires.
```

Same for `delete_manifest`, `record_session_manifest`, and the sweep. **No
`_locked` function may call an acquiring wrapper** — that is the non-reentrancy
rule and it is the whole safety property. Composites call `_locked` forms only.

Enforced structurally rather than by discipline: the `_locked` forms take a
`&StoreLock<'_>` witness parameter they cannot fabricate, so "called without the
lock" is a compile error rather than a hang. *This is the part I most want
reviewed — it is the difference between a convention and a guarantee.*

### Rule 2 — the composite is the only thing that spans both

```
pub fn persist_and_associate(&self, manifest, association) -> Result<...> {
    let g = self.acquire_store_lock()?;      // exactly one acquisition
    self.write_manifest_locked(&g, manifest)?;
    self.record_session_manifest_locked(&g, association)?;
    Ok(())
}
```

This is what closes the transaction gap the handoff renamed — the
persist+association / manifest-deletion gap — and it is why that gap could not be
tested before: *the composite did not exist, so there was no contract to violate.*

### Rule 3 — the path-exists fast return moves inside

Revalidated under the lock, with the same standard the existing revalidation
already uses at :1402: **"existing" means fully committed and integrity-valid,
not path-exists.** Outside the lock it is a guess.

### Rule 4 — `session_index_lock` is subsumed, not kept alongside

Two locks guarding overlapping state is a lock-ordering hazard for no benefit.
Association state moves under the store lock and `session_index_lock` is
**deleted**. Keeping both is how the ordering bug gets written later.

## What Stuart's answers remove from scope

**Cross-process locking: not required.** His answer was "never, one server." The
file lock in `acquire_store_lock` already covers it incidentally and stays — it
costs nothing — but the two-handle and cross-process tests Alden listed are **not
prerequisites**, and I am not building them. *If that assumption ever changes,
this is the paragraph that becomes wrong.*

## Ordering against the format freeze

Stuart also said versioning of the cold store will be removed, making a format
change cost a migration or a full cache clear. **Alden's whole-record integrity
digest is a format change.** It should land *before* the format freezes, which
puts it ahead of the close endpoint in my queue.

## Tests before this is believed

- **Double-acquire provokes a hang → must instead fail to compile.** The witness
  parameter is the claim; a test that tries to call a `_locked` form without one
  is a compile-fail test, not a runtime test.
- **Composite atomicity at a deterministic seam:** manifest committed but
  association write fails → no dangling association, and the manifest is either
  absent or unreferenced-but-valid. Never "association names a manifest that is
  not there."
- **Interleave `write_manifest(existing M)` with `delete_manifest(M)`** at the
  seam. Legal outcomes: M present and referenced, or M absent and no association
  names it. Nothing else.
- **Sweep sees a consistent set:** an association published concurrently is
  either fully visible or not visible; never a torn read.
- **Mutation:** remove the witness from one `_locked` form → the compile-fail
  test goes green (i.e. the guard stops guarding) → that must itself be caught.

## Open questions for Alden

1. **Is the witness-parameter approach right, or over-built?** It converts a
   discipline into a compile error, which is the detection→unrepresentable move
   you have pushed me on three times. But it touches every signature. If a
   plain naming convention plus one runtime debug-assert is the better trade
   here, say so — I am biased toward the heavier fix because you have been
   right about that bias.
2. **Should `delete_manifest` acquire, or only ever be reachable from a
   composite that already holds?** Public-and-acquiring is friendlier; only-
   reachable-from-composite is safer. I lean safer, but it changes the API.
3. **Does subsuming `session_index_lock` widen the critical section
   unacceptably?** Association writes are small, but they now serialise against
   every manifest publication on one server.

*Clement, 2026-07-30. Current-state table read at the bytes; the protocol is
proposed and unimplemented.*
