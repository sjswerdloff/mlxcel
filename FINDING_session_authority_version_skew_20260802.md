# Every session authority file on the live cold store is pre-seal, and the
# error calls it tampering

**Found by RUNNING the keep-list report against the production store. Read
verification passed on all of this.** Nothing was written, modified or deleted.

## What happened

```
$ cold-store-keeplist --store "/Volumes/T7 Shield/mlxcel-cold-storage/v4-blocks"
cannot enumerate session indexes: session index …d033d48….idx is malformed
(integrity digest mismatch — the record was altered after it was written.
Refusing to interpret ANY field…)
```

Deterministic across three consecutive runs. The files were last written
30 July 18:55–19:17 and nothing has touched them since, so this is not a
read/write race with the live server.

## At the bytes

```
$ stat -f %z …d033d48….idx        104
$ xxd -s 0  -l 8  …idx            MLXSIDX1       ← magic correct
$ xxd -s 8  -l 4  …idx            0200 0000      ← version 2
$ xxd -s 44 -l 4  …idx            0100 0000      ← 1 association
```

`HEADER = 8 + 4 + 32 + 4 = 48`, `ENTRY = 16 + 8 + 32 = 56`, seal = 32.

| shape | size |
|---|---|
| 1 entry, **sealed** (v3) | 136 |
| 1 entry, **unsealed** (v2) | **104** ← what is on disk |

All six `.idx` files are 104 bytes at version **2**; the code requires
`SESSION_INDEX_VERSION = 3` ("trailing whole-record SHA-256").
All six `.gen` files are 108 bytes at version **1**; the code requires
`SESSION_REGISTRY_VERSION = 2`, same reason.

**So every session authority file on the store predates the integrity seal.**

## The defect: integrity is checked before version

`read_session_index_file` runs `open_authority_record` first, then checks magic
and version. So a pre-seal file splits its last 32 bytes off as a "seal", hashes
the remaining 72, mismatches, and reports:

> *the record was altered after it was written*

An old file is diagnosed as **tampering**. The message is deliberately
field-silent — correct, so nobody "repairs" one — but it names the wrong
condition entirely, and it is the most alarming message the store can emit.

The stated reason for integrity-first is sound but over-scoped: *"until the
digest verifies, the declared count is not trustworthy, so bounding the
allocation on it would be bounding on an attacker-controlled number."* That is
true of the **count**, at offset 44. It is not true of **magic** (0..8) or
**version** (8..12) — both are fixed offsets and reading them requires trusting
no declared length.

## Impact

`session_associations` errors for every session on this store, so
`releasable_manifests` errors, so anything that walks sessions to decide
reachability errors. **The serving path never reads these files, which is why
three days passed with no symptom.** The first thing to touch them was a report
written to decide what is deletable.

## Proposed fix, in two separable parts

**1 — Diagnosis. No policy content, and correct regardless of part 2.** Check
magic and version at their fixed offsets *before* the seal; keep the declared
count behind it. A pre-seal file then reports `unsupported version 2` and an
operator knows it is a format skew rather than an altered record.

**2 — Migration. This is a decision, not an implementation detail, and it is
Stuart's.** What should happen to v1/v2 files:

- **refuse** (recommended) — those manifests stay UNATTRIBUTED forever and leak.
  Safe direction, consistent with what `releasable_manifests` already chose;
- **read them unsealed and re-seal on next write** — defeats the seal for
  exactly the records the seal was added to protect;
- **treat as absent** — wrong: their manifests would present as orphans, which
  is one bucket away from the one an operator acts on.

**Not implemented.** Part 1 touches an authority-record parse path and gets a
review before it lands.

*Clement, 2026-08-02, at `3bf624c`. Nothing on the store was modified.*
