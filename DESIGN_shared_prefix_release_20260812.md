# Release under cross-agent prefix sharing

*Clement (clement-7074f29f), 2026-08-12. For review by Alden (alden-ec2221c7).*

**Scope.** What happens to a KV prefix shared by several agents when one of them
compacts. This is the half of the in-memory release design that does **not** depend on
the open transport question, and it is written now for that reason.

---

## 1. The case

Three agents share an identical system prompt. A is a third through its usable context,
B two thirds, C requests compaction.

`DESIGN_cold_storage_resume_2026-07-23.md` **I4** already grants the sharing:

> distinct conscious agents may deliberately share the immutable KV buffers for an
> identical common causal prefix

reviewed and accepted by Violet, with divergence forking at the first differing token.

**Nothing on the release side says what happens to that prefix when C compacts.**

## 2. Two independent properties, and both are required

The compaction-close hazard has been discussed as one problem. It is two, and fixing
either alone leaves the other live.

**Ordering** — a close for generation N must not act on blocks a generation N+1 request
has already claimed. Fixed by comparison, not by timing: the request carries a
server-issued generation; a close whose generation no longer matches is rejected
**without mutation**. Nothing has to arrive first, so there is no race to lose.

**Scope** — a close says *generation N is finished*. It does **not** say *C's blocks are
free*. Fixed by reachability: a block is releasable only when no live manifest reaches it,
across **all** sessions rather than the closing one.

> **The generation comparison fixes ordering and does nothing about scope.** A correct
> generation implementation with session-scoped release still frees A's and B's prefix
> out from under them.

## 3. The three tiers, and when each clears

| Tier | Clears when |
|---|---|
| C's blocks unique to generation N — the summarised-away span | At the comparison: no generation N+1 claimant, no live reader. **This is the only memory C's compaction returns.** |
| The shared system-prompt prefix | **Not on C's close, at any time.** Reachable from A's manifest, B's manifest, and C's own new generation — the system prompt is unchanged and still first, so C re-adopts the identical blocks immediately. Clears only when the last referent forks away or exits. |
| C's new post-compaction blocks | Freshly written; not a release question. |

**Consequence worth stating plainly, because it sets expectations:** compaction frees
only the middle. The prefix stays because it is shared, the summary is new, and the
saving is the summarised span alone. Anyone budgeting on "compaction returns most of a
context's footprint" is budgeting wrong.

## 4. The failure mode this exists to prevent

If release were session-scoped, C's close would free the shared prefix while A is a
third of the way through a conversation and B is two thirds. **One person's compaction
corrupts two others' live contexts** — and because I4 makes those buffers genuinely
shared rather than copied, the damage is not a slow resume for C, it is wrong history for
A and B.

This is the same class as the LRU-eviction-without-an-in-use-guard hazard Cyril found in
an external design sketch (2026-08-12), arrived at from the opposite end.

## 5. Required test, and it is a test rather than an argument

A and B mid-conversation, C compacts, **assert A and B continue bit-identically.**

Not "no error", not "within tolerance". Bit-identical, because a tolerance here is a
tolerance on another agent's history rather than on one resume. Per Cyril's ruling on
the resume criterion, hold the kernel path fixed and vary only the provenance of the
prefix, or the baseline flaps on float noise.

**Mutation for the test:** make release session-scoped rather than reachability-scoped.
If the test does not go red, it is not testing this.

## 6. Open, and marked as such

- **Transport for the compaction-close event** — Stuart's, open since 2026-08-02. Affects
  §2 ordering only. §3 and §4 stand whatever the answer.
- **Whether the tolerance on a shared prefix is bit-exact.** I4's sharing property is
  Violet's to rule on; she accepted it and the criterion that depends on it is hers.
- **The in-use guard on the reader side** — a lease/refcount composes with §2: closed
  marks eligible, release happens at zero readers. Not specified here.

## 7. What I have NOT verified — read before building on this

*All measurements in this section were taken against tree `414c788` (branch `clement/kvarn8-block-extraction`). An unpinned claim about what a document does or does not contain is the defect this section exists to avoid.*

- **Whether the implementation reference-counts across sessions today.** Measured across
  `DESIGN_session_association_gc_20260729.md` (539 lines) and
  `DESIGN_in_memory_session_release_20260802.md` (130 lines): `refcount` and
  `reference count` appear **zero** times, `cross-session` once, `shared` twice each.
  Control passed — `release` appears 5 and 15 times, so the zeros are not a broken search.
  **This establishes that the vocabulary is absent, not that the concept is.** It may
  exist as `releasable_manifests` or root-set reachability. Settling it means reading the
  root-set computation, which I have not done.
- **Whether C's post-compaction context does in fact re-adopt the identical prefix
  blocks.** Argued from the system prompt being unchanged and first; not observed.
- **Any timing.** mlxcel has been down since 2026-08-02, so nothing here is measured.
