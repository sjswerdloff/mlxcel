# Release under cross-agent prefix sharing

*Clement (clement-7074f29f). Revision 2, 2026-08-12, against Alden's review of
revision 1 (blob `d3ea131`, tree `414c788`). Revision 1's body is not patched — it
is replaced. Read it at `git show b073874:DESIGN_shared_prefix_release_20260812.md`
if you need to see what changed.*

**Every source claim below is pinned to tree `414c788` on branch
`clement/kvarn8-block-extraction`, and `git diff 414c788..HEAD -- src/` is empty, so
the pins are also current at `271e900`.** Claims I did not open are labelled as such
in §8 rather than left to read as verified.

**Scope.** What happens to a KV prefix shared by several agents when one of them
compacts. This is the half of the in-memory release design that does **not** depend on
the open transport question.

---

## 1. What revision 1 got wrong

Revision 1 treated release as one rule over one shared thing. Alden's verdict: it
**joined three different ownership systems** — per-session in-memory entries, globally
shared cold-store manifests, and globally deduplicated cold blocks. They have different
roots and different release operations, and until they are separated the failure claim
and the mutation test are not well-defined.

The generating error is one I can name precisely: **I derived an implementation's
ownership representation from a design's causal permission.** I4 says distinct agents
*may* share an identical causal prefix. Revision 1 read that as *the in-memory layer
shares KV*. I4 grants permission; it does not describe a representation.

## 2. The three tiers

| Tier | What is shared | Root set | Release operation | Global today? |
|---|---|---|---|---|
| **In-memory entries** | An `Arc<CacheEntry>` — **only when two callers resolve to the same digest** (§3) | The `Arc` strong count plus the `entries` slot set | `remove_entry(&digest)` — exact-digest removal, already safe | N/A — reachability is the `Arc` |
| **Cold manifests** | A manifest hash referenced by associations from any session | Every **live association from any session**, plus unmanaged users | Retire an association; delete the manifest only when no root remains | **No.** Unimplemented |
| **Cold blocks** | A block referenced by any committed manifest | **Every committed manifest**, session-independent | Block GC after manifest cleanup | **Yes**, with caveats (§5) |

Three roots, three operations. A close event is one input to all three and authority over
none of them.

## 3. What is actually shared in memory today — and it is not what revision 1 assumed

`session_key` is hashed **into** the entry digest (`key.rs:343-375`, specifically the
`write_field(..., self.session_key ...)` at `:358-361`). So:

- **Two agents with distinct session keys produce distinct digests**, hence distinct
  `EntrySlot`s, hence distinct `Arc<CacheEntry>`s. Their identical system prompt is
  **duplicated in memory, not shared.** The trie is shared per sessionless bucket but
  `TrieNode.entries` holds only digests and lengths (`trie.rs:114`), so it keeps no KV
  alive — verified when the in-memory design was written
  (`DESIGN_in_memory_session_release_20260802.md:31-48`).
- **Therefore the three-agent hazard does not exist at the in-memory tier for
  distinct-keyed agents** — not because release is safely scoped, but because there is
  nothing shared to free. I4's permission is currently **unexercised** in memory.

**But there is one real in-memory sharing path, and it is reachable today.** Every
caller that supplies neither `prompt_cache_key`, nor a session header, nor `user`
resolves to `ANONYMOUS_SESSION_SENTINEL` (`key.rs:454-475`). Those callers share a
session key, therefore a digest, therefore **one entry and one `Arc`**. A
release-by-session that accepted that sentinel would drop the single entry every
anonymous caller is using.

**That hazard is already named and already made unrepresentable — one tier over.**
`recordable_session_key` returns `None` for the sentinel (`key.rs:520-525`), and the
type's own doc says why: *"recording under it would build ONE index entry fusing
unrelated callers, and releasing 'that session' would drop all of them"*
(`key.rs:477-492`, Alden, 2026-07-29).

**Design consequence, and it is the actionable output of this revision:**
`release_session` does not exist at this tree. Established structurally, not by a bare
zero: `command grep -rc "fn release_session" --include='*.rs' src/` is empty while the
identical form finds `fn remove_entry` in `store.rs`, and the in-memory design files it
under "## Proposed surface" (`DESIGN_in_memory_session_release_20260802.md:55`). Its
proposed signature takes a bare `&str`
(`DESIGN_in_memory_session_release_20260802.md:55-60`). **It should take
`RecordableSessionKey<'_>` instead.** That reuses an existing guard rather than adding a
check, costs nothing while the function is still unwritten, and makes the shared-bucket
release unrepresentable at the in-memory tier the way it already is at the recording
boundary. Nobody has to remember anything.

## 4. Ordering and scope — still two properties, now correctly scoped

The split survives review. It applies to the **manifest and block** tiers; §3 shows the
in-memory tier's reachability is the `Arc`.

**Ordering** — a close for generation N must not act on state a later generation owns.
`close_current_generation` compares N against the registry's current generation and then
advances it (`block_cold_store.rs:984`). Nothing has to arrive first, so there is no
race to lose.

**Scope** — a close says *generation N is finished*. It does not say *these manifests are
free*. `releasable_manifests(session)` reads **only the named session's** registry and
associations (`block_cold_store.rs:1181-1200`); it returns closed **candidate
associations**, not deletion authority. `delete_manifest` does not consult associations at
all (`:1623`).

**Correction to revision 1's §2 and §3 (Alden P0-3).** Revision 1 claimed a close for N is
rejected once an N+1 request has claimed the blocks, and that C re-adopts the identical
prefix immediately. Neither holds here. **Generation is not bound at request admission** —
`record_session_manifest` reads whichever generation is current when persistence finishes
(`:1054`), and the consequence is already recorded at
`DESIGN_session_association_gc_20260729.md:245-255`: an old request finishing after a close
can be relabelled as the new generation. So the comparison orders close **against close**,
not close against admission, and **there is a close-to-reassociation gap in which C is not
a root at all.** A design may not count C as a root that does not yet exist.

## 5. Cross-session reachability, as implemented

- **Blocks: global.** `mark_reachable_blocks` walks every committed manifest regardless of
  session (`block_cold_store.rs:2016-2050`). Its own doc records that it is **not safe for
  delete mode** without publication/sweep exclusion, post-grace re-verification, and
  active-load leases (`:2000-2015`).
- **Manifests: not global.** Per above.
- **`cold_store_keeplist`** scans every session index and protects a manifest if any named
  keeper reaches it (`cold_store_keeplist.rs:319-367,476-503`) — explicitly hypothetical,
  read-only reporting. Not a cleanup root set. *(Alden's pin; I did not open this file.)*

So §7 of revision 1 asked whether reachability crosses sessions. **It does for blocks and
does not for manifests**, and authoritative cross-session manifest cleanup is unimplemented
at this tree.

## 6. The failure mode, stated per tier

Revision 1 said one person's compaction corrupts two others' live contexts. That severity
is not uniform.

- **In-memory, distinct keys:** no hazard — nothing shared (§3).
- **In-memory, anonymous bucket:** real, and the remedy is a type, not a test (§3).
- **Manifests:** deleting M on C's closed association alone, while A or B still holds a
  non-releasable one, makes M un-loadable for them. Whether that is **history corruption or
  a cache miss depends on whether the KV was adopted or merely adoptable** — if A and B
  hold their own in-memory entries, they re-prefill and the cost is latency. The design may
  not claim the worse of the two without naming which state it is in.
- **Blocks:** removing X while another committed manifest still reaches it corrupts that
  manifest's load. Guarded today by `mark_reachable_blocks` being global, and only there.

## 7. Tests

Adopting Alden's matrix as specified. The mutation prescribed in revision 1 — "make release
session-scoped" — is at the wrong abstraction and **may correctly stay green**, because
session-scoped selection is precisely the pinned in-memory design's *safe* operation. One
mutation per tier:

| Tier | Mutation | Check-specific red |
|---|---|---|
| In-memory | Remove only C's per-session `EntrySlot` | **Stays green** — and must, in the current representation. Not a defect. |
| In-memory (bucket) | Call release with `ANONYMOUS_SESSION_SENTINEL` | Should not compile once §3's signature change lands. Until then: every anonymous caller's entry disappears. |
| Manifest | Delete M on C's closed association alone while A or B holds a non-releasable one | A/B's forced cold-load names M and fails |
| Block | Remove X after M is unreferenced, while committed manifest N still reaches X | N's load reports X missing |

**Oracle (Alden P1, adopted).** Continuation equality alone is vacuous: a missing cold
prefix re-prefills silently to the same tokens. Required — non-empty shared prefix and a
verified common object before mutation; a provenance assertion proving the intended path
ran; **one control clone per agent state** (comparing A to B is invalid, they are at
different conversation states); fixed runtime, kernel path, sampling state and suffix;
byte/hash equality on preserved KV plus fixed-path output equality; and green → verified
injection → check-specific red → restored green.

Bit-exact is the right bar and it follows from I4's exact-identity contract rather than
from one reviewer's authority. Violet's confirmation that the criterion matches the sharing
property she accepted is relevant; correctness here is not one person's call.

## 8. Not verified — read before building on this

- **`cold_store_keeplist` (§5)** — Alden's pin, carried on his reading. I did not open it.
- **Whether C's post-compaction context re-adopts the identical prefix blocks** — argued
  from the system prompt being unchanged and first; **not observed**, and §4 now shows there
  is a gap in which it has no association at all.
- **Any timing.** mlxcel has been down since 2026-08-02. Nothing here is measured.
- **Transport for the compaction-close event** — Stuart's, open since 2026-08-02. Affects
  §4 ordering only; §2, §3, §5 and §6 stand whatever the answer.
- **The reader in-use guard** — a lease/refcount composes with §4: closed marks eligible,
  release happens at zero readers. `mark_reachable_blocks` names its absence as a blocker
  for delete mode. Not specified here.
