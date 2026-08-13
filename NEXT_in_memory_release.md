# NEXT — where the cold-storage / GC work stands

*Rewritten 2026-08-02. **Status block updated 2026-08-07**; the body below still
dates from 08-05 and is superseded where the two conflict. Read this before
anything else in this worktree.*

## STATUS, 2026-08-13 evening — BUILT. Read this first.

**The eviction-after-compaction path is implemented and end-to-end on this branch.** Not
designed — built, tested, pushed.

- `4842c73` — `PromptCacheStore::release_session(session_key) -> ReleaseOutcome`
  (`src/server/prompt_cache/store.rs`). Selection by the session component of the bucket
  key; removal via the existing exact-digest primitives, so another session's entries cannot
  be touched whatever trie they share. Releases entries **and snapshots**.
- `ee914e6` — `POST /v1/cache/session/release` (`routes/cache.rs`, mounted in `app.rs`
  inside the existing `api_key_auth` layer — auth inherited, not reimplemented).

**3805 passed, 0 failed.** Eleven new tests; five mutation-checked (widen the session
filter, drop the sentinel guard, drop the snapshot loop, unmount the route, drop the
nothing-matched warning — each reddens its own test).

**Three properties a future me must not quietly "fix":**

1. **It refuses `ANONYMOUS_SESSION_SENTINEL`** without touching the store. Every caller with
   no session identity resolves to that one key; serving it would drop unrelated callers'
   entries. This is the only real cross-session hazard the in-memory tier has.
2. **`nothing_matched` is a discriminated status with a warning in the BODY.** A release
   that matches nothing and one that works are otherwise identical from outside. Do not
   collapse it into `released_now`.
3. **At-least-once, NOT idempotent.** A retry releases what is resident now, including
   post-compaction entries, because in-memory entries carry no generation. Stated in the doc
   comment and pinned by a test. Making it "idempotent" requires a generation dimension in
   the entry digest — a schema change, not a tidy-up.

**Why the narrow scope:** two of the full protocol's five blockers were missing mechanisms
in the cold store, not design defects. This tier touches none of them — `store.rs` has zero
references to `block_cold_store`, `BlockColdStore` or `SessionRegistry`. The full two-tier
protocol stays designed-not-built at `3b584b4`.

**NEXT, and Stuart named it: the opencode plugin.** Nothing calls the endpoint yet. The
plugin witnesses `session.compacted` and POSTs to it. Design at
`DESIGN_session_close_endpoint_20260812.md`; the `PREPARED -> COMMITTED` machine and the
`event_id` work there are for the COLD tier and are **not needed** for the narrow endpoint —
it takes a session key and nothing else.

**Xander (confirmed 2026-08-13, not assumed):** he is committing **on top of this branch**;
`xander/cold-store-v4` carries this design history. He is staying on it and will rebase. He
is **not** working near `src/server/prompt_cache/`. He owns the two remaining cold-store
tests (fsync fault-injection or mock-fs — **syscall tracing is ruled out**, SIP is enabled
and dtrace needs root; and two independently constructed stores for the lock). File-metadata
checks were eliminated: `stat` cannot distinguish fsynced from not.

## STATUS, 2026-08-12 (late) — the endpoint design, read this with the block above

**Transport and auth are SETTLED** (Stuart): HTTP, and the existing `api_key_auth`
middleware — `POST /v1/cache/reset` already sits behind it, so nothing new is built.

**`DESIGN_session_close_endpoint_20260812.md`, revision 2 at `3b584b4`.** Alden reviewed
revision 1 and found three P0s; all accepted. Not re-reviewed — he went to the waters.

**The three things that cost real thinking, do not re-derive:**

1. **A type cannot prove provenance its constructor did not observe.** Parsing a JSON
   string into `CompactionScopedSessionKey` launders a caller's assertion into apparent
   evidence. Either mlxcel mints a ticket when it *sees* `SessionKeySource::SessionHeader`,
   or the endpoint is an authenticated ADMIN operation over an ASSERTED scope — which is
   fine for a home server, but must be called that.
2. **`event_id` already exists.** `close_current_generation` takes it and records
   `last_event_id`. Without it a lost `200` and a stale close are the same request. It must
   consult the recorded event BEFORE rejecting on generation, or replay is unrecoverable.
3. **One `COMMITTED` bit cannot converge two tiers.** A cold close can legitimately match
   zero in-memory entries. The zero-match loudness rule I wrote to prevent a silent no-op
   would have STRANDED successful closes in `PREPARED` forever. Per-tier outcomes; cold CAS
   first, because memory removal is irreversible.

**Two open items are STUART'S, not a reviewer's:** which §2 posture (ticket vs asserted
scope — depends on what opencode can carry), and which §7 option for generation-blind
in-memory release.

## STATUS, 2026-08-12 — read this FIRST; the 08-07 block below is superseded where they conflict

**The transport question does NOT block as much as this file has been claiming.** I held
v2 of the in-memory release design as blocked for ten days. A large part of it was never
blocked: whether release is reachability-scoped or session-scoped is independent of how
the close event travels.

**APPROVED by Alden (alden-ec2221c7) 2026-08-12 at commit `ec5b3b4`, design blob
`beae8b754a4a552f2e782734f5f9570612a6fbb6`.** Four revisions: `b073874` (3 P0s),
`176c41d` (all five cleared, 3 raised), `0d4e258` (those closed, 1 provenance P2),
`ec5b3b4` (final). It separates the three ownership systems — in-memory entries, cold
manifests, cold blocks — each with its own root set and release operation.

**Release scope is a TYPE, and it is not `RecordableSessionKey`.** That proves only
non-empty and non-sentinel; it still admits `user` (end-user scope — a per-conversation
delete would remove every conversation that user cached, `key.rs:438-442`) and
uncontracted `prompt_cache_key`. Use `CompactionScopedSessionKey`, `SessionHeader` alone
at this tree (`key.rs:435-437`). This rule was already in `DESIGN_session_association_gc
_20260729.md:256-262` — my own document — before I proposed the weaker guard.

**`mark_reachable_blocks`'s doc comment (`:2000-2015`) is STALE and says the delete-mode
prerequisites are unimplemented. They are implemented** — `:1484`, `:2222`, `:2237-2243`,
`:699-740`, `:2410`. Judge the code. Fixing the comment is worth its own commit
(Xander's file).

Two answers worth not re-deriving, both pinned at `414c788`:

- **Reachability crosses sessions for BLOCKS, not for MANIFESTS.** `mark_reachable_blocks`
  walks every committed manifest (`block_cold_store.rs:2016`); `releasable_manifests` reads
  one session's registry and returns candidates, not authority (`:1181`).
- **The in-memory tier does not share KV between distinct-keyed agents — it DUPLICATES.**
  `session_key` is hashed into the entry digest (`key.rs:358-361`). I4's permission to share
  is currently unexercised in memory. The one real sharing path is the anonymous sentinel,
  and `recordable_session_key` already refuses it at the recording boundary (`key.rs:520`).
  **Actionable:** `release_session` is still unwritten, so type its parameter
  `RecordableSessionKey<'_>` rather than `&str` and the shared-bucket release becomes
  unrepresentable at no cost.

**Still genuinely open and still Stuart's: transport for the compaction-close event.**
It affects the ordering half only.

**Cyril's ruling, 2026-08-12:** satisfying the resume design's I1–I8 does NOT prove a
restored cache continues identically. I2 mandates that a restored run re-prefills at
least one token, so its path is prefill-then-decode against an in-memory run's
decode-only — different kernels. I4 constrains which BUILD executes, not which PATH is
taken. So the round-trip test is behavioural, not conformance. Correct baseline: Run A
prefills from cold and continues; Run B restores a shorter prefix, prefills the
remainder, continues. Both prefill, both decode, same shapes.

**mlxcel has been down since 2026-08-02.** Nothing here is measured.

## STATUS, 2026-08-07 — read this first, the body is older

**ONE decision is open, not two.** Stuart resolved the pre-seal question on
2026-08-05: there are no previous cold storage caches, so there is nothing to
migrate. Refuse pre-seal records; the 2.4 GB store can be cleared whenever
convenient. **The body below still says "two decisions" — it is wrong.**

**Still open, and it is the one that blocks the in-memory KV release:** transport
for the compaction-close event. Plugin witnesses via `session.compacted`,
Dashboard authorizes as Stuart ruled, MQTT recommended for the hop.

**The upstream proposals went four revisions with Alden**
(`kindled-opencode-plugins` PR #1, now at `c385fcc`). Revision 1 contained a
data-loss path — a surviving pre-compaction record has two indistinguishable
causes and it retried both, closing generations still being extended. Now
`PREPARED -> COMMITTED`, only the success hook marking `COMMITTED`, startup
retrying only `COMMITTED`. **Awaiting his re-review of revision 4.**

**P0-4 is settled in shape and Xander owns the implementation:**
`release_detached_paged` gains `Result<(), ReleaseError>` with `PoolUnavailable`
and `Partial`; `debug_assert!` in dev, `Err` (never `Ok`) in release, and at
least one caller must suppress the physical-release claim on `Err` or the
signature is ceremony.

**My branch `clement/kvarn8-block-extraction` has NO PR.** mlxcel is on GitHub
(`sjswerdloff/mlxcel`), the repo is **public**, so opening one is Stuart's to
authorize. Alden's six-round approval of the keep-list currently lives in
messages and his home office, attached to nothing a reader would find.

**Stuart's mlxcel server on 8890 has been down since 2026-08-02.** Untouched.

## Blocked on Stuart — the body from here down predates 08-07

**1. Transport for the compaction event.** Stuart ruled that **KindledDashboard
is the authoritative close producer**. That settles *who may*, not *who
witnesses*.

The Dashboard **cannot be the witness**: its only signal is a scraped tmux pane
(`scrape_claude_statusline_context.sh:19-33`). A percentage jumping back up is a
proxy, and `PROPOSAL_compaction_generation.md` argues the whole design turns on
closing only on evidence — closing early declares a live token sequence dead.
`session.compacted` *is* evidence (`compaction.ts:507`, published only
`if (result === "continue")`), and it exists only inside opencode.

So: **plugin = witness, Dashboard = authority.** The open question is how the
plugin's evidence reaches the Dashboard. Recommendation on the table: **MQTT**,
already the family bus, no new listener or secret. Until he answers, two
different designs are possible and neither should be written.

**2. Migration policy for pre-seal authority files.** Recommendation: refuse,
and let those manifests leak as UNATTRIBUTED — the safe direction, and what
`releasable_manifests` already chose. See
`FINDING_session_authority_version_skew_20260802.md`.

## Not blocked, not started: rev 3 of the lock protocol design

Owed to Alden. **Deliberately not written this cycle** — he spent six review
rounds on the keep-list across a compaction, and a design that still needs the
transport answer to be complete would be an incomplete document sent to a
depleted reviewer.

Must incorporate: Stuart's fix-forward ruling on `PruneMode::Delete`; the
per-turn / months-long-key arithmetic that killed the 4096-entry cap (reached in
~6 weeks) and makes **append-mostly recording mandatory**, reversing rev 2 §7.3;
the retired-key reframing; and in-memory release as a named requirement.

## In-memory KV release — design v2 still owed

Stuart, twice: *"letting go of the obsolete KV cache memory after compaction is
critical."* Ahead of the disk GC.

**RETRACTED 2026-08-05 — I claimed P0-2 had "a better answer than the one he
proposed" via `chat.headers`. It does not, and the claim was wrong in three
separate ways.** Alden's independent review of the upstream proposals
(kindled-opencode-plugins PR #1, REQUEST_CHANGES at `9e89e5ea`) source-resolved
all three at `anomalyco/opencode@1e17856`:

1. `prompt.ts:233`, which I cited for the sessionID→`X-Session-Id` identity, is
   **title-generation code and is not this path at all**. That citation was not
   merely unverified — it was wrong.
2. `X-Session-Id` is used **only for providers whose ID does not start with
   `opencode`**; OpenCode providers use `x-opencode-session`. The header I
   planned to test may not be the one in play.
3. `request.ts:202-203` spreads model headers and `chat.headers` **after** the
   default, so either can **replace** the session id rather than merely add
   beside it. I read "spread last" as additive; it is override.

And `session.ts:501-515` accepts a **caller-supplied** session ID, so after a
deletion the same string can name a new session while an old latch survives —
the namespace is incarnation-unsafe, and no client-supplied id is authority.

**So the close-ack ordering contract I claimed to have removed is still
required.** Alden's P0-2 stands, restated: the post-compaction request can reach
mlxcel while close N is in flight (`compaction.ts:507-510` publishes and
returns; `plugin/index.ts:251-258` dispatches with `void`;
`prompt.ts:1149-1158` continues the loop immediately).

**The header-arrival experiment is DEMOTED, not merely pending.** It can
establish today's configured behaviour; it cannot establish authority or the
invariant across plugin ordering, config, restart and ID reuse. Running it would
have produced a green result that licensed a design its scope never covered.

**The direction that replaces it (Alden's):** bind close and admission to a
**server-issued opaque incarnation/capability plus generation**, authenticate
the witness, and reject every unknown, stale, future, mismatched or
unauthorized tuple **without mutation**.

Still standing unchanged: P0-3 (`Arc::strong_count` is not liveness — the Arc
after `take_detached()` is a drained shell) and P0-4
(`release_detached_paged` returns `()` and cannot report partial failure, so no
outcome field can honestly claim physical release until that API changes —
a signature change in code Xander and Violet also touch).

His full findings: `~/ai/ClaudeInstanceHomeOffices/alden-ec2221c7/mlxcel_in_memory_release_design_review_20260802.md`

## Done and APPROVED this cycle

`cold-store-keeplist` — read-only diagnostic report, approved by Alden at
`ed1f6cc` after six rounds. See `DESIGN_keeplist_report_20260802.md`.

**His approval's own scope, quoted rather than paraphrased:** *"for a read-only
diagnostic report only. It does not authorize manual deletion, does not revive
the withdrawn artifact, and does not approve a future deletion consumer."*

**Three gaps remain and are explicitly not waived by the approval:**
1. no complete production-store report exists — every live authority record is
   pre-seal;
2. the running-server contradiction probe has no real-server positive control;
3. coherent quiescence cannot be established until session-index writes join
   authoritative cross-process locking. Today they are serialised by a
   **process-local `Mutex`** in `record_session_manifest`, so mutual exclusion
   across processes rests on the launch script's pre-flight check.

Also landed: `e777097`, magic/version checked before the seal, so a pre-seal
file reports an unsupported version rather than tampering.

*Clement, 2026-08-02.*
