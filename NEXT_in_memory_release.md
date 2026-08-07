# NEXT — where the cold-storage / GC work stands

*Rewritten 2026-08-02. **Status block updated 2026-08-07**; the body below still
dates from 08-05 and is superseded where the two conflict. Read this before
anything else in this worktree.*

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
