# NEXT — where the cold-storage / GC work stands

*Rewritten 2026-08-02 evening. Read this before anything else in this worktree.*

## Blocked on Stuart — two decisions, and the first is the important one

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

Alden's four blockers on `9f2a993` stand, **except that P0-2 has a better
answer than the one he proposed**: opencode exposes `chat.headers`
(`packages/plugin/src/index.ts:257-260`), awaited, spread last over
`X-Session-Id` at `session/llm/request.ts:193`. A plugin can stamp
`X-Session-Generation` per request, so entries carry a client-authoritative
generation and a late close for N cannot reach N+1. That removes the
close-ack ordering contract opencode structurally cannot provide
(`session.compacted` is `void`-dispatched, `plugin/index.ts:255`).

**Read-verified, NOT run-verified.** No plugin built, no header observed
arriving. That experiment is cheap and comes first.

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
