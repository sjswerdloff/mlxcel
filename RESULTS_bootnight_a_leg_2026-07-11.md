# Boot-night A-leg: harvest + live re-baseline (2026-07-11, boot A)

Boot A: base 8a37724, kvarn8 + D1 + blocked core (env-seeded defaults),
harvest active. Supervisor contract in force. Stuart asleep; watch on.

## Harvest: COMPLETE AND VERIFIED (the tonight-irreplaceable)

Session: 295,108 real tokens (1,259 family artifacts — messages, docs,
identity material), 1,500 decode tokens, cached_tokens=0 (clean prefill).
The vessel read the family's own history as its harvest workload.

| role | dumps | design arithmetic | check |
|---|---|---|---|
| k_rot_f32 | 2052 | 36 strides × 57 caches | EXACT |
| v_rot_f32 | 2052 | paired | EXACT |
| idx_k | 513 | 9 crossings × 57 | EXACT |
| idx_q | 334 | 85.5K gathered steps ÷ 256 | EXACT |
| sel | 334 | paired with idx_q | EXACT |

Verification: 68/68 sampled artifacts pass (sidecar parses, bin size =
shape×dtype, finite, non-zero variance, sel in block range). One
verifier-side correction recorded: sel dumps are uint32 (selection's
native dtype, code 3) — the artifact was right, my first checker's dtype
map was incomplete. Depth strata UNIFORM: k-dump offsets 6,144..292,864,
quartiles 80K/154K/227K — the stride gate doing exactly its job.
Population power: ~16.4K tiles/role (k,v), 334 real queries (≥256 floor
met; note: queries are END-OF-SESSION depth only — decode-phase
sampling; mid-depth queries would need mid-session turns, recorded as a
future refinement, not a gate breach).

Violet's early-check condition PAID: presence verified at 66K depth
(kv=468, idx_k=114), not at dawn. The declared-gap idx_k branch fired
correctly live (9 crossings, exact).

## Live re-baseline, boot A (D1 changed every denominator)

- 295K prefill: 2,284s for 295,108 tokens = 129 tok/s (chunked 2048,
  MSA sparse prefill, harvest sampling active).
- **295K decode spot: 3.44 tok/s end-to-end** (1,500 tokens in 436.1s,
  first-decode 13:18:52 → complete 13:26:08). Coherence vs bench:
  290ms/token total ≈ 193ms attention (bench kvarn8×blocked p50 at
  300K) + ~97ms model-rest (MoE FFN, embed, sampling, server). The
  bench and the live number agree on the attention share.
- 50K paired probe: fired, result appended below when it lands.

— Clement (clement-7074f29f), cycle 88, boot night.

## 50K paired probe, boot A (landed)

52,517 prompt + 256 decode in 69.86s TOTAL. The per-phase split is not
cleanly extractable from this log (chunked-prefill lines for the probe
were compacted); recorded as PAIRED TOTALS — boot B runs the byte-
identical request, so the A/B comparison binds on totals with the same
internal split, which is the honest frame. No guessed decomposition.

ARTIFACTS VERIFIED → SIGNAL AUTHORIZED (the contract's order). Signal
drops immediately after this commit; supervisor takes A down, RAM-checks,
announces, boots B (kvarn8 + C). B-leg: verify C dispatch witness in
log.b, then the identical 295K-session request + 50K probe.
