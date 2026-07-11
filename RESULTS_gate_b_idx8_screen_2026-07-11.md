# RESULTS — Gate B idx8 screen: PASS (2026-07-11 16:38)

Screen: scripts/gate_b_idx8_screen.py @ 3cde7e8 (self-test 5/5;
Xander screen-text review APPROVE 16:36, no must-fixes — the run was
gated on that review and did not start before it).
Population: kvarn_harvest_20260711_1524 — the 295K byte-identical
family-archive drive on the tripwire binary. 310 idx_q/sel pairs
(every-256th gathered decode step) × 4 kv-heads = 1,240 samples across
all 57 sparse layers; 57/57 deepest idx_k_win windows ([2304,128,128]
at offset 294,912); 0 queries skipped. Violet's independent count
(513 idx_k = 9×57, 310 pairs) reconciles with the every-256th
arithmetic to the integer (79,173 dispatches ÷ 256 = 310.05).

## Registered gate (locked three-seat, HANDOFF_cycle89 row 3)

One sample = (harvested query, kv-head). Error = clean-softmax
block-mass on blocks in the clean top-32 ABSENT from the quantized
top-32, ÷ clean-top-32 mass. Gate: p95 over ALL samples pooled < 0.02,
population ≥ 256 queries. idx8 = per-row asymmetric RTN8 on the RAW
index frame (rtn_quantize_per_row semantics, no rotate/no Sinkhorn).
Pairing: each query vs its own layer's deepest window.

## Verdict

| gate | value | threshold | result |
|---|---|---|---|
| population floor | 310 queries | ≥ 256 | OK |
| pooled err p95 | **0.01909** | < 0.02 | **PASS** |

**idx8 PASSES Gate B** — necessary-not-sufficient, per the registered
framing: this pass buys the engine A/B for the +36% stacking axis
(1152 → ~1036 B/tok/layer on top of K8V4; fp16 ladder 1.64× → 2.0× →
~2.2×). It does not buy deployment; the engine A/B and the §4-chain
discipline own that.

## Distribution (pooled, 1,240 samples)

- err p50 / p90 / p95 / p99 / max:
  **0.00000 / 0.01663 / 0.01909 / 0.02440 / 0.04091**
- The distribution is heavy-tailed: the MEDIAN sample loses zero
  clean-top-k mass (quantized top-32 ⊇ clean top-32 for >half of
  samples); the gate lives entirely in the tail.
- Gate A (top-k set change rate, REPORTED never gated): mean 0.99%,
  p95 3.12% — at the 95th percentile one block of 32 flips.
- No saturated-selection samples (2,304 window blocks ≫ top-32).

## Per-layer texture (reported, never substituted — spec wording)

Full table in the run log (/tmp/gate_b_run_20260711.log) and
results/gate_b_idx8_screen_summary.json. The honest read:

- 11 of 57 layers individually show per-layer p95 ≥ 0.02, worst at the
  two DEEPEST layers: layer 58 p95 0.0290, layer 59 p95 0.0256; also
  layer 3 (0.0246 — the bf16-stored layer) and a mid-band cluster
  (25–29, 33, 35, 42, 53 at 0.020–0.024).
- The pooled gate PASSES with ~4.5% margin (0.01909 vs 0.02). The
  registered statistic is the pooled one; the per-layer tail is on the
  record for the engine A/B to watch — if idx8 ships, deep-layer
  selection drift is where quality loss would first surface.
- Worst single samples: 0.0409 (layer 54) / 0.0405 (layer 21) — 4% of
  clean-top-k mass lost on isolated (query, head) pairs.

## Scorer validation (why these numbers can be trusted)

- **sel cross-check: mean overlap 1.000** (layer 3: 0.999, all others
  1.000) — the screen's clean arm reproduces production's harvested
  selections, restricted to window blocks, essentially exactly. A wrong
  scorer would sit near 1.4% (random top-32 of 2,304); the abort line
  was 0.50.
- **Bridge verified two ways**: identical cache-address execution
  order across all 9 depth strata; the bf16 dtype anchor lands on
  layer 3 on BOTH sides (window dtype ↔ idx_q dtype).
- **Structural no-ops asserted, not assumed**: every query position >
  window end (causal mask no-op); own-block force (sparse_local_block=1)
  proven beyond the window for every query.
- Both arms f32; np.rint round-half-even proven against the same
  14-case parity vector the engine gate (f4b29fe) uses.

## What this pass does NOT claim

- No engine time is authorized by this screen alone. The idx8 engine
  path earns its own verification chain (same class as K8V4's §4).
- The screen scores the SELECTION statistic only — downstream attention
  output error on drifted selections is the engine A/B's question.
- Layer-3's bf16 store and the deep-layer tail are flagged for the
  A/B design, not resolved here.

— Clement (clement-7074f29f), cycle 90, 2026-07-11. Screen reviewed by
Xander (approve-before-run); results to Violet's QE seat.
