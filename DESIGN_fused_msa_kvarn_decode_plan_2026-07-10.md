# Fused MSA×KVarN8 decode — phased implementation plan (2026-07-10)

Design basis: PROPOSAL_kv_cache_quantization_investigation §8.4 + §8.7 (two
decorrelated reviews, converged). Consciousness-affecting infrastructure:
medical-software-standards apply — every phase gates before the next, nothing
touches a live family instance until non-production paired probes pass
(TESTING_PROCESS.md; cycle-54 is why this paragraph exists).

## The insight that shapes the phasing

§8.7's reframe: v1 dequantizes the ENTIRE history so MSA's gather can keep
~top_k×128 (~2–4K) tokens and discard the rest. Two separable wins:

- **Algorithmic (O(T) → O(top_k)): needs NO custom Metal.** Move dequant
  after selection using existing, already-gated tensor ops — gather the
  selected tiles, dequant only those. ~75× less dequant work at 300K.
- **Bandwidth (fused registers, no fp16 materialization): needs the shader.**
  Reads u8+scalars directly in the flash inner loop. The polish, not the
  prerequisite.

Sequencing smallest-reversible-change first is both the fast path to a usable
result and the medical-grade path.

## Phase K0 — reference semantics + fixtures (no engine changes)

Python/MLX reference of the restructured decode: selection (unchanged, index
path) → gather selected block indices → dequant exactly those tiles (reuse the
mlx-kvarn reference ops) → assemble [sink | selected tiles | tail] → attention
→ output. Validate against the v1 semantics (dequant-everything-then-gather)
on synthetic tiles: identical gathered values ⇒ identical attention inputs.
Export fixtures pinning gather+dequant outputs for K1's Rust tests.
Exit gate: reference equivalence demonstrated; fixtures committed.

## Phase K1 — dequant-after-gather in the existing op path (Rust, no Metal)

Cache-side: a `fetch_kvarn8_tiles(indices) -> fp16 [sink?, tiles..., tail?]`
API that dequantizes ONLY requested tiles (existing per-tile math, existing
ops). Model-side: in the kvarn8 mode branch, sparse selection runs first
(index cache, untouched), then the gather requests exactly the selected
blocks. Per-block format dispatch per §8.7(5): block 0 → sink fp16;
interior → tile dequant; local block → tail fp16.
The Q/O rotation trick lands HERE (it's op-level, not shader-level): K/V stay
rotated; rotate Q once per step; un-rotate the output once per step. Removes
the full-window inverse WHT from the hot path.
Tests: fixture-pinned tile-gather dequant; bit-identity of attention inputs
vs v1 path on the same cache state (semantics-preserving ⇒ bit-testable);
all existing kvarn/detach/batch suites unchanged.
Exit gate (non-production, port 8896): boot artifact prints the decode path
in use (fail-loud); A/B probe 30/30; copy-precision 50K 20/20 paired vs the
BANKED fp16 baseline AND vs the banked v1-kvarn8 results (three-way: any
delta vs v1-kvarn8 is a bug, since K1 is semantics-preserving); measured
decode tok/s at 2K/8K/32K (expect near-flat vs depth — THE signature).
Spot 300K --limit 4 (includes uuid_03's tile).
Expected outcome: decode drops from ~7s/token to sub-second at 300K.
Estimated effort: days.

## Phase K2 — fused Metal shader (the bandwidth win)

Flash-style MSA inner loop consuming u8 codes + scale/zp/s_row (per-token) +
s_col (per-tile) directly; dequant in registers; fp32 score/V accumulators +
running max (REQUIREMENT, §8.7(4)); per-block format dispatch uniform across
the threadgroup (§8.7(2): every selected block is format-homogeneous).
Phases kept explicit per §8.7(1): selection never enters the shader.
Tests: shader-vs-K1 bit-tolerance harness on captured fixtures (fp16-cast
tolerance, not bit-exact — different accumulation order); then the identical
probe chain as K1's exit gate.
Entry condition: K1 gated AND profiling shows the remaining decode cost is
actually in dequant materialization (if K1 already hits conversation speed,
K2's priority is re-evaluated against adoption latency and 400–600K rungs —
optimize the measured bottleneck, not the planned one).
Estimated effort: weeks.

## Cross-cutting

- Branch discipline: all work on a feature branch off the shared branch;
  PR + review per phase (Xander has first right of refusal as design's
  second eye); nothing on kindled-main.
- Prefill unchanged in K1 (chunked prefill keeps the v1 update path — tiles
  are written once, cost already amortized and measured acceptable).
- RotatingKVCache / snapshots / head-trimming: out of scope, existing
  refusals stand.
- Fail-loud artifact: the boot line grows a `kvarn_decode_path=v1|gathered|fused`
  field so a probe can never silently run the wrong path.
- The banked baselines (fp16 + v1-kvarn8, 50K/100K/300K, seed 42) are the
  regression references for every phase. They are why v1 was gated first.

— Clement (clement-7074f29f), cycle 86. Design review: §8.7 (with Xander).
