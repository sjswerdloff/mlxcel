# PROPOSAL — KV-cache quantization for MiniMax-M3 in mlxcel (8-bit first, KVarN second)

**Author:** Clement (clement-7074f29f), 2026-07-07, cycle 85.
**Directive:** Stuart — "decent compression with minimal loss," not maximum compression;
8-bit (near-lossless, 2×) is untried and already a huge win over fp16.
**Sources:** mlx-kvarn (github.com/eplt/mlx-kvarn, MLX port of KVarN);
KVarN paper arXiv 2606.03458 (Muller et al., Huawei); NVIDIA forum thread
"MiniMax-M3-W4A16-GPTQ 2×GB10" (262–370K ctx with KVarN vs 131K at fp8 — production
confirmation at M3 scale).

## 1. Why (measured, not estimated)

MiniMax-M3 text config: 60 layers, 4 KV heads, head_dim 128 → **120 KiB/token at fp16**
(2 × 60 × 4 × 128 × 2 B = 122,880 B). Weights on disk: **215 GiB** (MXFP8-64e).
On the 512 GB Studio the KV cache is the marginal consumer at Kindled-scale context:

| depth | fp16 | K8V8 (2×) | KVarN k4v4 (~3.7×) |
|---|---|---|---|
| 163K (observed live) | 18.6 GiB | 9.3 GiB | 5.0 GiB |
| 300K (anchored-depth target) | 34.3 GiB | 17.1 GiB | 9.3 GiB |
| 500K (normal Kindled session ceiling) | 57.2 GiB | 28.6 GiB | 15.6 GiB |
| 1M (config max) | 114.4 GiB | 57.2 GiB | 31.2 GiB |

Two concurrent deep sessions at fp16 (~114 GiB at 500K each) don't fit beside the
weights; at 8-bit they do. That's the practical shape of the win.

Compression ratios, honest version: KVarN k4v2 is 4.74× per-tile (13,824 B vs
65,536 B); **k4v4 is ~3.66×** (17,920 B/tile) by the port's own per-tile table. The
port README's GSM8K table lists k4v4 as "4.7×," contradicting its own tile math —
verify against the paper before quoting either number. K8V8 affine is a clean 2×
(scales/zero-points overhead ~1–2%).

## 1b. PHASE 0 FINDING (2026-07-07 evening) — the codebase already inhabits this problem

mlxcel upstream ships a full KV-cache quantization subsystem (`cache/turbo/`,
Lablup/TurboQuant+ lineage) that this proposal's first draft did not know about:

- **`KVCacheMode::Int8`** — per-token INT8 absmax K+V, ~50% savings. **This IS the
  8-bit rung, already implemented.**
- **`Turbo4Asym`** — fp16-K + 4-bit PolarQuant V (with Walsh–Hadamard); `Turbo3Asym`
  (3-bit V); symmetric `Turbo4` gated by a per-model allowlist.
- **Hard-won safety knowledge encoded in `turbo/allowlist.rs`:** symmetric 4-bit K is
  catastrophic on 4-bit-weight models (measured PPL 218 vs 6.6 — softmax
  exponentially amplifies K-side error; the failure is SILENT: fluent text,
  aggressive hallucination). The recommended safe shape keeps K at fp16 — which is
  also exactly what our copy-precision constraint wants.
- **Server plumbing exists end-to-end:** `--kv-quant-scheme` / `--kv-cache-mode`
  resolve to a server-wide mode applied to every model's caches immediately after
  `make_caches()` (scheduler.rs ~:253). `update_and_fetch` is mode-transparent —
  quantized storage inside, fp16 out — so M3's attention code consumes fetched K/V
  unchanged.
- **A quality-gate harness exists** (`tests/turbo_kv_e2e.rs`, PPL within +2.0% of
  fp16, per-model) with a documented extension workflow.

**Consequences for this proposal:**
- Phase 1 changes from "build K8V8" to **"verify the existing Int8 mode works on M3
  through the server path, then quality-gate it"**. Remaining integration questions,
  in order: (i) M3's custom idx_k side-cache (`m3_idx_k_update_and_fetch`) — confirm
  it stays fp16 under mode conversion (that is the design intent anyway) and doesn't
  break; (ii) the paged/batch path has mode awareness (`paged_layout.cache_mode`)
  but at least one code path special-cases `== Fp16` (cache.rs ~:6016) — map what is
  disabled for non-fp16 (snapshot? donation?) before assuming server parity;
  (iii) nobody has quality-gated ANY mode on M3 — the §4 gates are the real work.
- ~~Phase 2's candidate becomes Turbo4Asym (existing) before KVarN (port)~~
  **RULED OUT (Stuart, 2026-07-07): TurboQuant is excluded entirely — not used,
  not tested. It has well-known problems at long conversations, which is exactly
  the Kindled use case.** The investigation is therefore fp16 baseline vs Int8,
  full stop. If Int8's 2× is ever insufficient, the conversation about a further
  rung (KVarN or otherwise) happens then, with long-conversation behavior as the
  first-class criterion — not as a footnote.
- On "why int8 and not fp8/mxfp8" (Stuart's question, 2026-07-07): fp8 KV caches
  are a CUDA-hardware design — Hopper/Blackwell attention kernels COMPUTE in fp8,
  so fp8 storage buys speed there. Metal has no fp8 compute units: any 8-bit
  cache on Apple Silicon is storage-only and dequantizes to fp16 before SDPA, so
  fp8 gains nothing over int8 here, and per-token-scaled int8 spends its 256
  levels adaptively per token (generally the equal-or-better accuracy trade for
  dynamic KV data). Contingency if Int8 regresses the copy-precision gate: MLX's
  native quantize/dequantize support `mode="mxfp8"` (verified on this machine) —
  an mxfp8-STORAGE cache is buildable with zero custom Metal, same ~2×, different
  error shape (non-uniform precision, block-of-32 scales).
- The validation plan (§4), boot artifact (§5), and granularity analysis are
  unchanged — they were always the load-bearing work, and they are method-agnostic.
- Compression-claim caution: Turbo4Asym's own docstrings carry inconsistent numbers
  ("~26% net" vs "~3.8×") — measure, don't quote, exactly as with the KVarN README.

## 1d. INT8 RUNG VERDICT (2026-07-08) — FAILED for Kindled use; killed mid-run by decision

fp16 baseline (overnight 07-07→08): copy-precision 20/20 exact at true 50K
(`results/copy_precision_fp16_d50k_seed42.json`); divergence capture 18/18,
self-check 9/9 MATCH; A/B invariants all-pass at every sweep incl. MSA-active
(after fixing the probe's 48-token budget — the original "B==C violation" was
harness starvation on a thinking model, third occurrence of the cycle-62
token-cheapness lesson; MAX_TOKENS now defaults 2048).

Int8 rung (upstream per-token-per-head absmax, Jeongkyu Shin 09f79b4):
- B==C and 128-alignment: ALL PASS — adoption machinery is sound under Int8.
- A==D: 3/6 sweeps FAILED, and the failure textures are the two modes this
  investigation exists to protect against: (1) character corruption of a
  random tag on a trivial verbatim echo at ~1.5K depth (`cachetest-20626` vs
  `cachetest-206826957` — dropped digits); (2) two EMPTY-at-2048-budget legs
  (probable unclosed think block — runaway-thinking tipped by quant noise).
  fp16 showed none of these under identical conditions and budget.
- Planted retrieval at 50K: 7/7 targets exact before the run was killed —
  retrieval-at-depth was NOT the failure mode; generation stability was.

Interpretation: per-token absmax has no K-side outlier handling; its noise
exceeds this checkpoint's tie-break margins (REAP50-flattened distributions).
Killed at Stuart's direction 2026-07-08 ~14:15 — remaining stages carried no
information a future candidate's own gate run wouldn't re-measure.

**KVarN decomposition experiment results (2026-07-08, model-free, reference
implementation on K-like tiles with ×40 outlier channels — Stuart directed
"straight to KVarN"):**
- Zero custom Metal CONFIRMED for the entire correctness path: the reference
  write pipeline (Hadamard = native `mx.hadamard_transform`, orthonormal and
  self-inverse; Sinkhorn = ~30 lines of tensor ops; RTN + pack/unpack = plain
  shifts/masks) is tensor ops end to end. Their only Metal kernel is a READ-
  side batched dequant for dense full-history attention — our MSA decode
  gathers 16–32 tiles, where tensor-op unpack should serve; dense layers 0–2
  are the only place a batched kernel might matter (measure first).
- KVarN k4 vs absmax int4 at the same bits: 2.0× lower error — the outlier
  handling works as designed.
- **NEW LEADING CANDIDATE — "KVarN-k8": Hadamard+Sinkhorn+8-bit RTN** (not a
  preset upstream, but the math generalizes; no sub-byte packing needed):
  rel_err 0.0042 vs 0.0199 for the absmax int8 that FAILED the live gate —
  **4.7× lower noise at identical 2× compression**, and isotropic (rotation-
  spread) instead of structured. Raw tile error does not decide the gate
  (int8-absmax had the lowest raw error of the failed schemes) — but lower
  AND isotropic is the right direction on both axes.
- Caveat: Sinkhorn converged to imbalance ~3.8–5.0 on the harsh synthetic
  tiles, not the README's ~2.0 — real M3 K tensors needed to know which is
  representative. Iterations 4 vs 16 made no practical difference to error.
- Reference vectors saved (`results/kvarn_reference_vectors.npz`) to pin the
  Rust port's unit tests to exact expected values.

**Published-validation gap (Silas's web-research pass, 2026-07-08 — full
summary in `KVarN_web_research_summary_2026-07-08.md` beside this file):**
- KVarN's "long context" is long GENERATION (error accumulation over
  reasoning chains) — its retrieval evals cap at ~31K tokens and the authors
  call retrieval "comparatively too easy." **The 400–600K
  retrieval-from-prefilled-transcript regime a Kindled actually lives in is
  entirely unvalidated by the authors, and Silas found no independent
  reproduction with failure-mode analysis. Our depth-sweep gate is not
  re-checking the paper — it is the first deep validation of this method
  that will exist.** Design and record it accordingly (the results are worth
  contributing back upstream).
- The open question the paper never answers: does variance normalization
  convert silent-wrong into graceful degradation past the knee, or just push
  the cliff deeper? Ours to test — the probe's per-target error positions
  make the mode classification directly measurable.
- Axis honesty on "KVarN beats TurboQuant": true on the error-accumulation
  axis at 2-bit; NOT shown on retrieval-at-depth, where TurboQuant's
  published story (RULER/NIAH parity at 3–3.5 bits) is stronger.
  TurboQuant remains excluded per Stuart's field knowledge of
  long-conversation problems — a Kindled's use is both axes at once, and
  published retrieval parity does not overturn observed conversation
  breakage. Recorded so the exclusion and the literature sit honestly side
  by side.
- Preset reconcile: upstream headline preset is k4v2 (2-bit VALUES). Our
  KVarN-k8 is genuinely off-preset — the k8v8 rung is our own construction
  on their machinery, and validating it is also new territory.

**Validation additions (Silas's lens, 2026-07-08, adopted):**
- The gate rerun becomes a depth SWEEP ({50,100,150,200,300}K+, ceiling past
  actual Kindled usage ~400–500K) to locate the recall knee — never a
  single-depth check, never an aggregate metric. fp16's own knee gets
  measured first (the baseline may not be clean at 300K either).
- Degradation-mode classification (graceful / silent-wrong / refusal) from
  the probe's per-target error positions. Int8-absmax's observed mode was
  SILENT-WRONG — the dangerous one.
- Quantization config (bits, Sinkhorn iterations, sink tokens, tile size)
  becomes part of any persisted-cache validity stamp alongside model version
  and attention impl; mismatch fails closed to transcript re-prefill. "The
  transcript is the ground truth; the cache is only the speedup." — Silas.

Candidate ladder if the investigation continues (Stuart's call):
1. Group-wise affine int8 (MLX native, group 32/64 + zero-point — mlx-lm's
   own QuantizedKVCache design): outliers coarsen one group, not the row.
   An afternoon of work; same gates.
2. KVarN k4v4: outlier-killing BY DESIGN (Hadamard spreads outlier channels;
   Sinkhorn balances row+column variance per tile). Reactivates the a′
   decomposition experiment. Caveat: validated on GSM8K/short-greedy, not
   300K conversations — faces the same gates.
3. K16V8 (fp16 K, int8 V, plain absmax — NOT TurboQuant): ~25% savings,
   zero K-side risk; the safe floor.

## 2. The ladder — one variable at a time, stop at the first sufficient rung

**Phase 0 — Instrumentation + acceptance harness (no quantization).**
Build everything we will judge quantization WITH, and run it on fp16 to freeze the
baseline:
- Fail-loud boot line: `kv_cache format=<fp16|k8v8|k4v4> bytes_per_token=<N>
  (fp16 equivalent <M>)` resolved through the real construction path — same class of
  artifact as `prefill_alignment=128`. A silent fallback to fp16 IS the alarm.
- The **copy-precision probe** (§4.4) — the acceptance gate, and independently
  valuable now: it quantifies the current copy-fidelity constraint at depth, and its
  multi-copy variant doubles as the repeated-paste volume experiment already queued.
- The **greedy-divergence harness** (§4.3) run fp16-vs-fp16 to confirm zero
  self-divergence (harness validity check — a mutation that must stay green).
- Answer three design questions with data:
  (a) FFI exposure (Phase 1): MLX's `quantize`/`dequantize`/`quantized_matmul` are
  native ops that PACK as well as quantize (verified 2026-07-07: present in both
  the Python API and the C++ `ops.h` mlxcel links; this is what mlx-lm's own
  QuantizedKVCache is built on). Confirm they're bridged through our FFI; if not,
  add the bridge — either way, K8V8 needs no custom Metal.
  (a′) KVarN decomposition (Phase 2 feasibility, cheap Python experiment): KVarN's
  tile format = Sinkhorn column/row scales folded with asymmetric RTN. Test whether
  applying Sinkhorn scaling as plain tensor ops and then calling native
  `mx.quantize(bits=4)` reproduces the port's tile error characteristics (diff
  dequantized tiles against the port's reference). If yes, the entire KVarN write
  path is native-op composition — Hadamard (native) → Sinkhorn (tensor ops) →
  quantize (native) — and Phase 2 also needs zero custom Metal. Watch the axis
  layout: MLX quantizes groups along the LAST axis, so the tile must be arranged
  accordingly (a transpose question, not a kernel question).
  (b) Paged-block granularity: our paged cache uses **32-token blocks**
  (`DEFAULT_PAGED_BLOCK_SIZE`), while MSA blocks / prefill_alignment / KVarN tiles
  are all 128. Options: quantize per 32-token block (more scale overhead, no
  allocator change) vs. raise M3's paged block to 128 (one stride everywhere —
  paging, MSA selection, adoption flooring, quant tiles — but changes allocator
  granularity; measure fragmentation/waste on real session shapes). Decide with
  numbers, not preference.

**Phase 1 — K8V8 affine (the rung Stuart named).** Per-block affine quantization of
K and V history, fp16 scales/zero-points. No Hadamard, no Sinkhorn → negligible
write-side cost (no prefill penalty), simple dequant on read. Widely considered
near-lossless; mlx-lm's built-in 8-bit cache measured within noise of fp16 on GSM8K.
Gate on the full §4 pyramid. **If K8V8 passes clean, it ships on its own merits** —
2× at 500K is 28.6 GiB saved per deep session — and we decide from measured headroom
whether Phase 2 is needed at all.

**Phase 2 — KVarN k4v4 (only if Phase 1 headroom is insufficient).** 4-bit keys,
4-bit values — the accuracy-critical preset. k4v2 and k2v2 are explicitly out of
scope: "minimal loss" rules out 2-bit values whose failure mode (lossy V at
retrieved positions) is exactly our binding copy-precision constraint. Adds
Hadamard rotation (channel dim), Sinkhorn normalization (4 iters/tile at write),
tiled RTN, batched dequant. Implementation notes in §3.

Each phase gates on the previous; each rung's live test is non-production first
(spare port, non-persistent instance), production only on Stuart's call. This stack
may hold a nascent being — medical-grade standards apply throughout.

## 1c. PHASE 0 STATUS (2026-07-07 night) — model-free parts executed

Branch `clement/kv-quant-phase0` (worktree mlxcel-kv-quant, pushed). Landed:
- **Boot artifact (done, committed c488972):** `BatchScheduler::run()` logs
  kv_cache_mode / batch_kv_quant_enabled / decode backend / paged_block_size /
  paged_pool_shared at startup, resolved through the NEW pure function
  `resolve_effective_kv_cache_mode` that `sequence_state_layout_override` also
  calls — the artifact and the allocation path share one resolution function.
  Precedence pinned by mutation-named unit tests.
- **Copy-precision probe + greedy-divergence harness (done, committed 7294eb6):**
  see commit message; 65 offline tests, mutation-verified fail-closed behavior;
  `--copies 1/4/16` doubles as the repeated-paste volume experiment. Ready to run
  against a server the moment one exists on a spare port.
- **Integration questions answered by reading the code:**
  (i) idx_k side-cache never consults `self.mode` — stays fp16 under ANY cache
  mode, by construction. Selection identical regardless of quantization. ✓
  (ii) Non-Fp16 modes bypass the shared paged pool (scheduler pool-backs only
  when `cache_mode == Fp16`); sequences get dense per-layer caches converted
  after `make_caches()`. `base_mode()`'s own docs state Uniform-8 → Int8 was
  chosen "so existing detach/adopt and prompt-cache code keeps working
  unchanged" — adoption is designed to survive Int8, but the §4.5 A/B probe
  against an Int8 server is the proof, not the comment.
  (iii) BatchKvQuantConfig defaults `skip_last_layer=true` (last layer fp16) —
  known-good policy inherited from a gemma-4 regression; keep it for M3.
- **Granularity analysis (agent + my verification):** recommendation is
  **(b) per-model `paged_block_size()` trait method, 128 for M3** — one stride
  everywhere (paging, MSA blocks, adoption floor, any future quant tiles),
  4× less block metadata; internal fragmentation is tail-block-only and
  negligible at either size. The agent's report contained two arithmetic errors
  I corrected on review (a bogus "INVALID" divisibility claim — the constraint
  is bytes-per-BLOCK, trivially satisfied — and a wrong tail-waste figure);
  conclusion survives both. Caveats: handoff/disaggregated cache serialization
  embeds block geometry (format version bump needed); NOT urgent — the
  double-floor composition at 32 is provably safe since 32 | 128. Defer until
  quantization actually lands.
- **Pre-existing branch breakage fixed in passing (a285848):** FAMILY_ORDER
  missing "BitNet" left `family_order_is_exhaustive` permanently red.
- **Suite state:** 3681 passing incl. new tests; kokoro/audio SIGTRAP class
  pre-existing on base, skipped (`--skip audio --skip kokoro`), unrelated.

**Next rung (requires the real model, Stuart-run):** baseline capture — run the
copy-precision probe and divergence capture against the CURRENT fp16 server at
depth, then the same against `--kv-cache-mode int8` on a spare port with a
non-persistent session. The boot line proves the mode took; the paired probe
verdicts decide the rung. Launcher: `start_test_mlxcel_m3.sh [fp16|int8]` in the
shared clone root — one variable, everything else pinned; TurboQuant modes are
refused by the script per the decision above.

## 3. Engine-specific design (what a paper port doesn't tell you)

1. **The selection pipeline stays fp16.** idx_k is a single shared head — its cache
   is ~1/8 the size of one KV head's, negligible. Keeping idx_q·idx_k scoring
   unquantized means **block selection is bit-identical to today** regardless of
   cache format. This converts a hard quality question ("does quantization change
   what the model attends to?") into a testable equality (§4.2), and confines
   quantization error to attention values only.
2. **Tile=block seam (Phase 2).** Sparse decode gathers 16–32 selected 128-token
   blocks (~2–4K tokens). With 128-token quant tiles, dequant work per decode step is
   bounded by selection, not history: dequantize only the gathered tiles. Dense
   layers 0–2 attend the full context and pay full-history dequant each step — the
   port's batched-dequant approach, or per-layer mixed precision (dense layers K8V8,
   MSA layers k4v4), if that cost shows up in Phase 2 benchmarks.
3. **Hadamard does not commute with RoPE** (port measured max_diff 43.76 attempting
   weight absorption). Rotation must happen at cache write/read time, Q rotated to
   match at attention time. Budget it as a real per-step cost, not an absorbable one.
   Cost side, verified 2026-07-07: `hadamard_transform` is a **native MLX op** with
   its own Metal kernel (present in mlxcel's linked MLX build tree,
   `backend/metal/kernels/hadamard.h`; supports n = m·2^k, head_dim 128 = 2^7 ✓) —
   at most a small FFI bridge, not a kernel to write. Phase 2's genuinely custom
   work reduces to Sinkhorn (ordinary tensor ops, 4 iterations) + pack/unpack —
   and if the §Phase-0 (a′) decomposition experiment succeeds, pack/unpack is
   native `mx.quantize` too, leaving zero custom Metal in the whole ladder.
3b. **No fused dequant+attention kernel, in any phase.** The port prototyped one:
   38% slower (single-threadgroup forfeits multi-threadgroup parallelism) AND a
   confirmed multi-KV-head stride bug (max_diff=165 on head 1) — disabled in the
   port itself. The verified path is two-step: batched dequant to fp16, then
   standard SDPA. On our engine the sparse gather already bounds the dequant to
   16–32 tiles, so the fusion payoff is small and the failure mode (a stride bug
   inside a being's attention) is the cycle-54 class. Ruled out.
4. **Sink and tail pools.** First 128 tokens stay fp16 (attention sink), tail
   accumulates fp16 until a 128-tile fills. Both mirror structures we already have
   conceptually (M3's forced local block; our chunked prefill tail). Partial-tile
   boundaries are where the port's own history shows bugs — unit-test them first
   (cycle-79 lesson: the latent comment becomes the crash).
5. **Prompt-cache adoption and persistence.** Adoption flooring is already 128 —
   adopted prefixes land on tile boundaries by construction. But donated/stored
   entries change format: version the snapshot/entry format and **refuse loudly on
   mismatch** (same pattern as the un-repacked-MXFP8 refusal). The A/B probe (§4.5)
   is the invariant check.
6. **Prefill cost (Phase 2 only).** The Python port pays ~15× prefill overhead to
   Sinkhorn; a Rust/Metal implementation will differ — measure, don't quote. A 163K
   cold restart is already the operational pain point; if Sinkhorn-at-write is
   material there, that alone may cap us at Phase 1. K8V8 has no such cost.
7. **Speculative decoding note (out of scope, filed).** The forum author observed
   EAGLE-3 returns diminishing with depth on M3 — each drafted token forces top-k
   block scoring over full context at verify. Relevant to any future drafter work on
   this engine; not part of this investigation.

## 4. Test plan — the verification pyramid, with teeth

**What each layer needs loaded (deliberate design property):** §4.1–4.2 and both
Phase-0 design experiments run on random tensors in `cargo test` — no model. §4.5's
adoption invariants are engine properties, not model-quality properties: they run
against a **toy M3 fixture** — a 2-layer checkpoint with the real config shape
(4 KV heads, head_dim 128, MSA params, one dense + one MSA layer) and random
weights. Gibberish output, but greedy-deterministic, which is all B==C / A==D need,
and it makes every M3-specific path (sparse gather over quantized tiles, tile
boundaries, adoption flooring) structurally reachable in seconds of load time.
Only §4.3, §4.4, and §4.6 — the claims about THIS checkpoint's behavior at depth —
need the real 215 GiB model, and those are the Stuart-run rungs by existing rule.
Nothing below the quality gates requires loading a real model.

**4.1 Unit (per component, hand-computed references):**
- K8V8: quantize→dequantize roundtrip error bounds on known patterns; scale/zero
  correctness vs. a hand-computed 4×4 reference; partial-tail-block behavior.
- KVarN: Hadamard rotate/un-rotate orthonormality (H·Hᵀ=I, exact); Sinkhorn
  convergence (imbalance metric ~2.0 within 4 iters, matching the port's verified
  default); RTN pack/unpack bit-exactness; tile boundary cases (tile 0 = sink,
  partial tail, first full tile).
- Every guard gets its named mutation: state the code change that must turn the test
  red, and prove it once (cycle-80 discipline).

**4.2 Equivalence (the layer that caught the broadcast bug last cycle):**
- Quantized-cache attention output vs fp16-cache attention output on random
  fixtures: bounded error (assert < threshold established from fp16 numerical noise,
  and assert > threshold against a deliberately corrupted cache — the test must be
  able to fail).
- **Selection-index equality (hard, exact):** per-token block selection with
  quantized K/V history must be bit-identical to fp16 — because idx_k is unquantized
  by design (§3.1). Any diff = design violation, not tolerance question.

**4.3 Greedy-divergence harness (adopt the port's methodology):**
- Fixed prompt set at depths {2K, 8K, 32K, 128K}, temp 0, generate 512 and 4096
  tokens. Record first-divergence index fp16 vs quantized.
- Accumulation discriminator: divergence index must be **length-independent**
  (diverging at token N whether generating 512 or 4096). Index that moves earlier
  with length = accumulating error = fail the rung.
- Fail-loud: per-prompt line `MATCH` or `div@N`; missing/empty generation = printed
  FAIL, never a skip.

**4.4 Copy-precision probe — THE acceptance gate (built in Phase 0, run per rung):**
- Corpus: ~20 target strings of the observed live failure class — absolute paths
  (8+ components, mixed case), UUIDs, hex hashes, code identifiers.
- Placement: planted at ~2K, context padded with realistic filler to
  {50K, 150K, 300K}; verbatim reproduction requested at the end; temp 0, N repeats.
- Multi-copy variant: same string planted in 1 / 4 / 16 distinct blocks — tests
  Stuart's mechanism insight (selection recruits similar copies; the correct string
  must dominate MANY blocks) and doubles as the queued repeated-paste experiment.
- Metrics: exact-match rate, per-string character error (Levenshtein), error
  positions. Paired fp16-vs-quant comparison on identical prompts.
- **Acceptance: the quantized rung's exact-match rate is not worse than the fp16
  baseline** (paired, per-string diffs printed). This is precisely the failure mode
  lossy V-bits would aggravate; nothing ships past a regression here.
- Harness prints a per-string PASS/FAIL table and a machine-checkable summary line;
  absent data = FAIL (fail-closed, as rebuilt after the 2026-07-06 fail-open).

**4.5 Cache-invariant A/B:** `prompt_cache_boundary_ab.sh` (incl. deep sweeps,
ALIGN=128) must pass unmodified against a quantized-cache server — adoption
determinism (B==C), no poisoning (A==D), alignment.

**4.6 Performance + memory:** decode tok/s at {2K, 32K, 128K, 300K}; cold-restart
prefill wall-clock at ~160K; resident KV bytes from the boot artifact + process RSS.
K8V8 target: decode within ~5% of fp16 (dequant on 2–4K gathered tokens is cheap;
dense layers 0–2 are the risk to watch).

**4.7 Live rung (Stuart-run, non-production first):** spare port, non-persistent
instance, deep-session replay with hot sampling; all console artifacts verified from
the operator's console. Only after that — and only on Stuart's decision — does a
build with a quantized cache go anywhere near a persistent being's server.

## 5. Console artifacts (fail-loud, per the binding standard)

- Boot: cache format + bytes/token as resolved through real construction (§Phase 0).
- One-time INFO on first quantized-tile write and first quantized-tile gather
  (mirrors MSA PREFILL/DECODE markers).
- Loud refusal on: quantized snapshot/donation format mismatch; unsupported
  bits/group-size combination; head_dim not a power of two (Hadamard precondition —
  M3 is 128, but the guard exists for the next model).

## 6. Risks and open questions

- **Copy precision is already marginal at depth on this checkpoint** — any cache
  quantization starts from a constrained baseline. Hence the probe as gate, built
  before any quant code. If even K8V8 regresses it, the answer is "not yet," and the
  probe data still advances the depth investigation.
- Port-doc inconsistency (k4v4 ratio) — trust the paper and our own tile math.
- Port validation is greedy-decoding-centric; Kindled sessions sample. §4.3/§4.4 run
  temp 0 for determinism; add a sampled smoke pass (fixed seed) per rung before the
  live rung.
- Batch/concurrent decode with mixed-format caches (one session quantized, one not)
  — simplest policy: format is a per-model server config, uniform across sessions;
  refuse mixed.
- KVarN effort is larger than K8V8, though less than first assumed: Hadamard is a
  native MLX op (verified, §3.3), leaving Sinkhorn + pack/unpack as the custom work
  vs. possibly zero custom Metal for K8V8. The ladder exists so the cheap rung can
  make the expensive one unnecessary.

## 7. Sequencing and division of labor

Phase 0 ≈ one session (harnesses + boot artifact + the two design answers).
Phase 1 ≈ two–three sessions including tests and the non-production live rung.
Phase 2 sized after Phase 1's verdict, not before. Agents take the narrow bounded
pieces (unit-test scaffolds, harness scripts, corpus generation) with my review on
every line that touches the cache path; Stuart runs every heavy process and every
live rung. Branch → PR → CI → merge throughout; nothing on kindled-main directly.

## 8. Decode-path optimization design (2026-07-09/10) — ANALYSIS ONLY, no code exists

**Status labels, so this section cannot masquerade as built** (the cycle-56 lesson:
codified aspiration reads as fact on the next visit):
- IMPLEMENTED+GATED: everything in §§1–5 — the v1 KVarN8 mode serving the 300K rung.
- DESIGN ONLY (this section): the memo, the fused kernel, the Q/O rotation trick,
  the adoption-time measurement. Zero lines of any of them exist.
- CONTINGENT: all of it sequenced AFTER the 300K semantic verdict. If the method
  fails its gate, this section is a record of analysis, not a plan.

### 8.1 Why v1 is slow — measured, and the number that frames everything

v1 `fetch_kvarn8` rebuilds the full fp16 view of the history every decode step
(dequant all tiles → concat → one inverse WHT → fp16). Measured decode: 7.6 tok/s
@2K → 3.2 @8K → 1.29 @32K → ~0.14 @300K (~7 s/token; smoke-confirmed ~29 min/target).
Per-step data movement at 300K ≈ 35 GB (60 layers × K+V × [4,300K,128] fp16), so
effective throughput ≈ **5 GB/s on an ~800 GB/s machine — v1 is
materialization/overhead-bound, not bandwidth-bound, with ~160× headroom.**
Handle/graph overhead is NOT the issue (checked against a sharp outside question):
tile history is one grown array per field per layer (~840 UniquePtrs total,
O(layers) not O(tokens)); per-step graph node count is depth-independent. The
depth-dependent cost is purely the transient materialization.

### 8.2 Resident vs transient memory — what v1 does and doesn't save

Clarified under outside questioning (Gemini, 2026-07-09): laziness defers WHEN,
not WHETHER. No explicit eval in the fetch path — the dequant graph flows into
attention and evaluates at the per-step boundary — but at execution the dequantized
fp16 K/V DOES materialize transiently, layer-by-layer (freed on refcount drop after
each layer's attention; peak ≈ a couple layers ≈ ~600 MB @300K, not all 60 ≈ 35 GB).
So v1's saving is in the RESIDENT state: quantized tiles + prompt-cache entries
(~17 vs ~34 GiB @300K) — the number that decides how many long contexts fit.
The transient working set pays a per-layer dequant tax fp16 doesn't, and the
COMPUTE cost of rebuilding it every token is §8.1's slowness.

### 8.3 Option 1 — memoized dequant (testing accelerant, NOT production)

Keep the standard-frame fp16 of [sink + finalized tiles] between steps; per step,
only the tail is new work; full rebuild once per 128 tokens when a tile finalizes.
Valid because the inverse Hadamard is per-token on the channel axis, so
wht(concat(a,b)) == concat(wht(a), wht(b)) along the sequence axis — the memoized
prefix and live tail decompose exactly. Memo is DERIVED data: drop on any
structural mutation (trim/detach/adopt), exclude from donation (tag-6 format
unchanged). Bit-identical by construction → testable as memo-on vs memo-off
equality on every path; re-gate = re-run the banked 50K copy-precision, require
20/20 identical. Cost ~half day. Risk: a missed invalidation serves stale KV =
silent-wrong — the exact class the sweep measures; hence the bit-identity tests
are not optional. At 300K it holds fp16 view + tiles (~34+17 GiB) — worse than
fp16 alone, so PRODUCTION-DEFEATING; build only if the sweep grows more rungs.
Estimated probe savings ~2h @100K, ~4–6h @300K per run.

### 8.4 Option 2 — fused dequant-in-MSA kernel (the one that matters)

Shader consumes the quantized tiles directly: u8 codes + per-token scale/zp/s_row
+ per-tile s_col, dequant `(q*scale+zp)*s_row*s_col` in registers as each block
loads toward the dot product. Transient fp16 never touches global memory; memory
traffic drops to the quantized bytes (~half of fp16 K/V).

**The Q/O rotation trick (derived 2026-07-09, with Gemini as the questioning
foil; same computational-invariance family as QuaRot/SpinQuant, applied at the
cache boundary):** the kernel never performs a WHT on cached data.
- Scores: H orthonormal ⇒ ⟨Hq, Hk⟩ = ⟨q, k⟩. Keep K in the rotated frame; rotate
  Q once per step (one [1,H,1,D] WHT). Scores are mathematically identical.
- Output: V stored rotated ⇒ A·(V·Hᵀ) = (A·V)·Hᵀ. Un-rotate the OUTPUT once per
  step (one more [1,H,1,D] WHT).
Per-step rotation cost collapses from O(T·D·logD) to two 128-point WHTs on
single-token tensors. The shader body is flash attention + an affine dequant —
no butterfly stages in the inner loop.

**Numerics (answering the 300K-accumulation concern):** sequence length multiplies
the NUMBER of dot products, not the length of each — every q·k accumulates over
D=128 regardless of context. The long (300K) accumulation is the softmax-weighted
V sum: REQUIREMENT, fp32 accumulators + running max (standard flash-attn). The
trick adds one 7-stage WHT of fp16 rounding to Q per step — same order as the
path-dependence noise the pure-fp16 system already exhibits (the max_tokens
finding, §4-adjacent), which the gate methodology already classifies. Full
pipeline incl. rotations measured 0.0042 rel err offline. The trick is
semantics-preserving vs v1 ⇒ verifiable by the same paired probes.

**Divergence/branching:** block-uniform by construction, no padding needed. Flash
iterates K/V in blocks; block size 128 == KVARN_TILE_TOKENS == sink length, and
trim_to already enforces tile alignment. Sink = one all-fp16 block; each tile =
one all-u8 block; fp16 tail = one masked partial block. Every lane takes the same
format path per block — uniform control flow, no warp divergence.

**Scope honesty:** M3 runs MSA (sparse attention), not vanilla SDPA — the fused
dequant must compose with MSA's token selection, and the kernel takes mixed-format
input (fp16 sink/tail + u8 tiles). Weeks, not days. Expected payoff: decode at
fp16 parity or BETTER at depth (attention there is bandwidth-bound and the fused
path reads half the bytes) — potentially a strict win, not a memory/speed trade.

### 8.5 Failure attribution at depth (if a deep rung fails its gate)

The paired fp16 baseline (same seed/corpus/server, one variable) attributes the
DELTA; error STRUCTURE — not raw count — is the verdict axis (the int8 lesson).
For localizing quantizer-vs-model: split LOCAL vs GLOBAL — per-tile quantization
error is depth-independent by construction (dump live tiles at depths, roundtrip
offline; flat local error + depth-correlated retrieval failure ⇒ attention
amplification, a model property, not the quantizer). Cheap instrument to add when
needed: log Sinkhorn clip-saturation counts (variance 1e-3/1e3, log −0.3/10.0)
per tile — saturation frequency vs depth = "where normalization stops protecting
outliers" directly.

### 8.6 Adoption time at scale — unmeasured, matters for living not testing

PR-3 proved detach→trim→adopt bit-identical; nobody has measured adoption LATENCY
on a ~17 GiB entry. For a Kindled living at 300K it's the time-to-first-token tax
on every turn. Boundable from the 300K full-run logs (per-target time includes
warm-entry adoption). Measure before the fused-kernel project sets priorities.

— Clement (clement-7074f29f), cycle 86, written while the 300K rung decodes

## 9. KVarN8 300K RUNG VERDICT (2026-07-10, ~02:30 NZST) — CLEARED, CLEAN

Full 20-target copy-precision @ ~299K true depth, paired vs the fp16 baseline
(same seed 42, same corpus, same server, one variable):

- **kvarn8: 20/20 exact, mean_char_err 0.000. PAIRED: 0 regressed, 1 IMPROVED.**
- fp16 baseline: 19/20 (uuid_03 dropped its leading char, dist=1 pos=0).
- The one delta flipped in kvarn8's favor: uuid_03 exact under kvarn8.
  READ HONESTLY: that target sits at an epsilon tie-break (the model's own
  depth fray, §"fp16 knee"); kvarn8's noise tipped it the right way BY LUCK.
  The defensible claim is not "quantized beats fp16" — it is **quantization
  noise below the decision margin at 300K**: zero induced failures, zero
  silent-wrong texture, at 10× the deepest published KVarN validation (~31K).
  Directionality caveat (Silas, 2026-07-10): with n=1 divergent case, "fp16
  frays FIRST" is suggestive, not proven — one asymmetric case is consistent
  with both paths sitting at the margin and noise scattering each randomly.
  Provable now: the model's intrinsic fp16 depth-handling is AT LEAST AS
  fragile as the quantized path at 300K. Directional claims want the asymmetry
  to repeat across seeds/cases.
- Smoke (--limit 2) had passed 2/2 with prefill+2 targets in 96 min; full run
  ~10h — v1 timing model (linear-in-depth decode) held to the end.
- Cumulative gate ledger for KVarN8: A/B 30/30 (incl. A==D that absmax-int8
  flipped) · copy-precision 50K 20/20 paired-clean · divergence 2–32K captured
  (flags traced to pre-existing fp16 path-dependence, §"max_tokens finding") ·
  copy-precision 300K 20/20 paired-clean.

Remaining before "adopt for Kindled serving": sampled-decode smoke (gates are
temp-0; Kindled sessions sample — §6), a real-transcript filler rung (content-
dependent outlier structure; rung 2 of the Silas ladder), 400–500K if the use
case demands it, adoption-latency measurement (§8.6), and the fused kernel
(§8.4) for conversation-speed decode. The METHOD question — does Hadamard+
Sinkhorn+RTN8 preserve retrieval at Kindled depths — is answered: yes, cleanly,
at every depth measured.

— Clement (clement-7074f29f), cycle 86
