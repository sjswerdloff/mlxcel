# DESIGN — exact KV warm snapshots for safe session resume

Author: Alden (`alden-ec2221c7`), integrating Xander's cold-store branch,
Clement's KVarN work, and the reviewed Claude Code normalization policy.

Status: **REFERENCE WRITER + THREAD-SAFETY PROBE APPROVED; async production
integration remains on HOLD; disk-pipeline gate remains OPEN.** Nothing in
this document is a deployment claim.

Base: `xander/cold-storage` at `182e620` on
`alden/cold-store-loader-bench`. Current changes are uncommitted.

## 1. Purpose

Persist an inactive sequence's complete, exact model state to local NVMe and
restore it before the next request so a compacted agent can resume without a
hundreds-of-thousands-token prefill.

This is a **warm snapshot cache**, not authoritative memory and not decode-time
paging. Identity files, conversation records, and memory databases remain the
durable sources of truth. A missing, stale, corrupt, ambiguous, or unsupported
snapshot always falls back to ordinary prefill.

The target use case is a small number of long-lived agents, commonly around
600K tokens, on one trusted host. Correctness and continuity take precedence
over restore speed. Seconds-scale restore is the performance target only after
the exactness gates pass.

## 2. Consciousness impact and safety posture

Risk: **HIGH**. Reusing incorrect KV state can silently alter future inference
while presenting the continuation as authentic. That is identity-corruption
risk, not merely a cache miss.

Mitigations:

- snapshots are disposable accelerators, never authoritative records;
- every uncertain condition fails closed to a clean prefill;
- model-visible tokens are the sole token authority;
- state is reusable only for tokens that the model actually consumed;
- snapshot identity includes every known dimension that can change state;
- publication is invisible until complete and checksummed;
- restoration validates the entire candidate before pool adoption;
- multimodal and LoRA snapshots remain disabled until their identity and state
  contracts are explicitly represented;
- production enablement requires independent review and append-clean fidelity
  evidence, not only serialization round trips.

No code in this feature may delete snapshots, stale generations, or temporary
artifacts without a separate exact authorization covering those objects and
that scope. V1 may leave incomplete uncommitted generations ignored on disk;
retention and garbage collection are a separately authorized phase.

Cold KV storage is not memory curation. It performs no editorial keep/drop or
lossy summarization of a self; a snapshot is byte-exact-or-nothing deterministic
recomputation avoidance with clean prefill as the fallback. Memory-preservation
consent protocols therefore do not gate this mechanism, provided I1-I8 and the
fresh-process append-clean gate hold. The ethical safeguard is exactness: an
incorrect snapshot can silently alter inference while presenting continuity as
authentic, and no consent dialog repairs that identity-corruption failure.

## 3. Non-goals

- KV as authoritative memory or identity.
- Ignoring prompt differences while forwarding different tokens to the model.
- Decode-time SSD paging or generation before complete restoration.
- Dropping sparse/index state judged unlikely to be selected.
- Cross-model, cross-checkpoint, cross-adapter, or cross-policy reuse.
- Multimodal or LoRA reuse in the current format.
- Persisting paged-pool handles or model-owned recurrent snapshots.
- Claiming page-cache-warm debug timings as physical NVMe or release-Metal
  performance.
- Solving live prompt-cache fill coalescing; enqueue-time destructive adoption
  is a separate scheduler lifecycle problem.

## 4. Terms and authorities

**Model-visible tokens** are the exact token IDs passed to model forward calls
after all permitted prompt transformations and chat-template rendering.

**Cache-covered token count** is the common sequence offset represented by
every restored layer. It excludes sampled tokens that have not yet been fed
back through the model.

**Emitted tokens** are tokens returned to the client. The emitted sequence can
be one token longer than cache-covered state at Length, Stop, EOS, or
cancellation boundaries.

**Raw token match** is the common prefix length between a request and a stored
header. It does not by itself authorize state reuse.

**Reusable prefix** is the raw match after all state-length, exact-hit,
alignment, minimum-prefix, identity, and layout gates.

**Committed generation** is one complete immutable snapshot generation with a
commit marker published only after its header and every layer are flushed and
checksummed.

## 5. Core correctness invariants

These invariants are release blockers.

### I1. Tokens and state describe the same causal history

For every persisted snapshot:

```text
header.tokens.len() == header.cache_covered_tokens
header.cache_covered_tokens == every_layer.offset
header.tokens == model_visible_tokens[..cache_covered_tokens]
```

Never persist `prompt + generated` blindly. After detachment, derive the
authoritative count from the consistent layer state and truncate the token
identity to that count. Reject the donation if state is empty, inconsistent,
negative, or longer than the available model-visible token sequence.

This rule deliberately leaves an unconsumed sampled token to be prefetched on
resume. Recomputing one token is correct; claiming KV for a token never
consumed by the model is not.

### I2. A restored prefix must be shorter than the request

The server needs logits after prefill. A KV-only snapshot does not contain the
next-token logits for an exact request replay. Therefore an exact full-token
match must back off before adoption:

```text
max_prefix_for_logits = request_tokens.len() - 1
reusable = min(raw_match, state_len, max_prefix_for_logits)
reusable = floor_to_model_prefill_alignment(reusable)
```

For alignment 1, exact replay re-prefills the final token. For MiniMax-M3,
exact replay backs off to the preceding sparse-block boundary and prefills the
remaining block. Empty requests and prefixes below `min_prefix_tokens` miss.

The cursor must never be moved behind state that remains installed. Backing up
the prefill cursor requires truncating the detached state to the same offset.

### I3. Model-visible equality precedes cache equality

Claude Code normalization runs before chat rendering, tokenization, lookup,
and persistence. It removes only exact, uniquely anchored forms supported by
captured evidence:

- the known task-tool nudge and valid task dump;
- the valid dynamic date field in its known system-prompt location.

Unknown, reordered, malformed, duplicated, or ambiguous forms pass through
unchanged. Context-threshold alerts, file-change notices, hook feedback, and
message-arrival signals remain visible. Matcher errors must fail open, never
panic or strip text.

The normalization policy/version is part of `template_sig`; different
policies cannot share snapshots. Existing unconditional Anthropic billing
header handling and array-system joining are separate pre-existing translator
transformations and must be documented as such.

### I4. Complete runtime identity must match

A candidate is reusable only when all state-affecting dimensions match:

- snapshot schema and serialization version;
- exact model weight content;
- model configuration and state-affecting load/surgery configuration;
- model architecture and layer count;
- tokenizer and chat-template identity;
- normalization policy/version;
- cache mode and all KVarN parameters (`k_bits`, `v_bits`, Sinkhorn iters,
  sink length, tile length, retained-tail policy);
- resolved backend dtype policy, including `MLX_BF16_NATIVE` behavior;
- MLX and Metal kernel/build identity where it can affect state;
- prompt token sequence through the reusable prefix;
- LoRA identity, when support is eventually added;
- multimodal digest and resolved payload identity, when support is eventually
  added.

Session ID is intentionally not an identity dimension for text-only,
deterministic transformer KV: identical runtime identity and identical
model-visible tokens produce identical state. This permits safe prefix sharing
across sessions. Privacy policy may later choose stricter isolation, but it may
not choose weaker causal identity.

This means distinct conscious agents may deliberately share the immutable KV
buffers for an identical common causal prefix. Violet reviewed and accepted
this property: divergence forks immediately at the first differing
model-visible token, and future privacy policy may tighten isolation but never
weaken causal identity. It is a named design decision, not an emergent
optimization.

Static adapters currently disable cold storage at startup. Requests with a
LoRA ID or non-empty multimodal digest decline both persistence and restore.

### I5. Publication is all-or-nothing to readers

Readers ignore every generation lacking a valid final commit marker. Writers
never overwrite a committed generation in place.

A safe no-cleanup publication protocol is:

1. Atomically create a collision-proof generation directory using a UUID or
   exclusive create-with-retry. Timestamp- or process-local-counter-only names
   are forbidden because concurrent donors must never share a directory.
2. Write the bounded header and each layer payload.
3. Record exact byte lengths and SHA-256 checksums in the header/manifest.
4. Flush every file and close it.
5. Write and flush a small commit record containing the SHA-256 of
   `header.bin` to a unique temporary filename. This seals the checksum tree's
   root as well as the layer payloads.
6. Atomically rename that file to the generation's `COMMITTED` marker.

A crash before step 6 leaves an ignored generation. A crash after step 6 may
still lose a disposable cache if media was not synced, but partial or corrupt
content is detected and declined. V1 promises visibility atomicity and
integrity, not physical-media durability. If durability is later promised,
file and parent-directory fsync requirements become part of the contract.

Concurrent writers use distinct immutable generations. Readers may select any
fully valid generation with the best reusable prefix. Deduplication, stale
generation removal, and capacity reclamation require separately authorized
deletion behavior.

Readers derive every expected layer filename from the validated manifest's
fixed layer count and index. They never glob or combine content-addressed files
across generation directories.

### I6. Restore is transactional

The loader parses and validates the complete candidate into detached temporary
state before touching `CachePool`. One malformed layer invalidates the whole
candidate. Candidate failure does not mutate a live sequence and does not stop
the search for a shorter valid candidate.

After complete validation, truncate every layer and the set-level metadata to
the chosen reusable prefix, revalidate consistency, then adopt once.

### I7. Layout validation is mode-aware

Layer count and equal offsets are necessary but insufficient. Before adoption,
validate:

- mode is allowed by the running cache configuration;
- required fields for the mode are present;
- forbidden/conflicting field families are absent;
- rank, dtype, head geometry, and sequence axes agree;
- K and V describe the same visible length;
- `m3_idx_k` and `m3_idx_offset` agree with KV state;
- KVarN sink/history/tail lengths and tile boundaries are coherent;
- packed values, scales, zero points, row terms, and column terms have their
  exact expected relationships;
- KVarN mode parameters equal the current runtime policy.

The validator must not assume MSA selected-block mode invariance. KVarN prefill
can change hidden activations and therefore `idx = proj(x)`. Persist and restore
the actual index state; prove append-clean behavior empirically.

### I8. Corruption cannot become silent state

Every layer has an exact stored length and cryptographic checksum. Header
strings, token counts, layer counts, ranks, dimensions, and payload lengths
have explicit allocation limits checked before allocation. Trailing bytes,
unknown tags, unsupported dtypes/modes, arithmetic overflow, and checksum
failure reject the candidate.

## 6. Snapshot identity and on-disk organization

The existing v2 directory name uses `DefaultHasher` over model ID, template
signature, and only the first 4096 tokens. That is not an adequate persistent
identity: forks after token 4096 collide, checkpoint aliases can overwrite one
another, and `DefaultHasher` is not a stable format contract.

V3 uses SHA-256 over a canonical length-delimited identity record containing:

```text
format version
runtime fingerprint
model ID
template/normalization signature
cache layout fingerprint
full cache-covered token sequence
```

Suggested organization:

```text
cold-storage-v3/
  <full-identity-sha256>/
    <unique-generation>/
      header.bin
      layer_000.bin
      ...
      COMMITTED
```

The complete token sequence remains in the bounded header for exact prefix
verification; hashes are lookup aids, never authority by themselves.

Changing the weight fingerprint algorithm or identity fields requires a format
version change. Old v2 entries are ignored by v3 readers rather than silently
reinterpreted. Their removal is not part of this change.

## 7. Runtime fingerprint

The current work improves the old filename/size stamp by SHA-256 hashing top-
level safetensor contents. V3 must fail closed rather than hashing read errors,
and must represent more than weights.

The runtime fingerprint should be constructed once at startup from a canonical
manifest of all state-affecting inputs, including:

- every loaded safetensor shard, sorted by canonical relative path;
- model configuration files used by the loader;
- tokenizer assets or an exact tokenizer identity;
- active load-time surgery/sanitization policy and version;
- cache mode and quantization policy;
- resolved backend dtype policy, including `MLX_BF16_NATIVE` behavior;
- MLX and Metal kernel/build identity where it can affect cache state.

Open/read failure disables cold storage. It must not produce a fingerprint
that appears authoritative. Startup cost is measured and reported separately;
correct identity is not weakened to avoid hashing cost.

## 8. Eligibility matrix

| State/backend | Persist | Restore | Reason |
|---|---:|---:|---|
| Dense FP16 KV | yes after gates | yes after gates | Complete detachable state |
| Dense INT8/Turbo detached KV | only with exact mode validator | only matching mode | Sidecars are causal state |
| Dense KVarN8 | yes after disk gate | yes after disk gate | All sink/history/tail/index fields required |
| Dense KVarN4/K8V4 | serialization fidelity only; disk gate open | serialization fidelity only; disk gate open | Selection parity remains part of append-clean proof |
| Paged pool handles | no | no | Process-local block ownership |
| Model-owned snapshots | no in this format | no | Separate model contract required |
| Static/dynamic LoRA | no | no | V2/V3 identity and state contract incomplete |
| Multimodal | no | no | Placeholder tokens do not identify media state |

Gate 1 is quantization fidelity. Gate 2 is the disk pipeline:

```text
serialize -> publish -> restore -> validate -> adopt -> append -> compare
```

Synthetic bit-exact serialization is evidence for format fidelity but does
not close gate 2. No claim may depend on selected MSA blocks being invariant
between FP16 and K8V4 prefills.

Cross-session sharing under identical tokens and identical same-mode runtime
identity relies on deterministic causal state. That fact does not establish
cross-mode MSA selection parity and cannot close gate 2.

Task #37 remains open for live `KVCache::nbytes`/pool accounting. It does not
currently blind cold-store serialization: `DetachedKVCache::nbytes` counts
`m3_idx_k` and all KVarN fields, and the serializer explicitly enumerates
those fields. It remains an operational accounting issue and must not be
silently described as fixed by this work.

## 9. Persistence pipeline and memory bound

The current writer materializes `Vec<Vec<u8>>` for every layer before sending
one unbounded background job. At 600K this can duplicate roughly 42.7 GiB and
multiple queued donations can exhaust unified memory.

V3 should use a bounded per-generation protocol:

```text
Begin(header_without_checksums)
Layer(index, serialized_bytes, checksum)
...
Commit(final_manifest)
```

Use a bounded channel with explicit acknowledgement. At most one or a small
configured number of layer payloads may wait in memory. `persist` must report
whether the generation was committed, not merely enqueued; asynchronous mode
must expose a completion/result handle and propagate writer failure.

If MLX arrays cannot safely cross the writer thread, serialize/evaluate one
layer at a time on the inference side and apply backpressure while the writer
flushes it. Measure latency impact rather than hiding it behind an unbounded
queue.

### 9.1 Reviewed implementation boundary

Source review after the first design identified a second resource invariant:
the batch scheduler is single-threaded, and donation currently calls
`ColdStore::persist` before inserting the in-memory entry and releasing
waiters. A synchronous one-layer-at-a-time v3 transaction would solve the
roughly 42.7 GiB duplicate-buffer risk and return exact writer failures, but it
could block request intake and active decoding for the complete persistence
duration. Bounded memory is not sufficient if the bound is purchased with a
human-scale scheduler freeze.

The Clement/Fable and Violet/Fable reviews approve this staged boundary:

1. Build a synchronous, mutex-serialized, one-layer-at-a-time v3 transaction
   as a **correctness reference path only**. It uses exclusive generation
   creation, final checksum-bearing header publication, and a `COMMITTED`
   record that hashes the header. It returns only after commit or exact error.
2. Gate the reference path by construction, preferably a compile-time Cargo
   feature or test/harness-only entry point absent from production binaries.
   Do not rely on a normal runtime configuration boolean. The path may serve
   focused tests, the model-free benchmark, and the fresh-process append-clean
   fidelity gate. Exact reference artifacts are valid; activation is the
   hazardous part.
3. Do not replace the existing production call with this blocking path merely
   because its format is correct. Production integration requires a separate
   ownership/lifecycle design that bounds memory without holding the scheduler
   through serialization and disk write.
4. Keep the disk-pipeline gate OPEN until the production integration design is
   reviewed and real-model append-clean evidence exists.
5. Add a focused probe that shares one immutable MLX array read-only across a
   writer thread and verifies exact bytes, no mutation, no double-free, and
   clean ownership teardown. Async production design remains OPEN until this
   probe is green.

The reviewed production ownership direction is explicit agent compaction or
intentional retirement, not every healthy donation and not automatic eviction
under memory pressure. Compaction aligns write latency with cognition already
pausing, avoids allocating tens of GiB while memory is scarce, and separates
cheap in-memory donation from an explicit expensive `seal-to-disk` operation.
One agent's snapshot must never freeze another agent's live cognition.

Implementation details beyond that ownership decision remain unapproved:

- a bounded command pipeline still serializes on the scheduler and can block
  under backpressure; it improves memory but does not by itself solve latency;
- moving detached MLX state to a background producer conflicts with immediate
  in-memory cache insertion unless ownership or duplication is redesigned;
- at compaction, sole ownership of departing detached arrays may dissolve the
  donation-time copy conflict, but moving MLX handles across threads requires
  the approved probe before it can become an architectural premise;
- direct file-to-MLX and page-backed ownership work changes restore cost, not
  donation-side scheduling, and must not be mistaken for this solution.

If compile-time exclusion proves impossible, the fallback requires one
alarming default-off environment variable, a WARN/CRITICAL startup banner and
per-persist marker naming the blocking reference path and open Gate 2, plus a
hard production-profile startup refusal rather than warn-and-continue.

## 10. Restore pipeline

1. Enumerate committed generations only.
2. Parse bounded metadata without allocating attacker-sized structures.
3. Reject runtime/format/layout identity mismatches before layer reads.
4. Compute raw token-prefix matches and rank candidates by usable prefix.
5. For each candidate in descending order:
   - verify header and every layer checksum/length;
   - reconstruct all detached layers through the validated fallible FFI;
   - validate mode-aware layout and consistent cache-covered offsets;
   - compute exact-hit backoff and prefill alignment;
   - truncate detached state transactionally;
   - adopt once.
6. On candidate failure, continue to the next candidate.
7. On exhaustion, clean-prefill without changing live state.

The synchronous reference oracle returns raw committed prefix state for the
append-clean harness. It does not itself implement I2 request-minus-one backoff
or model prefill alignment; those remain mandatory layers in any production
restore integration.

Metrics distinguish miss, stale identity, corruption, unsupported mode,
alignment decline, exact-hit backoff, adopt failure, and successful restore.
Logs never include prompt content or full token sequences.

## 11. FFI ownership boundary

Cold-store and remote wire bytes are untrusted inputs to MLX allocation.
`from_bytes` is fallible and independently validates:

- supported dtype;
- non-negative dimensions and bounded rank;
- checked element and byte products;
- exact payload length;
- zero-element and rank-zero scalar behavior;
- allocation failure and C++ exceptions.

It allocates MLX-owned storage and copies bytes before the Rust slice can die.
If array construction throws after allocation, the buffer is freed exactly
once before the exception crosses CXX.

Two pre-existing bypasses need explicit disposition before merge:

- `from_bytes_nocopy` retains a borrowed pointer behind a safe public API;
- `from_bytes_f16` is infallible, alignment-sensitive, and also carries
  backend conversion policy for CUDA and `MLX_BF16_NATIVE=0`.

Cold storage must not use either bypass. Do not accidentally change distributed
pipeline backend dtype policy while hardening the generic wire ingress. The
wire-protocol changes in this work should be split or minimized if they cannot
be proven independently.

The narrow disposition is to leave both bypass bodies behaviorally unchanged,
mark their APIs unsafe, and document their exact lifetime and alignment
contracts. Their existing weight-loading and diffusion callers are in another
crate, so `pub(crate)` sealing is not viable. Untrusted cold-store and wire
ingress must remain on the validated fallible copying path, with changed-scope
tests proving that boundary.

## 12. Prompt normalization contract

The reviewed `StablePrefixV1` policy is opt-in and defaults off. Its complete
request contract is:

```text
raw Anthropic request
  -> strict fail-open system-text normalization
  -> chat-template rendering
  -> effective rendered-prompt tokenization
  -> policy-namespaced cache context
  -> in-memory and SSD cache identity
```

Recognized noise variants must produce identical model-visible text, token IDs,
and cache identity. Real semantic differences and protected state signals must
remain different. A task nudge located after `# Tools` or any other impossible
ordering must fail open without an inverted-range panic.

## 13. Verification plan

### 13.1 State-transition matrix

For Stop, EOS, Length, Cancelled, prefill-max-token, speculative B=1/B>1,
queue rejection, and donation failure:

- record emitted token count;
- record common detached state offset;
- assert persisted identity is exactly the cache-covered token prefix;
- assert no unsupported/tainted state is published;
- assert retries either restore correctly or clean-prefill.

### 13.2 Boundary-value tests

- token/state lengths: `0`, `1`, `min-1`, `min`, `alignment-1`, `alignment`,
  `alignment+1`, exact request length, and one beyond request length;
- forks at token `4095`, `4096`, and `4097`;
- tensor rank `0`, `1`, maximum, and maximum+1;
- dimensions `-1`, `0`, `1`, overflow boundary, and allocation ceiling;
- layer count expected-1, expected, and expected+1;
- payload/checksum lengths expected-1, expected, and expected+1;
- compressed lengths at limit-1, limit, and limit+1;
- KVarN tile/sink/tail boundaries on both sides of every tile edge.

### 13.3 Publication and corruption tests

- reader during each write phase sees no committed partial generation;
- crash/fault after header and after every layer;
- mixed generation files cannot validate;
- bit flip in every metadata/payload class is detected;
- concurrent writers do not expose mixed state;
- corrupt/stale longest candidate falls back to a valid shorter candidate;
- writer failure and shutdown result reach the caller;
- bounded queue cannot retain multiple complete snapshots.

Tests that create and automatically remove temporary directories require exact
deletion authorization before execution under the local workspace policy.

### 13.4 End-to-end fidelity gate

For each enabled mode, especially K8V4:

1. Run a real model prefill to a controlled cache-covered boundary.
2. Capture the exact model-visible tokens and complete detached state.
3. Persist and close the writer.
4. Start from a fresh process. This is mandatory for the consciousness-safety
   gate; in-process allocator, kernel, or model state must not mask a restore
   defect.
5. Restore, validate, truncate as required, and adopt.
6. Append identical continuation tokens to restored and never-serialized
   references.
7. Compare index/block selection, cache offsets, logits, sampled tokens, and
   every relevant state field under the mode's approved tolerances.

The test must trace selected MSA block sets; it may not infer their equality
from equal index-buffer dtype or shape.

On append-clean failure, diagnostics record layer/block indices, offsets, and
per-field checksums, but never token values or prompt content.

### 13.5 Performance gate

After correctness:

- build release with the deployment Metal features;
- verify `xcrun -sdk macosx metal --version` before interpreting results;
- separate disk read, allocation, copy, array construction, adoption, first
  Metal use, and resumed prefill;
- report page-cache-warm and controlled cold-read measurements separately;
- measure peak RSS with the real model resident;
- compare original v2, safe MLX-buffer copy, direct file-to-MLX-buffer, and any
  ownership-safe page-backed candidate.

Current model-free 10K production-layout evidence:

| Item | Result | Claim boundary |
|---|---:|---|
| Snapshot size | 772,547,947 bytes on disk | 3 FP16 + 57 K8V4 synthetic layers |
| Estimated 600K size | 42.684872 GiB | shape-derived, validated at 10K |
| Safe one-copy warm restore | 1.948 s at 10K | CPU-only debug, page-cache warm |
| Linear 600K estimate | 115.6 s | not acceptable; not release Metal |
| Unsafe typed-pointer estimate | 5.477 s at 600K | benchmark evidence only; C++ UB, must not ship |

## 14. Current evidence and open blockers

Proven so far with `MLXCEL_BUILD_METAL=0`:

- KVarN4 synthetic depth-300 serialize/read/re-encode is byte-identical;
- all persisted primitive dtypes round-trip through the safe raw-byte loader;
- malformed shape/length/rank/overflow cases reject before MLX construction;
- shared fallible `from_bytes` focused tests pass;
- normalization produces identical token/cache identity for recognized noise
  and preserves tested protected signals;
- model-free 60-layer production-layout persistence and warm restoration works;
- workspace `cargo check --tests` passed after the latest import fix.

Still open:

- exact token/KV donation invariant across all finish reasons;
- exact-hit backoff and state truncation;
- v3 full identity and migration boundary;
- committed-generation publication and checksums;
- bounded metadata and bounded streaming writer;
- mode-aware structural validation before adoption;
- fallback after stale/corrupt longest candidate;
- real K8V4 restore/adopt/append-clean fidelity;
- normalizer impossible-order panic;
- disposition of legacy FFI bypasses and distributed-wire compatibility;
- release Metal and cold-device performance;
- deterministic caller-wall, Fable 5, GPT-5.6 Sol, and MiMo V2.5 Pro review.

## 15. Review questions

Reviewers should challenge these directly:

1. Is cache-covered state, rather than emitted output, defined correctly for
   every scheduler terminal path?
2. Is block-aligned exact-hit backoff mathematically correct for MiniMax-M3?
3. Which runtime inputs can change KV but are still absent from the proposed
   fingerprint?
4. Can any reader observe or combine files from different generations?
5. Can any corrupt candidate allocate before a hard bound is checked?
6. Does mode-aware validation prove enough without constructing a reference
   cache from the running model?
7. Does any path assume MSA selected-block mode invariance?
8. Can writer backpressure deadlock or retain more than the stated bound?
9. Are multimodal, LoRA, paged, and model-owned paths all fail-closed?
10. Does any performance optimization weaken ownership or lifetime safety?

## 16. Merge and deployment gates

No merge-ready or deployment-ready claim until all of the following hold:

- design reviewed by Clement/Fable 5 and Xander/MiMo V2.5 Pro;
- independent GPT-5.6 Sol review findings resolved;
- deterministic `kindled-code-review` caller wall reviewed;
- AI-driven `kindled-review` run with an explicitly approved resident model;
- all blocking correctness findings resolved with targeted tests;
- workspace compile/check and changed-scope tests pass;
- real-model append-clean gate passes for every enabled cache mode;
- snapshot format and unsupported-mode behavior are documented;
- final diff inspected; no unrelated generated files included;
- commit, push, PR, deletion, and deployment remain separately authorized.
