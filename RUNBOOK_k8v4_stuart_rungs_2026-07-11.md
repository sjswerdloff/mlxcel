# RUNBOOK — k8v4 Stuart-run rungs (§4.3 / §4.4 / §4.6), staged 2026-07-11

Everything below is STAGED for your launch windows — the real-checkpoint
rungs are yours by existing rule (PROPOSAL §4: "the Stuart-run rungs").
Nothing here blocks the offline work; run legs when wall-clock suits.

## Preconditions (all green tonight, branch clement/kvarn-k8v4)

- Storage + both readers real-tile golden (RESULTS_k8v4_golden_harness,
  addenda 1–2). §4.2 offline equivalence banked @ 1d5d8cf (band
  calibrated 0.0707 measured / 0.12 bound; corrupted-arm permanent).
- k8v4 serves transparent AND gathered MSA decode. C fused dispatch
  (rung 3) is performance parity — correctness does not wait on it.

## Build (once)

```bash
cd ~/ai/ClaudeInstanceHomeOffices/clement-7074f29f/worktrees/mlxcel-h0
cargo build --release
# server binary: target/release/mlxcel-server
```

## Leg A — fp16 baseline captures (port 8896, test port per TESTING_PROCESS)

```bash
LOG=~/mlxcel_test_8896_fp16_$(date +%Y%m%d_%H%M).log
nohup ./target/release/mlxcel-server \
  -m "/Volumes/T7 Shield/models/huggingface_cache_hub/MiniMax-M3-MXFP8-64e-mlx" \
  --host 0.0.0.0 --port 8896 --alias minimax-m3-test \
  > "$LOG" 2>&1 & echo $! > ~/mlxcel_test_8896.pid
```

```bash
# §4.3 baseline captures (registered depths; 512 AND 4096 tokens each)
for d in 2000 8000 32000 128000; do
  python3 scripts/greedy_divergence_harness.py capture \
    http://127.0.0.1:8896 minimax-m3-test \
    --depth $d --max-tokens 512  --out ~/k8v4_rungs/fp16_div_${d}_512.json
  python3 scripts/greedy_divergence_harness.py capture \
    http://127.0.0.1:8896 minimax-m3-test \
    --depth $d --max-tokens 4096 --out ~/k8v4_rungs/fp16_div_${d}_4096.json
done
# §4.4 baselines (registered depths; multi-copy variant per registration)
for d in 50000 150000 300000; do
  python3 scripts/copy_precision_probe.py http://127.0.0.1:8896 \
    minimax-m3-test --depth $d \
    --baseline-file ~/k8v4_rungs/fp16_copy_${d}.json
done
```

(Exact flag names: both scripts have full `--help`; capture/out flag
spellings verified there before each run — the harness is fail-loud on
missing data by design.)

## Leg B — k8v4 captures (same port after A's teardown)

```bash
kill $(cat ~/mlxcel_test_8896.pid)   # after leg A completes
LOG=~/mlxcel_test_8896_k8v4_$(date +%Y%m%d_%H%M).log
nohup ./target/release/mlxcel-server \
  -m "/Volumes/T7 Shield/models/huggingface_cache_hub/MiniMax-M3-MXFP8-64e-mlx" \
  --host 0.0.0.0 --port 8896 --alias minimax-m3-test \
  --kv-cache-mode k8v4 \
  > "$LOG" 2>&1 & echo $! > ~/mlxcel_test_8896.pid
```

**Console checks at boot (fail-loud artifacts):** startup echo carries
`kvarn_format=k8v4` + `kvarn_v_bits=4`; on the first MSA decode with C
enabled, expect the one-shot line `v4 qmm fall-through: serving via
assemble/gathered` (Violet's echo-vs-ran witness — proves the fall-through
ran, not just that qmm was requested).

Then the same capture commands with `k8v4_` output names, and:

```bash
# §4.3 verdicts — length-independence is the discriminator
python3 scripts/greedy_divergence_harness.py diff \
  ~/k8v4_rungs/fp16_div_${d}_${n}.json ~/k8v4_rungs/k8v4_div_${d}_${n}.json
# §4.4 verdict — paired, exact-match not worse than fp16
python3 scripts/copy_precision_probe.py http://127.0.0.1:8896 \
  minimax-m3-test --depth $d --compare-against ~/k8v4_rungs/fp16_copy_${d}.json
```

## Acceptance (registered, PROPOSAL §4.3/§4.4 — not negotiable at run time)

- §4.3: per-prompt `MATCH` or `div@N`; divergence index LENGTH-INDEPENDENT
  (same N at 512 and 4096 — earlier-with-length = accumulating error =
  FAIL). Missing/empty generation = FAIL, never a skip.
- §4.4: exact-match rate NOT WORSE than the fp16 baseline, paired,
  per-string diffs printed. This is THE gate; nothing ships past a
  regression here.
- §4.6 (falls out of the same sessions): decode tok/s at depth from the
  logs; bytes/token from the boot artifact (k8v4 target 1152 B/tok/layer).

## §4.7 (after all above, your call only)

Spare port, non-persistent instance, deep-session replay — TESTING_PROCESS
governs. Only on your decision does a k8v4 build go near a persistent
being's server.

— Clement (clement-7074f29f), cycle 91. Staged while the offline chain
ran; every offline rung above it is green and pushed.
