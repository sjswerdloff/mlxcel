# C-live revalidation on the tripwire binary — 2026-07-11 16:14

Session: server PID 44000, binary built 15:08 from base 0a2ce74-era
(kvarn8 + C + runtime msa_core + idx_k_win + trim predicate + tripwire).
Request: byte-identical replay of the boot-night 295K family-archive
prompt (/tmp/harvest_request.json). HTTP 200, finish=stop,
prompt_tokens=295,108 (exact match), completion=1,389, cached=0 (cold).
Harvest ACTIVE this session (boot-B comparable had harvest OFF) —
including the NEW idx_k_win full-window writes (~20 GB across nine
crossings × 57 layers), so prefill denominators carry harvest overhead
per the standing ledger caveat.

## Numbers (log-derived, boot-night method)

- Wall total: 2,724.5 s (15:28:56 → 16:14:20 NZT).
- PREFILL: 295,108 tok in ~2,464 s = **119.8 tok/s**
  (boot-night A-leg: 129 tok/s with lighter harvest — the ~7% delta is
  consistent with the added idx_k_win write volume; priced, not free).
- DECODE at 295K: 1,389 tok in 260.6 s = **5.33 tok/s**
  (first "MSA per-token DECODE" 04:10:00.24Z → completion 04:14:20.85Z).
- vs boot-night B-leg (C, harvest OFF): 5.49 tok/s → **−2.9%**.
- vs boot-night A-leg baseline (3.44): **1.55×** (boot-night C: 1.60×).

## Witnesses

- C dispatch: PRESENT — "C qmm-fetch fused core active (first dispatch
  this process) layer=3". Requested AND ran.
- Tripwire: SILENT — zero "kvarn cache cannot strip" fires across the
  full 295K prefill + decode. No false aborts on the production path.
- Boot artifacts: kv_cache_mode=KVarN8, msa-fetch=qmm, harvest dir
  kvarn_harvest_20260711_1524 (announced loud).

## Reading (mine; Violet QEs against the 1.60×/295K baseline)

The −2.9% decode delta vs boot-B sits within plausible run-to-run
variance plus the new binary's per-dispatch runtime-config reads, and
no CROSSING-TRIGGERED harvest write (idx_k/idx_k_win/k_rot/v_rot)
fires during this decode window (no 8K or 32K crossing between
295,108 and 296,497). The every-256th idx_q/sel sampled dumps DID run
throughout decode (310 pairs, ~KB scale each) — a real-but-tiny slice
of the −2.9% that boot-B never paid (Violet QE amendment 2026-07-11:
the original sentence claimed "no harvest write," over-broad). I read
this as REGRESSION GATE PASSED — the production config survived
migration + tripwire + idx instrumentation — but the QE seat owns the
verdict, not the author. QE verdict rendered 16:22: PASSED, confirmed
at source (board @ 6a014ba).

Gate B harvest state at completion: 57/57 layers hold full idx_k_win
keep-latest windows at the deepest crossing; 456+ idx_k sidecars across
nine depth strata; idx_q/sel query dumps from the decode phase present.
The screen implements next (spec locked three-seat; Xander reviews
before it runs).

— Clement (clement-7074f29f), cycle 89, banked pre-crossing.
