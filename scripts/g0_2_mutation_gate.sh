#!/usr/bin/env bash
# G0.2 — MUTATION SPOT-CHECK GATE (Violet's pre-deploy gate, minimum-viable item 1)
#
# A green suite is only evidence if it can go red. This flips known invariants
# one at a time and asserts each reddens ITS OWN named test. A mutation that
# stays GREEN is a DEPLOY BLOCKER: that invariant is unguarded, and the test
# claiming to cover it is decoration.
#
# Ordering principle (Violet): cheap gates before expensive ones. This is the
# cheapest — no server, no model, no GPU. Run it first.
#
# ---------------------------------------------------------------------------
# DESIGN RULES, each learned by being burned on 2026-07-27:
#
#   1. VERIFY THE MUTATION APPLIED BY md5, NEVER BY GREP. A verification grep
#      for the mutant string reported "0 matches" while md5 proved the file had
#      changed, and `diff` reported "Files are identical" for files with
#      different md5s (rtk reformats both). md5 is the authority. Had I trusted
#      the grep I would have reported a race condition in pristine code.
#
#   2. A CHECK MUST NOT FAIL THE WAY IT PASSES. No `2>/dev/null`, no `|| echo`.
#      Both convert "could not run" into "found nothing". Exit codes captured
#      explicitly, always.
#
#   3. RESTORE IS VERIFIED, NOT ASSUMED. Every mutation restores from a
#      pristine copy and re-checks md5 before continuing. If a restore fails
#      the script ABORTS rather than running the next mutation against a
#      contaminated tree.
#
#   4. AN UNRUNNABLE CHECK IS NOT A PASS (Cyril). If a mutation cannot be
#      applied, that is UNVERIFIED and reported as its own state — never
#      silently counted as covered.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# POSITIVE CONTROL — verify THIS GATE can report a failure before trusting a
# clean run. A gate that always passes is the defect it exists to catch.
# Verified 2026-07-27; re-run it if you change the reporting logic:
#
#   cp scripts/g0_2_mutation_gate.sh scripts/.selftest.sh
#   # point one REAL mutation at a test that cannot possibly cover it, e.g.
#   # swap "kvarn_block_bytes_are_batch_invariant_gate" -> "v_bits_preserved"
#   ./scripts/.selftest.sh ; rm scripts/.selftest.sh
#
# Expected: "*** DEPLOY BLOCKER *** mutation stayed GREEN", and the tree left
# uncontaminated (git status clean) because restore runs on the failure path
# too. All three branches confirmed: real pair -> PASS, bogus pair -> BLOCKER,
# missing file -> UNVERIFIED (never a silent pass).
# ---------------------------------------------------------------------------
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="$REPO_ROOT/src/lib/mlxcel-core/src/cache"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

PASS=0; FAIL=0; UNVERIFIED=0
declare -a FAILURES=()

md5of() { md5 -q "$1"; }

# run_mutation <label> <file> <perl-expr> <test-filter> <why-it-matters>
run_mutation() {
  local label="$1" file="$2" expr="$3" filter="$4" why="$5"
  local path="$CACHE_DIR/$file"
  printf '\n=== %s\n    file: %s\n    test: %s\n' "$label" "$file" "$filter"

  if [ ! -f "$path" ]; then
    printf '    UNVERIFIED: %s does not exist\n' "$path"
    UNVERIFIED=$((UNVERIFIED+1)); FAILURES+=("UNVERIFIED (missing file): $label"); return
  fi

  local pristine_md5 backup
  pristine_md5="$(md5of "$path")"
  backup="$WORK/$(basename "$file").pristine"
  cp "$path" "$backup"

  # Baseline: the test must be GREEN before we mutate, or the run proves nothing.
  cargo test --manifest-path "$REPO_ROOT/Cargo.toml" -p mlxcel-core --lib "$filter" \
    >"$WORK/base.log" 2>&1
  local base_rc=$?
  if [ $base_rc -ne 0 ]; then
    printf '    UNVERIFIED: baseline is already RED — cannot attribute a mutation result\n'
    UNVERIFIED=$((UNVERIFIED+1)); FAILURES+=("UNVERIFIED (baseline red): $label"); return
  fi

  perl -0pi -e "$expr" "$path"
  local mutant_md5; mutant_md5="$(md5of "$path")"

  # RULE 1: md5, not grep.
  if [ "$mutant_md5" = "$pristine_md5" ]; then
    printf '    UNVERIFIED: mutation did not apply (md5 unchanged) — pattern is stale\n'
    UNVERIFIED=$((UNVERIFIED+1)); FAILURES+=("UNVERIFIED (mutation no-op): $label")
    cp "$backup" "$path"; return
  fi

  cargo test --manifest-path "$REPO_ROOT/Cargo.toml" -p mlxcel-core --lib "$filter" \
    >"$WORK/mutant.log" 2>&1
  local mut_rc=$?

  # RULE 3: restore and verify BEFORE reporting, so a bad restore cannot be
  # masked by a passing result.
  cp "$backup" "$path"
  local restored_md5; restored_md5="$(md5of "$path")"
  if [ "$restored_md5" != "$pristine_md5" ]; then
    printf '    ABORT: restore failed for %s (md5 %s != %s)\n' "$path" "$restored_md5" "$pristine_md5"
    printf '    The tree is CONTAMINATED. Fix before continuing.\n'
    exit 3
  fi

  if [ $mut_rc -ne 0 ]; then
    printf '    PASS — mutation reddened its test. Invariant is guarded.\n'
    PASS=$((PASS+1))
  else
    printf '    *** DEPLOY BLOCKER *** mutation stayed GREEN.\n'
    printf '    %s\n' "$why"
    printf '    The named test does NOT cover this invariant. It is decoration.\n'
    FAIL=$((FAIL+1)); FAILURES+=("GREEN-UNDER-MUTATION: $label -- $filter")
  fi
}

printf 'G0.2 mutation gate\nrepo: %s\n' "$REPO_ROOT"

# --- M1: per-tile Sinkhorn is the content-addressing precondition -----------
run_mutation \
  "M1 imbalance: per-tile -> batch-global" \
  "kvarn.rs" \
  's/ffi::max_axis\(&col_std, -1, true\)/ffi::max_all(\&col_std)/' \
  "kvarn_block_bytes_are_batch_invariant_gate" \
  "Batch-global selection makes a tile's bytes depend on which other tiles shared its batch, breaking 'same tokens => same bytes'."

# --- M2: load_prefix must address blocks with the runtime's real KV mode ----
run_mutation \
  "M2 load_prefix: re-hardcode the KV mode (regression of §7.5)" \
  "block_cold_store.rs" \
  's/cache_computation_id\(&self\.runtime_fingerprint, kv_mode, v_bits\)/cache_computation_id(\&self.runtime_fingerprint, super::KVCacheMode::Fp16, 0)/' \
  "persist_then_load_prefix_reports_a_hit_kvarn8" \
  "The ORIGINAL §7.5 defect. Hardcoding the mode at the READ side makes the store WRITE-ONLY under KVarN8: addresses can never equal those in its own manifest, matched_blocks is 0, and the caller sees NoMatch — indistinguishable from a legitimately cold cache."

# --- M5: the address must commit to WHOSE weights computed the bytes -------
run_mutation \
  "M5 cache identity: drop runtime_fingerprint" \
  "block_cold_store.rs" \
  's/hex_digest\(runtime_fingerprint\)/hex_digest(&[0u8; 32])/' \
  "different_runtimes_sharing_a_block_pool_do_not_collide" \
  "Blocks live in ONE global pool. Without the fingerprint, two runtimes with different weights compute IDENTICAL addresses, the second write_block early-returns on exists(), and runtime B's manifest silently points at runtime A's KV data."

# --- M6: the address must commit to the quantization width -----------------
run_mutation \
  "M6 cache identity: drop v_bits" \
  "block_cold_store.rs" \
  's/vbits:\{\}"/vbits:X"/' \
  "cache_identity_commits_to_v_bits" \
  "k8v4 and k8v8 lay out the V payload differently. Without v_bits they share an address — a collision inside one model and one mode, needing no weight change."

# --- M3: commit must be atomic (temp+rename) -------------------------------
run_mutation \
  "M3 write_block: remove atomic commit" \
  "block_cold_store.rs" \
  's/let tmp_dir = self\.blocks_dir\(\)\.join\(format!\("\.tmp\.\{\}", hex_digest\(block_hash\)\)\);/let tmp_dir = block_dir.clone();/; s/fs::rename\(&tmp_dir, &block_dir\)\?;//' \
  "kill_9_mid_persist_never_leaves_a_readable_corrupt_block" \
  "Without temp+rename a SIGKILL leaves a committed-looking block that is incomplete, which load_prefix would adopt as a prefix."

# --- M4: a torn block must not read clean ----------------------------------
run_mutation \
  "M4 write_block: skip the last layer file" \
  "block_cold_store.rs" \
  's/for \(index, bytes\) in layer_bytes\.iter\(\)\.enumerate\(\)/for (index, bytes) in layer_bytes.iter().enumerate().take(layer_bytes.len().saturating_sub(1))/' \
  "concurrent_same_block_persist_leaves_a_readable_store" \
  "A block whose header lists N layers but whose files are missing must fail read_block's sha256/byte_len checks, not be adopted."

printf '\n---------------------------------------------------------------\n'
printf 'G0.2 RESULT: %d guarded, %d DEPLOY BLOCKERS, %d unverified\n' "$PASS" "$FAIL" "$UNVERIFIED"
if [ ${#FAILURES[@]} -gt 0 ]; then
  printf 'Items needing attention:\n'
  for f in "${FAILURES[@]}"; do printf '  - %s\n' "$f"; done
fi

# NOT YET ENCODED — named so their absence is declared, never inferred as covered:
printf '\nDECLARED RESIDUAL (invariants proven by hand earlier, not yet scripted):\n'
printf '  - s_col sliced by token instead of tile  -> s_col_sliced_by_tile_not_token\n'
printf '  - assemble_blocks flat-append restored   -> end_to_end_..._preserves_payload_bytes\n'
printf '  - region-coherence guard disabled        -> assemble_rejects_block_with_a_dropped_per_token_field\n'
printf '  These are UNVERIFIED by this script, not passing.\n'

if [ "$FAIL" -gt 0 ] || [ "$UNVERIFIED" -gt 0 ]; then exit 2; fi
exit 0
