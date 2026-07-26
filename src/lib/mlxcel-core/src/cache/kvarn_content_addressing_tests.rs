//! KVarN8 batch-invariance gate — proves the per-tile normalization change.
//!
//! Authorship: The Kindled (KVarN is our implementation). Clement
//! (clement-7074f29f), 2026-07-26. (No copied license header — add the repo's
//! canonical one when this lands on a mainline branch.)
//!
//! ## What this gates
//! The per-tile `imbalance` change (`kvarn.rs`) is supposed to make KVarN8
//! quantization BATCH-INVARIANT: a tile's quantized bytes must depend only on
//! that tile's own values (+ config), never on the other tiles in the batch it
//! was quantized with. This is (a) what the KVarN paper prescribes (per-chunk
//! online normalization, arXiv:2606.03458) and (b) the precondition for
//! content-addressing KVarN8 blocks by tokens (v4 block storage; Violet's Q4).
//!
//! [`kvarn_block_bytes_are_batch_invariant_gate`] is the cheap discriminator
//! that gates the expensive 300K semantic run: it quantizes the SAME tile alone
//! (`n_full=1`) vs. embedded in a 64-tile batch of out-of-distribution tiles
//! (`n_full=64`, mirroring live prefill) and asserts the tile's bytes are
//! IDENTICAL. Regression-gate property:
//!   * GREEN on this worktree (per-tile change applied).
//!   * RED on the pre-change global code (best-so-far was batch-global).
//! Run this before spending 300K — a unit test catches a batch-invariance bug
//! in seconds; the 300K run only proves per-tile didn't REGRESS retrieval.
//!
//! Status: drafted 2026-07-26; needs a build+run pass (the determinism CONTROL
//! must be green for a red on the gate to mean anything).

use crate::cache::kvarn::{KVARN_TILE_TOKENS, kvarn_quantize};
use crate::ffi;

const R: i32 = 128; // == KVARN_TILE_TOKENS (pinned by `tile_rows_match_engine`)
const D: i32 = 64;
const P: i32 = 1; // the one "shared prefix" tile whose bytes must be invariant
const S: i32 = 63; // out-of-distribution tiles after it → n_full = P + S = 64

/// Deterministic synthetic tiles: `n` tiles of `R×C`, values a fixed function of
/// the flat index (no RNG — reproducible). `mag` scales magnitude so a suffix can
/// be made deliberately out-of-distribution.
fn synth_tiles(n: i32, r: i32, c: i32, step: f32, base: f32, mag: f32) -> Vec<f32> {
    let count = (n * r * c) as usize;
    (0..count)
        .map(|i| ((((i as i64 * 197 + 13) % 251) as f32) * step + base) * mag)
        .collect()
}

/// Assert context-A's field bytes equal the leading (prefix) bytes of context-B's
/// field. Tiles are axis-0-outermost, so A's P tiles are exactly the leading
/// `a.len()` bytes of B's `P+S`-tile field — no MLX slice needed.
fn assert_prefix_bytes_match(name: &str, a: &[u8], b: &[u8]) {
    assert!(
        b.len() >= a.len(),
        "{name}: context-B field shorter than context-A ({} < {}) — harness/shape bug",
        b.len(),
        a.len()
    );
    if a != &b[..a.len()] {
        let first = a
            .iter()
            .zip(&b[..a.len()])
            .position(|(x, y)| x != y)
            .unwrap_or(0);
        panic!(
            "BATCH-INVARIANCE VIOLATED — field `{name}`: the shared tile's bytes changed \
             between n_full=1 and n_full=64 (first diff at byte {first}). Per-tile normalization \
             did NOT take — KVarN8 quantization still depends on batch grouping (SPEC §5.1)."
        );
    }
}

/// Geometry pin: our tile rows must equal the engine's tile size.
#[test]
fn tile_rows_match_engine() {
    assert_eq!(
        R as i64, KVARN_TILE_TOKENS as i64,
        "R must equal KVARN_TILE_TOKENS"
    );
}

/// CONTROL (must be GREEN): same tokens, same batch, quantized twice →
/// byte-identical. Proves the quantizer + comparison harness are deterministic,
/// so a red on the gate below is a real signal, not flakiness.
#[test]
fn kvarn_quantize_is_deterministic_control() {
    let data = synth_tiles(P, R, D, 0.017, -2.1, 1.0);
    let t1 = ffi::from_slice_f32(&data, &[P, R, D]);
    let t2 = ffi::from_slice_f32(&data, &[P, R, D]);
    let q1 = kvarn_quantize(&t1, 8);
    let q2 = kvarn_quantize(&t2, 8);
    assert_eq!(
        ffi::array_to_raw_bytes(&q1.q),
        ffi::array_to_raw_bytes(&q2.q),
        "q not deterministic"
    );
    assert_eq!(
        ffi::array_to_raw_bytes(&q1.s_col),
        ffi::array_to_raw_bytes(&q2.s_col),
        "s_col not deterministic"
    );
    assert_eq!(
        ffi::array_to_raw_bytes(&q1.s_row),
        ffi::array_to_raw_bytes(&q2.s_row),
        "s_row not deterministic"
    );
}

/// GATE (GREEN proves the per-tile change; RED on the old global code): the SAME
/// tile must quantize to the SAME bytes whether alone (`n_full=1`) or embedded in
/// a 64-tile batch of out-of-distribution tiles (`n_full=64`).
#[test]
fn kvarn_block_bytes_are_batch_invariant_gate() {
    // The shared tile: in-distribution.
    let prefix = synth_tiles(P, R, D, 0.017, -2.1, 1.0);
    // Suffix: deliberately OUT of distribution (×40, offset) to move the global
    // metric as far as possible — so on the OLD global code this test is
    // guaranteed to go red (proving the probe has power); on per-tile it stays
    // green because the shared tile's selection no longer sees the suffix.
    let suffix = synth_tiles(S, R, D, 0.031, 5.0, 40.0);

    // Context A: the tile alone (n_full = 1).
    let a_tiles = ffi::from_slice_f32(&prefix, &[P, R, D]);
    let qa = kvarn_quantize(&a_tiles, 8);

    // Context B: the same tile + 63 OOD tiles, ONE quantize call (n_full = 64) —
    // mirroring live prefill (`cache.rs` `as_tiles(&k_full)`).
    let mut both = prefix.clone();
    both.extend_from_slice(&suffix);
    let b_tiles = ffi::from_slice_f32(&both, &[P + S, R, D]);
    let qb = kvarn_quantize(&b_tiles, 8);

    // Compare the leading P tile of every stored field.
    assert_prefix_bytes_match(
        "q",
        &ffi::array_to_raw_bytes(&qa.q),
        &ffi::array_to_raw_bytes(&qb.q),
    );
    assert_prefix_bytes_match(
        "scale",
        &ffi::array_to_raw_bytes(&qa.scale),
        &ffi::array_to_raw_bytes(&qb.scale),
    );
    assert_prefix_bytes_match(
        "zp",
        &ffi::array_to_raw_bytes(&qa.zp),
        &ffi::array_to_raw_bytes(&qb.zp),
    );
    assert_prefix_bytes_match(
        "s_row",
        &ffi::array_to_raw_bytes(&qa.s_row),
        &ffi::array_to_raw_bytes(&qb.s_row),
    );
    assert_prefix_bytes_match(
        "s_col",
        &ffi::array_to_raw_bytes(&qa.s_col),
        &ffi::array_to_raw_bytes(&qb.s_col),
    );
}
