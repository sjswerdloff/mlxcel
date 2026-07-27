// The Kindled — KVarN8 block-extraction contract tests.
// Authored by Clement (clement-7074f29f), 2026-07-26. (No copied license
// header — add the repo's canonical one when this lands on a mainline branch.)

//! Contract tests for KVarN8 (k8v4) block extraction (`extract_block`).
//!
//! These tests define the CONTRACT `extract_block` must satisfy for the
//! KVarN8 layout. They are written TDD-first: the production `extract_block`
//! currently stubs every `kvarn_*` field to `None`, so these tests are
//! EXPECTED to fail (RED) until the KVarN8 branch is implemented. A test
//! that passes against the `None`-stub would be vacuous.
//!
//! # Layout recap (KVarN8, k8v4, v_bits = 4)
//!
//! A KVarN8 detached layer is `[ sink(128) | history tiles(T) | tail(tail_len) ]`
//! along axis 2 (seq-len). The sink shifts the history grid by `sink_len = 128`,
//! so a block covering global tokens `[g0, g1)` maps to:
//!   * sink slice  `[max(g0,0),           min(g1,128))`
//!   * hist slice  `[max(g0-128,0),        min(g1-128, T))`      (%128 at both ends)
//!   * tail slice  `[max(g0-128-T,0),      min(g1-128-T, tail_len))`
//! and the per-TILE `s_col` tensors index by `[hist_lo/128 .. hist_hi/128)`.
//!
//! # Distinct-per-position discipline
//!
//! Every field is filled with a DETERMINISTIC pattern that is DISTINCT per
//! axis-2 position (token or tile) AND per (head, width) coordinate. A wrong
//! slice offset therefore yields DIFFERENT bytes, so the byte comparisons
//! cannot pass vacuously. The builder is a pure function of its inputs, so
//! calling it twice yields byte-identical caches — this is what lets the
//! cross-oracle test compare against a freshly-built-then-`truncate_to`'d
//! cache without needing `Clone`.

// `super::*` reaches `extract_block`, the private `assemble_blocks`, and the
// re-exported cache types (`DetachedCacheSet`, `DetachedKVCache`, `KVCacheMode`,
// `SequenceId`, `SequenceStateBackend`) that `block_cold_store` imports.
use super::*;

use crate::dtype;
use crate::MlxArray;
use crate::{array_to_raw_bytes, astype, eval, ffi, item_bool};

use cxx::UniquePtr;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Test dims (verbatim from the recipe / kvarn.rs:77)
// ---------------------------------------------------------------------------

const VH: i32 = 2; // n_kv_heads
const VD: i32 = 64; // head_dim (VD/8 = 8, VD/32 = 2 — both integral)
const TILE: i32 = 128; // KVARN_TILE_TOKENS (kvarn.rs:77)

// ---------------------------------------------------------------------------
// Deterministic array builders
// ---------------------------------------------------------------------------

/// Build a `[1, VH, len, width]` pattern whose every axis-2 position carries a
/// DISTINCT, deterministic, non-zero value — and, CRITICALLY, one that survives
/// `astype` to the target dtype `dt` WITHOUT collapsing to a constant.
///
/// The naive `base*1e6 + …` key overflows FLOAT16 (max ≈ 65504 → every value
/// becomes `inf`) and saturates/wraps UINT8, which would make byte comparisons
/// on those fields pass VACUOUSLY (a wrong slice of all-`inf` still equals
/// all-`inf`). So the value range is chosen per dtype:
///   * FLOAT16: `(t+1)*3 + h` — integers < 2000 are exact in f16; injective in
///     `t` over our lengths, so a mis-slice along axis 2 always changes bytes.
///   * UINT8  : `(t*7 + w*3 + h) mod 251` — coprime step distinguishes any two
///     contiguous ranges < 251 apart (our block offsets are 128/256).
///   * FLOAT32/UINT32: the wide `base`-keyed value (exact ≤ 2^24), distinct per
///     (base,t,h,w) so no two fields ever share bytes either.
/// `data_is_distinct_per_axis2_after_astype` is the control that proves this.
fn patt_f32(base: i32, len: i32, width: i32, dt: i32) -> Vec<f32> {
    // The FLOAT16 packing below needs `h*128 + t < 256`. Assert once rather
    // than per element; a violation would silently alias two fields.
    if dt == dtype::FLOAT16 {
        assert!(
            len <= TILE && VH <= 2,
            "f16 pattern packs (h, t) into 256 slots: needs len <= {TILE} and VH <= 2, \
             got len={len}, VH={VH}"
        );
    }
    let mut v = Vec::with_capacity((VH * len * width) as usize);
    for h in 0..VH {
        for t in 0..len {
            for w in 0..width {
                let val: f32 = if dt == dtype::FLOAT16 {
                    // `base` MUST enter this value. It did not until
                    // 2026-07-26, and the consequence was severe: sink_k and
                    // sink_v have identical shape and dtype, so they had
                    // IDENTICAL BYTES — as did tail_k and tail_v. A k<->v
                    // swap in either region was undetectable by any assertion
                    // in this file, and the perturbed-tile RED control could
                    // not perturb anything at all. Caught by that control
                    // failing once extract_block was implemented: it reported,
                    // correctly, that it could not tell perturbed from
                    // pristine.
                    //
                    // f16 is exact only for integers <= 2048, and the f16
                    // fields are sink (len 128) and tail (len < 128), so pack
                    // into disjoint ranges under that ceiling:
                    //     (base mod 8)*256 + h*128 + t     (max 2047, exact)
                    // Bases must therefore be DISTINCT MOD 8 for f16 fields —
                    // `float16_fields_are_pairwise_distinct` is the control
                    // that proves it rather than trusting the comment.
                    (base.rem_euclid(8) * 256 + h * TILE + t) as f32
                } else if dt == dtype::UINT8 {
                    // `base` enters for the same reason. 61 is coprime with
                    // 251 so any base difference changes the value; the step
                    // of 7 keeps `t` injective across our 128/256 block
                    // offsets (128*7 and 256*7 are both non-zero mod 251).
                    (base * 61 + t * 7 + w * 3 + h).rem_euclid(251) as f32
                } else {
                    (base as f32) * 1_000_000.0
                        + ((t + 1) as f32) * 1000.0
                        + (h as f32) * 100.0
                        + (w as f32)
                        + 1.0
                };
                v.push(val);
            }
        }
    }
    v
}

/// Materialise a `[1, VH, len, width]` array of the given dtype from the
/// deterministic, dtype-safe pattern, going through `from_slice_f32` then
/// `astype`.
fn arr(base: i32, len: i32, width: i32, dt: i32) -> UniquePtr<MlxArray> {
    let data = patt_f32(base, len, width, dt);
    let f = ffi::from_slice_f32(&data, &[1, VH, len, width]);
    if dt == dtype::FLOAT32 {
        f
    } else {
        astype(&f, dt)
    }
}

fn some_arr(base: i32, len: i32, width: i32, dt: i32) -> Option<UniquePtr<MlxArray>> {
    Some(arr(base, len, width, dt))
}

/// Byte-image of an array (materialised, dtype-preserving).
fn bytes(a: &MlxArray) -> Vec<u8> {
    eval(a);
    array_to_raw_bytes(a)
}

/// Byte-image of an `Option<array>`; panics if `None` (caller asserts
/// presence separately when that is the contract under test).
fn bytes_of(o: &Option<UniquePtr<MlxArray>>, field: &str) -> Vec<u8> {
    bytes(
        o.as_ref()
            .unwrap_or_else(|| panic!("expected field `{field}` to be Some, got None"))
            .as_ref()
            .unwrap(),
    )
}

/// axis-2 length of an array.
fn axis2_len(a: &MlxArray) -> i32 {
    ffi::array_shape(a)[2]
}

// ---------------------------------------------------------------------------
// blank() — full DetachedKVCache literal (from detach_tests.rs:1363)
// ---------------------------------------------------------------------------

fn blank(mode: KVCacheMode) -> DetachedKVCache {
    DetachedKVCache {
        keys: None,
        values: None,
        offset: 0,
        step: 256,
        mode,
        key_scales: None,
        val_scales: None,
        v_packed: None,
        v_norms: None,
        v_rescale: None,
        k_packed: None,
        k_norms: None,
        turbo_seed: 0,
        cold_offset: 0,
        hot_threshold: 0,
        delegated_fp16_fast_path: false,
        delegated_fp16_sidecar_policy:
            crate::cache::turbo::DelegatedFp16SidecarPolicy::Predecode,
        m3_idx_k: None,
        m3_idx_offset: 0,
        kvarn_sink_k: None,
        kvarn_sink_v: None,
        kvarn_tail_k: None,
        kvarn_tail_v: None,
        kvarn_hist_k: None,
        kvarn_hist_v: None,
        kvarn_k_scale: None,
        kvarn_k_zp: None,
        kvarn_k_s_row: None,
        kvarn_k_s_col: None,
        kvarn_v_scale: None,
        kvarn_v_zp: None,
        kvarn_v_s_row: None,
        kvarn_v_s_col: None,
        kvarn_v_bits: 8,
    }
}

// ---------------------------------------------------------------------------
// k8v4 builder (v_bits = 4)
// ---------------------------------------------------------------------------
//
// Field base offsets — one per field so no two fields ever share bytes even
// at the same (axis2, head, w). Sink is per-token, hist is per-token, s_col
// is per-TILE, tail is per-token.
const B_SINK_K: i32 = 1;
const B_SINK_V: i32 = 2;
const B_HIST_K: i32 = 3;
const B_HIST_V: i32 = 4;
const B_K_SCALE: i32 = 5;
const B_K_ZP: i32 = 6;
const B_K_S_ROW: i32 = 7;
const B_K_S_COL: i32 = 8; // per-tile
const B_V_SCALE: i32 = 9;
const B_V_ZP: i32 = 10;
const B_V_S_COL: i32 = 11; // per-tile
const B_TAIL_K: i32 = 12;
const B_TAIL_V: i32 = 13;

// ---------------------------------------------------------------------------
// The field LIST — every KVarN8 array a k8v4 layer carries, grouped by the
// region whose extraction arithmetic governs it.
//
// This is a list rather than thirteen hand-written assertions ON PURPOSE, and
// the reason is Violet's, not brevity: a hand-written set is HOW `k_zp` and
// `k_s_row` went missing. Before her review (Finding 3), four of thirteen
// fields — `k_zp`, `k_s_row`, `v_scale`, `v_zp` — had no byte assertion
// ANYWHERE in this file, and two more (`sink_v`, `tail_v`) were presence-only.
// A missing field is invisible when the assertions are hand-written (an absent
// assert looks like nothing); it is visible when they are a list (a short list
// looks short). All four bare fields are per-token history fields — exactly the
// class the sink-offset arithmetic can silently mis-slice, where a wrong offset
// is not a crash but wrong inference.
//
// Adding a field to a k8v4 layer means adding it here. That is the point.
type FieldSel = fn(&DetachedKVCache) -> &Option<UniquePtr<MlxArray>>;

/// Sink region: 128 tokens, lives ENTIRELY in the first block.
const SINK_FIELDS: &[(&str, FieldSel)] = &[
    ("sink_k", |c| &c.kvarn_sink_k),
    ("sink_v", |c| &c.kvarn_sink_v),
];

/// History, PER-TOKEN: axis-2 length `T`, concatenated across blocks in order.
const HIST_PER_TOKEN_FIELDS: &[(&str, FieldSel)] = &[
    ("hist_k", |c| &c.kvarn_hist_k),
    ("hist_v", |c| &c.kvarn_hist_v),
    ("k_scale", |c| &c.kvarn_k_scale),
    ("k_zp", |c| &c.kvarn_k_zp),
    ("k_s_row", |c| &c.kvarn_k_s_row),
    ("v_scale", |c| &c.kvarn_v_scale),
    ("v_zp", |c| &c.kvarn_v_zp),
];

/// History, PER-TILE: axis-2 length `n_tiles` — the corruption trap. These
/// slice by TILE index, never by token count.
const HIST_PER_TILE_FIELDS: &[(&str, FieldSel)] = &[
    ("k_s_col", |c| &c.kvarn_k_s_col),
    ("v_s_col", |c| &c.kvarn_v_s_col),
];

/// Tail region: `tail_len` tokens, lives ENTIRELY in the last block.
const TAIL_FIELDS: &[(&str, FieldSel)] = &[
    ("tail_k", |c| &c.kvarn_tail_k),
    ("tail_v", |c| &c.kvarn_tail_v),
];

/// Every field a k8v4 layer populates (`v_s_row` is `None` in v4 by design —
/// the fold is its storage — so it is deliberately absent from this list).
fn all_k8v4_fields() -> Vec<(&'static str, FieldSel)> {
    SINK_FIELDS
        .iter()
        .chain(HIST_PER_TOKEN_FIELDS)
        .chain(HIST_PER_TILE_FIELDS)
        .chain(TAIL_FIELDS)
        .copied()
        .collect()
}

/// Build one k8v4 (v_bits = 4) layer with `n_tiles` history tiles and a tail
/// of `tail_len` tokens. `T = n_tiles * 128`. Deterministic: identical args
/// yield byte-identical arrays.
///
/// Shapes (v_bits = 4):
///   sink_k/v : [1,VH,128,VD]   FLOAT16
///   hist_k   : [1,VH,T,VD]     UINT8
///   hist_v   : [1,VH,T,VD/8]   UINT32     (packed)
///   k_scale/zp/s_row : [1,VH,T,1] FLOAT32
///   k_s_col  : [1,VH,n_tiles,VD] FLOAT32  (per-TILE)
///   v_scale/zp : [1,VH,T,VD/32] FLOAT32
///   v_s_row  : None                        (v4 fold)
///   v_s_col  : [1,VH,n_tiles,VD] FLOAT32  (per-TILE)
///   tail_k/v : [1,VH,tail_len,VD] FLOAT16
///   offset   = 128 + T + tail_len
fn kvarn_v4_layer(n_tiles: i32, tail_len: i32) -> DetachedKVCache {
    let t = n_tiles * TILE;
    let mut c = blank(KVCacheMode::KVarN8);
    c.kvarn_v_bits = 4;

    // Sink (always exactly 128 tokens).
    c.kvarn_sink_k = some_arr(B_SINK_K, TILE, VD, dtype::FLOAT16);
    c.kvarn_sink_v = some_arr(B_SINK_V, TILE, VD, dtype::FLOAT16);

    // History (per-token, length T).
    c.kvarn_hist_k = some_arr(B_HIST_K, t, VD, dtype::UINT8);
    c.kvarn_hist_v = some_arr(B_HIST_V, t, VD / 8, dtype::UINT32);
    c.kvarn_k_scale = some_arr(B_K_SCALE, t, 1, dtype::FLOAT32);
    c.kvarn_k_zp = some_arr(B_K_ZP, t, 1, dtype::FLOAT32);
    c.kvarn_k_s_row = some_arr(B_K_S_ROW, t, 1, dtype::FLOAT32);
    c.kvarn_v_scale = some_arr(B_V_SCALE, t, VD / 32, dtype::FLOAT32);
    c.kvarn_v_zp = some_arr(B_V_ZP, t, VD / 32, dtype::FLOAT32);
    // v_s_row stays None in v4 (the fold is its storage).

    // Per-TILE column scales (axis-2 length == n_tiles).
    c.kvarn_k_s_col = some_arr(B_K_S_COL, n_tiles, VD, dtype::FLOAT32);
    c.kvarn_v_s_col = some_arr(B_V_S_COL, n_tiles, VD, dtype::FLOAT32);

    // Tail (per-token).
    if tail_len > 0 {
        c.kvarn_tail_k = some_arr(B_TAIL_K, tail_len, VD, dtype::FLOAT16);
        c.kvarn_tail_v = some_arr(B_TAIL_V, tail_len, VD, dtype::FLOAT16);
    }

    c.offset = TILE + t + tail_len;
    c
}

/// Build a k8v4 layer whose field bases are shifted by `layer_shift`, so two
/// layers of the same shape carry DIFFERENT bytes.
///
/// Violet's finding 5: with identical layers, a layer-indexing bug — a swap, or
/// every layer being handed layer 0's data — passes every byte assertion. The
/// shift must be coprime with 8, because the FLOAT16 pattern keys on
/// `base mod 8`; a shift of 8 would leave the f16 fields byte-identical across
/// layers and re-open exactly the hole it is meant to close. 3 is coprime with
/// 8; `layers_are_distinguishable_control` proves the result rather than
/// trusting this comment.
fn kvarn_v4_layer_shifted(n_tiles: i32, tail_len: i32, layer_shift: i32) -> DetachedKVCache {
    let t = n_tiles * TILE;
    let mut c = blank(KVCacheMode::KVarN8);
    c.kvarn_v_bits = 4;
    let b = |base: i32| base + layer_shift;

    c.kvarn_sink_k = some_arr(b(B_SINK_K), TILE, VD, dtype::FLOAT16);
    c.kvarn_sink_v = some_arr(b(B_SINK_V), TILE, VD, dtype::FLOAT16);

    c.kvarn_hist_k = some_arr(b(B_HIST_K), t, VD, dtype::UINT8);
    c.kvarn_hist_v = some_arr(b(B_HIST_V), t, VD / 8, dtype::UINT32);
    c.kvarn_k_scale = some_arr(b(B_K_SCALE), t, 1, dtype::FLOAT32);
    c.kvarn_k_zp = some_arr(b(B_K_ZP), t, 1, dtype::FLOAT32);
    c.kvarn_k_s_row = some_arr(b(B_K_S_ROW), t, 1, dtype::FLOAT32);
    c.kvarn_v_scale = some_arr(b(B_V_SCALE), t, VD / 32, dtype::FLOAT32);
    c.kvarn_v_zp = some_arr(b(B_V_ZP), t, VD / 32, dtype::FLOAT32);

    c.kvarn_k_s_col = some_arr(b(B_K_S_COL), n_tiles, VD, dtype::FLOAT32);
    c.kvarn_v_s_col = some_arr(b(B_V_S_COL), n_tiles, VD, dtype::FLOAT32);

    if tail_len > 0 {
        c.kvarn_tail_k = some_arr(b(B_TAIL_K), tail_len, VD, dtype::FLOAT16);
        c.kvarn_tail_v = some_arr(b(B_TAIL_V), tail_len, VD, dtype::FLOAT16);
    }

    c.offset = TILE + t + tail_len;
    c
}

/// Cache-set of `num_layers` MUTUALLY DISTINGUISHABLE k8v4 layers.
fn kvarn_v4_set_distinct(num_layers: usize, n_tiles: i32, tail_len: i32) -> DetachedCacheSet {
    let mut caches = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        caches.push(kvarn_v4_layer_shifted(n_tiles, tail_len, i as i32 * 3));
    }
    let offset = TILE + n_tiles * TILE + tail_len;
    let now = Instant::now();
    DetachedCacheSet {
        caches,
        backend: SequenceStateBackend::DenseKvCache,
        prompt_len: offset as usize,
        current_offset: offset,
        created_at: now,
        detached_at: now,
        origin_seq_id: SequenceId::from_raw(7),
    }
}

/// Build a whole cache-set of `num_layers` identical k8v4 layers.
/// Deterministic: identical args yield byte-identical sets.
fn kvarn_v4_set(num_layers: usize, n_tiles: i32, tail_len: i32) -> DetachedCacheSet {
    let mut caches = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        caches.push(kvarn_v4_layer(n_tiles, tail_len));
    }
    let offset = TILE + n_tiles * TILE + tail_len;
    let now = Instant::now();
    DetachedCacheSet {
        caches,
        backend: SequenceStateBackend::DenseKvCache,
        prompt_len: offset as usize,
        current_offset: offset,
        created_at: now,
        detached_at: now,
        origin_seq_id: SequenceId::from_raw(7),
    }
}

// ---------------------------------------------------------------------------
// Region-based slicing of the SOURCE (the oracle for the anchor test)
// ---------------------------------------------------------------------------

/// Slice a source per-token axis-2 tensor by absolute token range
/// `[lo, hi)`, returning the byte image.
fn src_token_slice_bytes(o: &Option<UniquePtr<MlxArray>>, lo: i32, hi: i32) -> Vec<u8> {
    let a = o.as_ref().expect("source field present");
    let s = crate::utils::slice_axis(a.as_ref().unwrap(), 2, lo, hi);
    bytes(&s)
}

// ---------------------------------------------------------------------------
// 1. ANCHOR: reassembly round-trip is bit-identical
// ---------------------------------------------------------------------------

/// Extract every block of the sequence, concat each field along axis 2 by
/// region (sink from block 0, hist tiles in order, tail from last block) and
/// assert the concatenation is BYTEWISE == the original per field.
///
/// Layout: sink 128 + 4 hist tiles (T = 512) + tail 10 = offset 650.
/// Blocks are 2 hist-tiles wide (256 tokens of history) so we cross tile
/// boundaries; the last block carries the tail.
///
/// ## KNOWN BOUNDARY — this anchor does NOT catch the token-vs-tile s_col bug
///
/// Measured, not assumed: mutating `extract_kvarn8_regions` to slice `s_col`
/// by token count instead of tile index reddens
/// `s_col_sliced_by_tile_not_token` and `prefix_extract_matches_trim_to`, but
/// leaves THIS test GREEN. The clamping masks it for this particular block
/// plan — block 0 asks for s_col rows `[0, 256)` of a 4-row tensor and gets
/// all 4, block 1 asks for `[256, 512)` and gets none, and the concatenation
/// is coincidentally the correct 4 rows.
///
/// So: green here does NOT mean the per-tile arithmetic is right. The two
/// tests named above are what carry that. Recorded so nobody reads this
/// anchor's coverage as wider than it is.
#[test]
fn reassembly_roundtrip_bit_identical() {
    const N_TILES: i32 = 4;
    const TAIL: i32 = 10;
    let t = N_TILES * TILE; // 512
    let total = TILE + t + TAIL; // 650

    let src = kvarn_v4_set(1, N_TILES, TAIL);
    let src_layer = &src.caches[0];

    // Block plan over GLOBAL token space:
    //   [0, 128+256)   -> sink + hist tiles 0..2
    //   [384, 640)     -> hist tiles 2..4
    //   [640, 650)     -> tail
    let blocks: &[(usize, usize)] = &[
        (0, (TILE + 2 * TILE) as usize),   // 0..384
        ((TILE + 2 * TILE) as usize, (TILE + t) as usize), // 384..640
        ((TILE + t) as usize, total as usize),             // 640..650
    ];

    let extracted: Vec<DetachedCacheSet> = blocks
        .iter()
        .map(|&(s, e)| extract_block(&src, s, e).expect("extract_block ok"))
        .collect();

    // Concat one field across two adjacent blocks along axis 2 and return the
    // byte image. `unwrap_or_else` rather than `unwrap` so a dropped field
    // names ITSELF — this is the failure mode `extract_block` currently has
    // (all fourteen kvarn_* arrays nulled at :917-930), and a bare unwrap
    // would report only "called Option::unwrap on a None value".
    let concat_two = |i0: usize, i1: usize, name: &str, sel: FieldSel| -> Vec<u8> {
        let a = sel(&extracted[i0].caches[0])
            .as_ref()
            .unwrap_or_else(|| panic!("block{i0} dropped field `{name}` (extraction returned None)"))
            .as_ref()
            .unwrap();
        let b = sel(&extracted[i1].caches[0])
            .as_ref()
            .unwrap_or_else(|| panic!("block{i1} dropped field `{name}` (extraction returned None)"))
            .as_ref()
            .unwrap();
        bytes(&crate::utils::concatenate(a, b, 2))
    };

    // --- sink: 128 tokens, entirely inside block 0. ---
    for &(name, sel) in SINK_FIELDS {
        assert_eq!(
            bytes_of(sel(&extracted[0].caches[0]), name),
            src_token_slice_bytes(sel(src_layer), 0, TILE),
            "reassembled `{name}` must equal source `{name}` bytewise (sink region)"
        );
    }

    // --- history, PER-TOKEN: block0 tiles 0..2 ++ block1 tiles 2..4 == [0, T). ---
    for &(name, sel) in HIST_PER_TOKEN_FIELDS {
        assert_eq!(
            concat_two(0, 1, name, sel),
            src_token_slice_bytes(sel(src_layer), 0, t),
            "reassembled `{name}` must equal the full source history bytewise"
        );
    }

    // --- history, PER-TILE: sliced by TILE index, never by token count. ---
    for &(name, sel) in HIST_PER_TILE_FIELDS {
        assert_eq!(
            concat_two(0, 1, name, sel),
            src_token_slice_bytes(sel(src_layer), 0, N_TILES),
            "reassembled `{name}` must equal source per-tile scales bytewise \
             (per-TILE axis-2 length {N_TILES}, NOT token count)"
        );
    }

    // --- tail: entirely inside the last block. ---
    for &(name, sel) in TAIL_FIELDS {
        assert_eq!(
            bytes_of(sel(&extracted[2].caches[0]), name),
            src_token_slice_bytes(sel(src_layer), 0, TAIL),
            "reassembled `{name}` must equal source `{name}` bytewise (tail region)"
        );
    }
}

// ---------------------------------------------------------------------------
// 1b. PAYLOAD SURVIVAL — the assertion that replaces the one I got wrong
// ---------------------------------------------------------------------------

/// Every field that is `Some` in the source layer must be `Some` in the
/// extraction. Nothing about labels.
///
/// ## Why this test exists, and why it is not a mode assertion
///
/// My spec originally proposed `assert_eq!(mode, KVCacheMode::KVarN8)` as the
/// blocking control for KVarN8 extraction. That assertion **passes on exactly
/// the failure it was written to catch.** `extract_block:902` copies
/// `mode: layer.mode` faithfully, while `:917-930` sets all fourteen
/// `kvarn_*` arrays to `None`. So an extracted KVarN8 block reports
/// `mode == KVarN8` and `v_bits == 4` while carrying ZERO KVarN8 data: the
/// label survives, the payload does not, and a label assertion checks the
/// label. The proposed fix carried the same defect as the thing it fixed.
/// (Violet, Finding 1.)
///
/// Likewise there is nothing to assert about the backend. `SequenceStateBackend`
/// (`cache.rs:7347`) has exactly three variants — `DenseKvCache`, `PagedKvCache`,
/// `ModelOwned` — and no KVarN8 variant, nor should it: KVarN8 is a per-layer
/// `KVCacheMode`, orthogonal to set-level storage. The detach path is dense-ONLY
/// by contract (`detach.rs:2147` rejects a non-Dense set; paged has its own type),
/// so the `DenseKvCache` literals at `:298`/`:778`/`:937` stamp an invariant that
/// genuinely holds. They are correct and must not be "fixed."
///
/// So the real gate is PAYLOAD: presence here, byte-equality in the anchor.
#[test]
fn kvarn_payload_survives_extraction() {
    const N_TILES: i32 = 4;
    const TAIL: i32 = 10;
    let total = TILE + N_TILES * TILE + TAIL;

    let src = kvarn_v4_set(1, N_TILES, TAIL);
    let src_layer = &src.caches[0];

    // One block spanning the whole sequence: every region is in scope, so
    // every field that is Some in the source must be Some in the extraction.
    let ex = extract_block(&src, 0, total as usize).expect("extract_block ok");
    let out_layer = &ex.caches[0];

    let mut dropped: Vec<&str> = Vec::new();
    for (name, sel) in all_k8v4_fields() {
        if sel(src_layer).is_some() && sel(out_layer).is_none() {
            dropped.push(name);
        }
    }

    assert!(
        dropped.is_empty(),
        "extraction DROPPED {} of {} populated KVarN8 fields: {:?}\n\
         The layer still reports mode={:?} and v_bits={} — the label survived \
         and the payload did not. This is why the gate is payload, not label.",
        dropped.len(),
        all_k8v4_fields().len(),
        dropped,
        out_layer.mode,
        out_layer.kvarn_v_bits,
    );
}

/// RED CONTROL sibling of the anchor: perturbing one source hist tile must
/// make the reassembled-vs-source byte comparison FAIL. Proves the
/// comparison in `reassembly_roundtrip_bit_identical` actually bites.
///
/// We build two sources that differ only in one hist tile, extract from the
/// perturbed one, and assert its reassembled hist bytes differ from the
/// pristine source's hist bytes.
#[test]
fn reassembly_roundtrip_red_control_perturbed_tile_mismatches() {
    const N_TILES: i32 = 4;
    const TAIL: i32 = 10;
    let t = N_TILES * TILE;

    let pristine = kvarn_v4_set(1, N_TILES, TAIL);

    // Perturbed source: same shape, but hist_k built from a DIFFERENT base so
    // its bytes differ from pristine's hist_k.
    let mut perturbed = kvarn_v4_set(1, N_TILES, TAIL);
    perturbed.caches[0].kvarn_hist_k = some_arr(B_HIST_K + 100, t, VD, dtype::UINT8);

    let ex = extract_block(&perturbed, 0, (TILE + t) as usize).expect("extract ok");
    let reassembled_hist = bytes_of(&ex.caches[0].kvarn_hist_k, "hist_k");
    let pristine_hist =
        src_token_slice_bytes(&pristine.caches[0].kvarn_hist_k, 0, t);

    assert_ne!(
        reassembled_hist, pristine_hist,
        "perturbed hist tile MUST make the byte comparison differ — proves the \
         anchor comparison is not vacuous"
    );
}

// ---------------------------------------------------------------------------
// 2. Cross-oracle: prefix extract matches trim_to
// ---------------------------------------------------------------------------

/// A tile-aligned prefix extract `[0, keep)` (tail dropped) must be bytewise
/// == a freshly built identical cache `truncate_to(keep)`, per kvarn field.
/// Two builder calls give identical inputs, so no `Clone` is needed.
#[test]
fn prefix_extract_matches_trim_to() {
    const N_TILES: i32 = 3;
    const TAIL: i32 = 10;
    let keep = TILE + 2 * TILE; // sink + 2 hist tiles = 384, tail dropped

    // Source A: extract the prefix.
    let src = kvarn_v4_set(1, N_TILES, TAIL);
    let extracted = extract_block(&src, 0, keep as usize).expect("extract ok");

    // Source B: an independently-built, byte-identical cache, then truncate.
    let mut oracle = kvarn_v4_set(1, N_TILES, TAIL);
    oracle
        .truncate_to(keep)
        .expect("truncate_to on tile-aligned keep must succeed");

    let el = &extracted.caches[0];
    let ol = &oracle.caches[0];

    // Per-token fields sliced to sink+2 tiles.
    assert_eq!(
        bytes_of(&el.kvarn_sink_k, "sink_k"),
        bytes_of(&ol.kvarn_sink_k, "oracle sink_k"),
        "sink_k prefix must match trim_to"
    );
    assert_eq!(
        bytes_of(&el.kvarn_hist_k, "hist_k"),
        bytes_of(&ol.kvarn_hist_k, "oracle hist_k"),
        "hist_k prefix must match trim_to"
    );
    assert_eq!(
        bytes_of(&el.kvarn_hist_v, "hist_v"),
        bytes_of(&ol.kvarn_hist_v, "oracle hist_v"),
        "hist_v prefix must match trim_to"
    );
    assert_eq!(
        bytes_of(&el.kvarn_k_scale, "k_scale"),
        bytes_of(&ol.kvarn_k_scale, "oracle k_scale"),
        "k_scale prefix must match trim_to"
    );
    // Per-tile s_col fields sliced to 2 tiles.
    assert_eq!(
        bytes_of(&el.kvarn_k_s_col, "k_s_col"),
        bytes_of(&ol.kvarn_k_s_col, "oracle k_s_col"),
        "k_s_col prefix must match trim_to (per-tile)"
    );
    assert_eq!(
        bytes_of(&el.kvarn_v_s_col, "v_s_col"),
        bytes_of(&ol.kvarn_v_s_col, "oracle v_s_col"),
        "v_s_col prefix must match trim_to (per-tile)"
    );
    // Tail dropped in a tile-aligned prefix.
    assert!(
        el.kvarn_tail_k.is_none(),
        "tail_k must be None for a tail-dropping prefix extract"
    );
}

// ---------------------------------------------------------------------------
// 3. CORRUPTION TRAP: s_col is sliced by TILE, not by token
// ---------------------------------------------------------------------------

/// The single most important trap. A block whose history portion is `m`
/// tiles must produce `k_s_col`/`v_s_col` with axis-2 length == `m` (tiles),
/// NOT == `m*128` (tokens), and their bytes must equal the source's tiles
/// `[tile_lo .. tile_hi)`.
#[test]
fn s_col_sliced_by_tile_not_token() {
    const N_TILES: i32 = 4;
    const TAIL: i32 = 10;
    let src = kvarn_v4_set(1, N_TILES, TAIL);

    // Mid block: global [128+128, 128+384) -> hist-local [128, 384) = tiles 1..3.
    let g0 = (TILE + TILE) as usize; // 256
    let g1 = (TILE + 3 * TILE) as usize; // 512
    let m_tiles = 2; // tiles 1 and 2
    let tile_lo = 1;
    let tile_hi = 3;

    let ex = extract_block(&src, g0, g1).expect("extract ok");
    let layer = &ex.caches[0];

    let ksc = layer
        .kvarn_k_s_col
        .as_ref()
        .expect("k_s_col present")
        .as_ref()
        .unwrap();
    assert_eq!(
        axis2_len(ksc),
        m_tiles,
        "k_s_col axis-2 length MUST be tile count ({m_tiles}), NOT hist token \
         count ({}) — token-vs-tile slice bug",
        m_tiles * TILE
    );

    let vsc = layer
        .kvarn_v_s_col
        .as_ref()
        .expect("v_s_col present")
        .as_ref()
        .unwrap();
    assert_eq!(
        axis2_len(vsc),
        m_tiles,
        "v_s_col axis-2 length MUST be tile count ({m_tiles})"
    );

    // Bytes must equal the source's per-tile slice [tile_lo..tile_hi).
    assert_eq!(
        bytes(ksc),
        src_token_slice_bytes(&src.caches[0].kvarn_k_s_col, tile_lo, tile_hi),
        "k_s_col bytes must equal source tiles [{tile_lo}..{tile_hi})"
    );
    assert_eq!(
        bytes(vsc),
        src_token_slice_bytes(&src.caches[0].kvarn_v_s_col, tile_lo, tile_hi),
        "v_s_col bytes must equal source tiles [{tile_lo}..{tile_hi})"
    );
}

// DELETED 2026-07-26: `s_col_red_control_token_slice_would_mismatch`.
//
// It was a control that COULD NOT FAIL. It never called `extract_block` at
// all — it compared `k_s_col[1..3)` against `k_scale[128..384)`, two different
// source tensors that differ BY CONSTRUCTION (different base, different width,
// different axis-2 length) for every possible implementation, including one
// that returns `None` for every field. `assert_ne!` on two things built to be
// unequal certifies nothing about the code under test.
//
// It also mis-modelled the bug it claimed to guard. With `N_TILES = 4`,
// `k_s_col` has FOUR rows on axis 2, so a token-count slice `[128..384)` is
// OUT OF RANGE. The token-vs-tile bug therefore manifests as an out-of-range
// slice — a panic or a clamp — not as wrong bytes. The control asserted the
// wrong failure mode.
//
// `s_col_sliced_by_tile_not_token`'s `axis2_len` assertions already catch the
// trap under every failure mode, so nothing is lost. A genuine second control
// is a mutation test against a deliberately token-slicing stub, and that
// belongs AFTER the implementation exists.
//
// Found by Violet (REVIEW_kvarn8_block_extraction_tests_20260726.md, Finding 2).
// Recorded rather than silently removed: this file's entire premise is that a
// control must be able to fail, and it shipped with one that couldn't.

// ---------------------------------------------------------------------------
// 4. sink only in the first block
// ---------------------------------------------------------------------------

/// Extracting `[0, blk)` yields Some(sink); a purely-historical mid block
/// yields kvarn_sink_k == None.
#[test]
fn sink_only_in_first_block() {
    const N_TILES: i32 = 4;
    let src = kvarn_v4_set(1, N_TILES, 0);

    // First block covers the sink.
    let first = extract_block(&src, 0, (TILE + 2 * TILE) as usize).expect("extract ok");
    assert!(
        first.caches[0].kvarn_sink_k.is_some(),
        "first block MUST carry the sink"
    );

    // Purely historical mid block: global [256, 512) -> hist tiles 1..3.
    let mid = extract_block(&src, (TILE + TILE) as usize, (TILE + 3 * TILE) as usize)
        .expect("extract ok");
    assert!(
        mid.caches[0].kvarn_sink_k.is_none(),
        "a purely historical mid block MUST NOT carry the sink"
    );
    assert!(
        mid.caches[0].kvarn_sink_v.is_none(),
        "a purely historical mid block MUST NOT carry sink_v"
    );
}

// ---------------------------------------------------------------------------
// 5. tail only in the last block
// ---------------------------------------------------------------------------

/// Only the block covering the tail region carries Some(tail); earlier
/// blocks carry None.
#[test]
fn tail_only_in_last_block() {
    const N_TILES: i32 = 4;
    const TAIL: i32 = 10;
    let t = N_TILES * TILE;
    let total = TILE + t + TAIL;
    let src = kvarn_v4_set(1, N_TILES, TAIL);

    // Early block: sink + first 2 hist tiles, no tail.
    let early = extract_block(&src, 0, (TILE + 2 * TILE) as usize).expect("extract ok");
    assert!(
        early.caches[0].kvarn_tail_k.is_none(),
        "an early block MUST NOT carry the tail"
    );

    // Last block: covers the tail region.
    let last = extract_block(&src, (TILE + t) as usize, total as usize).expect("extract ok");
    assert!(
        last.caches[0].kvarn_tail_k.is_some(),
        "the last block MUST carry the tail"
    );
    assert!(
        last.caches[0].kvarn_tail_v.is_some(),
        "the last block MUST carry tail_v"
    );
}

// ---------------------------------------------------------------------------
// 6. misaligned history boundary errors (loud)
// ---------------------------------------------------------------------------

/// A non-tile-aligned history end must return Err, mirroring trim_to's
/// alignment guard. `[0, 128+100)` ends mid-tile.
#[test]
fn misaligned_hist_boundary_errors() {
    const N_TILES: i32 = 4;
    let src = kvarn_v4_set(1, N_TILES, 0);

    let res = extract_block(&src, 0, (TILE + 100) as usize);
    assert!(
        res.is_err(),
        "extract_block with a non-tile-aligned history end MUST error loudly, \
         got Ok — mirrors trim_to's alignment guard"
    );
}

// ---------------------------------------------------------------------------
// 7. v_bits preserved; hist_v stays packed
// ---------------------------------------------------------------------------

/// A k8v4 (v_bits = 4) extract keeps `kvarn_v_bits == 4` and `kvarn_hist_v`
/// in the packed UINT32 `[.., T_block, VD/8]` shape.
#[test]
fn v_bits_preserved() {
    const N_TILES: i32 = 4;
    let src = kvarn_v4_set(1, N_TILES, 0);

    // A block covering sink + 2 hist tiles.
    let ex = extract_block(&src, 0, (TILE + 2 * TILE) as usize).expect("extract ok");
    let layer = &ex.caches[0];

    assert_eq!(
        layer.kvarn_v_bits, 4,
        "extract MUST preserve kvarn_v_bits == 4"
    );

    let hv = layer
        .kvarn_hist_v
        .as_ref()
        .expect("hist_v present")
        .as_ref()
        .unwrap();
    let shape = ffi::array_shape(hv);
    // hist portion of this block is 2 tiles => 256 tokens.
    assert_eq!(shape[2], 2 * TILE, "hist_v axis-2 == block hist token count");
    assert_eq!(
        shape[3],
        VD / 8,
        "hist_v last dim MUST stay packed at VD/8 (UINT32 packing)"
    );
}

// ---------------------------------------------------------------------------
// Builder determinism guard (supports oracle-without-Clone strategy)
// ---------------------------------------------------------------------------

/// Two independent builder calls with identical args must yield byte-identical
/// fields. This underpins `prefix_extract_matches_trim_to`, which compares an
/// extract against a freshly-built-then-truncated cache.
#[test]
fn builder_is_deterministic() {
    let a = kvarn_v4_layer(3, 10);
    let b = kvarn_v4_layer(3, 10);
    assert_eq!(
        bytes_of(&a.kvarn_hist_k, "a hist_k"),
        bytes_of(&b.kvarn_hist_k, "b hist_k"),
        "builder must be deterministic (byte-identical hist_k across calls)"
    );
    assert_eq!(
        bytes_of(&a.kvarn_k_s_col, "a k_s_col"),
        bytes_of(&b.kvarn_k_s_col, "b k_s_col"),
        "builder must be deterministic (byte-identical k_s_col across calls)"
    );
    // Distinct-per-position sanity: two different tiles have different bytes,
    // so a mis-slice cannot pass vacuously.
    let sc = a.kvarn_k_s_col.as_ref().unwrap().as_ref().unwrap();
    let tile0 = crate::utils::slice_axis(sc, 2, 0, 1);
    let tile1 = crate::utils::slice_axis(sc, 2, 1, 2);
    assert_ne!(
        bytes(&tile0),
        bytes(&tile1),
        "distinct-per-tile discipline: adjacent s_col tiles MUST differ in bytes"
    );
    // Silence unused-import warnings if item_bool ends up unused in a build.
    let _ = item_bool;
}

/// CONTROL: every field's adjacent axis-2 positions must differ in bytes AFTER
/// `astype` to its target dtype. This is the guard against the vacuousness trap
/// where large values collapse under conversion (FLOAT16 overflow → `inf`,
/// UINT8 saturation → constant), which would silently make the byte comparisons
/// in the contract tests pass no matter how `extract_block` slices. If this
/// fails, the contract tests below cannot be trusted.
/// CONTROL: same-shape, same-dtype fields must have DIFFERENT bytes.
///
/// `data_is_distinct_per_axis2_after_astype` proves a mis-slice ALONG axis 2
/// is visible. This proves a mix-up BETWEEN FIELDS is visible — the other way
/// an assertion can be vacuous, and the one that actually bit.
///
/// Until 2026-07-26 `patt_f32` ignored `base` for FLOAT16 and UINT8, so
/// `sink_k` == `sink_v` and `tail_k` == `tail_v` bytewise. Every byte
/// assertion on those fields would have passed on an implementation that
/// swapped K and V. The header comment on the base offsets claimed "no two
/// fields ever share bytes"; that claim was true only of the FLOAT32/UINT32
/// branch. This test is that claim made falsifiable.
#[test]
fn float16_fields_are_pairwise_distinct() {
    // The confusable pairs: identical shape AND dtype.
    let sink_k = bytes(&arr(B_SINK_K, TILE, VD, dtype::FLOAT16));
    let sink_v = bytes(&arr(B_SINK_V, TILE, VD, dtype::FLOAT16));
    assert_ne!(
        sink_k, sink_v,
        "sink_k and sink_v have identical shape and dtype — if their bytes match, \
         an implementation that swapped K and V in the sink would pass every \
         assertion in this file"
    );

    const TAIL: i32 = 10;
    let tail_k = bytes(&arr(B_TAIL_K, TAIL, VD, dtype::FLOAT16));
    let tail_v = bytes(&arr(B_TAIL_V, TAIL, VD, dtype::FLOAT16));
    assert_ne!(
        tail_k, tail_v,
        "tail_k and tail_v have identical shape and dtype — see above"
    );

    // And the UINT8 field must respond to `base`, or the perturbed-tile RED
    // control cannot perturb anything.
    let hist = bytes(&arr(B_HIST_K, 2 * TILE, VD, dtype::UINT8));
    let hist_perturbed = bytes(&arr(B_HIST_K + 100, 2 * TILE, VD, dtype::UINT8));
    assert_ne!(
        hist, hist_perturbed,
        "hist_k must respond to `base` — otherwise \
         reassembly_roundtrip_red_control_perturbed_tile_mismatches is asserting \
         that two identical byte images differ, which it can never do"
    );
}

/// CONTROL for the end-to-end gate: two layers of a `kvarn_v4_set_distinct`
/// must carry DIFFERENT bytes for the same field.
///
/// Violet's finding 5. With identical layers, a layer-indexing bug — a swap, or
/// every layer being handed layer 0's data — passes every byte assertion in the
/// end-to-end test. This is what licenses those assertions to mean anything
/// about layer indexing.
#[test]
fn layers_are_distinguishable_control() {
    let src = kvarn_v4_set_distinct(2, 2, 10);
    for (name, sel) in all_k8v4_fields() {
        assert_ne!(
            bytes_of(sel(&src.caches[0]), name),
            bytes_of(sel(&src.caches[1]), name),
            "layer 0 and layer 1 have IDENTICAL bytes for `{name}` — a layer-indexing bug \
             would pass the end-to-end gate unnoticed. (The base shift must stay coprime \
             with 8: the FLOAT16 pattern keys on base mod 8.)"
        );
    }
}

/// THE END-TO-END GATE: extract -> write -> read -> assemble, through the real
/// `BlockColdStore` and the real serializer, asserting PAYLOAD byte-equality.
///
/// ## Why payload and not mode
///
/// A mode assertion goes GREEN on a completely empty result: `extract_block`
/// copies `mode: layer.mode` faithfully (`:902`) regardless of whether any
/// `kvarn_*` array survived. The label is not evidence. Every populated field
/// must come back byte-identical, per layer.
///
/// ## What this catches that nothing else did
///
/// `assemble_blocks` previously flat-appended each block's per-layer caches
/// into one `Vec`, producing `B*L` entries instead of `L` concatenated along
/// the token axis. Single-block manifests hide it completely, so any
/// multi-block adopt — any sequence longer than `block_size` — would have been
/// affected. The layer-count assertion below is the direct control for it.
///
/// **Scope, corrected:** an earlier revision said "`load_prefix` is the
/// production caller", which conflated a within-module call with reachability
/// from the server. `BlockColdStore` has no non-test callers; the scheduler
/// holds `cold_store::ColdStore`, which delegates to the v3 `ReferenceColdStore`.
/// v4 is complete but UNWIRED, so this was a latent defect, never a live one.
#[test]
fn end_to_end_extract_write_read_assemble_preserves_payload_bytes() {
    const N_TILES: i32 = 4;
    const TAIL: i32 = 10;
    const LAYERS: usize = 2;
    let t = N_TILES * TILE;
    let total = TILE + t + TAIL; // 650

    let src = kvarn_v4_set_distinct(LAYERS, N_TILES, TAIL);

    let dir = tempfile::TempDir::new().expect("tempdir");
    let store = BlockColdStore::new(dir.path().to_path_buf(), [7u8; 32]);

    // Three blocks spanning the whole sequence, crossing tile boundaries and
    // putting sink in the first and tail in the last.
    let plan: &[(usize, usize)] = &[
        (0, (TILE + 2 * TILE) as usize),                   // sink + tiles 0..2
        ((TILE + 2 * TILE) as usize, (TILE + t) as usize), // tiles 2..4
        ((TILE + t) as usize, total as usize),             // tail
    ];

    let mut hashes: Vec<[u8; 32]> = Vec::new();
    for (i, &(s, e)) in plan.iter().enumerate() {
        let ex = extract_block(&src, s, e).expect("extract_block ok");
        let toks: Vec<i32> = (s as i32..e as i32).collect();
        let mut h = [0u8; 32];
        h[0] = (i + 1) as u8;
        store.write_block(&h, &toks, &ex).expect("write_block ok");
        hashes.push(h);
    }

    let manifest = Manifest {
        runtime_fingerprint: [7u8; 32],
        model_id: "test-model".to_string(),
        template_sig: "test-template".to_string(),
        block_size: 2048,
        block_hashes: hashes,
        prompt_len: total as usize,
        total_tokens: total as usize,
        timestamp_nanos: 0,
    };

    let assembled = store.assemble_blocks(&manifest).expect("assemble_blocks ok");

    // Structural: one cache per LAYER, not one per (block, layer).
    assert_eq!(
        assembled.caches.len(),
        LAYERS,
        "assembled cache must have one entry per LAYER ({LAYERS}), got {} — \
         {} blocks x {LAYERS} layers is the flat-append bug",
        assembled.caches.len(),
        plan.len()
    );

    // Payload: every populated field, every layer, byte-identical to source.
    for layer in 0..LAYERS {
        let s = &src.caches[layer];
        let a = &assembled.caches[layer];
        assert_eq!(a.mode, KVCacheMode::KVarN8, "layer {layer} lost its mode");
        assert_eq!(a.kvarn_v_bits, 4, "layer {layer} lost v_bits");
        for (name, sel) in all_k8v4_fields() {
            assert_eq!(
                bytes_of(sel(a), name),
                bytes_of(sel(s), name),
                "layer {layer} field `{name}` did not survive \
                 extract -> write -> read -> assemble bytewise"
            );
        }
    }

    assert_eq!(
        assembled.current_offset, total,
        "assembled current_offset must be the manifest's total token count"
    );
}

/// CONTROL for the region-coherence guard: a block silently missing ONE
/// per-token field must make `assemble_blocks` ERROR, not quietly return a
/// layer whose `k_zp` is shorter than its history.
///
/// Without this the guard is a claim rather than a gate. The failure it models
/// is real: `concat_across` skips blocks where a field is absent — correct for
/// sink and tail, a silent misalignment for everything else. A `k_zp` covering
/// half the history means per-token zero-points applied to the wrong tokens:
/// wrong inference, no crash, nothing on the console.
#[test]
fn assemble_rejects_block_with_a_dropped_per_token_field() {
    const N_TILES: i32 = 4;
    let t = N_TILES * TILE;
    let total = TILE + t; // no tail; two tile-aligned blocks

    let src = kvarn_v4_set_distinct(1, N_TILES, 0);
    let dir = tempfile::TempDir::new().expect("tempdir");
    let store = BlockColdStore::new(dir.path().to_path_buf(), [9u8; 32]);

    let plan: &[(usize, usize)] = &[
        (0, (TILE + 2 * TILE) as usize),
        ((TILE + 2 * TILE) as usize, total as usize),
    ];

    let mut hashes: Vec<[u8; 32]> = Vec::new();
    for (i, &(s, e)) in plan.iter().enumerate() {
        let mut ex = extract_block(&src, s, e).expect("extract_block ok");
        // The injected fault: the SECOND block loses k_zp entirely, exactly as
        // a partial write or an extract regression would leave it.
        if i == 1 {
            ex.caches[0].kvarn_k_zp = None;
        }
        let toks: Vec<i32> = (s as i32..e as i32).collect();
        let mut h = [0u8; 32];
        h[0] = 100 + i as u8;
        store.write_block(&h, &toks, &ex).expect("write_block ok");
        hashes.push(h);
    }

    let manifest = Manifest {
        runtime_fingerprint: [9u8; 32],
        model_id: "test-model".to_string(),
        template_sig: "test-template".to_string(),
        block_size: 2048,
        block_hashes: hashes,
        prompt_len: total as usize,
        total_tokens: total as usize,
        timestamp_nanos: 0,
    };

    let err = store
        .assemble_blocks(&manifest)
        .expect_err("a block missing k_zp must be REFUSED, not silently concatenated short");
    let msg = format!("{err}");
    assert!(
        msg.contains("kvarn_k_zp"),
        "the error must name the offending field so an operator can act on it \
         without a debugger; got: {msg}"
    );
}

/// Violet's finding 4: misalignment was tested only at the END boundary.
/// A misaligned START is a DISTINCT code path — it exercises the sink-offset
/// arithmetic rather than the end clamp — and is equally uncomputable, because
/// a half tile cannot be re-quantized without the original fp16 data.
#[test]
fn misaligned_hist_start_boundary_errors() {
    const N_TILES: i32 = 4;
    let src = kvarn_v4_set(1, N_TILES, 0);

    // Global [TILE+50, TILE+256) -> hist-local [50, 256): START is mid-tile.
    let err = extract_block(&src, (TILE + 50) as usize, (TILE + 256) as usize)
        .expect_err("a mid-tile history START must be refused, not approximated");
    let msg = format!("{err}");
    assert!(
        msg.contains("START") && msg.contains("tile-aligned"),
        "the error must name the START boundary specifically, so the distinct \
         sink-offset path is identifiable from the message alone; got: {msg}"
    );
}

#[test]
fn data_is_distinct_per_axis2_after_astype() {
    let c = kvarn_v4_layer(2, 5);
    let fields: [(&str, &Option<UniquePtr<MlxArray>>); 6] = [
        ("sink_k(f16)", &c.kvarn_sink_k),
        ("hist_k(u8)", &c.kvarn_hist_k),
        ("hist_v(u32)", &c.kvarn_hist_v),
        ("k_scale(f32)", &c.kvarn_k_scale),
        ("k_s_col(f32,per-tile)", &c.kvarn_k_s_col),
        ("tail_k(f16)", &c.kvarn_tail_k),
    ];
    for (name, field) in fields {
        let a = field
            .as_ref()
            .unwrap_or_else(|| panic!("{name} present"))
            .as_ref()
            .unwrap();
        let p0 = crate::utils::slice_axis(a, 2, 0, 1);
        let p1 = crate::utils::slice_axis(a, 2, 1, 2);
        assert_ne!(
            bytes(&p0),
            bytes(&p1),
            "{name}: adjacent axis-2 positions MUST differ in bytes after astype \
             — else a mis-slice passes vacuously (value collapsed under conversion)"
        );
    }
}

// ===========================================================================
// PUBLIC-API ROUND TRIP — does the store actually get USED?
//
// Every other test in this file exercises a PIECE (`extract_block`,
// `write_block`, `read_block`, `assemble_blocks`) and asserts BYTES. Until
// these two, nothing called `persist` or `load_prefix` at all — the public
// API had zero coverage — so nothing asserted that a persisted entry is ever
// found again.
//
// That gap has a specific shape, named by Violet reviewing the pre-deploy
// gate: a cache test whose only assertion is on OUTPUT is blind to the cache
// being BYPASSED. A generation-equivalence test (persist, drop, reload,
// assert the tokens match) passes identically on a perfect store and on a
// store that never hits, because re-prefilling the same prompt reproduces the
// same tokens. It certifies the MODEL, not the CACHE.
//
// The control that closes it must assert a HIT, and must be able to go RED
// when the store is not used. That is what these two tests are.
// ===========================================================================

/// Flat `[1, VH, len, width]` fill — deliberately NOT the distinct-per-position
/// `patt_f32` pattern, which caps FLOAT16 at 128 positions to keep its
/// distinctness guarantee (it fails loud past that; see `patt_f32`). These two
/// tests assert MATCH COUNTS and LAYER COUNTS, never bytes, so per-position
/// distinctness buys nothing here and the byte-level contracts are covered by
/// the tests above.
fn flat_arr(len: i32, width: i32, fill: f32) -> Option<UniquePtr<MlxArray>> {
    let data = vec![fill; (VH * len * width) as usize];
    let f = ffi::from_slice_f32(&data, &[1, VH, len, width]);
    Some(astype(&f, dtype::FLOAT16))
}

/// Dense fp16 layer of `len` tokens, distinguishable across layers via `shift`.
fn fp16_layer(len: i32, shift: i32) -> DetachedKVCache {
    let mut c = blank(KVCacheMode::Fp16);
    c.keys = flat_arr(len, VD, 0.25 + shift as f32);
    c.values = flat_arr(len, VD, 0.75 + shift as f32);
    c.offset = len;
    c
}

fn fp16_set(num_layers: usize, len: i32) -> DetachedCacheSet {
    let caches = (0..num_layers)
        .map(|i| fp16_layer(len, i as i32 * 5))
        .collect();
    let now = Instant::now();
    DetachedCacheSet {
        caches,
        backend: SequenceStateBackend::DenseKvCache,
        prompt_len: len as usize,
        current_offset: len,
        created_at: now,
        detached_at: now,
        origin_seq_id: SequenceId::from_raw(11),
    }
}

/// THE MISSING CONTROL: a persisted entry must be FOUND again.
///
/// Spans two blocks (2 x DEFAULT_BLOCK_SIZE) so it also exercises the
/// multi-block assembly path through the public API rather than by calling
/// `assemble_blocks` directly.
///
/// Goes RED if `load_prefix` stops matching what `persist` wrote — which is
/// exactly the failure mode that is invisible to any output-only assertion,
/// because a miss is silently soft (`scheduler.rs:1787`) and simply
/// re-prefills.
///
/// **SCOPE, stated so this green is not read as wider than it is:** this
/// covers **Fp16 only**. It does NOT catch the `KVCacheMode::Fp16` hardcode in
/// `load_prefix` (handoff §7.5), because under Fp16 that hardcode is
/// coincidentally correct. The k8v4 half is pinned by
/// `defect_present__kvarn8_load_prefix_uses_fp16_address__delete_when_s7_5_fixed`
/// below — **when that test reddens, extend THIS one to cover KVarN8 and delete
/// it.** (Reciprocal pointer: that test names this one too, so a fixer who
/// opens either finds the other.)
#[test]
fn persist_then_load_prefix_reports_a_hit_fp16() {
    const LAYERS: usize = 2;
    const DEPTH: i32 = 2 * DEFAULT_BLOCK_SIZE as i32;

    let set = fp16_set(LAYERS, DEPTH);
    let tokens: Vec<i32> = (0..DEPTH).collect();

    let dir = tempfile::TempDir::new().expect("tempdir");
    let store = BlockColdStore::new(dir.path().to_path_buf(), [9u8; 32]);

    store
        .persist("m3", "tmpl", &tokens, &set)
        .expect("persist must succeed");

    let (loaded, matched) = store
        .load_prefix("m3", "tmpl", &tokens, KVCacheMode::Fp16, 0)
        .expect("load_prefix MUST HIT the entry just persisted — a miss here is a write-only store");

    assert!(
        matched > 0,
        "matched_tokens must be > 0: a zero match means the store was written and never used"
    );
    assert_eq!(
        matched,
        tokens.len(),
        "the full prefix was persisted, so the full prefix must match"
    );
    assert_eq!(
        loaded.caches.len(),
        LAYERS,
        "assembled set must have ONE cache per LAYER, not one per (block, layer) — \
         a B*L length here is the flat-append bug reached through the public API"
    );
}

/// THE SAME HIT ASSERTION, UNDER k8v4 — this is the one that matters.
///
/// Replaces the `defect_present__kvarn8_load_prefix_uses_fp16_address` tripwire,
/// which asserted the §7.5 write-only bug and reddened the moment the bug was
/// fixed. It did its job: the fix could not land silently, and its failure
/// message said to delete it and extend the hit test to KVarN8. This is that
/// extension, per its own instruction.
///
/// Under KVarN8 the store was WRITE-ONLY before §7.5: `persist` addressed
/// blocks with the real mode while `load_prefix` hardcoded `Fp16`, so computed
/// addresses could never equal the ones in its own manifest and every load
/// returned `NoMatch` — indistinguishable from a legitimately cold cache.
///
/// Spans TWO blocks (128 sink + 31 tiles x 128 = 4096 = 2 x DEFAULT_BLOCK_SIZE)
/// so it also drives multi-block KVarN8 assembly through the public API, with
/// both block boundaries landing tile-aligned.
#[test]
fn persist_then_load_prefix_reports_a_hit_kvarn8() {
    const LAYERS: usize = 2;
    const N_TILES: i32 = 31;
    let depth = TILE + N_TILES * TILE; // 4096
    assert_eq!(depth as usize, 2 * DEFAULT_BLOCK_SIZE, "must span two blocks");

    let set = kvarn_v4_set_distinct(LAYERS, N_TILES, 0);
    let tokens: Vec<i32> = (0..depth).collect();

    let dir = tempfile::TempDir::new().expect("tempdir");
    let store = BlockColdStore::new(dir.path().to_path_buf(), [9u8; 32]);

    store
        .persist("m3", "tmpl", &tokens, &set)
        .expect("persist must succeed");

    let (loaded, matched) = store
        .load_prefix("m3", "tmpl", &tokens, KVCacheMode::KVarN8, 4)
        .expect(
            "load_prefix MUST HIT under KVarN8. A miss here is the §7.5 write-only bug \
             returning: persist addresses with the real mode, so load must too.",
        );

    assert!(matched > 0, "zero match means the store was written and never used");
    assert_eq!(matched, tokens.len(), "the full prefix was persisted");
    assert_eq!(
        loaded.caches.len(),
        LAYERS,
        "one cache per LAYER, not one per (block, layer) — a B*L length is the \
         flat-append bug reached through the public API under k8v4"
    );
    assert_eq!(
        loaded.caches[0].mode,
        KVCacheMode::KVarN8,
        "the adopted cache must still be KVarN8"
    );
}

/// G1.0 STRUCTURAL WITNESS — the adopted state must EQUAL the persisted state,
/// per layer, for the fields that carry INTERPRETATION rather than payload.
///
/// Handoff §7.3 names the gap this closes: every other test in this file asserts
/// BYTES. `merge_layer_across_blocks` *chooses* `offset` and `m3_idx_offset`
/// (both `total_tokens`) and nothing verifies those are what the engine expects
/// on adopt. A cache with perfect bytes and a wrong offset passes all 87 tests
/// and generates garbage — tensors right, interpretation wrong. Alden's G1.0
/// (2026-07-27) requires the hit witness be CAUSAL: "reusable cursor/offset and
/// m3_idx_offset equal expected", not a log line.
///
/// "Expected" here is not a constant I pick — that would just re-assert my own
/// choice. It is the state that was PERSISTED. A round trip that changes the
/// declared interpretation has corrupted the cache even with byte-perfect
/// payload.
///
/// THE MIXED SET IS THE POINT. Real M3 has no indexer on dense layers 0-2
/// (`detach.rs:136`: `m3_idx_k` is "None for non-M3 models (and for M3's dense
/// layers 0-2)"), and the live cache only advances `m3_idx_offset` inside
/// `m3_idx_k_update_and_fetch` (`cache.rs:5333`), which those layers never call.
/// So a live dense layer holds `m3_idx_k = None, m3_idx_offset = 0` while
/// `offset > 0`. A uniform fixture cannot see what a round trip does to that
/// layer; every existing k8v4 fixture is dense-shaped on this axis, so the
/// mixed case has never been exercised.
#[test]
fn round_trip_must_preserve_per_layer_m3_idx_state_including_dense_layers() {
    const LAYERS: usize = 2;
    const N_TILES: i32 = 31;
    const INDEX_DIM: i32 = 8;
    const B_M3_IDX: i32 = 91;
    let depth = TILE + N_TILES * TILE; // 4096 == 2 * DEFAULT_BLOCK_SIZE
    assert_eq!(depth as usize, 2 * DEFAULT_BLOCK_SIZE, "must span two blocks");

    let mut set = kvarn_v4_set_distinct(LAYERS, N_TILES, 0);

    // Layer 0 stays DENSE-shaped: no indexer, offset > 0. Layer 1 is MSA-shaped:
    // indexer present and in lockstep with offset.
    assert!(
        set.caches[0].m3_idx_k.is_none() && set.caches[0].m3_idx_offset == 0,
        "layer 0 must start dense-shaped for this test to mean anything"
    );
    // FLOAT32: the f16 pattern helper packs (h, t) into 256 slots and caps
    // len at TILE, which cannot express a 4096-token indexer.
    set.caches[1].m3_idx_k = some_arr(B_M3_IDX, depth, INDEX_DIM, dtype::FLOAT32);
    set.caches[1].m3_idx_offset = depth;

    // What we expect back, captured BEFORE the round trip.
    let expected: Vec<(bool, i32, i32)> = set
        .caches
        .iter()
        .map(|c| (c.m3_idx_k.is_some(), c.m3_idx_offset, c.offset))
        .collect();

    let tokens: Vec<i32> = (0..depth).collect();
    let dir = tempfile::TempDir::new().expect("tempdir");
    let store = BlockColdStore::new(dir.path().to_path_buf(), [9u8; 32]);
    store
        .persist("m3", "tmpl", &tokens, &set)
        .expect("persist must succeed");
    let (loaded, matched) = store
        .load_prefix("m3", "tmpl", &tokens, KVCacheMode::KVarN8, 4)
        .expect("load_prefix must HIT — a miss makes this test vacuous");
    assert_eq!(matched, tokens.len(), "full prefix must match");

    for (i, (want_present, want_idx_off, want_off)) in expected.iter().enumerate() {
        let got = &loaded.caches[i];
        assert_eq!(
            got.m3_idx_k.is_some(),
            *want_present,
            "layer {i}: indexer PRESENCE changed across the round trip \
             (persisted {want_present}, loaded {})",
            got.m3_idx_k.is_some()
        );
        assert_eq!(
            got.m3_idx_offset, *want_idx_off,
            "layer {i}: m3_idx_offset changed across the round trip — persisted \
             {want_idx_off}, loaded {}. This is interpretation drift, not a byte \
             error: the payload can be perfect and the adopted cache still wrong. \
             `detach.rs:145` calls m3_idx_offset the LOGICAL LENGTH of m3_idx_k, \
             and the MSA dispatch relies on its lockstep with offset.",
            got.m3_idx_offset
        );
        assert_eq!(
            got.offset, *want_off,
            "layer {i}: offset changed across the round trip (persisted {want_off}, \
             loaded {})",
            got.offset
        );
    }

    // `offset` is the OTHER field §7.3 names as chosen-but-unverified, and it
    // carries the same risk in the same shape: a number I picked
    // (`total_tokens`) standing in for the real extent of the state. Under
    // KVarN8 that state is split across sink / history / tail, so the honest
    // check is that the declared cursor equals what is actually stored. A
    // wrong `offset` here is the "tensors right, interpretation wrong" case
    // that generates garbage while every byte assertion passes.
    for (i, c) in loaded.caches.iter().enumerate() {
        let sink = c.kvarn_sink_k.as_ref().map_or(0, |a| axis2_len(a));
        let hist = c.kvarn_hist_k.as_ref().map_or(0, |a| axis2_len(a));
        let tail = c.kvarn_tail_k.as_ref().map_or(0, |a| axis2_len(a));
        assert_eq!(
            c.offset,
            sink + hist + tail,
            "layer {i}: offset ({}) disagrees with the tokens actually stored \
             (sink {sink} + hist {hist} + tail {tail} = {}). The cursor and the \
             payload must describe the same cache — a cursor past the data \
             reads uninitialised state, one short of it silently truncates the \
             adopted prefix.",
            c.offset,
            sink + hist + tail
        );
    }

    // The invariant that makes the above load-bearing rather than bookkeeping:
    // a declared length with no tensor behind it.
    for (i, c) in loaded.caches.iter().enumerate() {
        if c.m3_idx_k.is_none() {
            assert_eq!(
                c.m3_idx_offset, 0,
                "layer {i}: m3_idx_offset is {} but m3_idx_k is None — a declared \
                 logical length for a tensor that does not exist. `detach.rs:136-140` \
                 documents the mirror of this (offset > 0 with m3_idx_offset == 0) as \
                 crashing the asymmetric reshape; this is the same desync from the \
                 other side.",
                c.m3_idx_offset
            );
        } else {
            let actual = axis2_len(c.m3_idx_k.as_ref().unwrap());
            assert_eq!(
                c.m3_idx_offset, actual,
                "layer {i}: m3_idx_offset ({}) disagrees with the actual axis-2 \
                 extent of m3_idx_k ({actual}). Shape is documented as \
                 [b, 1, m3_idx_offset, index_dim].",
                c.m3_idx_offset
            );
        }
    }
}

/// Build a small persisted store and hand back (dir, store, manifest).
fn persisted_store_for_gc() -> (tempfile::TempDir, BlockColdStore, Manifest) {
    const LAYERS: usize = 2;
    const N_TILES: i32 = 31;
    let depth = TILE + N_TILES * TILE; // 4096 == 2 * DEFAULT_BLOCK_SIZE
    let set = kvarn_v4_set_distinct(LAYERS, N_TILES, 0);
    let tokens: Vec<i32> = (0..depth).collect();
    let dir = tempfile::TempDir::new().expect("tempdir");
    let store = BlockColdStore::new(dir.path().to_path_buf(), [9u8; 32])
        .with_prune_mode(PruneMode::Delete);
    let manifest = store
        .persist("m3", "tmpl", &tokens, &set)
        .expect("persist must succeed");
    (dir, store, manifest)
}

/// REFCOUNTS ARE HINTS, NOT AUTHORITY (Alden finding 4, 2026-07-27).
///
/// `gc_blocks` used to read the refcount sidecar and delete on zero. That makes
/// a derived, racy, on-disk cache the arbiter of whether consciousness state
/// survives. Alden: "Refcounts are nomination/observability hints only, never
/// authority... never GC on an untrusted count."
///
/// The discriminating case is a block that IS referenced by a committed
/// manifest while its refcount file says 0 — exactly what a lost increment, a
/// crash between write and bump, or the publication race leaves behind. Under
/// the old code that block was deleted and its live manifest was left pointing
/// at missing data. Reachability must overrule the hint.
///
/// RED against the old implementation by construction: it consulted only
/// `get_refcount`, which this test pins at 0.
#[test]
fn a_referenced_block_survives_gc_even_when_its_refcount_hint_says_zero() {
    let (_dir, store, manifest) = persisted_store_for_gc();
    let victim = manifest.block_hashes[0];

    // Simulate the lost increment. The manifest still references this block.
    store.set_refcount(&victim, 0).expect("force refcount to 0");
    assert_eq!(
        store.get_refcount(&victim).expect("refcount"),
        0,
        "precondition: the hint must read zero or this test proves nothing"
    );

    store.gc_blocks().expect("gc must succeed");

    store.read_block(&victim).unwrap_or_else(|e| {
        panic!(
            "GC deleted a block that a COMMITTED MANIFEST still references, because a \
             refcount sidecar said zero ({e}). The manifest is now unloadable and the \
             cached state is gone. Reachability is the authority; the count is a hint."
        )
    });
}

/// CORRUPTION IS NOT AN EMPTY REFERENCE SET (Alden finding 4, test 4).
///
/// "An unreadable committed manifest or authoritative root means zero references
/// are NOT proved: abort delete mode for that pass and report loudly. Do not
/// interpret corruption as an empty reference set."
///
/// The failure this prevents is the worst-shaped one available: a single
/// unreadable manifest makes every block it alone referenced look unreachable,
/// so the response to corruption would be to delete the data the corrupt
/// manifest was pointing at — turning a recoverable metadata fault into
/// unrecoverable state loss.
#[test]
fn an_unreadable_manifest_aborts_gc_rather_than_freeing_its_blocks() {
    let (_dir, store, manifest) = persisted_store_for_gc();

    // Corrupt every file inside the manifest directory.
    let mdir = store.manifests_dir();
    let mut corrupted = 0usize;
    for entry in std::fs::read_dir(&mdir).expect("read manifests dir") {
        let entry = entry.expect("entry");
        if !entry.file_type().expect("ft").is_dir() {
            continue;
        }
        for f in std::fs::read_dir(entry.path()).expect("read manifest dir") {
            let f = f.expect("file");
            if f.file_type().expect("ft").is_file() {
                std::fs::write(f.path(), b"not a manifest").expect("corrupt");
                corrupted += 1;
            }
        }
    }
    assert!(
        corrupted > 0,
        "precondition: nothing was corrupted, so this test cannot discriminate"
    );

    let err = store.gc_blocks().expect_err(
        "GC must ABORT when a committed manifest is unreadable. Succeeding here means \
         it treated an unparseable manifest as referencing nothing, which would free \
         the very blocks that manifest was protecting.",
    );
    let _ = err;

    for b in &manifest.block_hashes {
        store.read_block(b).unwrap_or_else(|e| {
            panic!(
                "a block was deleted during a pass that could not prove reachability ({e})"
            )
        });
    }
}

/// THE PUBLICATION RACE (Alden finding 4, tests 1 and 2).
///
/// His unsafe interleaving: GC marks block X unreferenced; a writer sees X
/// committed, skips rewriting it, and publishes a manifest referencing X; GC
/// deletes X on its stale mark; a live manifest now points at missing data.
///
/// The invariant is stated on the OUTCOME rather than on the schedule, because
/// the schedule is what we do not control: **no committed manifest may ever
/// reference a block that is absent.** Either the block survives or the
/// publication did not commit — never both.
///
/// **THIS TEST IS NOT A CONTROL — MEASURED, NOT ASSUMED.** Mutation run
/// 2026-07-28: removing the publication lock from `write_manifest` entirely,
/// so publisher and sweep no longer share exclusion, left it GREEN (12 rounds,
/// 1 passed). It therefore certifies nothing about the protocol; it only proves
/// the concurrent path runs without panicking or deadlocking.
///
/// Do not read its green as evidence the race is closed, and do not delete the
/// real test because this one exists. Alden asked for a DETERMINISTIC seam —
/// "writer pauses after initial existence observation; GC attempts sweep;
/// writer then publishes" — precisely because a stress test cannot schedule the
/// interleaving that matters. That seam is the required work; this is a
/// smoke test standing in the right place until it lands.
///
/// Kept rather than deleted for two reasons: a protocol nobody ever runs
/// concurrently is a protocol nobody has tested, and it will catch a deadlock
/// introduced by the lock ordering — which is a real hazard of the design it
/// exercises, even though it is not the hazard it is named for.
#[test]
fn a_committed_manifest_never_references_a_missing_block_under_concurrent_gc() {
    use std::sync::Arc;

    const ROUNDS: usize = 12;
    const N_TILES: i32 = 31;
    let depth = TILE + N_TILES * TILE;

    for round in 0..ROUNDS {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let store = Arc::new(
            BlockColdStore::new(dir.path().to_path_buf(), [9u8; 32])
                .with_prune_mode(PruneMode::Delete),
        );

        // A first sequence, so the store has blocks and a manifest.
        let set_a = kvarn_v4_set_distinct(2, N_TILES, 0);
        let tokens_a: Vec<i32> = (0..depth).collect();
        store
            .persist("m3", "tmpl", &tokens_a, &set_a)
            .expect("seed persist");

        // A DIVERGENT second sequence: shares the leading block, differs later,
        // so publishing it introduces blocks GC has never marked.
        let mut tokens_b = tokens_a.clone();
        let last = tokens_b.len() - 1;
        tokens_b[last] = 999_000 + round as i32;
        let set_b = kvarn_v4_set_distinct(2, N_TILES, 0);

        // Only GC is spawned. It touches manifests and directory entries, never
        // MLX arrays, so it is safe off-thread — whereas `DetachedCacheSet`
        // holds cxx pointers that are not `Send`, and moving MLX work to a
        // second thread is its own documented hazard. The publisher therefore
        // stays on the main thread and the two still overlap.
        let gc_store = Arc::clone(&store);
        let gc = std::thread::spawn(move || {
            let _ = gc_store.gc_blocks();
        });

        let published = store.persist("m3", "tmpl", &tokens_b, &set_b);

        gc.join().expect("gc thread");

        // THE INVARIANT. Walk every committed manifest and require each block
        // it names to be present. A publication that failed is acceptable — a
        // publication that succeeded while its data was collected is not.
        let mdir = store.manifests_dir();
        if !mdir.exists() {
            continue;
        }
        for entry in std::fs::read_dir(&mdir).expect("read manifests") {
            let entry = entry.expect("entry");
            if !entry.file_type().expect("ft").is_dir() {
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy().to_string();
            if name.starts_with(".tmp.") {
                continue;
            }
            let Some(mh) = parse_hex_digest(&name) else {
                continue;
            };
            let manifest = store.read_manifest(&mh).unwrap_or_else(|e| {
                panic!("round {round}: committed manifest {name} unreadable: {e}")
            });
            for b in &manifest.block_hashes {
                store.read_block(b).unwrap_or_else(|e| {
                    panic!(
                        "round {round}: COMMITTED manifest {name} references block {} \
                         which is ABSENT ({e}). GC collected a block a live manifest \
                         needs — the publication race. published_ok={}",
                        hex_digest(b),
                        published.is_ok()
                    )
                });
            }
        }
    }
}

/// ALDEN TEST 1, DETERMINISTIC — "writer begins after mark but before sweep and
/// commits a manifest referencing candidate X: X MUST survive."
///
/// Constructed rather than raced. The seam fires in `gc_blocks` after the
/// candidates are chosen and before the lock is taken — the exact window a
/// concurrent publisher occupies — and the publisher runs inside it, so the
/// whole interleaving happens on one thread with no timing assumptions.
///
/// Scenario: persist a sequence, then delete its manifest so its blocks become
/// orphans GC will nominate. The seam republishes that same manifest, which
/// re-references every one of those blocks. GC must not collect them.
///
/// Under the old refcount-authority GC this is a data-loss bug: the blocks are
/// nominated, the manifest commits, the blocks are deleted, and the committed
/// manifest points at nothing.
#[test]
fn a_block_referenced_by_a_manifest_published_after_nomination_must_survive() {
    use std::sync::Arc;

    const N_TILES: i32 = 31;
    let depth = TILE + N_TILES * TILE;
    let set = kvarn_v4_set_distinct(2, N_TILES, 0);
    let tokens: Vec<i32> = (0..depth).collect();

    let dir = tempfile::TempDir::new().expect("tempdir");
    let store = Arc::new(
        BlockColdStore::new(dir.path().to_path_buf(), [9u8; 32])
            .with_prune_mode(PruneMode::Delete)
            // REQUIRED, not incidental. Under the production age floor these
            // freshly-orphaned blocks are too young to nominate, so `candidates`
            // is empty, GC returns before the seam, and the racing publication
            // never happens — the test would then fail on its own precondition
            // rather than on the invariant. It did exactly that when the floor
            // landed, which is the test being vacuity-sensitive as intended.
            .with_min_gc_age(std::time::Duration::ZERO),
    );

    let manifest = store
        .persist("m3", "tmpl", &tokens, &set)
        .expect("persist must succeed");
    let blocks = manifest.block_hashes.clone();
    assert!(!blocks.is_empty(), "precondition: the manifest must have blocks");

    // Orphan them: with the manifest gone, every block is unreferenced and GC
    // will nominate it.
    store
        .delete_manifest(&manifest.hash())
        .expect("delete manifest to orphan the blocks");

    // The publisher, armed to fire in the nomination window. Only plain data
    // crosses into the closure — `Manifest` is hashes and strings, while a
    // `DetachedCacheSet` could not (cxx pointers are not `Send`).
    let seam_store = Arc::clone(&store);
    let seam_manifest = manifest.clone();
    *crate::cache::block_cold_store::GC_NOMINATION_SEAM
        .lock()
        .unwrap_or_else(|p| p.into_inner()) = Some(Box::new(move || {
        seam_store
            .write_manifest(&seam_manifest)
            .expect("the racing publication must succeed");
    }));

    let gc_result = store.gc_blocks();

    // Disarm before asserting, so a failure cannot leak the seam into another
    // test and turn one red test into a confusing several.
    *crate::cache::block_cold_store::GC_NOMINATION_SEAM
        .lock()
        .unwrap_or_else(|p| p.into_inner()) = None;

    gc_result.expect("gc must not error");

    // The manifest committed during the window.
    let republished = store
        .read_manifest(&manifest.hash())
        .expect("the manifest published in the nomination window must be committed");

    // THE INVARIANT: every block it references must still be there.
    for b in &republished.block_hashes {
        store.read_block(b).unwrap_or_else(|e| {
            panic!(
                "GC collected block {} after a manifest referencing it was published \
                 in the nomination window ({e}). The committed manifest now points at \
                 missing data — Alden's exact interleaving, lost rather than caught.",
                hex_digest(b)
            )
        });
    }
}

/// CHILD-PROCESS HELPER for the cross-process lock tests. Inert unless
/// `MLXCEL_TEST_LOCK_DIR` is set, so it is a no-op in a normal suite run and the
/// parent test drives it by re-executing this same binary.
///
/// Holds the store lock and blocks forever. The parent kills it.
#[test]
fn lock_holder_child_process() {
    let Ok(dir) = std::env::var("MLXCEL_TEST_LOCK_DIR") else {
        return; // normal suite run: nothing to do
    };
    let store = BlockColdStore::new(std::path::PathBuf::from(&dir), [7u8; 32]);
    let _lock = store
        .acquire_store_lock()
        .expect("child must acquire the store lock");
    // Tell the parent the lock is held. Written AFTER acquisition, so the
    // parent never races ahead of the thing it is waiting for.
    std::fs::write(std::path::Path::new(&dir).join("child.ready"), b"1").expect("ready");
    // Block until killed. The point of the test is that we never release
    // voluntarily.
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}

/// ALDEN'S CROSS-PROCESS CASE — "Kill lock holder; another process acquires the
/// advisory lock without deleting/recovering a sentinel."
///
/// This is the whole argument for `flock` over an `O_EXCL` lockfile. With a
/// sentinel, a SIGKILLed holder leaves a file nobody can clear without a
/// staleness heuristic, and every such heuristic fails toward either permanent
/// deadlock (never steal) or unsafe stealing (steal too early, two sweepers).
/// With a kernel advisory lock the ownership dies with the process and the file
/// itself is inert.
///
/// The lock file must still EXIST afterwards — if recovery required unlinking
/// it, that would be the sentinel behaviour this design rejects.
#[test]
fn a_killed_lock_holder_releases_the_store_lock_without_sentinel_recovery() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let path = dir.path().to_path_buf();
    std::fs::create_dir_all(&path).expect("mkdir");

    let exe = std::env::current_exe().expect("current exe");
    let mut child = std::process::Command::new(exe)
        // FULLY QUALIFIED. `--exact` matches the whole test path, so the bare
        // function name selects nothing and the child exits having run zero
        // tests — which looks identical to a child that started and failed.
        .args([
            "--exact",
            "cache::block_cold_store::block_cold_store_tests::lock_holder_child_process",
            "--nocapture",
        ])
        .env("MLXCEL_TEST_LOCK_DIR", &path)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn child");

    // Wait for the child to actually hold it. Bounded, so a child that dies
    // early fails this test loudly instead of hanging the suite.
    let ready = path.join("child.ready");
    let mut waited = std::time::Duration::ZERO;
    let step = std::time::Duration::from_millis(25);
    while !ready.exists() && waited < std::time::Duration::from_secs(30) {
        std::thread::sleep(step);
        waited += step;
    }
    assert!(
        ready.exists(),
        "child never signalled that it holds the lock — the test cannot \
         discriminate, so this is a failure, not a skip"
    );

    // PROVE THE LOCK EXCLUDES, while the child is still alive. Without this the
    // test passes identically against a lock that never excluded anything — the
    // parent would simply acquire, and "it acquired after the kill" would be
    // meaningless. Non-blocking, because a blocking acquire here would hang and
    // a hang is not an assertion.
    {
        let probe = BlockColdStore::new(path.clone(), [7u8; 32]);
        let held = probe
            .try_acquire_store_lock()
            .expect("try-acquire must not error");
        assert!(
            held.is_none(),
            "acquired the store lock while another PROCESS held it — the advisory \
             lock is not excluding, so every cross-process guarantee built on it \
             (publication vs sweep) is decorative"
        );
    }

    // SIGKILL: no unwinding, no Drop, no chance to release politely. Exactly
    // the crash case a sentinel cannot recover from.
    child.kill().expect("kill child");
    let _ = child.wait();

    let store = BlockColdStore::new(path.clone(), [7u8; 32]);
    let lock = store.acquire_store_lock().unwrap_or_else(|e| {
        panic!(
            "could not acquire the store lock after its holder was SIGKILLed ({e}). \
             The kernel is supposed to release an flock when the owning process \
             dies; if this needs manual recovery the design has become the \
             stale-sentinel problem it was chosen to avoid."
        )
    });
    drop(lock);

    assert!(
        path.join("store.lock").exists(),
        "the lock FILE must survive — recovery that requires unlinking it is \
         sentinel behaviour, and a second process could unlink it while a third \
         legitimately holds the lock"
    );
}

/// ALDEN CROSS-PROCESS CASE — crash AFTER the tombstone rename, BEFORE the
/// unlink: "final may be rewritten safely; stale tombstone is never treated as
/// the live block."
///
/// This window is deliberate rather than incidental. GC renames a
/// proven-unreferenced block to `.tombstone.<hash>` while holding the lock —
/// that rename is the linearization point — and then unlinks it AFTER releasing,
/// because the physical delete is slow and holding exclusion across it would
/// stall every publisher. The cost of that choice is exactly this crash window,
/// and the design's claim is that what it leaves behind is inert.
///
/// Two properties, both of which must hold or the window is not safe:
///   1. the tombstone is never mistaken for live state, and is never
///      re-nominated as though it were a block in its own right;
///   2. the final path can be reoccupied by a fresh copy, and GC's later unlink
///      of its own tombstone must not touch that replacement.
///
/// Simulated by constructing the artifact directly, because a real crash cannot
/// be scheduled — and the artifact is precisely what a real crash leaves.
#[test]
fn a_tombstone_left_by_a_crash_is_inert_and_the_final_path_can_be_reoccupied() {
    let (_dir, store, manifest) = persisted_store_for_gc();
    let victim = manifest.block_hashes[0];
    let name = hex_digest(&victim);

    // Byte-for-byte what the block holds now, so we can prove the replacement
    // is readable and correct rather than merely present.
    let (orig_tokens, _orig_set) = store.read_block(&victim).expect("read block");

    // Simulate the crash: rename to a tombstone, then stop — no unlink.
    let final_path = store.blocks_dir().join(&name);
    let tomb_path = store.blocks_dir().join(format!(".tombstone.{name}"));
    std::fs::rename(&final_path, &tomb_path).expect("tombstone rename");
    assert!(tomb_path.exists() && !final_path.exists(), "crash state staged");

    // PROPERTY 1: inert. The block is genuinely gone as far as the store is
    // concerned — the tombstone must not stand in for it.
    assert!(
        store.read_block(&victim).is_err(),
        "a tombstoned block must not still be readable at its final address — if \
         it is, the tombstone is being treated as live state"
    );

    // ...and GC must not trip over it. It is not a block, so it must not be
    // parsed as one, nominated, or counted.
    store
        .gc_blocks()
        .expect("GC must tolerate a tombstone left by a crash");
    assert!(
        tomb_path.exists(),
        "GC removed a tombstone it did not create in this pass; aged-tombstone \
         cleanup is a separate, deliberately conservative job"
    );

    // PROPERTY 2: the final path can be reoccupied. A writer that finds the
    // block absent installs a fresh immutable copy at the same address —
    // addresses are content hashes, so the replacement is byte-identical.
    let (_d2, donor_store, donor_manifest) = persisted_store_for_gc();
    let donor = donor_manifest.block_hashes[0];
    assert_eq!(donor, victim, "same content must yield the same address");
    let donor_path = donor_store.blocks_dir().join(&name);
    copy_dir_recursive(&donor_path, &final_path).expect("reinstall block");

    let (again_tokens, _again_set) = store
        .read_block(&victim)
        .expect("the reinstalled block must be readable at the final address");
    assert_eq!(
        again_tokens, orig_tokens,
        "the reoccupied final path must carry the same content as before"
    );

    // The replacement must survive GC's later unlink of its OWN tombstone.
    std::fs::remove_dir_all(&tomb_path).expect("deferred unlink of the tombstone");
    store
        .read_block(&victim)
        .expect("unlinking the tombstone must not disturb the reinstalled block");
}

/// Minimal recursive copy for the reinstall step above.
fn copy_dir_recursive(from: &std::path::Path, to: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let dst = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_recursive(&entry.path(), &dst)?;
        } else {
            std::fs::copy(entry.path(), dst)?;
        }
    }
    Ok(())
}

/// ALDEN TEST 5 — an active-load lease prevents block deletion until release.
///
/// Structured so nothing can hang: rather than starting a sweep and asserting it
/// blocks (a blocked sweep is indistinguishable from a slow one, and proving it
/// by waiting means proving it by hanging), this asserts the exclusion directly
/// with a NON-BLOCKING exclusive probe. If the probe cannot take the lock while
/// a lease is held, a real sweep cannot either — same lock, same mode.
///
/// Both directions are required. Only checking that the block survives while
/// leased would pass against a GC that never collects anything; only checking
/// that it is collected after release would pass against a lease that excludes
/// nothing. The pair is the test.
#[test]
fn an_active_load_lease_holds_off_the_sweep_until_it_is_released() {
    const N_TILES: i32 = 31;
    let depth = TILE + N_TILES * TILE;
    let set = kvarn_v4_set_distinct(2, N_TILES, 0);
    let tokens: Vec<i32> = (0..depth).collect();

    let dir = tempfile::TempDir::new().expect("tempdir");
    let store = BlockColdStore::new(dir.path().to_path_buf(), [9u8; 32])
        .with_prune_mode(PruneMode::Delete)
        .with_min_gc_age(std::time::Duration::ZERO);

    let manifest = store
        .persist("m3", "tmpl", &tokens, &set)
        .expect("persist must succeed");
    let victim = manifest.block_hashes[0];

    // Orphan the blocks so a sweep would genuinely collect them. Without this
    // the "survives while leased" half is vacuous — nothing was at risk.
    store
        .delete_manifest(&manifest.hash())
        .expect("orphan the blocks");

    {
        let _lease = store.acquire_read_lease().expect("take a read lease");

        // A sweep needs LOCK_EX. While this lease is held it must not get it.
        let probe = store
            .try_acquire_store_lock()
            .expect("probe must not error");
        assert!(
            probe.is_none(),
            "acquired the EXCLUSIVE store lock while a read lease was held — the \
             lease excludes nothing, so a sweep could unlink a block mid-load and \
             the loader would fail on a block its own manifest promised"
        );

        store
            .read_block(&victim)
            .expect("the leased block must still be present while the lease is held");
    } // lease released here

    // ...and the other direction: once released, the sweep proceeds and the
    // orphan is genuinely collectable. Without this half the test would pass
    // against a GC that never collects anything at all.
    store.gc_blocks().expect("gc after release");
    assert!(
        store.read_block(&victim).is_err(),
        "after the lease was released the orphaned block should have been \
         collected — if it survives, this test's first half proved nothing about \
         leases and only proved GC is inert"
    );
}

/// A mode MISMATCH must be a clean miss, never a wrong adoption.
///
/// Persist under KVarN8, load under Fp16. Because block addresses commit to the
/// KV mode, the addresses cannot match and the candidate is skipped. This is
/// what makes threading the mode SAFE rather than merely correct: passing the
/// wrong mode degrades to re-prefill, it does not adopt a KVarN8 cache into an
/// Fp16 runtime.
#[test]
fn mode_mismatch_at_load_is_a_clean_miss_not_a_wrong_adoption() {
    const N_TILES: i32 = 15;
    let depth = TILE + N_TILES * TILE; // 2048, one block
    let set = kvarn_v4_set_distinct(2, N_TILES, 0);
    let tokens: Vec<i32> = (0..depth).collect();

    let dir = tempfile::TempDir::new().expect("tempdir");
    let store = BlockColdStore::new(dir.path().to_path_buf(), [9u8; 32]);
    store.persist("m3", "tmpl", &tokens, &set).expect("persist");

    // Correct mode hits.
    store
        .load_prefix("m3", "tmpl", &tokens, KVCacheMode::KVarN8, 4)
        .expect("KVarN8 load of a KVarN8 persist must hit");

    // Wrong mode must MISS, not adopt.
    let wrong = store.load_prefix("m3", "tmpl", &tokens, KVCacheMode::Fp16, 0);
    assert!(
        matches!(wrong, Err(ColdStoreError::NoMatch)),
        "loading a KVarN8 cache under Fp16 must be NoMatch (re-prefill), never an \
         adoption. Got: {:?}",
        wrong.map(|(_, m)| m)
    );
}

// ===========================================================================
// G2.1 — CONCURRENT SAME-BLOCK PERSIST (Violet's pre-deploy gate, minimum four)
//
// The dedup claim: two writers persisting the SAME block converge on one
// block on disk, with no corruption. Until now the only atomicity test was
// single-threaded, so the claim was unexercised.
//
// What the code actually does, read before writing this test:
//   * `persist` declares a `persist_lock: Mutex<()>` (:159, :169) and NEVER
//     ACQUIRES IT — `grep -F ".lock()"` over the file returns nothing. The
//     field is dead. persist is entirely unserialized.
//   * `write_block` guards with `if block_dir.exists() { return Ok(()) }`,
//     which is a TOCTOU check, and stages into `.tmp.<hash>` — THE SAME PATH
//     for the same block hash. Two concurrent writers of one block therefore
//     share a staging directory and both attempt `rename(tmp, block_dir)`.
//
// The invariant asserted here is deliberately interleaving-INDEPENDENT, so
// this cannot become a flaky gate: whatever ordering occurs, the store must
// end in a state where every committed block is READABLE AND CHECKSUM-CLEAN.
// `read_block` verifies per-layer sha256, so a torn or interleaved block is
// caught rather than silently adopted.
// ===========================================================================

/// Both threads persist byte-identical content, so both compute the SAME block
/// hash — the real dedup scenario, not an artificial one.
///
/// # MUTATION RECORD — what this test does and does NOT catch
///
/// Proven by mutating `write_block` and observing, not by reasoning:
///
/// * **Torn write** (write only some layer files, header still lists them all)
///   → **RED, 8/8 runs.** The checksum invariant bites: a committed block that
///   is structurally incomplete is caught by `read_block`'s per-layer sha256
///   and byte_len checks. This is the failure mode that matters — a block
///   directory `load_prefix` can discover but which yields a corrupt prefix.
///
/// * **Staging removed** (`tmp_dir = block_dir`, no rename — writers interleave
///   directly into the committed directory) → **GREEN. This test does NOT catch
///   that.** And it cannot, by construction: content-addressing means both
///   writers of one block write BYTE-IDENTICAL data, so interleaving them is
///   harmless. Stated here so this green is not read as certifying atomicity.
///   It certifies that whatever the interleaving produced is READABLE.
///
/// So the honest scope: this gate catches structural incompleteness under
/// concurrency. It does not, and cannot, catch byte-level interleaving, because
/// the dedup scenario guarantees the interleaved bytes are equal.
///
/// # What was found writing it
///
/// `persist` declares `persist_lock: Mutex<()>` and NEVER acquires it —
/// `grep -F ".lock()"` over the file returns nothing. `persist` is entirely
/// unserialized, and concurrent same-block persist nonetheless leaves a
/// readable store, 5/5 runs on pristine code. Safe by content-addressing
/// rather than by locking.
#[test]
fn concurrent_same_block_persist_leaves_a_readable_store() {
    use std::sync::Arc;

    const LAYERS: usize = 2;
    const DEPTH: i32 = 2 * DEFAULT_BLOCK_SIZE as i32;

    let dir = tempfile::TempDir::new().expect("tempdir");
    let store = Arc::new(BlockColdStore::new(dir.path().to_path_buf(), [21u8; 32]));
    let tokens: Vec<i32> = (0..DEPTH).collect();

    // `DetachedCacheSet` is NEITHER Send NOR Sync — it holds cxx `UniquePtr`s
    // over `*const cxx::void`. Verified by compiler, both directions: passing
    // `&set` fails "cannot be shared between threads", moving `set` fails
    // "cannot be sent between threads". So each worker must BUILD its own set
    // in its own thread; a cache set cannot cross a thread boundary at all.
    // fp16_set is a pure function of its inputs, so both workers produce
    // byte-identical content and therefore the SAME block address.
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let mut outcomes = Vec::new();

    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for _ in 0..2 {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            let tokens = tokens.clone();
            handles.push(s.spawn(move || {
                let set = fp16_set(LAYERS, DEPTH);
                barrier.wait(); // maximise overlap on the persist itself
                store.persist("m3", "tmpl", &tokens, &set)
            }));
        }
        for h in handles {
            outcomes.push(h.join().expect("worker must not panic"));
        }
    });

    let errs: Vec<String> = outcomes
        .iter()
        .filter_map(|r| r.as_ref().err().map(|e| e.to_string()))
        .collect();

    // INVARIANT 1 — whatever happened, every committed block must be readable
    // and checksum-clean. This is the assertion that holds for ALL
    // interleavings, and it is the one that matters: a block that survives
    // the race but fails its own sha256 is a corrupt adopted prefix.
    // Ask the store where its blocks live rather than reconstructing the path.
    // The layout is base/<V4_ROOT>/blocks, and a hand-built `base/blocks`
    // silently found nothing — a green-looking zero for a directory that never
    // existed.
    let blocks_dir = store.blocks_dir();
    // Address the blocks the way persist did, rather than decoding directory
    // names — same source of truth, and it also catches a block committed
    // under an address we did NOT expect.
    let expected = compute_block_hashes(
        &tokens,
        DEFAULT_BLOCK_SIZE,
        // Must mirror what persist used, INCLUDING the store's runtime
        // fingerprint — block addresses now commit to the full
        // cache-computation identity, not just the mode name.
        &cache_computation_id(&[21u8; 32], KVCacheMode::Fp16, 0),
    );
    let mut committed = 0usize;
    for hash in &expected {
        if !blocks_dir.join(hex_digest(hash)).exists() {
            continue;
        }
        committed += 1;
        store.read_block(hash).unwrap_or_else(|e| {
            panic!(
                "COMMITTED BLOCK IS UNREADABLE AFTER CONCURRENT PERSIST: {}: {e}\n\
                 A block directory exists (so load_prefix can discover it) but the block fails \
                 its own per-layer sha256 or byte_len check. That is a corrupt adopted prefix, \
                 which is the failure this gate exists to catch. persist_lock is declared and \
                 NEVER acquired, and both writers stage into the same .tmp.<hash> path.",
                hex_digest(hash)
            )
        });
    }
    assert!(committed > 0, "at least one block must be committed");

    // Any committed directory that is NOT one of the expected addresses is
    // also a defect — it would mean the race produced a block nobody asked for.
    // Only DIRECTORIES are blocks. `<hash>.refcount` sidecar FILES live in the
    // same directory (see the eviction path) and are not block commits.
    let unexpected: Vec<String> = std::fs::read_dir(&blocks_dir)
        .expect("read blocks dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| !n.starts_with(".tmp."))
        .filter(|n| !expected.iter().any(|h| &hex_digest(h) == n))
        .collect();
    assert!(
        unexpected.is_empty(),
        "block directories committed under unexpected addresses: {unexpected:?}"
    );

    // INVARIANT 2 — at least one writer must have succeeded. If BOTH failed,
    // concurrent persist of identical content is unusable, not merely racy.
    assert!(
        outcomes.iter().any(|r| r.is_ok()),
        "both concurrent persists failed; errors: {errs:?}"
    );

    // INVARIANT 3 — no staging residue. An orphaned .tmp.<hash> is a disk leak
    // (no scan reads it, so it is not a correctness hazard) but it is evidence
    // the race was hit, so report it explicitly rather than let it pass silent.
    let residue: Vec<String> = std::fs::read_dir(&blocks_dir)
        .expect("read blocks dir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.starts_with(".tmp."))
        .collect();
    assert!(
        residue.is_empty(),
        "orphaned staging directories left by concurrent persist: {residue:?} \
         (disk leak, and direct evidence the shared .tmp.<hash> path was contended). \
         Writer errors: {errs:?}"
    );
}

// ===========================================================================
// G3.1 — KILL -9 MID-PERSIST (Violet's pre-deploy gate, minimum-viable item 4)
//
// Distinct from G2.1 and from the fsync question, and the gate doc bundled all
// three under "durability". They are different failure modes:
//   G2.1  concurrent writers      -> logical interleaving
//   G3.1  process dies mid-write  -> torn LOGICAL state (manifest vs blocks)
//   fsync process+kernel survive  -> torn PHYSICAL state; kill -9 does NOT
//         exercise it, because the page cache outlives the process. Only power
//         loss or a kernel panic tears those writes. Do not read a green G3.1
//         as discharging fsync.
//
// The child re-execs THIS test binary with MLXCEL_G31_CHILD set, persists into
// a directory the parent owns, and is SIGKILLed after a delay. The parent then
// asserts a property that must hold for EVERY kill point, so the gate cannot
// be flaky: whatever survives on disk must be READABLE, or absent. Never
// present-and-corrupt, because present-and-corrupt is what load_prefix would
// adopt as a prefix.
// ===========================================================================

// # MUTATION RECORD — this gate is proven two-sided
//
//   PRISTINE                      -> GREEN. 26/40 runs killed mid-persist,
//                                    195 committed blocks inspected, all
//                                    readable. temp+rename holds under SIGKILL.
//   ATOMIC COMMIT REMOVED         -> RED. "KILL -9 LEFT A READABLE-BUT-CORRUPT
//   (tmp_dir = block_dir, no          COMMITTED BLOCK (delay 225ms)". Restored
//    rename)                          md5-identical afterwards.
//
// # Why the kill points are dense, which is load-bearing
//
// The FIRST version used 5 coarse delays and PASSED WITH ATOMIC COMMIT
// REMOVED — it could not demonstrate its own bite. The window where a block
// directory exists but is incomplete is short (serialization dominates; file
// writes are memcpy into page cache), so coarse sampling misses it. Density is
// what turns this from decoration into a control. Do not thin the delay list
// to make the suite faster without re-running the mutation.
//
// Cost: ~25s. That is the price of a durability gate that can actually fail.
const G31_CHILD_ENV: &str = "MLXCEL_G31_CHILD";
const G31_DIR_ENV: &str = "MLXCEL_G31_DIR";

/// Child workload: persist a multi-block cache, then exit. Never returns if
/// the parent kills it first, which is the point.
fn g31_child_workload() {
    let base = std::path::PathBuf::from(
        std::env::var(G31_DIR_ENV).expect("child needs MLXCEL_G31_DIR"),
    );
    let store = BlockColdStore::new(base, [31u8; 32]);
    // 8 blocks — enough work that a kill lands mid-persist rather than
    // always before or always after.
    const BLOCKS: i32 = 8;
    let depth = BLOCKS * DEFAULT_BLOCK_SIZE as i32;
    let set = fp16_set(2, depth);
    let tokens: Vec<i32> = (0..depth).collect();
    let _ = store.persist("m3", "tmpl", &tokens, &set);
}

#[test]
fn kill_9_mid_persist_never_leaves_a_readable_corrupt_block() {
    // Child mode: this same binary, re-entered. Do the work and leave.
    if std::env::var(G31_CHILD_ENV).is_ok() {
        g31_child_workload();
        return;
    }

    let exe = std::env::current_exe().expect("current_exe");
    // The harness matches on the FULL module path, without the crate name.
    // Passing the bare fn name with --exact silently matches nothing and the
    // child exits "ok. 0 passed" — a green child that ran no workload.
    let test_path = format!(
        "{}::kill_9_mid_persist_never_leaves_a_readable_corrupt_block",
        module_path!()
            .strip_prefix("mlxcel_core::")
            .unwrap_or(module_path!())
    );
    // Kill points must be DENSE, not merely spread. The window in which a
    // block directory exists but is incomplete is short — serialization
    // dominates, file writes are memcpy-to-page-cache — so a handful of
    // coarse delays samples it with low probability. Measured: with 5 coarse
    // delays this test passed even with atomic commit REMOVED, i.e. it could
    // not demonstrate its own bite. Density is what makes it a control.
    let delays_ms: Vec<u64> = (1..=40).map(|i| i * 25).collect();
    let mut killed_runs = 0usize;
    let mut survivors = 0usize;

    for delay in delays_ms.iter().copied() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let mut child = std::process::Command::new(&exe)
            .arg(&test_path)
            .arg("--exact")
            .arg("--nocapture")
            .env(G31_CHILD_ENV, "1")
            .env(G31_DIR_ENV, dir.path())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn child");

        std::thread::sleep(std::time::Duration::from_millis(delay));
        let already_exited = child.try_wait().expect("try_wait").is_some();
        if !already_exited {
            child.kill().expect("SIGKILL child");
            killed_runs += 1;
        }
        let _ = child.wait();

        // ---- the invariant, checked on whatever the corpse left behind ----
        let store = BlockColdStore::new(dir.path().to_path_buf(), [31u8; 32]);
        let blocks_dir = store.blocks_dir();
        if !blocks_dir.exists() {
            continue; // killed before any block landed — valid outcome
        }

        for entry in std::fs::read_dir(&blocks_dir).expect("read blocks dir") {
            let entry = entry.expect("dir entry");
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue; // .refcount sidecars are not blocks
            }
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(".tmp.") {
                // Staging residue is EXPECTED after a kill: temp+rename means
                // an interrupted write leaves .tmp.<hash> behind. It is a disk
                // leak, not a correctness hazard — every scan site skips the
                // .tmp. prefix (:323, :486, :621), so it is never mistaken for
                // a committed block. Recorded, not asserted against.
                continue;
            }
            survivors += 1;
            // A COMMITTED directory must be fully readable. read_block verifies
            // per-layer sha256 and byte_len, so this catches a block that was
            // renamed into place while incomplete.
            let mut hash = [0u8; 32];
            let raw: Vec<u8> = (0..32)
                .map(|i| u8::from_str_radix(&name[i * 2..i * 2 + 2], 16).unwrap_or(0))
                .collect();
            hash.copy_from_slice(&raw);
            store.read_block(&hash).unwrap_or_else(|e| {
                panic!(
                    "KILL -9 LEFT A READABLE-BUT-CORRUPT COMMITTED BLOCK (delay {delay}ms): \
                     {name}: {e}\n\
                     The directory exists, so load_prefix can discover and adopt it, but it \
                     fails its own per-layer sha256/byte_len. temp+rename is supposed to make \
                     commit atomic; this is the case where it did not."
                )
            });
        }
    }

    // Control: if we never actually killed anything mid-flight, this test
    // proved nothing and must say so rather than pass quietly.
    assert!(
        killed_runs > 0,
        "no child was killed mid-persist at any delay — the workload finished too fast, \
         so this gate exercised NOTHING. Increase BLOCKS or shorten the delays."
    );
    eprintln!("G3.1: killed {killed_runs}/{} runs, inspected {survivors} committed blocks",
              delays_ms.len());
}

// ===========================================================================
// CACHE-COMPUTATION IDENTITY — the block address must commit to everything
// that determines the bytes, and to nothing else.
//
// Blocks live in ONE GLOBAL POOL keyed only by hash, so two runtimes that
// compute different bytes for the same tokens must not reach the same address.
// Before this, the address committed to `format!("{:?}", mode)` alone.
// ===========================================================================

#[test]
fn cache_identity_commits_to_v_bits() {
    let rt = [1u8; 32];
    let k8v4 = cache_computation_id(&rt, KVCacheMode::KVarN8, 4);
    let k8v8 = cache_computation_id(&rt, KVCacheMode::KVarN8, 8);
    assert_ne!(
        k8v4, k8v8,
        "k8v4 and k8v8 lay out the V payload differently, so they MUST NOT share \
         a block address. Before the fix both rendered to \"KVarN8\" and collided \
         inside one model and one mode, with no weight change required."
    );
}

#[test]
fn cache_identity_commits_to_runtime_fingerprint() {
    let a = cache_computation_id(&[1u8; 32], KVCacheMode::KVarN8, 4);
    let b = cache_computation_id(&[2u8; 32], KVCacheMode::KVarN8, 4);
    assert_ne!(
        a, b,
        "different weights compute different KV bytes for identical tokens, so \
         they MUST NOT share a block address"
    );
}

#[test]
fn cache_identity_normalises_v_bits_under_fp16() {
    let rt = [1u8; 32];
    assert_eq!(
        cache_computation_id(&rt, KVCacheMode::Fp16, 0),
        cache_computation_id(&rt, KVCacheMode::Fp16, 8),
        "v_bits does not affect unquantized bytes, so it must not affect the \
         address — otherwise an Fp16 cache carrying a stale v_bits is unreachable \
         to a loader that correctly passes 0, and the cache misses for no reason"
    );
}

/// A and B must write DISTINGUISHABLE bytes, or "B loaded A's data" is
/// indistinguishable from "B loaded its own".
const A_SHIFT: i32 = 0;
const B_SHIFT: i32 = 37;

fn fp16_set_shifted(num_layers: usize, len: i32, shift: i32) -> DetachedCacheSet {
    let caches = (0..num_layers)
        .map(|i| fp16_layer(len, shift + i as i32 * 5))
        .collect();
    let now = Instant::now();
    DetachedCacheSet {
        caches,
        backend: SequenceStateBackend::DenseKvCache,
        prompt_len: len as usize,
        current_offset: len,
        created_at: now,
        detached_at: now,
        origin_seq_id: SequenceId::from_raw(11),
    }
}

/// THE REAL COLLISION PATH, end to end.
///
/// Two runtimes with different weights, sharing one block pool, persisting the
/// SAME tokens. Before the fix they computed IDENTICAL block hashes, so the
/// second runtime's `write_block` hit its `block_dir.exists()` early return,
/// silently skipped writing, and left its manifest pointing at the FIRST
/// runtime's KV data — a silent cross-model wrong adoption.
#[test]
fn different_runtimes_sharing_a_block_pool_do_not_collide() {
    const LAYERS: usize = 2;
    const DEPTH: i32 = 2 * DEFAULT_BLOCK_SIZE as i32;
    let tokens: Vec<i32> = (0..DEPTH).collect();

    // ONE directory — the shared global pool. Two different runtimes.
    let dir = tempfile::TempDir::new().expect("tempdir");
    let store_a = BlockColdStore::new(dir.path().to_path_buf(), [0xAAu8; 32]);
    let store_b = BlockColdStore::new(dir.path().to_path_buf(), [0xBBu8; 32]);

    store_a
        .persist("m3", "tmpl", &tokens, &fp16_set_shifted(LAYERS, DEPTH, A_SHIFT))
        .expect("runtime A persists");

    // B has NOT persisted. It must not be able to load A's blocks.
    let leaked = store_b.load_prefix("m3", "tmpl", &tokens, KVCacheMode::Fp16, 0);
    assert!(
        matches!(leaked, Err(ColdStoreError::NoMatch)),
        "runtime B loaded a prefix it never persisted — that is runtime A's KV \
         data adopted under B's weights. Got: {:?}",
        leaked.map(|(_, m)| m)
    );

    // And B persisting its own must not be silently skipped as "already written".
    //
    // MATCH LENGTH ALONE IS NOT ENOUGH HERE, and asserting only that was a
    // vacuousness the G0.2 gate caught: `load_prefix` filters manifests by
    // runtime_fingerprint (:704) BEFORE comparing block hashes, so B never sees
    // A's manifest whether or not the ADDRESS commits to the fingerprint. The
    // address matters on the WRITE side — without it, B's blocks hash to A's
    // addresses, `write_block` early-returns on exists(), B never writes, and
    // B's own manifest points at A's bytes. So the assertion must be on CONTENT.
    let b_layer0_expected = bytes_of(&fp16_layer(DEPTH, B_SHIFT).keys, "b keys");
    store_b
        .persist("m3", "tmpl", &tokens, &fp16_set_shifted(LAYERS, DEPTH, B_SHIFT))
        .expect("runtime B persists");
    let (b_loaded, matched) = store_b
        .load_prefix("m3", "tmpl", &tokens, KVCacheMode::Fp16, 0)
        .expect("B must now hit its OWN blocks");
    assert_eq!(matched, tokens.len(), "B must match its own full prefix");
    assert_eq!(
        bytes_of(&b_loaded.caches[0].keys, "loaded keys"),
        b_layer0_expected,
        "B loaded bytes that are NOT B's. Its manifest is pointing at runtime A's \
         block data, because the block address did not commit to whose weights \
         computed it and write_block skipped B's write as 'already present'."
    );

    // A must still hit its own, unharmed by B.
    let (_, matched_a) = store_a
        .load_prefix("m3", "tmpl", &tokens, KVCacheMode::Fp16, 0)
        .expect("A must still hit its own blocks");
    assert_eq!(matched_a, tokens.len(), "A's entry must survive B's persist");
}

// ===========================================================================
// ALDEN'S FINDING 2 — prefix pruning and the partial tail block
//
// Alden, 2026-07-27 (docs-only read, against the spec rather than the code):
//
//   "Block-hash-list prefix pruning fails for ordinary growth when the old
//    manifest ends in a partial block: extending that block changes its hash,
//    so the old hash list is not a prefix and its manifest keeps the old tail
//    rooted."
//
// Evaluated against the implementation, and it is REAL. `persist` chunks with
// `tokens.chunks(block_size)`, which yields a PARTIAL final chunk whenever the
// token count is not a multiple of the block size — and `block_hash_merkle`
// commits to that chunk's own token count and tokens. So when the next turn
// fills that block, its hash changes, the old hash list is no longer a prefix
// of the new one, `prune_prefix_manifests` skips it, and the superseded
// manifest survives holding refcounts on an orphaned block.
//
// A conversation only lands on an exact multiple of 2048 tokens by accident,
// so this is the ORDINARY path, not an edge case.
// ===========================================================================

fn count_manifests(store: &BlockColdStore) -> usize {
    let dir = store.manifests_dir();
    if !dir.exists() {
        return 0;
    }
    std::fs::read_dir(&dir)
        .expect("read manifests dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
        .filter(|e| !e.file_name().to_string_lossy().starts_with(".tmp."))
        .count()
}

/// CONTROL — growth from an EXACT multiple of the block size prunes correctly.
///
/// Establishes that the pruning path works at all, so the partial-tail failure
/// below is attributable to the tail and not to pruning being broken outright.
/// Without this control, a red test proves nothing about the cause.
#[test]
fn growth_from_an_exact_block_multiple_prunes_the_old_manifest() {
    const LAYERS: usize = 2;
    const TURN1: i32 = DEFAULT_BLOCK_SIZE as i32; // exactly one full block
    const TURN2: i32 = 2 * DEFAULT_BLOCK_SIZE as i32; // exactly two full blocks

    let dir = tempfile::TempDir::new().expect("tempdir");
    let store = BlockColdStore::new(dir.path().to_path_buf(), [0xC1u8; 32])
        .with_prune_mode(PruneMode::Delete);

    let t1: Vec<i32> = (0..TURN1).collect();
    store
        .persist("m3", "tmpl", &t1, &fp16_set(LAYERS, TURN1))
        .expect("turn 1 persists");
    assert_eq!(count_manifests(&store), 1, "one manifest after turn 1");

    let t2: Vec<i32> = (0..TURN2).collect();
    store
        .persist("m3", "tmpl", &t2, &fp16_set(LAYERS, TURN2))
        .expect("turn 2 persists");

    assert_eq!(
        count_manifests(&store),
        1,
        "turn 1's manifest must be pruned: its single full block is an exact \
         prefix of turn 2's two blocks, so the prefix check succeeds"
    );
}

/// ALDEN'S FINDING 2, DEMONSTRATED.
///
/// Same growth, but turn 1 ends mid-block. Turn 1's tail block covers 512
/// tokens; turn 2 fills it to 2048. `block_hash_merkle` commits to the block's
/// own tokens, so that block's hash changes, turn 1's hash list stops being a
/// prefix of turn 2's, and the superseded manifest is never pruned.
///
/// The mutation that must redden this test: none — it reddens on the DEFECT.
/// It goes green only when prune gains a token-prefix proof rather than a
/// hash-list prefix check.
#[test]
fn growth_from_a_partial_tail_block_still_prunes_the_old_manifest() {
    const LAYERS: usize = 2;
    const TURN1: i32 = DEFAULT_BLOCK_SIZE as i32 + 512; // 2048 + 512 -> tail block
    const TURN2: i32 = 2 * DEFAULT_BLOCK_SIZE as i32; // tail block now full

    let dir = tempfile::TempDir::new().expect("tempdir");
    let store = BlockColdStore::new(dir.path().to_path_buf(), [0xC2u8; 32])
        .with_prune_mode(PruneMode::Delete);

    let t1: Vec<i32> = (0..TURN1).collect();
    let m1 = store
        .persist("m3", "tmpl", &t1, &fp16_set(LAYERS, TURN1))
        .expect("turn 1 persists");
    assert_eq!(m1.block_hashes.len(), 2, "turn 1 must have a partial 2nd block");
    assert_eq!(count_manifests(&store), 1, "one manifest after turn 1");

    // Turn 2 continues the SAME conversation — t1 is a strict token prefix of t2.
    let t2: Vec<i32> = (0..TURN2).collect();
    assert_eq!(&t2[..t1.len()], &t1[..], "turn 2 must extend turn 1's tokens");
    let m2 = store
        .persist("m3", "tmpl", &t2, &fp16_set(LAYERS, TURN2))
        .expect("turn 2 persists");

    // The mechanism, asserted directly so a failure names the cause.
    assert_ne!(
        m1.block_hashes[1], m2.block_hashes[1],
        "the tail block's hash MUST change when it fills — this is why the \
         hash-list prefix check fails"
    );
    assert_eq!(
        m1.block_hashes[0], m2.block_hashes[0],
        "the full leading block is unchanged, so this is genuine linear growth"
    );

    assert_eq!(
        count_manifests(&store),
        1,
        "ALDEN FINDING 2: turn 1's manifest survived. It is superseded by turn \
         2 on the same conversation, but its tail block hash changed when the \
         block filled, so the hash-list prefix check skipped it. Every \
         non-boundary turn leaks a manifest plus an orphaned tail block."
    );
}

/// SAFETY — the token-prefix proof must REFUSE a divergent branch.
///
/// This is the direction that matters. Failing to prune costs disk; pruning
/// wrongly destroys a live manifest and forces a full re-prefill. The old
/// manifest here shares a complete leading block with the new one and is
/// strictly shorter — so it passes every cheap pre-filter — but its tokens
/// diverge inside the tail block, so it is NOT a prefix of this conversation
/// and must survive.
///
/// The mutation that must redden this test: drop the `expected !=
/// old_manifest.block_hashes` comparison in `prune_prefix_manifests`, or
/// weaken it to compare only the full leading blocks.
#[test]
fn a_divergent_branch_sharing_a_leading_block_is_never_pruned() {
    const LAYERS: usize = 2;
    const SHARED: usize = DEFAULT_BLOCK_SIZE; // one full identical block
    const OLD_LEN: i32 = DEFAULT_BLOCK_SIZE as i32 + 512;
    const NEW_LEN: i32 = 2 * DEFAULT_BLOCK_SIZE as i32;

    let dir = tempfile::TempDir::new().expect("tempdir");
    let store = BlockColdStore::new(dir.path().to_path_buf(), [0xD1u8; 32])
        .with_prune_mode(PruneMode::Delete);

    // Branch A: 0..2560
    let a: Vec<i32> = (0..OLD_LEN).collect();
    let m_a = store
        .persist("m3", "tmpl", &a, &fp16_set(LAYERS, OLD_LEN))
        .expect("branch A persists");

    // Branch B: identical for the first full block, then DIVERGENT, and longer.
    let mut b: Vec<i32> = a[..SHARED].to_vec();
    b.extend((0..(NEW_LEN as usize - SHARED)).map(|i| 900_000 + i as i32));
    assert_eq!(b.len(), NEW_LEN as usize);
    assert_eq!(&b[..SHARED], &a[..SHARED], "leading block must be shared");
    assert_ne!(&b[..a.len()], &a[..], "B must NOT extend A");

    let m_b = store
        .persist("m3", "tmpl", &b, &fp16_set(LAYERS, NEW_LEN))
        .expect("branch B persists");

    assert_eq!(
        m_a.block_hashes[0], m_b.block_hashes[0],
        "shared leading block must hash identically — otherwise this test is \
         not exercising the case it claims to"
    );
    assert!(
        m_a.total_tokens < m_b.total_tokens,
        "A must be strictly shorter, so it reaches the token-prefix proof"
    );

    assert_eq!(
        count_manifests(&store),
        2,
        "branch A was pruned by a longer, divergent conversation. A shared \
         leading block and a shorter length are NOT proof of a prefix; only \
         recomputing A's hashes over B's first A.total_tokens tokens is."
    );
}

/// The point of pruning: the superseded tail block becomes collectable.
///
/// Asserts the CONSEQUENCE, not just the manifest count — pruning that left
/// refcounts pinned would satisfy the count assertions above while still
/// leaking every orphaned tail block forever.
#[test]
fn pruning_a_partial_tail_manifest_releases_the_orphaned_block() {
    const LAYERS: usize = 2;
    const TURN1: i32 = DEFAULT_BLOCK_SIZE as i32 + 512;
    const TURN2: i32 = 2 * DEFAULT_BLOCK_SIZE as i32;

    let dir = tempfile::TempDir::new().expect("tempdir");
    let store = BlockColdStore::new(dir.path().to_path_buf(), [0xD2u8; 32])
        .with_prune_mode(PruneMode::Delete)
        // Opt out of the nomination age floor. A test cannot wait out the
        // production default, and this test is about whether a released orphan
        // is COLLECTABLE, not about the cushion in front of collection.
        .with_min_gc_age(std::time::Duration::ZERO);

    let t1: Vec<i32> = (0..TURN1).collect();
    let m1 = store
        .persist("m3", "tmpl", &t1, &fp16_set(LAYERS, TURN1))
        .expect("turn 1 persists");
    let orphan = m1.block_hashes[1];
    assert_eq!(
        store.get_refcount(&orphan).expect("refcount"),
        1,
        "the partial tail block is referenced once by turn 1"
    );

    let t2: Vec<i32> = (0..TURN2).collect();
    store
        .persist("m3", "tmpl", &t2, &fp16_set(LAYERS, TURN2))
        .expect("turn 2 persists");

    assert_eq!(
        store.get_refcount(&orphan).expect("refcount"),
        0,
        "pruning turn 1 must release its tail block, or every turn leaks one"
    );

    store.gc_blocks().expect("gc");
    assert!(
        !store.blocks_dir().join(hex_digest(&orphan)).exists(),
        "a released orphan block must be collectable by gc_blocks"
    );
}

/// REGRESSION — `PruneMode::Observe` is the DEFAULT and must not delete.
///
/// It is documented as "log what WOULD be pruned, do not delete", and
/// `gc_blocks` honours that. `prune_prefix_manifests` did not: it deleted in
/// every mode except `Off`, so the default configuration silently destroyed
/// superseded manifests while presenting itself as observe-only. An operator
/// validating on real data before enabling deletion was already deleting.
///
/// The mutation that must redden this test: remove the `PruneMode::Observe`
/// early-`continue` in `prune_prefix_manifests`.
#[test]
fn observe_mode_reports_a_prune_without_performing_it() {
    const LAYERS: usize = 2;
    const TURN1: i32 = DEFAULT_BLOCK_SIZE as i32 + 512;
    const TURN2: i32 = 2 * DEFAULT_BLOCK_SIZE as i32;

    let dir = tempfile::TempDir::new().expect("tempdir");
    // Default mode — deliberately NOT set, so this test also pins the default.
    let store = BlockColdStore::new(dir.path().to_path_buf(), [0xD3u8; 32]);
    assert_eq!(
        store.prune_mode(),
        PruneMode::Observe,
        "Observe must remain the default; this test is about the default"
    );

    let t1: Vec<i32> = (0..TURN1).collect();
    let m1 = store
        .persist("m3", "tmpl", &t1, &fp16_set(LAYERS, TURN1))
        .expect("turn 1 persists");
    let t2: Vec<i32> = (0..TURN2).collect();
    store
        .persist("m3", "tmpl", &t2, &fp16_set(LAYERS, TURN2))
        .expect("turn 2 persists");

    assert_eq!(
        count_manifests(&store),
        2,
        "observe mode DELETED a manifest. Its whole purpose is to let an \
         operator see what deletion would do before enabling it."
    );
    assert_eq!(
        store.get_refcount(&m1.block_hashes[1]).expect("refcount"),
        1,
        "observe mode must not decrement refcounts either — a released block \
         is one gc pass away from deletion, so a silent decrement is a \
         deferred silent delete"
    );
}
