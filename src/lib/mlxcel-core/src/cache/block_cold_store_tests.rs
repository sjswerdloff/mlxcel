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
    let mut v = Vec::with_capacity((VH * len * width) as usize);
    for h in 0..VH {
        for t in 0..len {
            for w in 0..width {
                let val: f32 = if dt == dtype::FLOAT16 {
                    ((t + 1) * 3 + h) as f32
                } else if dt == dtype::UINT8 {
                    (t * 7 + w * 3 + h).rem_euclid(251) as f32
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

    // --- sink: entirely inside block 0. ---
    let sink_k = bytes_of(&extracted[0].caches[0].kvarn_sink_k, "sink_k");
    assert_eq!(
        sink_k,
        src_token_slice_bytes(&src_layer.kvarn_sink_k, 0, TILE),
        "reassembled sink_k must equal source sink_k bytewise"
    );

    // --- hist_k: concat block0 hist (tiles 0..2) then block1 hist (tiles 2..4). ---
    let hk0 = extracted[0].caches[0]
        .kvarn_hist_k
        .as_ref()
        .expect("block0 hist_k")
        .as_ref()
        .unwrap();
    let hk1 = extracted[1].caches[0]
        .kvarn_hist_k
        .as_ref()
        .expect("block1 hist_k")
        .as_ref()
        .unwrap();
    let hk_cat = crate::utils::concatenate(hk0, hk1, 2);
    assert_eq!(
        bytes(&hk_cat),
        src_token_slice_bytes(&src_layer.kvarn_hist_k, 0, t),
        "reassembled hist_k must equal the full source history bytewise"
    );

    // --- k_s_col: per-TILE. block0 tiles 0..2, block1 tiles 2..4. ---
    let sc0 = extracted[0].caches[0]
        .kvarn_k_s_col
        .as_ref()
        .expect("block0 k_s_col")
        .as_ref()
        .unwrap();
    let sc1 = extracted[1].caches[0]
        .kvarn_k_s_col
        .as_ref()
        .expect("block1 k_s_col")
        .as_ref()
        .unwrap();
    let sc_cat = crate::utils::concatenate(sc0, sc1, 2);
    assert_eq!(
        bytes(&sc_cat),
        src_token_slice_bytes(&src_layer.kvarn_k_s_col, 0, N_TILES),
        "reassembled k_s_col must equal source per-tile s_col bytewise"
    );

    // --- tail: entirely inside the last block. ---
    let tail_k = bytes_of(&extracted[2].caches[0].kvarn_tail_k, "tail_k");
    assert_eq!(
        tail_k,
        src_token_slice_bytes(&src_layer.kvarn_tail_k, 0, TAIL),
        "reassembled tail_k must equal source tail_k bytewise"
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

/// RED control for the trap: prove the TILE interpretation and the TOKEN
/// interpretation of the mid-block `s_col` slice yield DIFFERENT bytes, so an
/// impl that (wrongly) sliced `s_col` by token count instead of tile count is
/// distinguishable. This demonstrates `s_col_sliced_by_tile_not_token` can go
/// red under the specific token-vs-tile bug (not vacuously green).
///
/// For the mid block global `[256, 512)` -> hist-local `[128, 384)`:
///   * CORRECT (per-tile):  `k_s_col[1..3)`  — 2 rows, width VD
///   * BUGGY  (per-token):  a `[128..384)`-indexed read of a token-length
///     tensor — 256 rows. We stand in `k_scale` (which IS token-length, width
///     1) as the shape a token-count slice would have.
/// The two byte images differ in both length and content, so the comparison
/// bites.
#[test]
fn s_col_red_control_token_slice_would_mismatch() {
    const N_TILES: i32 = 4;
    let src = kvarn_v4_set(1, N_TILES, 0);

    // CORRECT per-tile slice: tiles [1..3) of the per-tile s_col tensor.
    let per_tile = src_token_slice_bytes(&src.caches[0].kvarn_k_s_col, 1, 3);
    // BUGGY token-indexed slice: tokens [128..384) of a token-length tensor.
    let per_token = src_token_slice_bytes(&src.caches[0].kvarn_k_scale, 128, 384);

    // Sanity on the two shapes.
    let sc = src.caches[0]
        .kvarn_k_s_col
        .as_ref()
        .unwrap()
        .as_ref()
        .unwrap();
    assert_eq!(axis2_len(sc), N_TILES, "source s_col is per-tile (N_TILES rows)");
    let ks = src.caches[0]
        .kvarn_k_scale
        .as_ref()
        .unwrap()
        .as_ref()
        .unwrap();
    assert_eq!(axis2_len(ks), N_TILES * TILE, "source k_scale is per-token");

    // The discriminating claim: the tile-sliced bytes are NOT equal to the
    // token-sliced bytes — a token-vs-tile bug changes both length and value.
    assert_ne!(
        per_tile, per_token,
        "per-tile s_col slice MUST differ from a per-token slice — proves the \
         token-vs-tile trap is discriminating, not vacuous"
    );
    assert_ne!(
        per_tile.len(),
        per_token.len(),
        "per-tile (2 rows x VD) and per-token (256 rows x 1) byte counts differ"
    );
}

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
