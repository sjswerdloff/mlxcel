// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! K8V4 GOLDEN HARNESS — §4 golden vectors, the screen→engine bridge
//! (DESIGN_kvarn_k8v4_engine_2026-07-11 §4; RESULTS_kvarn4_realtile_2026-07-11).
//!
//! Drives the REAL engine write path (`update_kvarn8`, one fresh
//! `kvarn_v_bits=4` cache per harvest event) on the SAME real-tile
//! population the boot-night screen licensed, and pins every stored field
//! — K side and V side — BITWISE against the dual-approved math layer
//! ([`kvarn_quantize`] at 8-bit, [`kvarn_quantize_v4`] at gs=32) computed
//! on the identical event batch. This is what makes the screen verdict
//! TRANSFER to the engine: same tiles, same ops, zero tolerance.
//!
//! ## Scope — what a green run gates (and what it does not)
//!
//! * GATES: the STORAGE roundtrip only — what `update_kvarn8` stores is
//!   bit-for-bit what the math layer produces on the tiles the update
//!   built (codes, folded params, s_col; K fields; sink/tail placement).
//!   Read paths do not exist yet; this harness is the bridge they will
//!   build against (§5 order).
//! * DOES NOT gate `gather_qmm`'s FUSED consumption of that storage —
//!   tolerance-gated separately under §4.2 (coverage-mapping pin, Violet).
//! * DOES NOT claim live-session bit-equality, BY CONSTRUCTION, for two
//!   reasons recorded so nobody hunts a phantom: (1) the harvest sampled
//!   ≤ [`harvest::TILES_PER_EVENT`] spread tiles per finalization event,
//!   so the live batches (e.g. n_full=64) cannot be reconstructed; and
//!   (2) Sinkhorn best-so-far selection is batch-GLOBAL (`imbalance`,
//!   kvarn.rs), so per-tile outputs depend on the whole batch grouping.
//!   The contract here is engine-vs-reference on IDENTICAL per-event
//!   batches, which the harness guarantees by feeding both chains the
//!   same event.
//! * ALL comparisons in this harness are BITWISE. No dequant-level
//!   tolerance comparison exists here — the cancellation-pricing birth
//!   constraint (§4 ⊕⊕: near tile minima, `q·s + zp` error rides the
//!   INTERMEDIATE magnitude ~ulp·qmax·scale, so relative-to-final bounds
//!   are wrong-shaped) is honored by not opening that door at all.
//!
//! ## Population (pinned to the screen, scripts/kvarn4_realtile_screen.py)
//!
//! Same selection, replicated exactly: sorted `harvest_*_k_rot_f32.json`
//! basenames, `np.linspace(0, n_files-1, 512).astype(int)` event indices
//! (truncation semantics — pinned by unit test below), V dump = K seq + 1
//! with matching `cache` and `offset` sidecar fields. Tiles are stored
//! ROTATED f32 (exactly the tensors quantize received, cache.rs harvest
//! hook); the harness unrotates once on load (Hadamard self-inverse) so
//! the engine's own per-token rotate runs on standard-frame rows, exactly
//! as live. Expected population: 512 events / 4,096 tiles per role —
//! deviation fails loud rather than silently gating a different bank.
//!
//! ## First act (structural gate)
//!
//! The gating test's first statement is
//! [`kvarn::assert_round_half_even_parity`] — the hand-built half-case
//! vector runs before any tile is touched (Violet PM pin 1, Xander
//! confirmation: a test-code dependency, not prose).
//!
//! ## Invocation (ignored: needs the harvest bank on disk)
//!
//! ```text
//! MLXCEL_KVARN_GOLDEN_DIR=$HOME/kvarn_harvest_20260710 \
//!   cargo test -p mlxcel-core --release k8v4_golden_harness_real_tiles \
//!   -- --ignored --nocapture
//! ```

use cxx::UniquePtr;

use super::{KVCache, KVCacheMode};
use crate::cache::kvarn::{
    self, KVARN_TILE_TOKENS, KVARN_V4_GROUP_SIZE, kvarn_quantize, kvarn_quantize_v4, kvarn_rotate,
};
use crate::dtype;
use crate::ffi::{self, MlxArray};

const GOLDEN_DIR_ENV: &str = "MLXCEL_KVARN_GOLDEN_DIR";
/// Event count the screen drew (`MAX_PAIRS`); the bank holds more K jsons
/// than this, so linspace picks 512 distinct, evenly-spread events.
const MAX_EVENTS: usize = 512;
/// The licensed population (RESULTS_kvarn4_realtile_2026-07-11): 512
/// events × 8 sampled tiles. The harness pins the count exactly — a
/// different bank must be a conscious re-registration, not a drift.
const EXPECTED_TILES: usize = 4096;
const R: i32 = 128; // tile rows == KVARN_TILE_TOKENS
const C: i32 = 128; // head_dim (M3)

/// One harvest dump event: token-aligned K/V tile batches (same `cache`
/// and `offset` sidecar fields), still in the stored ROTATED f32 frame.
struct HarvestEvent {
    seq: u64,
    offset: i64,
    /// `[n_tiles, R, C]` row-major f32, rotated frame (as dumped).
    k_rot: Vec<f32>,
    v_rot: Vec<f32>,
    n_tiles: i32,
}

/// `np.linspace(0, n-1, k).astype(int)` — evenly spaced floats over the
/// closed range, TRUNCATED toward zero (numpy `astype(int)`), computed in
/// f64 to mirror numpy exactly. `n <= k` degenerates to `0..n`.
fn linspace_trunc_indices(n: usize, k: usize) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    if n <= k {
        return (0..n).collect();
    }
    let step = (n - 1) as f64 / (k - 1) as f64;
    (0..k).map(|j| (j as f64 * step) as usize).collect()
}

fn sidecar_str(json: &serde_json::Value, key: &str) -> String {
    json.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| panic!("harvest sidecar missing string field '{key}': {json}"))
        .to_string()
}

fn sidecar_i64(json: &serde_json::Value, key: &str) -> i64 {
    json.get(key)
        .and_then(|v| v.as_i64())
        .unwrap_or_else(|| panic!("harvest sidecar missing integer field '{key}': {json}"))
}

/// Read one `harvest_NNNNNN_<role>` dump (json sidecar + raw f32 LE bin).
/// Returns (tiles, n_tiles, offset, cache_key). Fails loud on any
/// integrity violation — a golden population is verified, never coerced.
fn read_dump(dir: &std::path::Path, seq: u64, role: &str) -> (Vec<f32>, i32, i64, String) {
    let base = dir.join(format!("harvest_{seq:06}_{role}"));
    let json_path = base.with_extension("json");
    let sidecar: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&json_path)
            .unwrap_or_else(|e| panic!("golden bank read failed: {}: {e}", json_path.display())),
    )
    .unwrap_or_else(|e| panic!("golden sidecar parse failed: {}: {e}", json_path.display()));

    assert_eq!(sidecar_str(&sidecar, "role"), role, "role mismatch at seq {seq}");
    let shape = sidecar
        .get("shape")
        .and_then(|v| v.as_array())
        .unwrap_or_else(|| panic!("harvest sidecar missing shape: {sidecar}"));
    assert_eq!(shape.len(), 3, "tile dump must be [n, R, C], got {shape:?}");
    let n = shape[0].as_i64().expect("shape[0]") as i32;
    assert_eq!(shape[1].as_i64().expect("shape[1]") as i32, R, "tile rows");
    assert_eq!(shape[2].as_i64().expect("shape[2]") as i32, C, "tile channels");

    let bin_path = base.with_extension("bin");
    let bytes = std::fs::read(&bin_path)
        .unwrap_or_else(|e| panic!("golden bank read failed: {}: {e}", bin_path.display()));
    let expected_len = (n as usize) * (R as usize) * (C as usize) * 4;
    assert_eq!(
        bytes.len(),
        expected_len,
        "bin size mismatch at seq {seq}: shape says {expected_len} bytes"
    );
    let tiles: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let non_finite = tiles.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(
        non_finite, 0,
        "golden population integrity: {non_finite} non-finite values at seq {seq} ({role})"
    );
    (
        tiles,
        n,
        sidecar_i64(&sidecar, "offset"),
        sidecar_str(&sidecar, "cache"),
    )
}

/// Replicates the screen's `load_pairs()` selection exactly (see module
/// docs): sorted K sidecars → linspace event indices → pair with seq+1 V
/// dump requiring identical `cache` and `offset`. Non-pairing events are
/// skipped exactly as the screen skipped them.
fn load_events(dir: &std::path::Path) -> Vec<HarvestEvent> {
    let mut k_seqs: Vec<u64> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("golden bank dir unreadable: {}: {e}", dir.display()))
        .filter_map(|entry| {
            let name = entry.expect("dir entry").file_name();
            let name = name.to_string_lossy().into_owned();
            let seq = name
                .strip_prefix("harvest_")?
                .strip_suffix("_k_rot_f32.json")?
                .parse::<u64>()
                .ok()?;
            Some(seq)
        })
        .collect();
    k_seqs.sort_unstable();
    assert!(
        !k_seqs.is_empty(),
        "no harvest_*_k_rot_f32.json dumps found in {}",
        dir.display()
    );

    let mut events = Vec::with_capacity(MAX_EVENTS);
    for &i in &linspace_trunc_indices(k_seqs.len(), MAX_EVENTS) {
        let seq = k_seqs[i];
        let (k_rot, k_n, k_off, k_cache) = read_dump(dir, seq, "k_rot_f32");
        let v_json = dir.join(format!("harvest_{:06}_v_rot_f32.json", seq + 1));
        if !v_json.exists() {
            continue; // screen rule: unpaired K dump is skipped
        }
        let (v_rot, v_n, v_off, v_cache) = read_dump(dir, seq + 1, "v_rot_f32");
        if v_cache != k_cache || v_off != k_off {
            continue; // screen rule: pairs must share (cache, offset)
        }
        let n = k_n.min(v_n);
        let take = (n as usize) * (R as usize) * (C as usize);
        events.push(HarvestEvent {
            seq,
            offset: k_off,
            k_rot: k_rot[..take].to_vec(),
            v_rot: v_rot[..take].to_vec(),
            n_tiles: n,
        });
    }
    events
}

/// One recorded divergence. The harness collects across the whole
/// population before failing, so a single red event cannot hide the rest
/// of the run's evidence.
struct Mismatch {
    event_seq: u64,
    field: &'static str,
    detail: String,
}

/// Byte-level comparison with length checked FIRST — a zip-shortest
/// comparison would silently pass a truncated field, which is exactly the
/// class of false green the must-fail arms below prove impossible.
fn compare_field(
    event_seq: u64,
    field: &'static str,
    engine: &[u8],
    reference: &[u8],
    out: &mut Vec<Mismatch>,
) {
    if engine.len() != reference.len() {
        out.push(Mismatch {
            event_seq,
            field,
            detail: format!(
                "length mismatch: engine {} bytes vs reference {} bytes",
                engine.len(),
                reference.len()
            ),
        });
        return;
    }
    if let Some(pos) = engine.iter().zip(reference).position(|(a, b)| a != b) {
        out.push(Mismatch {
            event_seq,
            field,
            detail: format!(
                "first divergence at byte {pos} of {}: engine 0x{:02x} vs reference 0x{:02x}",
                engine.len(),
                engine[pos],
                reference[pos]
            ),
        });
    }
}

fn raw_field(a: &Option<UniquePtr<MlxArray>>, field: &'static str) -> Vec<u8> {
    ffi::array_to_raw_bytes(
        a.as_ref()
            .unwrap_or_else(|| panic!("engine cache field '{field}' unexpectedly None")),
    )
}

/// Build the standard-frame feed for one event: a 128-row all-zero sink
/// preamble (the fp16 sink absorbs the first KVARN_TILE_TOKENS rows of a
/// fresh cache — without the preamble every tile boundary would shift by
/// one tile) followed by the event's tiles unrotated back to the standard
/// frame, concatenated on the length axis as `[1, 1, 128 + n·128, C]`.
fn build_feed(tiles_rot: &[f32], n_tiles: i32) -> UniquePtr<MlxArray> {
    let rot = ffi::from_slice_f32(tiles_rot, &[n_tiles, R, C]);
    // Hadamard is self-inverse: one more rotate IS the unrotation.
    let std_frame = kvarn_rotate(&rot);
    let as_len = ffi::reshape(&std_frame, &[1, 1, n_tiles * R, C]);
    let preamble = ffi::full_f32(&[1, 1, KVARN_TILE_TOKENS, C], 0.0, dtype::FLOAT32);
    crate::ops::concatenate(&preamble, &as_len, 2)
}

/// Reference tile batch for the same feed, replicating the update path's
/// exact preprocessing (the dual-approved 10222 wiring-test idiom): rotate
/// the f16-cast feed, drop the sink prefix, reshape to the
/// `[n_tiles, R, C]` f32 batch that `update_kvarn8` hands to quantize.
fn reference_tiles(feed: &MlxArray, n_tiles: i32) -> UniquePtr<MlxArray> {
    let rot = kvarn_rotate(&ffi::astype(feed, dtype::FLOAT16));
    let total = KVARN_TILE_TOKENS + n_tiles * R;
    let past_sink = ffi::slice(&rot, &[0, 0, KVARN_TILE_TOKENS, 0], &[1, 1, total, C]);
    ffi::reshape(&ffi::astype(&past_sink, dtype::FLOAT32), &[n_tiles, R, C])
}

/// Drive one event through a fresh v4 cache and pin every stored field
/// bitwise against the math layer on the identical batch.
fn run_event(ev: &HarvestEvent, out: &mut Vec<Mismatch>) {
    let k_feed = build_feed(&ev.k_rot, ev.n_tiles);
    let v_feed = build_feed(&ev.v_rot, ev.n_tiles);

    // Reference chains on the SAME feeds (batch grouping == engine's).
    let k_ref = kvarn_quantize(&reference_tiles(&k_feed, ev.n_tiles), 8);
    let v_ref = kvarn_quantize_v4(&reference_tiles(&v_feed, ev.n_tiles), KVARN_V4_GROUP_SIZE);

    let mut cache = KVCache::new_with_mode(KVCacheMode::KVarN8);
    cache.kvarn_v_bits = 4;
    cache.update_only(k_feed, v_feed);

    // Feed-construction pins: sink took exactly the zero preamble, every
    // event row finalized (no tail), offset advanced by the whole feed.
    let sink = raw_field(&cache.kvarn_sink_k, "kvarn_sink_k");
    if sink.iter().any(|&b| b != 0) {
        out.push(Mismatch {
            event_seq: ev.seq,
            field: "kvarn_sink_k",
            detail: "sink holds non-zero bytes — preamble did not land in the sink".into(),
        });
    }
    if cache.kvarn_tail_k.is_some() || cache.kvarn_tail_v.is_some() {
        out.push(Mismatch {
            event_seq: ev.seq,
            field: "kvarn_tail",
            detail: "tail unexpectedly Some — event rows must finalize exactly".into(),
        });
    }
    let expected_offset = KVARN_TILE_TOKENS + ev.n_tiles * R;
    if cache.offset != expected_offset {
        out.push(Mismatch {
            event_seq: ev.seq,
            field: "offset",
            detail: format!("offset {} != expected {expected_offset}", cache.offset),
        });
    }
    if cache.kvarn_v_s_row.is_some() {
        out.push(Mismatch {
            event_seq: ev.seq,
            field: "kvarn_v_s_row",
            detail: "must stay None on V4 — the fold IS its storage".into(),
        });
    }

    // K side (design §1: K does not change — pinned on real tiles).
    compare_field(
        ev.seq,
        "kvarn_hist_k",
        &raw_field(&cache.kvarn_hist_k, "kvarn_hist_k"),
        &ffi::array_to_raw_bytes(&k_ref.q),
        out,
    );
    compare_field(
        ev.seq,
        "kvarn_k_scale",
        &raw_field(&cache.kvarn_k_scale, "kvarn_k_scale"),
        &ffi::array_to_raw_bytes(&k_ref.scale),
        out,
    );
    compare_field(
        ev.seq,
        "kvarn_k_zp",
        &raw_field(&cache.kvarn_k_zp, "kvarn_k_zp"),
        &ffi::array_to_raw_bytes(&k_ref.zp),
        out,
    );
    compare_field(
        ev.seq,
        "kvarn_k_s_row",
        &raw_field(&cache.kvarn_k_s_row, "kvarn_k_s_row"),
        &ffi::array_to_raw_bytes(&k_ref.s_row),
        out,
    );
    compare_field(
        ev.seq,
        "kvarn_k_s_col",
        &raw_field(&cache.kvarn_k_s_col, "kvarn_k_s_col"),
        &ffi::array_to_raw_bytes(&k_ref.s_col),
        out,
    );

    // V side (the K8V4 candidate: packed codes + FOLDED params, §3.1).
    compare_field(
        ev.seq,
        "kvarn_hist_v",
        &raw_field(&cache.kvarn_hist_v, "kvarn_hist_v"),
        &ffi::array_to_raw_bytes(&v_ref.q_packed),
        out,
    );
    compare_field(
        ev.seq,
        "kvarn_v_scale",
        &raw_field(&cache.kvarn_v_scale, "kvarn_v_scale"),
        &ffi::array_to_raw_bytes(&v_ref.scale_folded),
        out,
    );
    compare_field(
        ev.seq,
        "kvarn_v_zp",
        &raw_field(&cache.kvarn_v_zp, "kvarn_v_zp"),
        &ffi::array_to_raw_bytes(&v_ref.zp_folded),
        out,
    );
    compare_field(
        ev.seq,
        "kvarn_v_s_col",
        &raw_field(&cache.kvarn_v_s_col, "kvarn_v_s_col"),
        &ffi::array_to_raw_bytes(&v_ref.s_col),
        out,
    );
}

/// THE GATING RUN (ignored: requires the harvest bank; see module docs
/// for invocation). Green means: on the licensed real-tile population,
/// the engine write path stores bit-for-bit what the dual-approved math
/// layer computes — the §4 golden-vector rung. It gates STORAGE only
/// (coverage mapping in module docs).
#[test]
#[ignore]
fn k8v4_golden_harness_real_tiles() {
    // FIRST ACT — structural parity gate (hand-built half-case vector).
    kvarn::assert_round_half_even_parity();

    let dir = std::env::var(GOLDEN_DIR_ENV).unwrap_or_else(|_| {
        panic!(
            "{GOLDEN_DIR_ENV} must point at the harvest bank named by \
             RESULTS_kvarn4_realtile_2026-07-11 (kvarn_harvest_20260710)"
        )
    });
    let dir = std::path::PathBuf::from(dir);
    let events = load_events(&dir);
    let total_tiles: usize = events.iter().map(|e| e.n_tiles as usize).sum();
    assert_eq!(
        (events.len(), total_tiles),
        (MAX_EVENTS, EXPECTED_TILES),
        "population drifted from the licensed bank (RESULTS_kvarn4_realtile_2026-07-11: \
         512 events / 4096 tiles per role) — re-registration must be conscious"
    );

    let mut mismatches = Vec::new();
    for ev in &events {
        run_event(ev, &mut mismatches);
    }

    for m in &mismatches {
        eprintln!(
            "GOLDEN MISMATCH event_seq={} field={} — {}",
            m.event_seq, m.field, m.detail
        );
    }
    let off_min = events.iter().map(|e| e.offset).min().unwrap_or(0);
    let off_max = events.iter().map(|e| e.offset).max().unwrap_or(0);
    println!(
        "golden harness: {} events / {} tiles per role, offsets {off_min}..{off_max}, \
         9 stored fields + 4 structure pins per event, {} mismatches",
        events.len(),
        total_tiles,
        mismatches.len()
    );
    assert!(
        mismatches.is_empty(),
        "{} bitwise divergences between engine storage and the math layer (listed above)",
        mismatches.len()
    );
}

// ---------------------------------------------------------------------------
// Harness self-pins (run in the normal suite, no bank needed): the
// comparator and the population selection must themselves be provably
// able to fail — a gate that cannot go red certifies nothing.
// ---------------------------------------------------------------------------

/// MUST-FAIL ARM: a single flipped byte anywhere in a field is caught.
#[test]
fn golden_comparator_catches_single_byte_divergence() {
    let engine = vec![0u8, 1, 2, 3, 4, 5, 6, 7];
    let mut reference = engine.clone();
    reference[5] ^= 0x10;
    let mut out = Vec::new();
    compare_field(7, "probe", &engine, &reference, &mut out);
    assert_eq!(out.len(), 1, "flipped byte must be reported");
    assert!(out[0].detail.contains("byte 5"), "detail names the position: {}", out[0].detail);

    // And the identical case stays clean — the arm proves red AND green.
    let mut clean = Vec::new();
    compare_field(7, "probe", &engine, &engine.clone(), &mut clean);
    assert!(clean.is_empty(), "identical bytes must not report");
}

/// MUST-FAIL ARM: a truncated field is a reported mismatch, never a
/// zip-shortest false green.
#[test]
fn golden_comparator_refuses_length_mismatch() {
    let engine = vec![9u8; 16];
    let reference = vec![9u8; 12];
    let mut out = Vec::new();
    compare_field(3, "probe", &engine, &reference, &mut out);
    assert_eq!(out.len(), 1, "length mismatch must be reported");
    assert!(out[0].detail.contains("length mismatch"), "{}", out[0].detail);
}

/// Population-selection pin: the f64 truncation replication of
/// `np.linspace(0, n-1, k).astype(int)` agrees with exact integer floor
/// division for the ACTUAL bank geometry (and endpoints hold). numpy
/// computes `j·(n-1)/(k-1)` in f64 and truncates; for n·k at this scale
/// the f64 product error is far below the distance to the nearest
/// integer except where the quotient IS an integer, and gcd(4103, 511)=1
/// makes j=0 and j=511 the only integer quotients — both exact in f64.
#[test]
fn linspace_indices_match_integer_floor_for_the_bank() {
    let (n, k) = (4104usize, 512usize);
    let got = linspace_trunc_indices(n, k);
    assert_eq!(got.len(), k);
    assert_eq!(got[0], 0, "left endpoint");
    assert_eq!(*got.last().unwrap(), n - 1, "right endpoint (inclusive linspace)");
    for (j, &idx) in got.iter().enumerate() {
        assert_eq!(idx, j * (n - 1) / (k - 1), "index {j} diverged from exact floor");
    }
    assert!(got.windows(2).all(|w| w[0] < w[1]), "strictly ascending — distinct events");

    // Degenerate contract: small banks take every event.
    assert_eq!(linspace_trunc_indices(3, 512), vec![0, 1, 2]);
    assert_eq!(linspace_trunc_indices(0, 512), Vec::<usize>::new());
}

/// Feed/reference construction self-check on synthetic tiles (no bank):
/// the harness's own plumbing — unrotate-feed-rotate replication, sink
/// preamble, slicing — reproduces the engine bitwise on a small event.
/// This is the 10222 wiring pin re-proven THROUGH the harness helpers,
/// so a green on real tiles cannot be a green of broken helpers.
#[test]
fn golden_helpers_roundtrip_synthetic_event_bitwise() {
    let n_tiles = 2i32;
    let count = (n_tiles * R * C) as usize;
    let tiles_rot: Vec<f32> = (0..count)
        .map(|i| (((i as i32 * 197 + 13) % 251) as f32) * 0.017 - 2.1)
        .collect();
    let ev = HarvestEvent {
        seq: 0,
        offset: 0,
        k_rot: tiles_rot.clone(),
        v_rot: tiles_rot,
        n_tiles,
    };
    let mut out = Vec::new();
    run_event(&ev, &mut out);
    for m in &out {
        eprintln!("synthetic self-check mismatch: {} — {}", m.field, m.detail);
    }
    assert!(out.is_empty(), "harness helpers must reproduce the engine on synthetic tiles");
}
