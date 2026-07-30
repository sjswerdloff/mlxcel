// Copyright 2026 Lablup Inc. and Jeongkyu Shin
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

//! Content-addressed KV cache cold-storage for fast session restart.
//!
//! Caches are identified by token content, not session IDs. The directory
//! name is a hash of `(model_id, template_sig, token_prefix)`. On restore,
//! the longest matching prefix is found by scanning stored token sequences.
//!
//! This design allows:
//! - Same conversation across requests → same cache (automatic)
//! - Different sessions with shared prefix → shared cache (efficient)
//! - Forked conversations → shared cache until divergence point
//!
//! # Storage layout (v3, via `cold_store_reference.rs`)
//!
//! ```text
//! ~/.cache/mlxcel/cold-storage/
//!   cold-storage-v3/
//!     <hex(identity_sha256)>/            — runtime+model+template+layout+tokens
//!       gen-<pid>-<nanos>-<counter>/     — immutable generation
//!         layer_0000.bin                 — per-layer DetachedKVCache tensors
//!         layer_0001.bin
//!         ...
//!         header.bin                     — identity, per-layer SHA-256 + lengths
//!         COMMITTED                      — atomic publication marker (magic +
//!                                          header SHA-256); written last
//! ```
//!
//! # Safety
//!
//! The runtime fingerprint (derived from the model-weight fingerprint) is part
//! of the identity: a cache written by a different checkpoint never matches.
//! Loads see only fully COMMITTED generations, verify every layer's length and
//! SHA-256 checksum, and run `validate_consistency` before any FFI shape op.

use std::collections::hash_map::DefaultHasher;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};

use cxx::UniquePtr;
use sha2::{Digest, Sha256};

use crate::dtype;
use crate::ffi;
use crate::ffi::MlxArray;

use super::detach::{DetachedCacheSet, DetachedKVCache};
use super::KVCacheMode;

#[path = "cold_store_reference.rs"]
mod reference;
pub use reference::{ReferenceColdStore, ReferenceSnapshot, runtime_fingerprint_from_manifest, PruneMode};
pub(crate) use reference::serialize_cache_set_layers;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// v2 header format: production-dead since persist/load moved to the v3
// reference machinery, retained for the v2 round-trip + bound-fix tests.
#[allow(dead_code)]
const FORMAT_VERSION: u32 = 2; // v2: content-addressed, tokens in header
const DEFAULT_BASE_DIR: &str = ".cache/mlxcel/cold-storage";
const MAX_SERIALIZED_TENSOR_RANK: usize = 32;

/// Header-scan resource bounds, mirroring the reference implementation
/// (cold_store_reference.rs `MAX_TOKENS` / `MAX_LAYERS`). A corrupt header
/// must produce an `Err` the load_prefix scan loop can skip — never an
/// allocation abort that takes down every subsequent restore scan.
#[allow(dead_code)]
const MAX_HEADER_TOKENS: usize = 2_000_000;
#[allow(dead_code)]
const MAX_HEADER_LAYERS: usize = 1024;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum ColdStoreError {
    Io(io::Error),
    WeightMismatch { stored: String, current: String },
    FormatVersionMismatch { stored: u32, current: u32 },
    CorruptLayer { layer: usize, detail: String },
    NoMatch,
    WorkerDied,
}

impl std::fmt::Display for ColdStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "cold-store I/O: {e}"),
            Self::WeightMismatch { stored, current } => {
                write!(f, "cold-store weight mismatch: stored={stored}, current={current}")
            }
            Self::FormatVersionMismatch { stored, current } => {
                write!(f, "cold-store format version mismatch: stored={stored}, current={current}")
            }
            Self::CorruptLayer { layer, detail } => {
                write!(f, "cold-store corrupt layer {layer}: {detail}")
            }
            Self::NoMatch => write!(f, "cold-store: no matching prefix found"),
            Self::WorkerDied => write!(f, "cold-store background writer died"),
        }
    }
}

impl std::error::Error for ColdStoreError {}

impl From<io::Error> for ColdStoreError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

// ---------------------------------------------------------------------------
// Content hashing
// ---------------------------------------------------------------------------

/// Compute a content hash for a token prefix.
///
/// The hash incorporates model_id and template_sig so that the same tokens
/// under different models or templates get different cache entries.
pub fn compute_content_hash(model_id: &str, template_sig: &str, tokens: &[i32]) -> String {
    let mut hasher = DefaultHasher::new();
    model_id.hash(&mut hasher);
    template_sig.hash(&mut hasher);
    // Hash a bounded prefix to keep hashing fast for long conversations.
    // The full token sequence is stored in the header for exact matching.
    let prefix_len = tokens.len().min(4096);
    tokens[..prefix_len].hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Compute a weight fingerprint for the currently loaded model.
pub fn compute_weight_fingerprint(model_path: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(model_path.as_bytes());
    if let Ok(entries) = fs::read_dir(model_path) {
        let mut files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        files.sort_by_key(|entry| entry.file_name());
        let mut buffer = vec![0u8; 1024 * 1024];
        for entry in files {
            hasher.update(entry.file_name().to_string_lossy().as_bytes());
            match File::open(entry.path()) {
                Ok(mut file) => loop {
                    match file.read(&mut buffer) {
                        Ok(0) => break,
                        Ok(count) => hasher.update(&buffer[..count]),
                        Err(error) => {
                            hasher.update(format!("read-error:{error}").as_bytes());
                            break;
                        }
                    }
                },
                Err(error) => hasher.update(format!("open-error:{error}").as_bytes()),
            }
        }
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

// ---------------------------------------------------------------------------
// Tensor serialization
// ---------------------------------------------------------------------------

fn write_array(w: &mut impl Write, arr: &MlxArray) -> io::Result<()> {
    let dt = ffi::array_dtype(arr);
    let shape = ffi::array_shape(arr);
    let ndim = shape.len() as i32;
    let bytes = ffi::array_to_raw_bytes(arr);
    w.write_all(&dt.to_le_bytes())?;
    w.write_all(&ndim.to_le_bytes())?;
    for &d in &shape {
        w.write_all(&d.to_le_bytes())?;
    }
    w.write_all(&(bytes.len() as u64).to_le_bytes())?;
    w.write_all(&bytes)?;
    Ok(())
}

fn read_array(r: &mut impl Read) -> io::Result<UniquePtr<MlxArray>> {
    let dt = read_i32(r)?;
    let ndim = usize::try_from(read_i32(r)?).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidData, "negative tensor rank")
    })?;
    if ndim > MAX_SERIALIZED_TENSOR_RANK {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("tensor rank {ndim} exceeds limit {MAX_SERIALIZED_TENSOR_RANK}"),
        ));
    }
    let mut shape = vec![0i32; ndim];
    for s in &mut shape {
        *s = read_i32(r)?;
    }
    let byte_len = usize::try_from(read_u64(r)?).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidData, "tensor byte length exceeds usize")
    })?;
    validate_array_metadata(byte_len, dt, &shape)?;
    if byte_len > isize::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("cannot allocate {byte_len}-byte tensor payload"),
        ));
    }
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(byte_len).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("cannot allocate {byte_len}-byte tensor payload: {error}"),
        )
    })?;
    bytes.resize(byte_len, 0);
    r.read_exact(&mut bytes)?;
    array_from_raw_bytes(&bytes, dt, &shape)
}

fn write_opt_array(w: &mut impl Write, opt: &Option<UniquePtr<MlxArray>>) -> io::Result<()> {
    match opt {
        Some(arr) => {
            w.write_all(&[1u8])?;
            write_array(w, arr)?;
        }
        None => {
            w.write_all(&[0u8])?;
        }
    }
    Ok(())
}

fn read_opt_array(r: &mut impl Read) -> io::Result<Option<UniquePtr<MlxArray>>> {
    let mut tag = [0u8; 1];
    r.read_exact(&mut tag)?;
    if tag[0] == 0 {
        Ok(None)
    } else {
        Ok(Some(read_array(r)?))
    }
}

// ---------------------------------------------------------------------------
// DetachedKVCache serialization
// ---------------------------------------------------------------------------

/// Collect raw pointers to all populated arrays in a DetachedKVCache.
/// Used to batch-evaluate all arrays before serialization (single GPU→CPU sync).
fn collect_array_ptrs(cache: &DetachedKVCache, out: &mut Vec<*const MlxArray>) {
    let arrays: Vec<Option<&UniquePtr<MlxArray>>> = vec![
        cache.keys.as_ref(),
        cache.values.as_ref(),
        cache.key_scales.as_ref(),
        cache.val_scales.as_ref(),
        cache.v_packed.as_ref(),
        cache.v_norms.as_ref(),
        cache.v_rescale.as_ref(),
        cache.k_packed.as_ref(),
        cache.k_norms.as_ref(),
        cache.m3_idx_k.as_ref(),
        cache.kvarn_sink_k.as_ref(),
        cache.kvarn_sink_v.as_ref(),
        cache.kvarn_tail_k.as_ref(),
        cache.kvarn_tail_v.as_ref(),
        cache.kvarn_hist_k.as_ref(),
        cache.kvarn_hist_v.as_ref(),
        cache.kvarn_k_scale.as_ref(),
        cache.kvarn_k_zp.as_ref(),
        cache.kvarn_k_s_row.as_ref(),
        cache.kvarn_k_s_col.as_ref(),
        cache.kvarn_v_scale.as_ref(),
        cache.kvarn_v_zp.as_ref(),
        cache.kvarn_v_s_row.as_ref(),
        cache.kvarn_v_s_col.as_ref(),
    ];
    for arr in arrays.into_iter().flatten() {
        out.push(arr.as_ptr() as *const _);
    }
}

fn write_kv_cache(w: &mut impl Write, cache: &DetachedKVCache) -> io::Result<()> {
    w.write_all(&(cache.mode as u8).to_le_bytes())?;
    w.write_all(&cache.offset.to_le_bytes())?;
    w.write_all(&cache.step.to_le_bytes())?;
    w.write_all(&cache.turbo_seed.to_le_bytes())?;
    w.write_all(&cache.cold_offset.to_le_bytes())?;
    w.write_all(&cache.hot_threshold.to_le_bytes())?;
    w.write_all(&(cache.delegated_fp16_fast_path as u8).to_le_bytes())?;
    w.write_all(&(cache.delegated_fp16_sidecar_policy as u8).to_le_bytes())?;
    w.write_all(&cache.kvarn_v_bits.to_le_bytes())?;
    w.write_all(&cache.m3_idx_offset.to_le_bytes())?;
    write_opt_array(w, &cache.keys)?;
    write_opt_array(w, &cache.values)?;
    write_opt_array(w, &cache.key_scales)?;
    write_opt_array(w, &cache.val_scales)?;
    write_opt_array(w, &cache.v_packed)?;
    write_opt_array(w, &cache.v_norms)?;
    write_opt_array(w, &cache.v_rescale)?;
    write_opt_array(w, &cache.k_packed)?;
    write_opt_array(w, &cache.k_norms)?;
    write_opt_array(w, &cache.m3_idx_k)?;
    write_opt_array(w, &cache.kvarn_sink_k)?;
    write_opt_array(w, &cache.kvarn_sink_v)?;
    write_opt_array(w, &cache.kvarn_tail_k)?;
    write_opt_array(w, &cache.kvarn_tail_v)?;
    write_opt_array(w, &cache.kvarn_hist_k)?;
    write_opt_array(w, &cache.kvarn_hist_v)?;
    write_opt_array(w, &cache.kvarn_k_scale)?;
    write_opt_array(w, &cache.kvarn_k_zp)?;
    write_opt_array(w, &cache.kvarn_k_s_row)?;
    write_opt_array(w, &cache.kvarn_k_s_col)?;
    write_opt_array(w, &cache.kvarn_v_scale)?;
    write_opt_array(w, &cache.kvarn_v_zp)?;
    write_opt_array(w, &cache.kvarn_v_s_row)?;
    write_opt_array(w, &cache.kvarn_v_s_col)?;
    Ok(())
}

pub fn read_kv_cache(r: &mut impl Read, layer_idx: usize) -> Result<DetachedKVCache, ColdStoreError> {
    let mode_tag = read_u8(r)?;
    let offset = read_i32(r)?;
    let step = read_i32(r)?;
    let turbo_seed = read_u32(r)?;
    let cold_offset = read_i32(r)?;
    let hot_threshold = read_i32(r)?;
    let delegated_fp16_fast_path = read_u8(r)? != 0;
    let sidecar_policy_raw = read_u8(r)?;
    let kvarn_v_bits = read_u8(r)?;
    let m3_idx_offset = read_i32(r)?;

    let mode = match mode_tag {
        0 => KVCacheMode::Fp16,
        1 => KVCacheMode::Int8,
        2 => KVCacheMode::Turbo4Asym,
        3 => KVCacheMode::Turbo3Asym,
        4 => KVCacheMode::Turbo4,
        5 => KVCacheMode::Turbo4Delegated,
        6 => KVCacheMode::KVarN8,
        _ => return Err(ColdStoreError::CorruptLayer {
            layer: layer_idx,
            detail: format!("unknown mode tag: {mode_tag}"),
        }),
    };

    let delegated_fp16_sidecar_policy = match sidecar_policy_raw {
        0 => super::turbo::DelegatedFp16SidecarPolicy::Predecode,
        1 => super::turbo::DelegatedFp16SidecarPolicy::Lazy,
        _ => return Err(ColdStoreError::CorruptLayer {
            layer: layer_idx,
            detail: format!("unknown sidecar policy: {sidecar_policy_raw}"),
        }),
    };

    Ok(DetachedKVCache {
        keys: read_opt_array(r)?,
        values: read_opt_array(r)?,
        offset,
        step,
        mode,
        key_scales: read_opt_array(r)?,
        val_scales: read_opt_array(r)?,
        v_packed: read_opt_array(r)?,
        v_norms: read_opt_array(r)?,
        v_rescale: read_opt_array(r)?,
        k_packed: read_opt_array(r)?,
        k_norms: read_opt_array(r)?,
        turbo_seed,
        cold_offset,
        hot_threshold,
        delegated_fp16_fast_path,
        delegated_fp16_sidecar_policy,
        m3_idx_k: read_opt_array(r)?,
        m3_idx_offset,
        kvarn_sink_k: read_opt_array(r)?,
        kvarn_sink_v: read_opt_array(r)?,
        kvarn_tail_k: read_opt_array(r)?,
        kvarn_tail_v: read_opt_array(r)?,
        kvarn_hist_k: read_opt_array(r)?,
        kvarn_hist_v: read_opt_array(r)?,
        kvarn_k_scale: read_opt_array(r)?,
        kvarn_k_zp: read_opt_array(r)?,
        kvarn_k_s_row: read_opt_array(r)?,
        kvarn_k_s_col: read_opt_array(r)?,
        kvarn_v_scale: read_opt_array(r)?,
        kvarn_v_zp: read_opt_array(r)?,
        kvarn_v_s_row: read_opt_array(r)?,
        kvarn_v_s_col: read_opt_array(r)?,
        kvarn_v_bits,
    })
}

// ---------------------------------------------------------------------------
// Post-read semantic validation (PRE-FFI guard)
// ---------------------------------------------------------------------------

/// Per-layer semantic + shape validation of a freshly deserialized
/// [`DetachedKVCache`]. MUST run after [`read_kv_cache`] and BEFORE the
/// cache is adopted (`install_detached`) or fetched.
///
/// Why this exists (issue-6): the layer format is positional with no
/// cross-field validation, and the downstream first-use paths assume these
/// invariants without checking:
/// * `fetch_kvarn8` (cache.rs:1522) panics on `kvarn_v_bits ∉ {8,4}`;
/// * the assemble closures `.unwrap()` the hist sidecars (cache.rs:1408-1412,
///   1467-1472) and index `shape[0..4]` unchecked (cache.rs:1414/1461);
/// * `ffi::slice`/`reshape`/`concatenate` are bare-`UniquePtr` cxx calls —
///   an MLX shape exception there is `std::terminate`, uncatchable by both
///   `Err => continue` AND `catch_unwind`. The abort class can only be
///   PREVENTED, which is this function's job.
///
/// Only `ffi::array_shape` / `ffi::array_dtype` (metadata reads) are called
/// here — no shape-sensitive MLX op touches the tensors before validation
/// passes. Every violation is `ColdStoreError::CorruptLayer`; this function
/// never panics on any input.
pub fn validate_consistency(
    cache: &DetachedKVCache,
    layer_idx: usize,
) -> Result<(), ColdStoreError> {
    use super::kvarn::{KVARN_TILE_TOKENS, KVARN_V4_GROUP_SIZE};

    let corrupt = |detail: String| ColdStoreError::CorruptLayer {
        layer: layer_idx,
        detail,
    };

    // ── Scalar sanity ────────────────────────────────────────────────────
    if cache.offset < 0 {
        return Err(corrupt(format!("negative offset {}", cache.offset)));
    }
    if cache.m3_idx_offset < 0 {
        return Err(corrupt(format!(
            "negative m3_idx_offset {}",
            cache.m3_idx_offset
        )));
    }

    // ── Rank guard ───────────────────────────────────────────────────────
    // Every tensor in this format is rank 4 with seq axis 2 (detach.rs
    // trim paths, cache.rs append paths, install_detached's own
    // `shape.len() == 4` probe). A wrong-rank tensor reaching ffi::slice
    // aborts the process; refuse it here.
    let rank4 = |name: &str,
                 opt: &Option<UniquePtr<MlxArray>>|
     -> Result<Option<Vec<i32>>, ColdStoreError> {
        match opt {
            None => Ok(None),
            Some(a) => {
                let s = ffi::array_shape(a);
                if s.len() != 4 {
                    Err(corrupt(format!(
                        "{name}: expected rank 4, got rank {} (shape {s:?})",
                        s.len()
                    )))
                } else {
                    Ok(Some(s))
                }
            }
        }
    };

    let keys = rank4("keys", &cache.keys)?;
    let values = rank4("values", &cache.values)?;
    let key_scales = rank4("key_scales", &cache.key_scales)?;
    let val_scales = rank4("val_scales", &cache.val_scales)?;
    let _v_packed = rank4("v_packed", &cache.v_packed)?;
    let _v_norms = rank4("v_norms", &cache.v_norms)?;
    let _v_rescale = rank4("v_rescale", &cache.v_rescale)?;
    let _k_packed = rank4("k_packed", &cache.k_packed)?;
    let _k_norms = rank4("k_norms", &cache.k_norms)?;
    let m3 = rank4("m3_idx_k", &cache.m3_idx_k)?;
    let sink_k = rank4("kvarn_sink_k", &cache.kvarn_sink_k)?;
    let sink_v = rank4("kvarn_sink_v", &cache.kvarn_sink_v)?;
    let tail_k = rank4("kvarn_tail_k", &cache.kvarn_tail_k)?;
    let tail_v = rank4("kvarn_tail_v", &cache.kvarn_tail_v)?;
    let hist_k = rank4("kvarn_hist_k", &cache.kvarn_hist_k)?;
    let hist_v = rank4("kvarn_hist_v", &cache.kvarn_hist_v)?;
    let k_scale = rank4("kvarn_k_scale", &cache.kvarn_k_scale)?;
    let k_zp = rank4("kvarn_k_zp", &cache.kvarn_k_zp)?;
    let k_s_row = rank4("kvarn_k_s_row", &cache.kvarn_k_s_row)?;
    let k_s_col = rank4("kvarn_k_s_col", &cache.kvarn_k_s_col)?;
    let v_scale = rank4("kvarn_v_scale", &cache.kvarn_v_scale)?;
    let v_zp = rank4("kvarn_v_zp", &cache.kvarn_v_zp)?;
    let v_s_row = rank4("kvarn_v_s_row", &cache.kvarn_v_s_row)?;
    let v_s_col = rank4("kvarn_v_s_col", &cache.kvarn_v_s_col)?;

    // ── M3 indexer (mode-agnostic) ───────────────────────────────────────
    // Detached contract: exact-length `[b, 1, m3_idx_offset, index_dim]`
    // (clone_handle slice-to-fill, detach.rs:701-708; field doc :136).
    // A desync here is the cycle-79 asymmetric-reshape crash class.
    if let Some(s) = &m3 {
        if s[1] != 1 {
            return Err(corrupt(format!(
                "m3_idx_k: head axis must be 1, got {} (shape {s:?})",
                s[1]
            )));
        }
        if s[2] != cache.m3_idx_offset {
            return Err(corrupt(format!(
                "m3_idx_k seq length {} != m3_idx_offset {}",
                s[2], cache.m3_idx_offset
            )));
        }
    }

    // ── Mode-specific ────────────────────────────────────────────────────
    let kvarn_fields_present = sink_k.is_some()
        || sink_v.is_some()
        || tail_k.is_some()
        || tail_v.is_some()
        || hist_k.is_some()
        || hist_v.is_some()
        || k_scale.is_some()
        || k_zp.is_some()
        || k_s_row.is_some()
        || k_s_col.is_some()
        || v_scale.is_some()
        || v_zp.is_some()
        || v_s_row.is_some()
        || v_s_col.is_some();

    if cache.mode != KVCacheMode::KVarN8 && kvarn_fields_present {
        // "KVarN8 state (mode == KVCacheMode::KVarN8; all None otherwise)"
        // — cache.rs:631. Kvarn droppings under another mode mean the mode
        // byte and the tensor table disagree: reject, don't guess which lies.
        return Err(corrupt(format!(
            "kvarn tensors present under non-KVarN8 mode {:?}",
            cache.mode
        )));
    }

    match cache.mode {
        KVCacheMode::Fp16 => {
            if cache.offset > 0 {
                let k = keys
                    .as_ref()
                    .ok_or_else(|| corrupt("Fp16: keys missing with offset > 0".into()))?;
                let v = values
                    .as_ref()
                    .ok_or_else(|| corrupt("Fp16: values missing with offset > 0".into()))?;
                // keys/values are CAPACITY buffers (clone_handle moves the
                // step-grown buffer unsliced): capacity >= offset, batch and
                // head axes in lockstep. capacity < offset would make the
                // downstream fetch slice read out of bounds → MLX abort.
                if k[2] < cache.offset || v[2] < cache.offset {
                    return Err(corrupt(format!(
                        "Fp16: seq capacity (K {}, V {}) below offset {}",
                        k[2], v[2], cache.offset
                    )));
                }
                if k[0] != v[0] || k[1] != v[1] {
                    return Err(corrupt(format!(
                        "Fp16: K/V batch-head mismatch ({:?} vs {:?})",
                        &k[..2],
                        &v[..2]
                    )));
                }
            }
        }
        KVCacheMode::Int8 => {
            if cache.offset > 0 {
                let k = keys
                    .as_ref()
                    .ok_or_else(|| corrupt("Int8: keys missing with offset > 0".into()))?;
                let v = values
                    .as_ref()
                    .ok_or_else(|| corrupt("Int8: values missing with offset > 0".into()))?;
                let ks = key_scales.as_ref().ok_or_else(|| {
                    corrupt("Int8: key_scales missing with offset > 0".into())
                })?;
                let vs = val_scales.as_ref().ok_or_else(|| {
                    corrupt("Int8: val_scales missing with offset > 0".into())
                })?;
                for (name, s) in [("keys", k), ("values", v), ("key_scales", ks), ("val_scales", vs)]
                {
                    if s[2] < cache.offset {
                        return Err(corrupt(format!(
                            "Int8: {name} seq capacity {} below offset {}",
                            s[2], cache.offset
                        )));
                    }
                }
            }
        }
        KVCacheMode::KVarN8 => {
            // The single validated gate for the cache.rs:1522 panic class:
            // kvarn_v_bits comes off disk unvalidated; a flipped byte is a
            // guaranteed panic on first fetch_kvarn8 without this check.
            if cache.kvarn_v_bits != 8 && cache.kvarn_v_bits != 4 {
                return Err(corrupt(format!(
                    "kvarn_v_bits must be 8 or 4, got {}",
                    cache.kvarn_v_bits
                )));
            }

            // K/V pairing: the writers fill sink/hist/tail K and V in
            // lockstep (cache.rs:1217-1224, 1317-1358; trim_to slices in
            // lockstep). A lone side would panic/abort in assemble.
            for (name, a, b) in [
                ("kvarn_sink_k/kvarn_sink_v", sink_k.is_some(), sink_v.is_some()),
                ("kvarn_hist_k/kvarn_hist_v", hist_k.is_some(), hist_v.is_some()),
                ("kvarn_tail_k/kvarn_tail_v", tail_k.is_some(), tail_v.is_some()),
            ] {
                if a != b {
                    return Err(corrupt(format!("{name} presence mismatch")));
                }
            }

            // Reference geometry (b, h, d) from the first present tensor.
            let geom = sink_k.as_ref().or(tail_k.as_ref()).or(hist_k.as_ref());
            let (b, h, d) = match geom {
                Some(s) => (s[0], s[1], s[3]),
                None => {
                    // Empty kvarn state is only consistent with offset 0.
                    if cache.offset != 0 {
                        return Err(corrupt(format!(
                            "KVarN8: offset {} with no kvarn tensors \
                             (fetch would panic on an empty cache)",
                            cache.offset
                        )));
                    }
                    return Ok(());
                }
            };
            // v4 hist_k carries full D; d itself must be group-divisible
            // for the v4 dequant reshape (checked under the v4 arm below).

            let check_bhd = |name: &str, s: &Option<Vec<i32>>| -> Result<(), ColdStoreError> {
                if let Some(s) = s {
                    if s[0] != b || s[1] != h || s[3] != d {
                        return Err(corrupt(format!(
                            "{name}: shape {s:?} inconsistent with [b={b}, h={h}, ·, d={d}]"
                        )));
                    }
                }
                Ok(())
            };
            check_bhd("kvarn_sink_k", &sink_k)?;
            check_bhd("kvarn_sink_v", &sink_v)?;
            check_bhd("kvarn_tail_k", &tail_k)?;
            check_bhd("kvarn_tail_v", &tail_v)?;
            check_bhd("kvarn_hist_k", &hist_k)?;

            // Sink: fills first, capped at one tile, never quantized
            // (cache.rs:1200-1231; the #36 cap does not stop the sink fill)
            // ⇒ sink_len == min(offset, KVARN_TILE_TOKENS) always.
            let sink_len = sink_k.as_ref().map_or(0, |s| s[2]);
            if let (Some(sk), Some(sv)) = (&sink_k, &sink_v) {
                if sk[2] != sv[2] {
                    return Err(corrupt(format!(
                        "sink K len {} != sink V len {}",
                        sk[2], sv[2]
                    )));
                }
            }
            if cache.offset > 0 && sink_k.is_none() {
                return Err(corrupt(
                    "KVarN8: sink missing with offset > 0 (sink fills first)".into(),
                ));
            }
            if sink_len != cache.offset.min(KVARN_TILE_TOKENS) {
                return Err(corrupt(format!(
                    "sink length {} != min(offset {}, tile {})",
                    sink_len, cache.offset, KVARN_TILE_TOKENS
                )));
            }

            // Tail: present ⇒ non-empty, K/V lengths in lockstep.
            let tail_len = tail_k.as_ref().map_or(0, |s| s[2]);
            if let (Some(tk), Some(tv)) = (&tail_k, &tail_v) {
                if tk[2] != tv[2] {
                    return Err(corrupt(format!(
                        "tail K len {} != tail V len {}",
                        tk[2], tv[2]
                    )));
                }
                if tk[2] <= 0 {
                    return Err(corrupt("tail present but empty".into()));
                }
            }

            // History + sidecars.
            let hist_len = hist_k.as_ref().map_or(0, |s| s[2]);
            if let Some(hk) = &hist_k {
                let t = hk[2];
                if t <= 0 || t % KVARN_TILE_TOKENS != 0 {
                    return Err(corrupt(format!(
                        "hist length {t} not a positive multiple of tile {KVARN_TILE_TOKENS}"
                    )));
                }
                let n_tiles = t / KVARN_TILE_TOKENS;

                // K side is 8-bit in BOTH V widths (cache.rs:1313-1315).
                if ffi::array_dtype(cache.kvarn_hist_k.as_ref().unwrap()) != dtype::UINT8 {
                    return Err(corrupt("kvarn_hist_k: expected u8 codes".into()));
                }
                let require = |name: &str,
                               s: &Option<Vec<i32>>|
                 -> Result<Vec<i32>, ColdStoreError> {
                    s.clone()
                        .ok_or_else(|| corrupt(format!("{name} missing while hist present")))
                };
                // Per-token row params [b,h,T_hist,1] in lockstep with hist
                // dim2; per-tile s_col [b,h,n_tiles,d] (cache.rs:643-646,
                // 1317-1323). assemble unwraps these (cache.rs:1408-1412).
                for (name, s) in [
                    ("kvarn_k_scale", require("kvarn_k_scale", &k_scale)?),
                    ("kvarn_k_zp", require("kvarn_k_zp", &k_zp)?),
                    ("kvarn_k_s_row", require("kvarn_k_s_row", &k_s_row)?),
                ] {
                    if s != vec![b, h, t, 1] {
                        return Err(corrupt(format!(
                            "{name}: shape {s:?} != [{b}, {h}, {t}, 1] (hist lockstep)"
                        )));
                    }
                }
                let sc = require("kvarn_k_s_col", &k_s_col)?;
                if sc != vec![b, h, n_tiles, d] {
                    return Err(corrupt(format!(
                        "kvarn_k_s_col: shape {sc:?} != [{b}, {h}, n_tiles={n_tiles}, {d}]"
                    )));
                }

                // V side per kvarn_v_bits (cache.rs:1324-1359, :678-684).
                let hv = hist_v
                    .as_ref()
                    .expect("pairing check guarantees hist_v when hist_k present");
                if hv[2] != t {
                    return Err(corrupt(format!(
                        "hist V length {} != hist K length {t} \
                         (K and V tiles finalize together)",
                        hv[2]
                    )));
                }
                match cache.kvarn_v_bits {
                    8 => {
                        if hv[3] != d {
                            return Err(corrupt(format!(
                                "kvarn_hist_v (v8): trailing dim {} != d {d}",
                                hv[3]
                            )));
                        }
                        if ffi::array_dtype(cache.kvarn_hist_v.as_ref().unwrap())
                            != dtype::UINT8
                        {
                            return Err(corrupt("kvarn_hist_v (v8): expected u8 codes".into()));
                        }
                        for (name, s) in [
                            ("kvarn_v_scale", require("kvarn_v_scale", &v_scale)?),
                            ("kvarn_v_zp", require("kvarn_v_zp", &v_zp)?),
                            ("kvarn_v_s_row", require("kvarn_v_s_row", &v_s_row)?),
                        ] {
                            if s != vec![b, h, t, 1] {
                                return Err(corrupt(format!(
                                    "{name}: shape {s:?} != [{b}, {h}, {t}, 1] (hist lockstep)"
                                )));
                            }
                        }
                    }
                    4 => {
                        // Packed u32 nibbles [B,H,T,D/8]; folded per-group
                        // params [B,H,T,D/gs]; s_row MUST be None (the fold
                        // IS its storage — kvarn.rs:476-481). A v4 entry
                        // with s_row present is a mislabeled/mixed record.
                        if d % 8 != 0 || d % KVARN_V4_GROUP_SIZE != 0 {
                            return Err(corrupt(format!(
                                "v4: head_dim {d} not divisible by 8 and \
                                 group size {KVARN_V4_GROUP_SIZE}"
                            )));
                        }
                        if hv[3] != d / 8 {
                            return Err(corrupt(format!(
                                "kvarn_hist_v (v4): trailing dim {} != d/8 = {}",
                                hv[3],
                                d / 8
                            )));
                        }
                        if ffi::array_dtype(cache.kvarn_hist_v.as_ref().unwrap())
                            != dtype::UINT32
                        {
                            return Err(corrupt(
                                "kvarn_hist_v (v4): expected packed u32 codes".into(),
                            ));
                        }
                        let g = d / KVARN_V4_GROUP_SIZE;
                        for (name, s) in [
                            ("kvarn_v_scale", require("kvarn_v_scale", &v_scale)?),
                            ("kvarn_v_zp", require("kvarn_v_zp", &v_zp)?),
                        ] {
                            if s != vec![b, h, t, g] {
                                return Err(corrupt(format!(
                                    "{name}: shape {s:?} != [{b}, {h}, {t}, {g}] \
                                     (folded per-group lockstep)"
                                )));
                            }
                        }
                        if v_s_row.is_some() {
                            return Err(corrupt(
                                "kvarn_v_s_row present under v_bits=4 \
                                 (the fold IS its storage; must be None)"
                                    .into(),
                            ));
                        }
                    }
                    _ => unreachable!("v_bits validated above"),
                }
                let vsc = require("kvarn_v_s_col", &v_s_col)?;
                if vsc != vec![b, h, n_tiles, d] {
                    return Err(corrupt(format!(
                        "kvarn_v_s_col: shape {vsc:?} != [{b}, {h}, n_tiles={n_tiles}, {d}]"
                    )));
                }
            } else {
                // No hist ⇒ no orphan sidecars (no writer produces that
                // state; trim_to clears them together, detach.rs:527-536).
                for (name, present) in [
                    ("kvarn_k_scale", k_scale.is_some()),
                    ("kvarn_k_zp", k_zp.is_some()),
                    ("kvarn_k_s_row", k_s_row.is_some()),
                    ("kvarn_k_s_col", k_s_col.is_some()),
                    ("kvarn_v_scale", v_scale.is_some()),
                    ("kvarn_v_zp", v_zp.is_some()),
                    ("kvarn_v_s_row", v_s_row.is_some()),
                    ("kvarn_v_s_col", v_s_col.is_some()),
                ] {
                    if present {
                        return Err(corrupt(format!("{name} present without history")));
                    }
                }
            }

            // THE window-sum invariant (trim_to arithmetic detach.rs:540-541,
            // update accounting cache.rs:1226/1381, synth cache.rs:2054-2115):
            // offset == sink_len + hist_len + tail_len. Violation means the
            // assembled attention window disagrees with the logical length —
            // silent wrong attention or an abort at the concat/adopt boundary.
            if cache.offset != sink_len + hist_len + tail_len {
                return Err(corrupt(format!(
                    "offset {} != sink {} + hist {} + tail {}",
                    cache.offset, sink_len, hist_len, tail_len
                )));
            }
        }
        KVCacheMode::Turbo4Asym
        | KVCacheMode::Turbo3Asym
        | KVCacheMode::Turbo4
        | KVCacheMode::Turbo4Delegated => {
            // Rank-4 guards above already cover the abort class for these
            // modes' tensors. Presence contracts are NOT asserted here:
            // the delegated fast-path variants condition the interpretation
            // of `values`/`v_packed` on `delegated_fp16_fast_path` and
            // sidecar policy (detach.rs:92-104, 404-434), and I could not
            // confirm a single presence rule across all of them from source.
            if cache.mode == KVCacheMode::Turbo4Delegated
                && (cache.cold_offset < 0 || cache.cold_offset > cache.offset)
            {
                return Err(corrupt(format!(
                    "Turbo4Delegated: cold_offset {} outside [0, offset {}]",
                    cache.cold_offset, cache.offset
                )));
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Session header (v2: content-addressed)
//
// Production-dead since persist/load moved to the v3 reference machinery.
// Retained (allow(dead_code)) because the v2 round-trip test and the
// read_header bound-fix regression tests still exercise it.
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct SessionHeader {
    format_version: u32,
    content_hash: String,
    weight_fingerprint: String,
    model_id: String,
    template_sig: String,
    layer_count: u32,
    prompt_len: usize,
    current_offset: i32,
    timestamp_secs: u64,
    tokens: Vec<i32>,
}

#[allow(dead_code)]
fn write_header(w: &mut impl Write, hdr: &SessionHeader) -> io::Result<()> {
    w.write_all(&hdr.format_version.to_le_bytes())?;
    write_string(w, &hdr.content_hash)?;
    write_string(w, &hdr.weight_fingerprint)?;
    write_string(w, &hdr.model_id)?;
    write_string(w, &hdr.template_sig)?;
    w.write_all(&hdr.layer_count.to_le_bytes())?;
    w.write_all(&(hdr.prompt_len as u64).to_le_bytes())?;
    w.write_all(&hdr.current_offset.to_le_bytes())?;
    w.write_all(&hdr.timestamp_secs.to_le_bytes())?;
    // Token sequence for prefix matching.
    w.write_all(&(hdr.tokens.len() as u64).to_le_bytes())?;
    for &tok in &hdr.tokens {
        w.write_all(&tok.to_le_bytes())?;
    }
    Ok(())
}

#[allow(dead_code)]
fn read_header(r: &mut impl Read) -> Result<SessionHeader, ColdStoreError> {
    let format_version = read_u32(r)?;
    if format_version != FORMAT_VERSION {
        return Err(ColdStoreError::FormatVersionMismatch {
            stored: format_version,
            current: FORMAT_VERSION,
        });
    }
    let content_hash = read_string(r)?;
    let weight_fingerprint = read_string(r)?;
    let model_id = read_string(r)?;
    let template_sig = read_string(r)?;
    let layer_count = read_u32(r)?;
    bounded_usize(u64::from(layer_count), MAX_HEADER_LAYERS, "layer count")?;
    let prompt_len = bounded_usize(read_u64(r)?, MAX_HEADER_TOKENS, "prompt length")?;
    let current_offset = read_i32(r)?;
    let timestamp_secs = read_u64(r)?;
    // A flipped/hostile token count must NOT reach `vec![0i32; n]`: a huge n
    // is an allocation abort (handle_alloc_error), which `Err => continue`
    // in the scan loop cannot catch — one corrupt header would kill every
    // restore scan. Bound first, then fallibly reserve.
    let token_count = bounded_usize(read_u64(r)?, MAX_HEADER_TOKENS, "token count")?;
    let mut tokens = Vec::new();
    tokens.try_reserve_exact(token_count).map_err(|error| {
        ColdStoreError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("cannot allocate {token_count} tokens: {error}"),
        ))
    })?;
    for _ in 0..token_count {
        tokens.push(read_i32(r)?);
    }
    Ok(SessionHeader {
        format_version,
        content_hash,
        weight_fingerprint,
        model_id,
        template_sig,
        layer_count,
        prompt_len,
        current_offset,
        timestamp_secs,
        tokens,
    })
}

// ---------------------------------------------------------------------------
// ColdStore
// ---------------------------------------------------------------------------

/// Content-addressed cold-storage for KV caches.
///
/// Caches are identified by token content, not session IDs. On restore,
/// the longest matching prefix is found by scanning stored token sequences.
///
/// Persist/restore is routed through the v3 [`ReferenceColdStore`] machinery:
/// per-layer SHA-256 checksums, atomic `COMMITTED` publication, full-token +
/// layout identity, and bounded/validated loads. The Q3 thread-affinity split
/// is preserved: serialization (`MlxArray` -> bytes) happens on the calling
/// (inference) thread in [`ColdStore::persist`]; only `Vec<u8>` crosses to the
/// background writer, which publishes via
/// [`ReferenceColdStore::persist_serialized`].
pub struct ColdStore {
    base_dir: PathBuf,
    #[allow(dead_code)] // retained model-weight identity; folded into ref_store's runtime fingerprint
    weight_fingerprint: String,
    ref_store: std::sync::Arc<ReferenceColdStore>,
    writer_tx: Option<Sender<WriteJob>>,
    writer_handle: Option<JoinHandle<()>>,
}

/// Pre-serialized persist work handed to the background writer.
///
/// Q3 constraint: `MlxArray` -> raw bytes is thread-affine under Metal, so
/// only already-serialized `Vec<u8>` plus identity inputs may cross threads.
struct WriteJob {
    model_id: String,
    template_sig: String,
    covered_tokens: Vec<i32>,
    prompt_len: usize,
    layout_fingerprint: [u8; 32],
    layer_bytes: Vec<Vec<u8>>,
}

impl ColdStore {
    pub fn new(model_path: &str) -> Self {
        Self::with_base_dir(default_base_dir(), model_path)
    }

    pub fn with_base_dir(base_dir: PathBuf, model_path: &str) -> Self {
        let weight_fingerprint = compute_weight_fingerprint(model_path);
        // Known scope gap: the runtime manifest currently captures only weight
        // identity (model path + safetensors bytes). Backend-dtype policy and
        // MLX/Metal build identity are NOT yet folded in. Mode + kvarn layout
        // ARE covered: layout_fingerprint is folded into the v3 identity hash,
        // so a mode mismatch already fails to match.
        // TODO(fuller-manifest): extend the manifest with backend dtype policy
        // and MLX build identity.
        let runtime_fingerprint =
            runtime_fingerprint_from_manifest(weight_fingerprint.as_bytes());
        let ref_store = std::sync::Arc::new(ReferenceColdStore::new(
            base_dir.clone(),
            runtime_fingerprint,
        ));
        let (tx, handle) = spawn_writer(std::sync::Arc::clone(&ref_store));
        ColdStore {
            base_dir,
            weight_fingerprint,
            ref_store,
            writer_tx: Some(tx),
            writer_handle: Some(handle),
        }
    }

    /// Persist a DetachedCacheSet to SSD, keyed by token content.
    ///
    /// `tokens` contains the full model-visible history available at donation;
    /// persistence records only the prefix covered by every detached layer.
    /// `model_id` and `template_sig` are included in the content hash
    /// so the same tokens under different models/templates get separate entries.
    pub fn persist(
        &self,
        model_id: &str,
        template_sig: &str,
        tokens: &[i32],
        cache_set: &DetachedCacheSet,
    ) -> Result<(), ColdStoreError> {
        let tx = self.writer_tx.as_ref().ok_or(ColdStoreError::WorkerDied)?;
        if cache_set.caches.is_empty() || !cache_set.has_consistent_seq_len() {
            return Err(ColdStoreError::CorruptLayer {
                layer: 0,
                detail: "cache-covered layer offsets are empty or inconsistent".into(),
            });
        }
        let cache_covered_len = usize::try_from(cache_set.seq_len()).map_err(|_| {
            ColdStoreError::CorruptLayer {
                layer: 0,
                detail: "cache-covered layer offset is negative".into(),
            }
        })?;
        if cache_covered_len == 0 || cache_covered_len > tokens.len() {
            return Err(ColdStoreError::CorruptLayer {
                layer: 0,
                detail: format!(
                    "cache-covered length {cache_covered_len} is invalid for {} model-visible tokens",
                    tokens.len()
                ),
            });
        }
        let tokens = &tokens[..cache_covered_len];

        // Q3 constraint: MlxArray -> raw bytes is thread-affine under Metal.
        // Serialize every layer (and hash the layout) ON THIS (inference)
        // thread; only Vec<u8> crosses to the background writer.
        // serialize_cache_set_layers batch-evaluates all arrays first (single
        // GPU→CPU sync instead of one per array per layer).
        let layout_fingerprint = reference::layout_fingerprint(cache_set);
        let layer_bytes = reference::serialize_cache_set_layers(cache_set)?;

        tx.send(WriteJob {
            model_id: model_id.to_string(),
            template_sig: template_sig.to_string(),
            covered_tokens: tokens.to_vec(),
            prompt_len: cache_set.prompt_len.min(cache_covered_len),
            layout_fingerprint,
            layer_bytes,
        })
        .map_err(|_| ColdStoreError::WorkerDied)
    }

    /// Load the best matching cache for a given token prefix.
    ///
    /// Delegates to the v3 [`ReferenceColdStore::load_prefix`]: only fully
    /// COMMITTED generations are visible, every layer is length- and
    /// SHA-256-checked, `validate_consistency` runs pre-FFI, and the runtime
    /// fingerprint (weight identity) is validated before any candidate is
    /// eligible. Returns the longest committed matching prefix.
    pub fn load_prefix(
        &self,
        model_id: &str,
        template_sig: &str,
        tokens: &[i32],
    ) -> Result<(DetachedCacheSet, usize), ColdStoreError> {
        self.ref_store.load_prefix(model_id, template_sig, tokens)
    }

    /// Invalidate a specific cache entry by v3 identity hash (the directory
    /// name under `<base_dir>/cold-storage-v3/`, as returned by
    /// [`ColdStore::list_entries`]). Removes every generation of that identity.
    // TODO(v3-admin): richer admin surface (per-generation invalidation, GC).
    pub fn invalidate(&self, identity_hex: &str) -> Result<(), ColdStoreError> {
        let identity_dir = self.ref_store.root_dir().join(identity_hex);
        if identity_dir.exists() {
            fs::remove_dir_all(&identity_dir)?;
        }
        Ok(())
    }

    /// List all stored v3 identity hashes (directory names under
    /// `<base_dir>/cold-storage-v3/`).
    // TODO(v3-admin): expose committed-generation detail, not just identities.
    pub fn list_entries(&self) -> Result<Vec<String>, ColdStoreError> {
        let mut entries = Vec::new();
        let root = self.ref_store.root_dir();
        if root.exists() {
            for entry in fs::read_dir(&root)? {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    if let Some(name) = entry.file_name().to_str() {
                        entries.push(name.to_string());
                    }
                }
            }
        }
        Ok(entries)
    }

    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    pub fn shutdown(&mut self) {
        drop(self.writer_tx.take());
        if let Some(handle) = self.writer_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for ColdStore {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ---------------------------------------------------------------------------
// Background writer
// ---------------------------------------------------------------------------

fn spawn_writer(
    ref_store: std::sync::Arc<ReferenceColdStore>,
) -> (Sender<WriteJob>, JoinHandle<()>) {
    let (tx, rx) = mpsc::channel::<WriteJob>();
    let handle = thread::spawn(move || {
        while let Ok(job) = rx.recv() {
            // The job carries PRE-SERIALIZED bytes (Q3: serialization stayed
            // on the inference thread); publication here is pure I/O with
            // per-layer checksums and an atomic COMMITTED marker.
            match ref_store.persist_serialized(
                &job.model_id,
                &job.template_sig,
                &job.covered_tokens,
                job.prompt_len,
                job.layout_fingerprint,
                &job.layer_bytes,
            ) {
                Ok(snapshot) => {
                    tracing::info!(
                        identity = %snapshot.identity_hex,
                        generation = %snapshot.generation,
                        layers = job.layer_bytes.len(),
                        "cold-store: entry persisted"
                    );
                }
                Err(e) => {
                    tracing::error!(
                        model_id = %job.model_id,
                        covered_tokens = job.covered_tokens.len(),
                        error = %e,
                        "cold-store: failed to write entry"
                    );
                }
            }
        }
    });
    (tx, handle)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn default_base_dir() -> PathBuf {
    if let Ok(env_path) = std::env::var("MLXCEL_COLD_STORE_DIR") {
        return PathBuf::from(env_path);
    }
    if let Some(home) = dirs::home_dir() {
        return home.join(DEFAULT_BASE_DIR);
    }
    PathBuf::from("/tmp/mlxcel/cold-storage")
}

/// Why a configured cold-store directory was refused. Carries the resolved path
/// so the operator sees the thing that was checked, not the thing they typed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColdStoreDirRefusal {
    /// The directory does not exist. We deliberately do NOT create it: see
    /// [`validate_configured_base_dir`].
    Missing(PathBuf),
    /// The path exists but is not a directory.
    NotADirectory(PathBuf),
    /// Under `/Volumes/<name>` whose root is NOT a mount point — i.e. the
    /// external drive is unmounted and this is a stub on the boot disk.
    VolumeNotMounted { path: PathBuf, volume: PathBuf },
}

impl std::fmt::Display for ColdStoreDirRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing(p) => write!(
                f,
                "MLXCEL_COLD_STORE_DIR={} does not exist. It is NOT created \
                 automatically: creating it is exactly how an unmounted drive \
                 ends up filling the boot disk. Create it yourself (or mount \
                 the drive) and restart.",
                p.display()
            ),
            Self::NotADirectory(p) => write!(
                f,
                "MLXCEL_COLD_STORE_DIR={} exists but is not a directory.",
                p.display()
            ),
            Self::VolumeNotMounted { path, volume } => write!(
                f,
                "MLXCEL_COLD_STORE_DIR={} is under {}, which is NOT a mount \
                 point — the drive is not mounted. Writing here would land on \
                 the BOOT DISK, and would become invisible the moment the \
                 drive mounts over it. Mount the drive and restart.",
                path.display(),
                volume.display()
            ),
        }
    }
}

/// Validate an operator-configured `MLXCEL_COLD_STORE_DIR`.
///
/// Returns `Ok(None)` when the variable is unset — the default under `$HOME` is
/// always acceptable and this check does not apply to it.
///
/// # Why this refuses instead of creating
///
/// `acquire_store_lock` calls `fs::create_dir_all(&self.base_dir)`
/// unconditionally. That is correct for the default path and catastrophic for a
/// configured one: on macOS, `/Volumes/<name>` is an ordinary directory on the
/// boot disk until something mounts over it. Point the store at an unmounted
/// drive and `create_dir_all` silently materialises the tree **on the boot
/// disk** — writes succeed, the cache works, nothing complains, and the volume
/// the operator was trying to protect fills up anyway. When the drive is later
/// mounted it covers those bytes, which then consume space while being
/// unreachable through that path.
///
/// The failure mode is therefore indistinguishable from success at every point
/// where anyone would look. So the configured directory must ALREADY EXIST and,
/// if it lives under `/Volumes`, its volume root must be a real mount point.
/// An unmounted drive becomes a loud refusal rather than a silent redirect.
///
/// This validates *configuration*, not permissions: a writability probe would
/// have to create something to be meaningful, which is the act being prevented.
/// An unwritable directory still fails loudly at first use.
pub fn validate_configured_base_dir() -> Result<Option<PathBuf>, ColdStoreDirRefusal> {
    let Ok(raw) = std::env::var("MLXCEL_COLD_STORE_DIR") else {
        return Ok(None);
    };
    let path = PathBuf::from(raw);

    if !path.exists() {
        return Err(ColdStoreDirRefusal::Missing(path));
    }
    if !path.is_dir() {
        return Err(ColdStoreDirRefusal::NotADirectory(path));
    }

    // `/Volumes/<name>/...` -> require `<name>` to be a genuine mount point.
    // Compared by device id rather than by consulting a mount table: a mount
    // point is precisely a directory whose device differs from its parent's,
    // which is the property that matters and needs no external command.
    if let Some(volume) = volume_root_of(&path) {
        if !is_mount_point(&volume) {
            return Err(ColdStoreDirRefusal::VolumeNotMounted { path, volume });
        }
    }
    Ok(Some(path))
}

/// `/Volumes/T7 Shield/mlxcel` -> `Some(/Volumes/T7 Shield)`; anything not under
/// `/Volumes` -> `None`, because the mount-point rule only applies there.
fn volume_root_of(path: &Path) -> Option<PathBuf> {
    let mut components = path.components();
    if components.next()? != std::path::Component::RootDir {
        return None;
    }
    if components.next()?.as_os_str() != "Volumes" {
        return None;
    }
    let name = components.next()?;
    Some(Path::new("/Volumes").join(name))
}

/// True when `dir` sits on a different device than its parent — the definition
/// of a mount point. Unreadable metadata returns `false`, which routes to a
/// refusal: unknown must not read as mounted.
fn is_mount_point(dir: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    let Some(parent) = dir.parent() else {
        return false;
    };
    match (fs::metadata(dir), fs::metadata(parent)) {
        (Ok(d), Ok(p)) => d.dev() != p.dev(),
        _ => false,
    }
}

#[allow(dead_code)] // v2 header helper; kept for the retained v2 tests
fn write_string(w: &mut impl Write, s: &str) -> io::Result<()> {
    w.write_all(&(s.len() as u32).to_le_bytes())?;
    w.write_all(s.as_bytes())
}

#[allow(dead_code)] // v2 header helper; kept for the retained v2 tests
fn read_string(r: &mut impl Read) -> io::Result<String> {
    let len = read_u32(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn read_u8(r: &mut impl Read) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i32(r: &mut impl Read) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// Bound an untrusted length field read from disk. Mirrors the reference's
/// `bounded_usize` (cold_store_reference.rs:763). Returns an InvalidData
/// `Io` error so the header-scan `Err(_) => continue` guard skips the entry.
#[allow(dead_code)] // v2 header helper; kept for the retained v2 tests
fn bounded_usize(value: u64, maximum: usize, field: &str) -> Result<usize, ColdStoreError> {
    let value = usize::try_from(value).map_err(|_| {
        ColdStoreError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{field} does not fit usize"),
        ))
    })?;
    if value > maximum {
        return Err(ColdStoreError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{field} {value} exceeds limit {maximum}"),
        )));
    }
    Ok(value)
}

fn validate_array_metadata(
    byte_len: usize,
    dt: i32,
    shape: &[i32],
) -> io::Result<usize> {
    let item_size = match dt {
        dtype::FLOAT32
        | dtype::FLOAT16
        | dtype::BFLOAT16
        | dtype::INT8
        | dtype::UINT8
        | dtype::INT32
        | dtype::UINT32 => dtype::size_bytes(dt).expect("supported dtype has a size"),
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported dtype: {dt}"),
            ));
        }
    };

    let element_count = shape.iter().try_fold(1usize, |count, &dim| {
        let dim = usize::try_from(dim).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("negative tensor dimension: {dim}"),
            )
        })?;
        count.checked_mul(dim).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "tensor element count overflow")
        })
    })?;
    let expected_len = element_count
        .checked_mul(item_size)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "tensor byte length overflow"))?;
    if byte_len != expected_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "tensor byte length mismatch: dtype {dt}, shape {shape:?}, expected {expected_len}, got {}",
                byte_len
            ),
        ));
    }
    Ok(item_size)
}

fn array_from_raw_bytes(
    bytes: &[u8],
    dt: i32,
    shape: &[i32],
) -> Result<UniquePtr<MlxArray>, io::Error> {
    validate_array_metadata(bytes.len(), dt, shape)?;
    ffi::from_bytes(bytes, shape, dt).map_err(io::Error::other)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use super::super::SequenceStateBackend;

    #[test]
    fn content_hash_is_deterministic() {
        let tokens = vec![1, 2, 3, 4, 5];
        let h1 = compute_content_hash("model-a", "tmpl-1", &tokens);
        let h2 = compute_content_hash("model-a", "tmpl-1", &tokens);
        assert_eq!(h1, h2);
    }

    #[test]
    fn content_hash_differs_by_model() {
        let tokens = vec![1, 2, 3];
        let h1 = compute_content_hash("model-a", "tmpl-1", &tokens);
        let h2 = compute_content_hash("model-b", "tmpl-1", &tokens);
        assert_ne!(h1, h2);
    }

    #[test]
    fn content_hash_differs_by_template() {
        let tokens = vec![1, 2, 3];
        let h1 = compute_content_hash("model-a", "tmpl-1", &tokens);
        let h2 = compute_content_hash("model-a", "tmpl-2", &tokens);
        assert_ne!(h1, h2);
    }

    #[test]
    fn weight_fingerprint_is_stable() {
        let fp1 = compute_weight_fingerprint("/path/to/model");
        let fp2 = compute_weight_fingerprint("/path/to/model");
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn weight_fingerprint_differs_for_different_paths() {
        let fp1 = compute_weight_fingerprint("/path/to/model-a");
        let fp2 = compute_weight_fingerprint("/path/to/model-b");
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn session_header_v2_round_trip() {
        let hdr = SessionHeader {
            format_version: FORMAT_VERSION,
            content_hash: "abc123".to_string(),
            weight_fingerprint: "def456".to_string(),
            model_id: "m3".to_string(),
            template_sig: "tmpl".to_string(),
            layer_count: 32,
            prompt_len: 1024,
            current_offset: 512,
            timestamp_secs: 1700000000,
            tokens: vec![1, 2, 3, 4, 5],
        };
        let mut buf = Vec::new();
        write_header(&mut buf, &hdr).unwrap();
        let mut reader = &buf[..];
        let restored = read_header(&mut reader).unwrap();
        assert_eq!(restored.format_version, FORMAT_VERSION);
        assert_eq!(restored.content_hash, "abc123");
        assert_eq!(restored.layer_count, 32);
        assert_eq!(restored.tokens, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn string_round_trip() {
        let mut buf = Vec::new();
        write_string(&mut buf, "hello world").unwrap();
        let mut reader = &buf[..];
        let restored = read_string(&mut reader).unwrap();
        assert_eq!(restored, "hello world");
    }

    #[test]
    fn cold_store_list_empty() {
        let dir = tempfile::tempdir().unwrap();
        let cs = ColdStore::with_base_dir(dir.path().to_path_buf(), "/dummy");
        let entries = cs.list_entries().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn cold_store_invalidate_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let cs = ColdStore::with_base_dir(dir.path().to_path_buf(), "/dummy");
        cs.invalidate("nonexistent").unwrap();
    }

    #[test]
    fn cold_store_load_prefix_no_match() {
        let dir = tempfile::tempdir().unwrap();
        let cs = ColdStore::with_base_dir(dir.path().to_path_buf(), "/dummy");
        let result = cs.load_prefix("model", "tmpl", &[1, 2, 3]);
        assert!(matches!(result, Err(ColdStoreError::NoMatch)));
    }

    // End-to-end tests with synthetic data.

    fn synth_tensor(shape: &[i32], seed: u32) -> UniquePtr<MlxArray> {
        let total: usize = shape.iter().map(|&d| d as usize).product();
        let mut state = if seed == 0 { 0xCAFE_BABE } else { seed };
        let mut data = Vec::with_capacity(total);
        for _ in 0..total {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let x = (state >> 1) as f32 / (i32::MAX as f32);
            data.push(x);
        }
        ffi::from_slice_f32(&data, shape)
    }

    fn flatten_fp32(arr: &MlxArray) -> Vec<f32> {
        let a = ffi::astype(arr, dtype::FLOAT32);
        ffi::eval(&a);
        let bytes = ffi::array_to_raw_bytes(&a);
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    pub(crate) fn make_test_cache_set(
        num_layers: usize,
        seq_len: i32,
        head_dim: i32,
    ) -> DetachedCacheSet {
        use super::super::SequenceStateBackend;
        use std::time::Instant;

        let mut caches = Vec::new();
        for layer in 0..num_layers {
            let seed_k = 1000 + layer as u32;
            let seed_v = 2000 + layer as u32;
            let k_f32 = synth_tensor(&[1, 2, seq_len, head_dim], seed_k);
            let v_f32 = synth_tensor(&[1, 2, seq_len, head_dim], seed_v);
            let k = ffi::astype(&k_f32, dtype::FLOAT16);
            let v = ffi::astype(&v_f32, dtype::FLOAT16);
            caches.push(DetachedKVCache {
                keys: Some(k),
                values: Some(v),
                offset: seq_len,
                step: 0,
                mode: KVCacheMode::Fp16,
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
                delegated_fp16_sidecar_policy: super::super::turbo::DelegatedFp16SidecarPolicy::default(),
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
            });
        }
        DetachedCacheSet {
            caches,
            backend: SequenceStateBackend::DenseKvCache,
            prompt_len: seq_len as usize,
            current_offset: seq_len,
            created_at: Instant::now(),
            detached_at: Instant::now(),
            origin_seq_id: super::super::SequenceId(0),
        }
    }

    // ===================== Sharpened KV concurrency probe (Q3 gate) =====================
    // Author: Clement (clement-7074f29f). Review: Xander (dev/MLX) + Violet (QE), K=2 decorrelated.
    // Design: DESIGN_sharpened_concurrency_probe_20260723.md (6 revisions, both sign-offs).
    // Reader models the off-thread serializer: runs the REAL write_kv_cache over a NON-CONTIGUOUS set
    // (transpose forces contiguous()+eval() = real Metal work, not a trivial memcpy). Evaluator models
    // live inference: own generation stream + heavy/varied alloc churn (saturates the shared thread pool).
    // TWO axes, each with a validity control: correctness (byte-compare; negative control proves teeth) and
    // latency decoupling (evaluator per-iter latency with/without writer; sensitivity control proves the
    // instrument can see a stall). Output tagged "PROBE". A green is only trusted if its control fired.
    struct SharedSet(DetachedCacheSet);
    // SAFETY: evidence for a future ownership design ONLY — NOT a claim MLX ops are Send+Sync. The set is
    // immutable and read-only across threads; the reader serializes it, never mutates it.
    unsafe impl Send for SharedSet {}
    unsafe impl Sync for SharedSet {}

    fn probe_serialize_set(set: &DetachedCacheSet) -> Vec<u8> {
        let mut buf = Vec::new();
        for c in &set.caches {
            write_kv_cache(&mut buf, c).expect("write_kv_cache");
        }
        buf
    }

    // Xander adjudication (SIGTRAP): cross-thread contiguous()+eval() is unsafe under Metal. The transposed
    // (lazy, non-contiguous) S made the reader do cross-thread graph construction+eval → crash. The SAFE
    // off-thread path is memcpy of PRE-MATERIALIZED CONTIGUOUS arrays — eval on the inference thread first
    // (what eval_all already does), then the writer memcpy+disk-writes off-thread. This builder materializes
    // S on the MAIN thread so the reader's array_to_raw_bytes is a pure memcpy (no cross-thread eval).
    fn probe_make_materialized_set(num_layers: usize, seq_len: i32, head_dim: i32) -> DetachedCacheSet {
        let set = make_test_cache_set(num_layers, seq_len, head_dim); // contiguous fp16 (astype), no transpose
        for c in &set.caches {
            if let Some(k) = &c.keys {
                ffi::eval(k); // materialize on the main (origin) thread
            }
            if let Some(v) = &c.values {
                ffi::eval(v);
            }
        }
        set
    }

    // Evaluator (inference-proxy): own generation stream + heavy varied churn; returns wall time.
    fn probe_evaluator_work(iters: usize) -> std::time::Duration {
        if let Some(s) = crate::streams::new_thread_local_generation_stream() {
            crate::streams::install_thread_local_default_stream(Some(&s));
        }
        let start = std::time::Instant::now();
        for i in 0..iters {
            let sz = 128 + ((i % 8) as i32) * 64; // varied alloc sizes → stress the global allocator + pool
            let t = synth_tensor(&[sz, sz], 4242 + i as u32);
            let a = ffi::astype(&t, dtype::FLOAT16);
            let b = ffi::transpose(&a);
            let c = ffi::reshape(&b, &[sz, sz]); // reshape of a transposed view forces a real contiguous copy
            ffi::eval(&c); // eval is synchronous → wall time captures GPU work (per CLAUDE.md timing note)
        }
        start.elapsed()
    }

    // Run `readers` serializer threads + 1 evaluator concurrently, watchdog-bounded.
    // Returns (all_readers_ok, hang, evaluator_latency).
    fn probe_run(
        shared: std::sync::Arc<SharedSet>,
        expected: std::sync::Arc<Vec<u8>>,
        iters: usize,
        readers: usize,
        isolate_reader_stream: bool,
        watchdog: std::time::Duration,
    ) -> (bool, bool, std::time::Duration) {
        let (done_tx, done_rx) = std::sync::mpsc::channel::<(&'static str, bool)>();
        for _ in 0..readers {
            let shared = shared.clone();
            let expected = expected.clone();
            let done_tx = done_tx.clone();
            std::thread::spawn(move || {
                if isolate_reader_stream {
                    if let Some(s) = crate::streams::new_thread_local_generation_stream() {
                        crate::streams::install_thread_local_default_stream(Some(&s));
                    }
                }
                let mut ok = true;
                for _ in 0..iters {
                    if probe_serialize_set(&shared.0) != *expected {
                        ok = false;
                        break;
                    }
                }
                let _ = done_tx.send(("reader", ok));
            });
        }
        let (lat_tx, lat_rx) = std::sync::mpsc::channel();
        {
            let done_tx = done_tx.clone();
            std::thread::spawn(move || {
                let elapsed = probe_evaluator_work(iters);
                let _ = lat_tx.send(elapsed);
                let _ = done_tx.send(("eval", true));
            });
        }
        drop(done_tx);
        let expected_msgs = readers + 1;
        let mut got = 0usize;
        let mut readers_ok = true;
        let deadline = std::time::Instant::now() + watchdog;
        while got < expected_msgs {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            match done_rx.recv_timeout(remaining) {
                Ok((who, ok)) => {
                    got += 1;
                    if who == "reader" && !ok {
                        readers_ok = false;
                    }
                }
                Err(_) => break,
            }
        }
        let hang = got < expected_msgs;
        let latency = lat_rx.try_recv().unwrap_or(watchdog);
        (readers_ok, hang, latency)
    }

    // Q3 concurrency-gate probe (Clement, cycle 94): DELIBERATELY crashes the
    // process (SIGTRAP) to demonstrate that off-thread serialization of
    // Metal-resident MlxArrays is thread-affine and unsafe. The verdict is
    // settled; this stays as a manually-runnable artifact (`--ignored`) and is
    // #[ignore]'d so it never aborts the default test suite.
    #[ignore = "Q3 thread-affinity probe: intentionally SIGTRAPs; run with --ignored"]
    #[test]
    fn sharpened_kv_concurrency_probe() {
        use std::sync::Arc;
        use std::time::Duration;
        let iters = 100usize;
        let watchdog = Duration::from_secs(60);

        // Pre-materialized-contiguous S (Xander fix): reader = pure memcpy, no cross-thread eval.
        // (The lazy/non-contiguous S variant SIGTRAPs — recorded separately as the cross-thread-eval finding.)
        println!("PROBE variant=materialized-contiguous (reader does memcpy, no cross-thread eval)");
        let set = probe_make_materialized_set(8, 128, 64);
        let expected = Arc::new(probe_serialize_set(&set));
        let shared = Arc::new(SharedSet(set));
        assert!(!expected.is_empty(), "PROBE: ground truth serialization empty");
        println!("PROBE ground-truth-bytes={}", expected.len());

        // AXIS-1 NEGATIVE CONTROL: a corrupted buffer MUST compare unequal → byte-compare has teeth.
        let mut corrupt = (*expected).clone();
        corrupt[expected.len() / 2] ^= 0xFF;
        let neg_fires = corrupt != *expected;
        println!("PROBE axis1-negative-control-fired={}", neg_fires);
        assert!(neg_fires, "PROBE: negative control did NOT fire — byte-compare invalid");

        // BASELINE: evaluator alone (no writer).
        let baseline = {
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let _ = tx.send(probe_evaluator_work(iters));
            });
            rx.recv().unwrap()
        };
        println!("PROBE baseline-eval-ms={}", baseline.as_millis());

        // READER-ALONE (Violet's discriminator): reader memcpy-serializes materialized S on an isolated
        // stream, NO concurrent evaluator. PASS -> the crash needs concurrent eval (Xander's scheduler
        // mechanism = real limit). CRASH -> cross-thread sharing/access itself faults (probe artifact).
        {
            let shared_ra = shared.clone();
            let expected_ra = expected.clone();
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                // DECISIVE STREAM-INSTALL VERIFICATION (Xander's fix):
                // The prior form silently no-op'd if new_thread_local_generation_stream()
                // returned None — leaving the reader thread WITHOUT a Metal stream. We
                // remove that ambiguity: install an explicit GPU stream and print
                // confirmation BEFORE touching S. If we see stream-installed=true here and
                // STILL fault during serialization -> real thread-affinity -> off-thread
                // dead. If it now passes -> the earlier crash was a missing-context
                // artifact -> off-thread optimization is back on the table.
                //
                // !! VERDICT CONTAMINATION WARNING (clement, 2026-07-27) !!
                // This probe previously called `streams::init_thread()` here to "remove
                // the ambiguity" by arming a teardown finalizer. That finalizer was ITSELF
                // a deterministic crash source: its `Drop` called into MLX after MLX's own
                // C++ thread-locals were destroyed, killing the process at THREAD EXIT
                // (SIGTRAP) regardless of anything this probe measured. Any prior verdict
                // of "off-thread dead / real thread-affinity limit" drawn from a crash in
                // this probe MUST be re-derived: arming did the opposite of what its
                // comment claimed. The arming call is removed; the probe is otherwise
                // unchanged and still `#[ignore]`d. See streams.rs "Per-thread MLX
                // teardown" notes for the measurement.
                let gpu = crate::ffi::is_gpu_available();
                let installed = if let Some(s) = crate::streams::new_thread_local_generation_stream() {
                    crate::streams::install_thread_local_default_stream(Some(&s));
                    true
                } else {
                    false
                };
                use std::io::Write as _;
                println!(
                    "PROBE reader-alone is-gpu-available={} stream-installed={} finalizer-armed=false (BEFORE touching S)",
                    gpu, installed
                );
                let _ = std::io::stdout().flush();
                let mut ok = true;
                for _ in 0..iters {
                    if probe_serialize_set(&shared_ra.0) != *expected_ra {
                        ok = false;
                        break;
                    }
                }
                // FLUSHED loop-completed marker: distinguishes a crash DURING serialization
                // (real cross-thread Metal thread-affinity — this line never prints) from a
                // crash at thread teardown (serialization survived — this line prints, then
                // the finalizer/static-destructor path faults). Block-buffered file stdout
                // makes the flush mandatory: without it, absence of this line is uninformative.
                println!(
                    "PROBE reader-alone loop-completed ok={} (serialization SURVIVED; any crash after this is teardown, not thread-affinity)",
                    ok
                );
                let _ = std::io::stdout().flush();
                let _ = tx.send(ok);
            });
            match rx.recv_timeout(watchdog) {
                Ok(ok) => println!("PROBE reader-alone-materialized ok={} (no concurrent evaluator)", ok),
                Err(_) => println!("PROBE reader-alone-materialized HANG"),
            }
        }

        // VARIANT B — isolated streams (the production question).
        let (b_ok, b_hang, b_lat) =
            probe_run(shared.clone(), expected.clone(), iters, 1, true, watchdog);
        println!(
            "PROBE variantB-isolated ok={} hang={} eval-ms={}",
            b_ok, b_hang, b_lat.as_millis()
        );

        // VARIANT A — same default stream (CONTROL: null hyp says this is the stressed path).
        let (a_ok, a_hang, a_lat) =
            probe_run(shared.clone(), expected.clone(), iters, 1, false, watchdog);
        println!(
            "PROBE variantA-samestream ok={} hang={} eval-ms={}",
            a_ok, a_hang, a_lat.as_millis()
        );

        // AXIS-2 SENSITIVITY CONTROL: forced heavy contention (4 same-stream readers) MUST spike
        // evaluator latency vs baseline, or the latency instrument is too coarse to trust a flat result.
        let (_f_ok, _f_hang, f_lat) =
            probe_run(shared.clone(), expected.clone(), iters, 4, false, watchdog);
        let spiked = f_lat.as_millis() > baseline.as_millis() * 3 / 2;
        println!(
            "PROBE axis2-sensitivity forced-ms={} baseline-ms={} spiked={}",
            f_lat.as_millis(), baseline.as_millis(), spiked
        );

        // TWO-AXIS VERDICT + RED-origin attribution.
        let correctness = b_ok && !b_hang;
        let latency_flat = (b_lat.as_millis() as f64) < (baseline.as_millis() as f64) * 1.5;
        let verdict = if !correctness {
            "OFF-THREAD-OUT (fallback: serialize-on-inference-thread)"
        } else if latency_flat {
            "OPTIMIZATION-VIABLE (off-thread decouples)"
        } else {
            "CORRECT-BUT-STALLS -> compaction-boundary sync (clean landing)"
        };
        println!(
            "PROBE VERDICT correctness={} latency-flat={} => {}",
            correctness, latency_flat, verdict
        );
        if b_hang {
            println!("PROBE RED-origin=HANG (scheduler-completion)");
        } else if !b_ok {
            println!("PROBE RED-origin=MISMATCH (corruption)");
        } else if !latency_flat {
            println!("PROBE RED-origin=LATENCY-only (contention)");
        }
        // Construct-validity guard: the sensitivity control must be able to show a stall, else a flat
        // axis-2 means "couldn't measure one," not "no contention." Report, don't hard-fail (Metal-dependent).
        println!("PROBE construct-validity sensitivity-control-usable={}", spiked || !latency_flat);
    }

    #[test]
    fn cold_store_persist_load_exact_match() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model");
        fs::create_dir_all(&model_path).unwrap();
        let cs = ColdStore::with_base_dir(dir.path().join("cs").to_path_buf(), model_path.to_str().unwrap());

        let tokens = vec![100, 200, 300, 400, 500];
        let original = make_test_cache_set(2, 5, 32);

        cs.persist("m3", "tmpl", &tokens, &original).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));

        let (loaded, match_len) = cs.load_prefix("m3", "tmpl", &tokens).unwrap();
        assert_eq!(match_len, 5);
        assert_eq!(loaded.caches.len(), 2);
        assert_eq!(loaded.prompt_len, 5);
    }

    #[test]
    fn cold_store_persists_only_cache_covered_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model");
        fs::create_dir_all(&model_path).unwrap();
        let mut cs = ColdStore::with_base_dir(
            dir.path().join("cs").to_path_buf(),
            model_path.to_str().unwrap(),
        );
        let tokens = vec![100, 200, 300, 400, 500, 600];
        let original = make_test_cache_set(2, 5, 32);

        cs.persist("m3", "tmpl", &tokens, &original).unwrap();
        cs.shutdown();

        let (loaded, match_len) = cs.load_prefix("m3", "tmpl", &tokens).unwrap();
        assert_eq!(match_len, 5);
        assert_eq!(loaded.seq_len(), 5);
        assert_eq!(loaded.current_offset, 5);
    }

    #[test]
    fn production_coldstore_v3_roundtrip_persists_and_restores() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model");
        fs::create_dir_all(&model_path).unwrap();
        let base = dir.path().join("cs");
        let mut cs = ColdStore::with_base_dir(base.clone(), model_path.to_str().unwrap());

        let tokens = vec![11, 22, 33, 44, 55];
        let original = make_test_cache_set(2, 5, 32);
        cs.persist("m3", "tmpl", &tokens, &original).unwrap();

        // The writer is async; shutdown() drops the channel and JOINS the
        // writer thread, so the COMMITTED marker is durably published before
        // any load is attempted (no sleep/polling race).
        cs.shutdown();

        // The entry landed in the v3 layout with an atomic COMMITTED marker.
        let root = base.join("cold-storage-v3");
        let identity_dirs: Vec<_> = fs::read_dir(&root)
            .expect("v3 root exists")
            .map(|e| e.unwrap().path())
            .collect();
        assert_eq!(identity_dirs.len(), 1, "one identity expected");
        let generation_dirs: Vec<_> = fs::read_dir(&identity_dirs[0])
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        assert_eq!(generation_dirs.len(), 1, "one generation expected");
        assert!(
            generation_dirs[0].join("COMMITTED").is_file(),
            "generation must be atomically committed"
        );

        // Verified load through the production API on the same store.
        let (loaded, matched) = cs.load_prefix("m3", "tmpl", &tokens).unwrap();
        assert_eq!(matched, 5);
        assert_eq!(loaded.seq_len(), 5);
        assert_eq!(loaded.caches.len(), 2);

        // A FRESH ColdStore over the same base_dir (same model weights ->
        // same runtime fingerprint) also loads it: on-disk durability +
        // checksum-verified restore through the production API.
        let cs2 = ColdStore::with_base_dir(base, model_path.to_str().unwrap());
        let (reloaded, rematched) = cs2.load_prefix("m3", "tmpl", &tokens).unwrap();
        assert_eq!(rematched, 5);
        assert_eq!(reloaded.seq_len(), 5);
        assert_eq!(reloaded.caches.len(), 2);
    }

    #[test]
    fn cold_store_rejects_state_beyond_available_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model");
        fs::create_dir_all(&model_path).unwrap();
        let cs = ColdStore::with_base_dir(
            dir.path().join("cs").to_path_buf(),
            model_path.to_str().unwrap(),
        );
        let original = make_test_cache_set(2, 5, 32);

        let result = cs.persist("m3", "tmpl", &[100, 200, 300, 400], &original);

        assert!(matches!(result, Err(ColdStoreError::CorruptLayer { .. })));
    }

    #[test]
    fn cold_store_rejects_inconsistent_or_empty_layer_offsets() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model");
        fs::create_dir_all(&model_path).unwrap();
        let cs = ColdStore::with_base_dir(
            dir.path().join("cs").to_path_buf(),
            model_path.to_str().unwrap(),
        );
        let tokens = vec![100, 200, 300, 400, 500];

        let mut inconsistent = make_test_cache_set(2, 5, 32);
        inconsistent.caches[1].offset = 4;
        assert!(matches!(
            cs.persist("m3", "tmpl", &tokens, &inconsistent),
            Err(ColdStoreError::CorruptLayer { .. })
        ));

        let mut empty = make_test_cache_set(2, 5, 32);
        for cache in &mut empty.caches {
            cache.offset = 0;
        }
        assert!(matches!(
            cs.persist("m3", "tmpl", &tokens, &empty),
            Err(ColdStoreError::CorruptLayer { .. })
        ));
    }

    #[test]
    fn cold_store_longest_prefix_match() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model");
        fs::create_dir_all(&model_path).unwrap();
        let cs = ColdStore::with_base_dir(dir.path().join("cs").to_path_buf(), model_path.to_str().unwrap());

        // Store a short prefix.
        let short_tokens = vec![100, 200, 300];
        let short_cache = make_test_cache_set(1, 3, 16);
        cs.persist("m3", "tmpl", &short_tokens, &short_cache).unwrap();

        // Store a longer prefix.
        let long_tokens = vec![100, 200, 300, 400, 500];
        let long_cache = make_test_cache_set(1, 5, 16);
        cs.persist("m3", "tmpl", &long_tokens, &long_cache).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(100));

        // Request with the long prefix should match the long cache.
        let (loaded, match_len) = cs.load_prefix("m3", "tmpl", &long_tokens).unwrap();
        assert_eq!(match_len, 5);
        assert_eq!(loaded.prompt_len, 5);

        // Request with an extended prefix should match the long cache (5 tokens match).
        let extended = vec![100, 200, 300, 400, 500, 600];
        let (loaded, match_len) = cs.load_prefix("m3", "tmpl", &extended).unwrap();
        assert_eq!(match_len, 5);
        assert_eq!(loaded.prompt_len, 5);
    }

    #[test]
    fn cold_store_cross_session_sharing() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model");
        fs::create_dir_all(&model_path).unwrap();
        let cs = ColdStore::with_base_dir(dir.path().join("cs").to_path_buf(), model_path.to_str().unwrap());

        // Session A stores tokens [1,2,3,4,5].
        let tokens_a = vec![1, 2, 3, 4, 5];
        let cache_a = make_test_cache_set(1, 5, 16);
        cs.persist("m3", "tmpl", &tokens_a, &cache_a).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(100));

        // Session B has tokens [1,2,3,6,7] — shares prefix [1,2,3].
        let tokens_b = vec![1, 2, 3, 6, 7];
        let (loaded, match_len) = cs.load_prefix("m3", "tmpl", &tokens_b).unwrap();
        assert_eq!(match_len, 3); // 3 tokens match
        assert_eq!(loaded.prompt_len, 5); // But we get the full 5-token cache
    }

    #[test]
    fn cold_store_different_model_no_match() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model");
        fs::create_dir_all(&model_path).unwrap();
        let cs = ColdStore::with_base_dir(dir.path().join("cs").to_path_buf(), model_path.to_str().unwrap());

        let tokens = vec![1, 2, 3];
        let cache = make_test_cache_set(1, 3, 16);
        cs.persist("m3", "tmpl", &tokens, &cache).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Different model should not match.
        let result = cs.load_prefix("other-model", "tmpl", &tokens);
        assert!(matches!(result, Err(ColdStoreError::NoMatch)));
    }

    #[test]
    fn cold_store_fp16_tensor_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model");
        fs::create_dir_all(&model_path).unwrap();
        let cs = ColdStore::with_base_dir(dir.path().join("cs").to_path_buf(), model_path.to_str().unwrap());

        // Create FP16 tensors with known values.
        let num_layers = 2;
        let seq_len = 4;
        let head_dim = 8;
        let tokens = vec![100, 200, 300, 400];

        let mut caches = Vec::new();
        for layer in 0..num_layers {
            // Create deterministic FP16 data.
            let k_f32 = synth_tensor(&[1, 2, seq_len, head_dim], 1000 + layer as u32);
            let v_f32 = synth_tensor(&[1, 2, seq_len, head_dim], 2000 + layer as u32);
            let k = ffi::astype(&k_f32, dtype::FLOAT16);
            let v = ffi::astype(&v_f32, dtype::FLOAT16);

            // Get the raw bytes of the original FP16 tensors.
            ffi::eval(&k);
            ffi::eval(&v);
            let k_bytes_orig = ffi::array_to_raw_bytes(&k);
            let v_bytes_orig = ffi::array_to_raw_bytes(&v);

            caches.push(DetachedKVCache {
                keys: Some(k),
                values: Some(v),
                offset: seq_len,
                step: 0,
                mode: KVCacheMode::Fp16,
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
                delegated_fp16_sidecar_policy: super::super::turbo::DelegatedFp16SidecarPolicy::default(),
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
            });
        }

        let original = DetachedCacheSet {
            caches,
            backend: SequenceStateBackend::DenseKvCache,
            prompt_len: seq_len as usize,
            current_offset: seq_len,
            created_at: std::time::Instant::now(),
            detached_at: std::time::Instant::now(),
            origin_seq_id: super::super::SequenceId(0),
        };

        // Persist to cold-store.
        cs.persist("m3", "tmpl", &tokens, &original).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Load from cold-store.
        let (loaded, match_len) = cs.load_prefix("m3", "tmpl", &tokens).unwrap();
        assert_eq!(match_len, 4);
        assert_eq!(loaded.caches.len(), num_layers);

        // Verify FP16 tensor bytes are identical.
        for layer in 0..num_layers {
            let orig_cache = &original.caches[layer];
            let loaded_cache = &loaded.caches[layer];

            // Check keys.
            let orig_k = orig_cache.keys.as_ref().unwrap();
            let loaded_k = loaded_cache.keys.as_ref().unwrap();
            ffi::eval(orig_k);
            ffi::eval(loaded_k);
            let orig_k_bytes = ffi::array_to_raw_bytes(orig_k);
            let loaded_k_bytes = ffi::array_to_raw_bytes(loaded_k);
            assert_eq!(
                orig_k_bytes, loaded_k_bytes,
                "Layer {layer}: FP16 key tensor bytes differ after round-trip"
            );

            // Check values.
            let orig_v = orig_cache.values.as_ref().unwrap();
            let loaded_v = loaded_cache.values.as_ref().unwrap();
            ffi::eval(orig_v);
            ffi::eval(loaded_v);
            let orig_v_bytes = ffi::array_to_raw_bytes(orig_v);
            let loaded_v_bytes = ffi::array_to_raw_bytes(loaded_v);
            assert_eq!(
                orig_v_bytes, loaded_v_bytes,
                "Layer {layer}: FP16 value tensor bytes differ after round-trip"
            );
        }
    }

    #[test]
    fn raw_byte_loader_round_trips_every_persisted_dtype() {
        let cases = [
            (dtype::FLOAT32, vec![0, 0, 192, 63, 0, 0, 32, 192]),
            (dtype::FLOAT16, vec![0, 60, 0, 192]),
            (dtype::BFLOAT16, vec![128, 63, 0, 192]),
            (dtype::INT8, vec![0, 127, 128, 255]),
            (dtype::UINT8, vec![0, 127, 128, 255]),
            (dtype::INT32, vec![0, 0, 0, 128, 255, 255, 255, 127]),
            (dtype::UINT32, vec![0, 0, 0, 0, 239, 190, 173, 222]),
        ];

        for (dt, bytes) in cases {
            let item_size = dtype::size_bytes(dt).unwrap();
            let shape = [(bytes.len() / item_size) as i32];
            let arr = array_from_raw_bytes(&bytes, dt, &shape).unwrap();
            assert_eq!(ffi::array_dtype(&arr), dt, "dtype {dt}");
            assert_eq!(ffi::array_shape(&arr), shape, "dtype {dt}");
            assert_eq!(ffi::array_to_raw_bytes(&arr), bytes, "dtype {dt}");
        }
    }

    #[test]
    fn raw_byte_loader_accepts_misaligned_multibyte_input() {
        #[repr(align(4))]
        struct AlignedBytes([u8; 5]);

        let storage = AlignedBytes([0, 239, 190, 173, 222]);
        let bytes = &storage.0[1..];
        assert_ne!(bytes.as_ptr() as usize % std::mem::align_of::<u32>(), 0);

        let arr = array_from_raw_bytes(bytes, dtype::UINT32, &[1]).unwrap();
        assert_eq!(ffi::array_dtype(&arr), dtype::UINT32);
        assert_eq!(ffi::array_to_raw_bytes(&arr), bytes);
    }

    #[test]
    fn raw_byte_loader_rejects_malformed_metadata() {
        let error = |result: Result<UniquePtr<MlxArray>, io::Error>| match result {
            Ok(_) => panic!("malformed tensor metadata unexpectedly succeeded"),
            Err(error) => error,
        };

        let short = error(array_from_raw_bytes(&[0; 7], dtype::FLOAT32, &[2]));
        assert_eq!(short.kind(), io::ErrorKind::InvalidData);
        assert!(short.to_string().contains("expected 8, got 7"));

        let long = error(array_from_raw_bytes(&[0; 9], dtype::FLOAT32, &[2]));
        assert_eq!(long.kind(), io::ErrorKind::InvalidData);
        assert!(long.to_string().contains("expected 8, got 9"));

        let negative = error(array_from_raw_bytes(&[], dtype::UINT8, &[-1]));
        assert_eq!(negative.kind(), io::ErrorKind::InvalidData);
        assert!(negative.to_string().contains("negative tensor dimension"));

        let overflow = error(array_from_raw_bytes(&[], dtype::UINT8, &[i32::MAX; 3]));
        assert_eq!(overflow.kind(), io::ErrorKind::InvalidData);
        assert!(overflow.to_string().contains("overflow"));

        let unsupported = error(array_from_raw_bytes(&[0], dtype::BOOL, &[1]));
        assert_eq!(unsupported.kind(), io::ErrorKind::InvalidData);
        assert!(unsupported.to_string().contains("unsupported dtype"));
    }

    #[test]
    fn read_array_rejects_malformed_metadata_before_payload_allocation() {
        let error = |bytes: &[u8]| match read_array(&mut io::Cursor::new(bytes)) {
            Ok(_) => panic!("malformed serialized tensor unexpectedly succeeded"),
            Err(error) => error,
        };

        let mut negative_rank = Vec::new();
        negative_rank.extend_from_slice(&dtype::UINT8.to_le_bytes());
        negative_rank.extend_from_slice(&(-1i32).to_le_bytes());
        assert!(error(&negative_rank).to_string().contains("negative tensor rank"));

        let mut excessive_rank = Vec::new();
        excessive_rank.extend_from_slice(&dtype::UINT8.to_le_bytes());
        excessive_rank
            .extend_from_slice(&((MAX_SERIALIZED_TENSOR_RANK + 1) as i32).to_le_bytes());
        assert!(error(&excessive_rank).to_string().contains("exceeds limit"));

        let mut impossible_payload = Vec::new();
        impossible_payload.extend_from_slice(&dtype::UINT8.to_le_bytes());
        impossible_payload.extend_from_slice(&1i32.to_le_bytes());
        impossible_payload.extend_from_slice(&1i32.to_le_bytes());
        impossible_payload.extend_from_slice(&u64::MAX.to_le_bytes());
        assert!(
            error(&impossible_payload)
                .to_string()
                .contains("tensor byte length")
        );

        #[cfg(target_pointer_width = "64")]
        {
            let huge_dim = i32::MAX;
            let huge_len = (huge_dim as u64) * (huge_dim as u64) * 4;
            let mut self_consistent_huge = Vec::new();
            self_consistent_huge.extend_from_slice(&dtype::FLOAT32.to_le_bytes());
            self_consistent_huge.extend_from_slice(&2i32.to_le_bytes());
            self_consistent_huge.extend_from_slice(&huge_dim.to_le_bytes());
            self_consistent_huge.extend_from_slice(&huge_dim.to_le_bytes());
            self_consistent_huge.extend_from_slice(&huge_len.to_le_bytes());
            assert!(
                error(&self_consistent_huge)
                    .to_string()
                    .contains("cannot allocate")
            );
        }
    }

    #[test]
    fn kvarn4_cache_serialization_round_trip_is_bit_exact() {
        let mut cache =
            super::super::KVCache::synth_kvarn_state(1, 1, 32, 300, 16, 7, 4);
        let original = cache.clone_handle();
        let mut encoded = Vec::new();
        write_kv_cache(&mut encoded, &original).unwrap();

        let restored = read_kv_cache(&mut io::Cursor::new(&encoded), 0).unwrap();
        let mut reencoded = Vec::new();
        write_kv_cache(&mut reencoded, &restored).unwrap();

        assert_eq!(reencoded, encoded);
    }

    // ================= issue-6: post-read validation (regression tests) =================
    // Contract per test: the SAME fixture that passes green is mutated one
    // field at a time; each mutation must produce Err(CorruptLayer) from a
    // plain call — red-on-bug-present, green-on-bug-absent, and never a panic.

    fn kvarn_fixture(v_bits: u8) -> DetachedKVCache {
        // Same construction as kvarn4_cache_serialization_round_trip_is_bit_exact:
        // b=1, h=1, d=32, total=300 → sink 128 + hist 128 (1 tile) + tail 44.
        let mut live = super::super::KVCache::synth_kvarn_state(1, 1, 32, 300, 16, 7, v_bits);
        live.clone_handle()
    }

    fn fp16_fixture() -> DetachedKVCache {
        make_test_cache_set(1, 5, 32).caches.remove(0)
    }

    /// Serialize + deserialize so the mutation provably survives the on-disk
    /// positional format (read_kv_cache accepts it) and is caught only by
    /// validate_consistency.
    fn roundtrip(cache: &DetachedKVCache) -> DetachedKVCache {
        let mut buf = Vec::new();
        write_kv_cache(&mut buf, cache).unwrap();
        read_kv_cache(&mut io::Cursor::new(&buf), 0).unwrap()
    }

    fn assert_corrupt(result: Result<(), ColdStoreError>, what: &str) {
        assert!(
            matches!(result, Err(ColdStoreError::CorruptLayer { .. })),
            "{what}: expected Err(CorruptLayer), got {result:?}"
        );
    }

    #[test]
    fn validate_consistency_accepts_healthy_fixtures() {
        // Green control: without it every red test below could be red for
        // the wrong reason (a validator that rejects everything).
        validate_consistency(&fp16_fixture(), 0).expect("healthy fp16");
        validate_consistency(&roundtrip(&fp16_fixture()), 0).expect("healthy fp16 roundtrip");
        validate_consistency(&kvarn_fixture(8), 0).expect("healthy k8v8");
        validate_consistency(&kvarn_fixture(4), 0).expect("healthy k8v4");
        validate_consistency(&roundtrip(&kvarn_fixture(4)), 0).expect("healthy k8v4 roundtrip");
    }

    #[test]
    fn validate_rejects_bad_v_bits_after_disk_roundtrip() {
        // The cache.rs:1522 panic class: a single flipped v_bits byte.
        for bad in [0u8, 5, 255] {
            let mut c = kvarn_fixture(4);
            c.kvarn_v_bits = bad;
            let restored = roundtrip(&c); // read_kv_cache accepts it unvalidated
            assert_corrupt(
                validate_consistency(&restored, 3),
                &format!("v_bits={bad}"),
            );
        }
    }

    #[test]
    fn validate_rejects_wrong_rank_tensor() {
        // The cxx std::terminate class: rank-2 tensor reaching ffi::slice.
        let mut c = fp16_fixture();
        c.keys = Some(ffi::astype(&synth_tensor(&[5, 32], 11), dtype::FLOAT16));
        assert_corrupt(validate_consistency(&roundtrip(&c), 0), "rank-2 keys");

        let mut c = kvarn_fixture(8);
        c.kvarn_sink_k = Some(ffi::astype(&synth_tensor(&[128, 32], 12), dtype::FLOAT16));
        assert_corrupt(validate_consistency(&c, 0), "rank-2 kvarn_sink_k");
    }

    #[test]
    fn validate_rejects_offset_window_sum_mismatch() {
        // offset != sink + hist + tail (300 = 128 + 128 + 44 in the fixture).
        let mut c = kvarn_fixture(8);
        c.offset += 1;
        assert_corrupt(validate_consistency(&roundtrip(&c), 0), "offset+1");

        let mut c = kvarn_fixture(4);
        c.offset = 172; // drops exactly the hist tile from the sum
        assert_corrupt(validate_consistency(&c, 0), "offset shrunk past hist");

        let mut c = fp16_fixture();
        c.offset = -1;
        assert_corrupt(validate_consistency(&c, 0), "negative offset");
    }

    #[test]
    fn validate_rejects_m3_idx_offset_mismatch() {
        // The cycle-79 desync class: m3_idx_offset != m3_idx_k seq length.
        let mut c = kvarn_fixture(8);
        assert!(c.m3_idx_k.is_some(), "fixture must carry m3 state");
        c.m3_idx_offset -= 1;
        assert_corrupt(validate_consistency(&roundtrip(&c), 0), "m3 desync");
    }

    #[test]
    fn validate_rejects_missing_mode_sidecars() {
        // KVarN8 v8: assemble unwraps these (cache.rs:1408-1412).
        let mut c = kvarn_fixture(8);
        c.kvarn_v_scale = None;
        assert_corrupt(validate_consistency(&roundtrip(&c), 0), "v8 missing v_scale");

        let mut c = kvarn_fixture(8);
        c.kvarn_k_s_row = None;
        assert_corrupt(validate_consistency(&c, 0), "missing k_s_row");

        let mut c = kvarn_fixture(8);
        c.kvarn_sink_k = None;
        c.kvarn_sink_v = None;
        assert_corrupt(validate_consistency(&c, 0), "missing sink with offset>0");

        // Fp16: fetch requires both K and V.
        let mut c = fp16_fixture();
        c.values = None;
        assert_corrupt(validate_consistency(&c, 0), "fp16 missing values");

        // Int8: scales are mandatory for dequantization.
        let mut c = fp16_fixture();
        c.mode = KVCacheMode::Int8; // keys/values present, scales absent
        assert_corrupt(validate_consistency(&roundtrip(&c), 0), "int8 missing scales");
    }

    #[test]
    fn validate_rejects_v4_fold_violation() {
        // v4 contract: kvarn_v_s_row MUST be None (the fold IS its storage).
        let mut c = kvarn_fixture(4);
        assert!(c.kvarn_v_s_row.is_none(), "fixture contract");
        c.kvarn_v_s_row = Some(synth_tensor(&[1, 1, 128, 1], 13));
        assert_corrupt(validate_consistency(&roundtrip(&c), 0), "v4 s_row present");
    }

    #[test]
    fn validate_rejects_param_lockstep_mismatch() {
        // scale dim2 out of lockstep with hist dim2 (128 in the fixture).
        let mut c = kvarn_fixture(8);
        c.kvarn_v_scale = Some(synth_tensor(&[1, 1, 64, 1], 14));
        assert_corrupt(validate_consistency(&c, 0), "v_scale dim2 != hist dim2");

        // s_col dim2 != n_tiles (1 in the fixture).
        let mut c = kvarn_fixture(8);
        c.kvarn_v_s_col = Some(synth_tensor(&[1, 1, 2, 32], 15));
        assert_corrupt(validate_consistency(&c, 0), "s_col dim2 != n_tiles");

        // v4: folded params dim3 != d/gs (32/32 = 1 in the fixture).
        let mut c = kvarn_fixture(4);
        c.kvarn_v_zp = Some(synth_tensor(&[1, 1, 128, 4], 16));
        assert_corrupt(validate_consistency(&c, 0), "v4 zp dim3 != d/gs");
    }

    #[test]
    fn validate_rejects_unaligned_history() {
        // hist dim2 must be a positive multiple of KVARN_TILE_TOKENS (128).
        let mut c = kvarn_fixture(8);
        c.kvarn_hist_k = Some(ffi::astype(
            &synth_tensor(&[1, 1, 100, 32], 17),
            dtype::UINT8,
        ));
        assert_corrupt(validate_consistency(&c, 0), "hist not tile-aligned");
    }

    #[test]
    fn validate_rejects_capacity_below_offset() {
        // Fp16 keys capacity (seq axis) below the logical offset ⇒ the
        // downstream fetch slice would read out of bounds → MLX abort.
        let mut c = fp16_fixture(); // keys seq capacity == 5
        c.offset = 6;
        assert_corrupt(validate_consistency(&roundtrip(&c), 0), "capacity < offset");
    }

    #[test]
    fn validate_rejects_kvarn_state_under_wrong_mode() {
        // Mode byte and tensor table disagree — never guess which lies.
        let mut c = fp16_fixture();
        c.kvarn_sink_k = Some(ffi::astype(&synth_tensor(&[1, 2, 5, 32], 18), dtype::FLOAT16));
        assert_corrupt(validate_consistency(&c, 0), "kvarn droppings on fp16");
    }

    #[test]
    fn read_header_bounds_token_count_without_panicking() {
        let hdr = SessionHeader {
            format_version: FORMAT_VERSION,
            content_hash: "abc".to_string(),
            weight_fingerprint: "def".to_string(),
            model_id: "m3".to_string(),
            template_sig: "tmpl".to_string(),
            layer_count: 2,
            prompt_len: 0,
            current_offset: 0,
            timestamp_secs: 0,
            tokens: vec![], // token count is the final 8 bytes of the header
        };
        let mut buf = Vec::new();
        write_header(&mut buf, &hdr).unwrap();

        // Hostile count: previously `vec![0i32; u64::MAX as usize]` — an
        // allocation abort that Err=>continue in the scan loop cannot catch,
        // so ONE corrupt header killed every subsequent restore scan.
        for hostile in [u64::MAX, (MAX_HEADER_TOKENS as u64) + 1] {
            let mut corrupt = buf.clone();
            let n = corrupt.len();
            corrupt[n - 8..].copy_from_slice(&hostile.to_le_bytes());
            let result = read_header(&mut &corrupt[..]); // plain call: must NOT panic/abort
            assert!(
                result.is_err(),
                "token_count={hostile}: expected Err, got Ok"
            );
        }
    }

    #[test]
    fn read_header_bounds_layer_count_and_prompt_len() {
        let base = |layer_count: u32, prompt_len: usize| SessionHeader {
            format_version: FORMAT_VERSION,
            content_hash: "abc".to_string(),
            weight_fingerprint: "def".to_string(),
            model_id: "m3".to_string(),
            template_sig: "tmpl".to_string(),
            layer_count,
            prompt_len,
            current_offset: 3,
            timestamp_secs: 0,
            tokens: vec![1, 2, 3],
        };

        let mut buf = Vec::new();
        write_header(&mut buf, &base((MAX_HEADER_LAYERS as u32) + 1, 3)).unwrap();
        assert!(read_header(&mut &buf[..]).is_err(), "layer_count over limit");

        let mut buf = Vec::new();
        write_header(&mut buf, &base(2, usize::MAX)).unwrap();
        assert!(read_header(&mut &buf[..]).is_err(), "prompt_len over limit");

        // Bounds must not reject healthy headers (regression guard for the fix).
        let mut buf = Vec::new();
        write_header(&mut buf, &base(2, 3)).unwrap();
        assert!(read_header(&mut &buf[..]).is_ok(), "healthy header rejected");
    }
}

// ---------------------------------------------------------------------------
// Configured-base-dir validation tests
//
// These guard a SILENT failure: pointing the store at an unmounted drive makes
// `create_dir_all` materialise the tree on the boot disk, where every
// observable signal says success. There is no red test to be had after the
// fact, so the check is what has to be tested.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod configured_base_dir_tests {
    use super::*;

    /// Saves and restores `MLXCEL_COLD_STORE_DIR`. The suite runs
    /// `--test-threads=1`, so a process-global env var is safe here; the
    /// restore exists so these tests cannot leak into the ones that construct
    /// a real store from the default path.
    struct EnvGuard(Option<String>);
    impl EnvGuard {
        fn set(value: Option<&str>) -> Self {
            let prior = std::env::var("MLXCEL_COLD_STORE_DIR").ok();
            match value {
                Some(v) => unsafe { std::env::set_var("MLXCEL_COLD_STORE_DIR", v) },
                None => unsafe { std::env::remove_var("MLXCEL_COLD_STORE_DIR") },
            }
            Self(prior)
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.0 {
                Some(v) => unsafe { std::env::set_var("MLXCEL_COLD_STORE_DIR", v) },
                None => unsafe { std::env::remove_var("MLXCEL_COLD_STORE_DIR") },
            }
        }
    }

    #[test]
    fn unset_does_not_apply() {
        let _g = EnvGuard::set(None);
        assert_eq!(validate_configured_base_dir(), Ok(None));
    }

    #[test]
    fn existing_directory_outside_volumes_is_accepted() {
        let tmp = tempfile::tempdir().unwrap();
        let _g = EnvGuard::set(Some(tmp.path().to_str().unwrap()));
        assert_eq!(
            validate_configured_base_dir(),
            Ok(Some(tmp.path().to_path_buf()))
        );
    }

    /// The load-bearing one. A missing directory must NOT be created, because
    /// creating it is precisely how an unmounted drive fills the boot disk.
    #[test]
    fn missing_directory_is_refused_and_not_created() {
        let tmp = tempfile::tempdir().unwrap();
        let absent = tmp.path().join("not-there");
        let _g = EnvGuard::set(Some(absent.to_str().unwrap()));

        assert_eq!(
            validate_configured_base_dir(),
            Err(ColdStoreDirRefusal::Missing(absent.clone()))
        );
        assert!(
            !absent.exists(),
            "validation must not create the directory it is refusing"
        );
    }

    #[test]
    fn a_file_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("regular-file");
        fs::write(&file, b"x").unwrap();
        let _g = EnvGuard::set(Some(file.to_str().unwrap()));
        assert_eq!(
            validate_configured_base_dir(),
            Err(ColdStoreDirRefusal::NotADirectory(file))
        );
    }

    #[test]
    fn volume_root_is_extracted_only_under_volumes() {
        assert_eq!(
            volume_root_of(Path::new("/Volumes/T7 Shield/mlxcel/cold")),
            Some(PathBuf::from("/Volumes/T7 Shield"))
        );
        // The volume root itself, with nothing below it.
        assert_eq!(
            volume_root_of(Path::new("/Volumes/T7 Shield")),
            Some(PathBuf::from("/Volumes/T7 Shield"))
        );
        // Not under /Volumes -> the mount-point rule does not apply.
        assert_eq!(volume_root_of(Path::new("/Users/x/.cache/mlxcel")), None);
        assert_eq!(volume_root_of(Path::new("/Volumes")), None);
        assert_eq!(volume_root_of(Path::new("relative/path")), None);
    }

    /// An ordinary directory is not a mount point — structurally identical to
    /// the `/Volumes/<name>` stub left behind when a drive is unmounted.
    #[test]
    fn ordinary_directory_is_not_a_mount_point() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("sub");
        fs::create_dir(&sub).unwrap();
        assert!(!is_mount_point(&sub));
    }

    /// Positive control for `is_mount_point`: without one, every "not a mount
    /// point" result in this module could equally mean the device comparison
    /// never works at all.
    ///
    /// Do NOT reach for `/System/Volumes/Data` here — it looks like the obvious
    /// choice and it is wrong. Measured on this host: it reports dev=16777233,
    /// *identical* to its parent `/System/Volumes` and to `/`, because it is an
    /// APFS **firmlink** within one volume rather than a device mount. A real
    /// external drive does differ: `/Volumes/T7 Shield` dev=16777244 against
    /// `/Volumes` dev=16777233. The firmlink cost this control one red run.
    ///
    /// So the control locates a genuine device mount at runtime instead of
    /// naming one, and skips loudly when the host has none attached.
    #[test]
    fn known_mount_point_is_detected() {
        // Candidates come from the MOUNT TABLE, never from `is_mount_point`.
        // Selecting with the function under test would make this tautological:
        // it would pass on whatever it selected, and a wholly broken
        // implementation would select nothing and silently skip.
        let Ok(out) = std::process::Command::new("/sbin/mount").output() else {
            eprintln!("SKIPPED: could not run /sbin/mount; negatives UNCONTROLLED");
            return;
        };
        let table = String::from_utf8_lossy(&out.stdout);
        let mounted: Vec<PathBuf> = table
            .lines()
            .filter_map(|l| l.split(" on ").nth(1))
            .filter_map(|rest| rest.rsplit_once(" ("))
            .map(|(mount_point, _opts)| PathBuf::from(mount_point))
            .filter(|p| p.starts_with("/Volumes/"))
            .collect();

        if mounted.is_empty() {
            eprintln!(
                "SKIPPED: mount table lists no volume under /Volumes on this \
                 host, so the negative mount-point results in this module are \
                 UNCONTROLLED."
            );
            return;
        }
        for m in &mounted {
            assert!(
                is_mount_point(m),
                "the mount table lists {} as mounted but the device comparison \
                 disagrees — every negative result in this module would then be \
                 meaningless",
                m.display()
            );
        }
    }

    #[test]
    fn refusals_name_the_path_they_checked() {
        let msg = ColdStoreDirRefusal::VolumeNotMounted {
            path: PathBuf::from("/Volumes/T7 Shield/mlxcel"),
            volume: PathBuf::from("/Volumes/T7 Shield"),
        }
        .to_string();
        assert!(msg.contains("/Volumes/T7 Shield/mlxcel"));
        assert!(msg.contains("BOOT DISK"));
    }
}
