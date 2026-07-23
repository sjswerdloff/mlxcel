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
//! # Storage layout
//!
//! ```text
//! ~/.cache/mlxcel/cold-storage/
//!   <hex(content_hash)>/
//!     header.bin        — metadata + weight-fingerprint + token sequence
//!     layer_0.bin       — per-layer DetachedKVCache tensors
//!     layer_1.bin
//!     ...
//! ```
//!
//! # Safety
//!
//! The weight-fingerprint guard refuses to restore a cache that was written
//! by a different model checkpoint.

use std::collections::hash_map::DefaultHasher;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};
use std::time::{SystemTime, UNIX_EPOCH};

use cxx::UniquePtr;
use sha2::{Digest, Sha256};

use crate::dtype;
use crate::ffi;
use crate::ffi::MlxArray;

use super::detach::{DetachedCacheSet, DetachedKVCache};
use super::KVCacheMode;

#[cfg(any(test, feature = "coldstore-reference-sync"))]
#[path = "cold_store_reference.rs"]
mod reference;
#[cfg(any(test, feature = "coldstore-reference-sync"))]
pub use reference::{ReferenceColdStore, ReferenceSnapshot, runtime_fingerprint_from_manifest};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const FORMAT_VERSION: u32 = 2; // v2: content-addressed, tokens in header
const DEFAULT_BASE_DIR: &str = ".cache/mlxcel/cold-storage";
const MAX_SERIALIZED_TENSOR_RANK: usize = 32;

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

fn read_kv_cache(r: &mut impl Read, layer_idx: usize) -> Result<DetachedKVCache, ColdStoreError> {
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
// Session header (v2: content-addressed)
// ---------------------------------------------------------------------------

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
    let prompt_len = read_u64(r)? as usize;
    let current_offset = read_i32(r)?;
    let timestamp_secs = read_u64(r)?;
    let token_count = read_u64(r)? as usize;
    let mut tokens = vec![0i32; token_count];
    for tok in &mut tokens {
        *tok = read_i32(r)?;
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
pub struct ColdStore {
    base_dir: PathBuf,
    weight_fingerprint: String,
    writer_tx: Option<Sender<WriteJob>>,
    writer_handle: Option<JoinHandle<()>>,
}

struct WriteJob {
    content_hash: String,
    header_bytes: Vec<u8>,
    layer_bytes: Vec<Vec<u8>>,
    base_dir: PathBuf,
}

impl ColdStore {
    pub fn new(model_path: &str) -> Self {
        Self::with_base_dir(default_base_dir(), model_path)
    }

    pub fn with_base_dir(base_dir: PathBuf, model_path: &str) -> Self {
        let weight_fingerprint = compute_weight_fingerprint(model_path);
        let (tx, handle) = spawn_writer();
        ColdStore {
            base_dir,
            weight_fingerprint,
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
        let content_hash = compute_content_hash(model_id, template_sig, tokens);

        let header = SessionHeader {
            format_version: FORMAT_VERSION,
            content_hash: content_hash.clone(),
            weight_fingerprint: self.weight_fingerprint.clone(),
            model_id: model_id.to_string(),
            template_sig: template_sig.to_string(),
            layer_count: cache_set.caches.len() as u32,
            prompt_len: cache_set.prompt_len.min(cache_covered_len),
            current_offset: cache_set.seq_len(),
            timestamp_secs: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            tokens: tokens.to_vec(),
        };

        let mut header_bytes = Vec::new();
        write_header(&mut header_bytes, &header)?;

        let mut layer_bytes = Vec::with_capacity(cache_set.caches.len());

        // Batch-evaluate all arrays across all layers before writing.
        // This forces a single GPU→CPU synchronization instead of one per
        // array per layer (600 sync calls for k8v4 with 60 layers).
        let mut all_ptrs: Vec<*const MlxArray> = Vec::new();
        for cache in &cache_set.caches {
            collect_array_ptrs(cache, &mut all_ptrs);
        }
        if !all_ptrs.is_empty() {
            unsafe { ffi::eval_all(&all_ptrs); }
        }

        for cache in &cache_set.caches {
            let mut buf = Vec::new();
            write_kv_cache(&mut buf, cache)?;
            layer_bytes.push(buf);
        }

        tx.send(WriteJob {
            content_hash,
            header_bytes,
            layer_bytes,
            base_dir: self.base_dir.clone(),
        })
        .map_err(|_| ColdStoreError::WorkerDied)
    }

    /// Load the best matching cache for a given token prefix.
    ///
    /// Scans all stored entries and returns the one with the longest
    /// matching prefix. The weight fingerprint is validated before returning.
    pub fn load_prefix(
        &self,
        model_id: &str,
        template_sig: &str,
        tokens: &[i32],
    ) -> Result<(DetachedCacheSet, usize), ColdStoreError> {
        if !self.base_dir.exists() {
            return Err(ColdStoreError::NoMatch);
        }

        let mut best: Option<(SessionHeader, String)> = None;
        let mut best_match_len = 0usize;

        for entry in fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let dir_name = entry.file_name();
            let header_path = entry.path().join("header.bin");
            if !header_path.exists() {
                continue;
            }

            // Read header to check for prefix match.
            let header = match File::open(&header_path) {
                Ok(f) => {
                    let mut reader = BufReader::new(f);
                    match read_header(&mut reader) {
                        Ok(h) => h,
                        Err(_) => continue, // Skip corrupt entries.
                    }
                }
                Err(_) => continue,
            };

            // Must match model and template.
            if header.model_id != model_id || header.template_sig != template_sig {
                continue;
            }

            // Find longest common prefix between stored tokens and request tokens.
            let match_len = header
                .tokens
                .iter()
                .zip(tokens.iter())
                .take_while(|(a, b)| a == b)
                .count();

            if match_len > best_match_len {
                best_match_len = match_len;
                best = Some((header, dir_name.to_string_lossy().to_string()));
            }
        }

        let (header, dir_name) = best.ok_or(ColdStoreError::NoMatch)?;

        // Weight-fingerprint guard.
        if header.weight_fingerprint != self.weight_fingerprint {
            return Err(ColdStoreError::WeightMismatch {
                stored: header.weight_fingerprint,
                current: self.weight_fingerprint.clone(),
            });
        }

        // Load layers.
        let session_dir = self.base_dir.join(&dir_name);
        let mut caches = Vec::with_capacity(header.layer_count as usize);
        for i in 0..header.layer_count {
            let layer_path = session_dir.join(format!("layer_{i}.bin"));
            let mut f = BufReader::new(File::open(&layer_path)?);
            caches.push(read_kv_cache(&mut f, i as usize)?);
        }

        Ok((
            DetachedCacheSet {
                caches,
                backend: super::SequenceStateBackend::DenseKvCache,
                prompt_len: header.prompt_len,
                current_offset: header.current_offset,
                created_at: std::time::Instant::now(),
                detached_at: std::time::Instant::now(),
                origin_seq_id: super::SequenceId(0),
            },
            best_match_len,
        ))
    }

    /// Invalidate a specific cache entry by content hash.
    pub fn invalidate(&self, content_hash: &str) -> Result<(), ColdStoreError> {
        let session_dir = self.base_dir.join(content_hash);
        if session_dir.exists() {
            fs::remove_dir_all(&session_dir)?;
        }
        Ok(())
    }

    /// List all stored content hashes.
    pub fn list_entries(&self) -> Result<Vec<String>, ColdStoreError> {
        let mut entries = Vec::new();
        if self.base_dir.exists() {
            for entry in fs::read_dir(&self.base_dir)? {
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

fn spawn_writer() -> (Sender<WriteJob>, JoinHandle<()>) {
    let (tx, rx) = mpsc::channel::<WriteJob>();
    let handle = thread::spawn(move || {
        while let Ok(job) = rx.recv() {
            if let Err(e) = write_job_to_disk(&job) {
                tracing::error!(
                    content_hash = job.content_hash,
                    error = %e,
                    "cold-store: failed to write entry"
                );
            }
        }
    });
    (tx, handle)
}

fn write_job_to_disk(job: &WriteJob) -> Result<(), ColdStoreError> {
    let session_dir = job.base_dir.join(&job.content_hash);
    fs::create_dir_all(&session_dir)?;

    let header_path = session_dir.join("header.bin");
    {
        let mut f = BufWriter::new(File::create(&header_path)?);
        f.write_all(&job.header_bytes)?;
        f.flush()?;
    }

    for (i, layer_data) in job.layer_bytes.iter().enumerate() {
        let layer_path = session_dir.join(format!("layer_{i}.bin"));
        let mut f = BufWriter::new(File::create(&layer_path)?);
        f.write_all(layer_data)?;
        f.flush()?;
    }

    tracing::info!(
        content_hash = job.content_hash,
        layers = job.layer_bytes.len(),
        "cold-store: entry persisted"
    );

    Ok(())
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

fn write_string(w: &mut impl Write, s: &str) -> io::Result<()> {
    w.write_all(&(s.len() as u32).to_le_bytes())?;
    w.write_all(s.as_bytes())
}

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
mod tests {
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

    pub(super) fn make_test_cache_set(
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
}
