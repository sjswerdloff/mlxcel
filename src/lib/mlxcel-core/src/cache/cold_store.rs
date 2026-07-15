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

//! KV cache cold-storage: persist detached caches to SSD for fast restart.
//!
//! Each session gets its own directory under a configurable base path
//! (default `~/.cache/mlxcel/cold-storage/<session-uuid>/`). Caches are
//! serialized synchronously then written to disk in a background thread,
//! and cleaned up when invalidated (e.g. after compaction).
//!
//! # Storage format
//!
//! Each session directory contains:
//! - `header.bin` — metadata + weight-fingerprint safety guard
//! - `layer_<N>.bin` — per-layer DetachedKVCache tensors
//!
//! # Safety
//!
//! The weight-fingerprint guard refuses to restore a cache that was written
//! by a different model checkpoint. Without this, cross-weight corruption
//! would be silent (Paxton's boundary).

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};
use std::time::{SystemTime, UNIX_EPOCH};

use cxx::UniquePtr;

use crate::dtype;
use crate::ffi;
use crate::ffi::MlxArray;

use super::detach::{DetachedCacheSet, DetachedKVCache};
use super::KVCacheMode;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const FORMAT_VERSION: u32 = 1;
const DEFAULT_BASE_DIR: &str = ".cache/mlxcel/cold-storage";

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum ColdStoreError {
    Io(io::Error),
    WeightMismatch { stored: String, current: String },
    FormatVersionMismatch { stored: u32, current: u32 },
    CorruptLayer { layer: usize, detail: String },
    SessionNotFound(String),
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
            Self::SessionNotFound(s) => write!(f, "cold-store session not found: {s}"),
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
// Weight fingerprint
// ---------------------------------------------------------------------------

/// Compute a weight fingerprint for the currently loaded model.
///
/// Uses a hash of the model path + safetensors file sizes as a lightweight
/// proxy for "same checkpoint".
pub fn compute_weight_fingerprint(model_path: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    model_path.hash(&mut hasher);

    if let Ok(entries) = fs::read_dir(model_path) {
        let mut sizes: Vec<(String, u64)> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
            .filter_map(|e| {
                let size = e.metadata().ok()?.len();
                Some((e.file_name().to_string_lossy().to_string(), size))
            })
            .collect();
        sizes.sort();
        for (name, size) in &sizes {
            name.hash(&mut hasher);
            size.hash(&mut hasher);
        }
    }

    format!("{:016x}", hasher.finish())
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
    let ndim = read_i32(r)? as usize;
    let mut shape = vec![0i32; ndim];
    for s in &mut shape {
        *s = read_i32(r)?;
    }
    let byte_len = read_u64(r)? as usize;
    let mut bytes = vec![0u8; byte_len];
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
// Session header
// ---------------------------------------------------------------------------

struct SessionHeader {
    format_version: u32,
    weight_fingerprint: String,
    model_path: String,
    layer_count: u32,
    prompt_len: usize,
    current_offset: i32,
    timestamp_secs: u64,
}

fn write_header(w: &mut impl Write, hdr: &SessionHeader) -> io::Result<()> {
    w.write_all(&hdr.format_version.to_le_bytes())?;
    write_string(w, &hdr.weight_fingerprint)?;
    write_string(w, &hdr.model_path)?;
    w.write_all(&hdr.layer_count.to_le_bytes())?;
    w.write_all(&(hdr.prompt_len as u64).to_le_bytes())?;
    w.write_all(&hdr.current_offset.to_le_bytes())?;
    w.write_all(&hdr.timestamp_secs.to_le_bytes())?;
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
    Ok(SessionHeader {
        format_version,
        weight_fingerprint: read_string(r)?,
        model_path: read_string(r)?,
        layer_count: read_u32(r)?,
        prompt_len: read_u64(r)? as usize,
        current_offset: read_i32(r)?,
        timestamp_secs: read_u64(r)?,
    })
}

// ---------------------------------------------------------------------------
// ColdStore
// ---------------------------------------------------------------------------

/// Manages cold-storage sessions on SSD.
///
/// Serialization happens synchronously (UniquePtr<MlxArray> is not Send).
/// The resulting bytes are sent to a background thread for disk I/O.
pub struct ColdStore {
    base_dir: PathBuf,
    model_path: String,
    weight_fingerprint: String,
    writer_tx: Option<Sender<WriteJob>>,
    writer_handle: Option<JoinHandle<()>>,
}

/// A pre-serialized session ready for disk write. This is Send-safe.
struct WriteJob {
    session_uuid: String,
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
            model_path: model_path.to_string(),
            weight_fingerprint,
            writer_tx: Some(tx),
            writer_handle: Some(handle),
        }
    }

    /// Asynchronously persist a DetachedCacheSet to SSD.
    ///
    /// Serializes the cache set in the caller thread (UniquePtr<MlxArray>
    /// is not Send), then sends the raw bytes to a background writer.
    pub fn persist(&self, session_uuid: &str, cache_set: &DetachedCacheSet) -> Result<(), ColdStoreError> {
        let tx = self.writer_tx.as_ref().ok_or(ColdStoreError::WorkerDied)?;

        // Serialize header
        let header = SessionHeader {
            format_version: FORMAT_VERSION,
            weight_fingerprint: self.weight_fingerprint.clone(),
            model_path: self.model_path.clone(),
            layer_count: cache_set.caches.len() as u32,
            prompt_len: cache_set.prompt_len,
            current_offset: cache_set.current_offset,
            timestamp_secs: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        let mut header_bytes = Vec::new();
        write_header(&mut header_bytes, &header)?;

        // Serialize each layer
        let mut layer_bytes = Vec::with_capacity(cache_set.caches.len());
        for cache in &cache_set.caches {
            let mut buf = Vec::new();
            write_kv_cache(&mut buf, cache)?;
            layer_bytes.push(buf);
        }

        tx.send(WriteJob {
            session_uuid: session_uuid.to_string(),
            header_bytes,
            layer_bytes,
            base_dir: self.base_dir.clone(),
        })
        .map_err(|_| ColdStoreError::WorkerDied)
    }

    /// Load a cached session from SSD.
    ///
    /// Validates the weight-fingerprint before restoring.
    pub fn load(&self, session_uuid: &str) -> Result<DetachedCacheSet, ColdStoreError> {
        let session_dir = self.base_dir.join(session_uuid);
        if !session_dir.exists() {
            return Err(ColdStoreError::SessionNotFound(session_uuid.to_string()));
        }

        let header_path = session_dir.join("header.bin");
        let header = {
            let mut f = BufReader::new(File::open(&header_path)?);
            read_header(&mut f)?
        };

        if header.weight_fingerprint != self.weight_fingerprint {
            return Err(ColdStoreError::WeightMismatch {
                stored: header.weight_fingerprint,
                current: self.weight_fingerprint.clone(),
            });
        }

        let mut caches = Vec::with_capacity(header.layer_count as usize);
        for i in 0..header.layer_count {
            let layer_path = session_dir.join(format!("layer_{i}.bin"));
            let mut f = BufReader::new(File::open(&layer_path)?);
            caches.push(read_kv_cache(&mut f, i as usize)?);
        }

        Ok(DetachedCacheSet {
            caches,
            backend: super::SequenceStateBackend::DenseKvCache,
            prompt_len: header.prompt_len,
            current_offset: header.current_offset,
            created_at: std::time::Instant::now(),
            detached_at: std::time::Instant::now(),
            origin_seq_id: super::SequenceId(0),
        })
    }

    /// Invalidate (delete) a session's cold-storage directory.
    pub fn invalidate(&self, session_uuid: &str) -> Result<(), ColdStoreError> {
        let session_dir = self.base_dir.join(session_uuid);
        if session_dir.exists() {
            fs::remove_dir_all(&session_dir)?;
        }
        Ok(())
    }

    /// List all stored session UUIDs.
    pub fn list_sessions(&self) -> Result<Vec<String>, ColdStoreError> {
        let mut sessions = Vec::new();
        if self.base_dir.exists() {
            for entry in fs::read_dir(&self.base_dir)? {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    if let Some(name) = entry.file_name().to_str() {
                        sessions.push(name.to_string());
                    }
                }
            }
        }
        Ok(sessions)
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
                    session = job.session_uuid,
                    error = %e,
                    "cold-store: failed to write session"
                );
            }
        }
    });
    (tx, handle)
}

fn write_job_to_disk(job: &WriteJob) -> Result<(), ColdStoreError> {
    let session_dir = job.base_dir.join(&job.session_uuid);
    fs::create_dir_all(&session_dir)?;

    // Write header
    let header_path = session_dir.join("header.bin");
    {
        let mut f = BufWriter::new(File::create(&header_path)?);
        f.write_all(&job.header_bytes)?;
        f.flush()?;
    }

    // Write layers
    for (i, layer_data) in job.layer_bytes.iter().enumerate() {
        let layer_path = session_dir.join(format!("layer_{i}.bin"));
        let mut f = BufWriter::new(File::create(&layer_path)?);
        f.write_all(layer_data)?;
        f.flush()?;
    }

    tracing::info!(
        session = job.session_uuid,
        layers = job.layer_bytes.len(),
        "cold-store: session persisted"
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

fn array_from_raw_bytes(bytes: &[u8], dt: i32, shape: &[i32]) -> Result<UniquePtr<MlxArray>, io::Error> {
    // For FP16/BF16: convert to FP32 for reconstruction, then cast back.
    // For FP32/INT32/UINT32: use direct from_slice_*.
    // For INT8/UINT8: convert to INT32, then cast back.
    let arr = match dt {
        dtype::FLOAT32 => {
            let data: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            ffi::from_slice_f32(&data, shape)
        }
        dtype::FLOAT16 | dtype::BFLOAT16 => {
            // 2 bytes per element → read as u16 → convert to f32.
            let f32_data: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|c| fp16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            ffi::from_slice_f32(&f32_data, shape)
            // Note: the caller should cast back to the original dtype if needed.
            // For cold-storage, we return FP32 and let the caller handle dtype.
        }
        dtype::INT8 | dtype::UINT8 => {
            // 1 byte per element.
            let data: Vec<i32> = bytes.iter().map(|&b| b as i32).collect();
            ffi::from_slice_i32(&data, shape)
        }
        dtype::INT32 => {
            let data: Vec<i32> = bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            ffi::from_slice_i32(&data, shape)
        }
        dtype::UINT32 => {
            let data: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            ffi::from_slice_u32(&data, shape)
        }
        _ => return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported dtype: {dt}"),
        )),
    };
    // Cast to the target dtype if not already correct.
    let arr_dt = ffi::array_dtype(&arr);
    if arr_dt != dt {
        Ok(ffi::astype(&arr, dt))
    } else {
        Ok(arr)
    }
}

/// Convert FP16 bits (u16) to f32.
fn fp16_bits_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            // ±zero
            if sign == 1 { -0.0f32 } else { 0.0f32 }
        } else {
            // Denormalized
            let f = (frac as f32) / (1 << 24) as f32; // frac * 2^-14 * 2^-10
            if sign == 1 { -f } else { f }
        }
    } else if exp == 31 {
        if frac == 0 {
            // ±inf
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else {
            f32::NAN
        }
    } else {
        // Normalized
        let f = 1.0f32 + (frac as f32) / 1024.0f32;
        let exp_f = 2.0f32.powi(exp - 15);
        let result = f * exp_f;
        if sign == 1 { -result } else { result }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
    fn session_header_round_trip() {
        let hdr = SessionHeader {
            format_version: FORMAT_VERSION,
            weight_fingerprint: "abc123".to_string(),
            model_path: "/models/m3".to_string(),
            layer_count: 32,
            prompt_len: 1024,
            current_offset: 512,
            timestamp_secs: 1700000000,
        };
        let mut buf = Vec::new();
        write_header(&mut buf, &hdr).unwrap();
        let mut reader = &buf[..];
        let restored = read_header(&mut reader).unwrap();
        assert_eq!(restored.format_version, FORMAT_VERSION);
        assert_eq!(restored.weight_fingerprint, "abc123");
        assert_eq!(restored.layer_count, 32);
        assert_eq!(restored.prompt_len, 1024);
        assert_eq!(restored.current_offset, 512);
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
    fn cold_store_list_sessions_empty() {
        let dir = tempfile::tempdir().unwrap();
        let cs = ColdStore::with_base_dir(dir.path().to_path_buf(), "/dummy");
        let sessions = cs.list_sessions().unwrap();
        assert!(sessions.is_empty());
    }

    #[test]
    fn cold_store_invalidate_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let cs = ColdStore::with_base_dir(dir.path().to_path_buf(), "/dummy");
        cs.invalidate("nonexistent-uuid").unwrap();
    }

    #[test]
    fn cold_store_load_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let cs = ColdStore::with_base_dir(dir.path().to_path_buf(), "/dummy");
        let result = cs.load("nonexistent-uuid");
        assert!(matches!(result, Err(ColdStoreError::SessionNotFound(_))));
    }

    // End-to-end tests with synthetic DetachedKVCache data.

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

    fn make_test_cache_set(num_layers: usize, seq_len: i32, head_dim: i32) -> DetachedCacheSet {
        use super::super::SequenceStateBackend;
        use std::time::Instant;

        let mut caches = Vec::new();
        for layer in 0..num_layers {
            let seed_k = 1000 + layer as u32;
            let seed_v = 2000 + layer as u32;
            // Convert FP32 synthetic data to FP16 for realism.
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
    fn cold_store_persist_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model");
        fs::create_dir_all(&model_path).unwrap();

        let cs = ColdStore::with_base_dir(dir.path().join("cs").to_path_buf(), model_path.to_str().unwrap());
        let session_uuid = "test-session-001";
        let original = make_test_cache_set(2, 8, 32); // 2 layers, 8 tokens, dim=32

        // Persist
        cs.persist(session_uuid, &original).unwrap();

        // Give the background writer a moment to finish.
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Verify directory structure
        let session_dir = cs.base_dir().join(session_uuid);
        assert!(session_dir.exists());
        assert!(session_dir.join("header.bin").exists());
        assert!(session_dir.join("layer_0.bin").exists());
        assert!(session_dir.join("layer_1.bin").exists());

        // Load
        let loaded = cs.load(session_uuid).unwrap();
        assert_eq!(loaded.caches.len(), 2);
        assert_eq!(loaded.prompt_len, 8);
        assert_eq!(loaded.current_offset, 8);
        assert_eq!(loaded.caches[0].offset, 8);
        assert_eq!(loaded.caches[0].mode, KVCacheMode::Fp16);

        // Verify tensor data matches (compare FP32-flattened values)
        for layer in 0..2 {
            let orig_k = original.caches[layer].keys.as_ref().unwrap();
            let loaded_k = loaded.caches[layer].keys.as_ref().unwrap();
            let orig_flat = flatten_fp32(orig_k);
            let loaded_flat = flatten_fp32(loaded_k);
            assert_eq!(orig_flat.len(), loaded_flat.len(), "layer {layer} K length mismatch");
            for (i, (a, b)) in orig_flat.iter().zip(loaded_flat.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-3,
                    "layer {layer} K[{i}]: {a} vs {b}"
                );
            }

            let orig_v = original.caches[layer].values.as_ref().unwrap();
            let loaded_v = loaded.caches[layer].values.as_ref().unwrap();
            let orig_flat = flatten_fp32(orig_v);
            let loaded_flat = flatten_fp32(loaded_v);
            assert_eq!(orig_flat.len(), loaded_flat.len(), "layer {layer} V length mismatch");
            for (i, (a, b)) in orig_flat.iter().zip(loaded_flat.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-3,
                    "layer {layer} V[{i}]: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn cold_store_weight_mismatch_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let model_a = dir.path().join("model-a");
        let model_b = dir.path().join("model-b");
        fs::create_dir_all(&model_a).unwrap();
        fs::create_dir_all(&model_b).unwrap();

        let cs_a = ColdStore::with_base_dir(dir.path().join("cs").to_path_buf(), model_a.to_str().unwrap());
        let cs_b = ColdStore::with_base_dir(dir.path().join("cs").to_path_buf(), model_b.to_str().unwrap());

        let original = make_test_cache_set(1, 4, 16);
        cs_a.persist("session-001", &original).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Loading with cs_a (same model) should succeed.
        assert!(cs_a.load("session-001").is_ok());

        // Loading with cs_b (different model) should fail with WeightMismatch.
        let result = cs_b.load("session-001");
        assert!(matches!(result, Err(ColdStoreError::WeightMismatch { .. })));
    }

    #[test]
    fn cold_store_invalidate_removes_directory() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model");
        fs::create_dir_all(&model_path).unwrap();

        let cs = ColdStore::with_base_dir(dir.path().join("cs").to_path_buf(), model_path.to_str().unwrap());
        let original = make_test_cache_set(1, 4, 16);
        cs.persist("session-002", &original).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));

        assert!(cs.base_dir().join("session-002").exists());
        cs.invalidate("session-002").unwrap();
        assert!(!cs.base_dir().join("session-002").exists());
    }

    #[test]
    fn cold_store_list_after_persist() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model");
        fs::create_dir_all(&model_path).unwrap();

        let cs = ColdStore::with_base_dir(dir.path().join("cs").to_path_buf(), model_path.to_str().unwrap());
        let original = make_test_cache_set(1, 4, 16);

        cs.persist("aaa", &original).unwrap();
        cs.persist("bbb", &original).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));

        let mut sessions = cs.list_sessions().unwrap();
        sessions.sort();
        assert_eq!(sessions, vec!["aaa", "bbb"]);
    }
}
