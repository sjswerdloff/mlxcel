//! Construction-gated synchronous v3 cold-store reference implementation.
//!
//! This is the correctness oracle for fresh-process append-clean validation.
//! It is intentionally absent from normal production builds and must not be
//! wired into the serving scheduler's donation path.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Cursor, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use super::{
    ColdStoreError, DetachedCacheSet, DetachedKVCache, collect_array_ptrs, read_kv_cache,
    write_kv_cache,
};
use crate::cache::KVCacheMode;
use crate::cache::kvarn::{
    KVARN_SINKHORN_ITERS, KVARN_TILE_TOKENS, KVARN_V4_GROUP_SIZE,
};
use crate::ffi;

const V3_FORMAT_VERSION: u32 = 3;
const V3_ROOT: &str = "cold-storage-v3";
const COMMIT_MAGIC: &[u8; 16] = b"MLXCEL-COMMIT-V3";
const MAX_HEADER_BYTES: u64 = 32 * 1024 * 1024;
const MAX_STRING_BYTES: usize = 1024 * 1024;
const MAX_TOKENS: usize = 2_000_000;
const MAX_LAYERS: usize = 1024;
const MAX_LAYER_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const MAX_GENERATIONS_SCANNED: usize = 4096;
const MAX_CANDIDATES: usize = 128;
const MAX_RETAINED_CANDIDATE_BYTES: usize = 64 * 1024 * 1024;

static GENERATION_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, PartialEq, Eq)]
struct LayerMetadata {
    index: u32,
    byte_len: u64,
    sha256: [u8; 32],
}

#[derive(Clone, Debug)]
struct V3Header {
    identity_sha256: [u8; 32],
    runtime_fingerprint: [u8; 32],
    layout_fingerprint: [u8; 32],
    model_id: String,
    template_sig: String,
    prompt_len: usize,
    cache_covered_tokens: usize,
    timestamp_nanos: u128,
    tokens: Vec<i32>,
    layers: Vec<LayerMetadata>,
}

#[derive(Clone, Debug)]
struct Candidate {
    header: V3Header,
    generation_dir: PathBuf,
    matched_tokens: usize,
}

/// Published reference snapshot location and identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceSnapshot {
    pub identity_hex: String,
    pub generation: String,
    pub path: PathBuf,
}

/// Synchronous, one-layer-at-a-time v3 writer and reader used only by tests,
/// benchmarks, and the fresh-process fidelity harness.
pub struct ReferenceColdStore {
    base_dir: PathBuf,
    runtime_fingerprint: [u8; 32],
    persist_lock: Mutex<()>,
}

impl ReferenceColdStore {
    pub fn new(base_dir: PathBuf, runtime_fingerprint: [u8; 32]) -> Self {
        Self {
            base_dir,
            runtime_fingerprint,
            persist_lock: Mutex::new(()),
        }
    }

    pub fn root_dir(&self) -> PathBuf {
        self.base_dir.join(V3_ROOT)
    }

    /// Serialize and publish one immutable generation. This call blocks until
    /// `COMMITTED` is visible or returns the exact failure. It is not safe for
    /// the production scheduler hot path.
    pub fn persist(
        &self,
        model_id: &str,
        template_sig: &str,
        tokens: &[i32],
        cache_set: &DetachedCacheSet,
    ) -> Result<ReferenceSnapshot, ColdStoreError> {
        let _guard = self.persist_lock.lock().map_err(|_| {
            ColdStoreError::Io(io::Error::other("reference cold-store persist lock poisoned"))
        })?;
        let cache_covered_tokens = validate_cache_covered_tokens(tokens, cache_set)?;
        let tokens = &tokens[..cache_covered_tokens];
        if cache_set.caches.len() > MAX_LAYERS {
            return Err(invalid_data(format!(
                "layer count {} exceeds limit {MAX_LAYERS}",
                cache_set.caches.len()
            )));
        }

        let layout_fingerprint = layout_fingerprint(cache_set);
        let identity_sha256 = identity_hash(
            &self.runtime_fingerprint,
            model_id,
            template_sig,
            &layout_fingerprint,
            tokens,
        );
        let identity_hex = hex_digest(&identity_sha256);
        let identity_dir = self.root_dir().join(&identity_hex);
        fs::create_dir_all(&identity_dir)?;
        let generation_dir = create_generation_dir(&identity_dir)?;

        let mut all_ptrs = Vec::new();
        for cache in &cache_set.caches {
            collect_array_ptrs(cache, &mut all_ptrs);
        }
        if !all_ptrs.is_empty() {
            unsafe { ffi::eval_all(&all_ptrs) };
        }

        let mut layers = Vec::with_capacity(cache_set.caches.len());
        for (index, cache) in cache_set.caches.iter().enumerate() {
            let mut bytes = Vec::new();
            write_kv_cache(&mut bytes, cache)?;
            let byte_len = u64::try_from(bytes.len()).map_err(|_| {
                invalid_data("serialized layer length does not fit u64".into())
            })?;
            if byte_len > MAX_LAYER_BYTES {
                return Err(invalid_data(format!(
                    "layer {index} length {byte_len} exceeds limit {MAX_LAYER_BYTES}"
                )));
            }
            let metadata = LayerMetadata {
                index: index as u32,
                byte_len,
                sha256: sha256(&bytes),
            };
            write_file(&generation_dir.join(layer_filename(index)), &bytes, false)?;
            layers.push(metadata);
        }

        let header = V3Header {
            identity_sha256,
            runtime_fingerprint: self.runtime_fingerprint,
            layout_fingerprint,
            model_id: model_id.to_string(),
            template_sig: template_sig.to_string(),
            prompt_len: cache_set.prompt_len.min(cache_covered_tokens),
            cache_covered_tokens,
            timestamp_nanos: now_nanos(),
            tokens: tokens.to_vec(),
            layers,
        };
        let header_bytes = encode_header(&header)?;
        if header_bytes.len() as u64 > MAX_HEADER_BYTES {
            return Err(invalid_data(format!(
                "header length {} exceeds limit {MAX_HEADER_BYTES}",
                header_bytes.len()
            )));
        }
        write_file(&generation_dir.join("header.bin"), &header_bytes, false)?;

        let mut commit = Vec::with_capacity(COMMIT_MAGIC.len() + 32);
        commit.extend_from_slice(COMMIT_MAGIC);
        commit.extend_from_slice(&sha256(&header_bytes));
        let commit_tmp = generation_dir.join(format!(
            "COMMITTED.tmp.{}",
            GENERATION_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        write_file(&commit_tmp, &commit, true)?;
        fs::rename(&commit_tmp, generation_dir.join("COMMITTED"))?;

        let generation = generation_dir
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| invalid_data("generation path is not UTF-8".into()))?
            .to_string();
        tracing::warn!(
            identity = %identity_hex,
            generation = %generation,
            layers = header.layers.len(),
            "COLD-STORE REFERENCE WRITER persisted a generation; Gate 2 remains OPEN"
        );
        Ok(ReferenceSnapshot {
            identity_hex,
            generation,
            path: generation_dir,
        })
    }

    /// Load the best fully committed, checksummed candidate. Invalid longest
    /// candidates are skipped so a valid shorter generation can still resume.
    /// This oracle returns the raw committed prefix; a production caller must
    /// separately apply request-minus-one backoff and model prefill alignment.
    pub fn load_prefix(
        &self,
        model_id: &str,
        template_sig: &str,
        tokens: &[i32],
    ) -> Result<(DetachedCacheSet, usize), ColdStoreError> {
        let mut candidates = self.collect_candidates(model_id, template_sig, tokens)?;
        candidates.sort_by(|left, right| {
            right
                .matched_tokens
                .cmp(&left.matched_tokens)
                .then_with(|| right.header.timestamp_nanos.cmp(&left.header.timestamp_nanos))
        });
        candidates.truncate(MAX_CANDIDATES);

        for candidate in candidates {
            if let Ok(cache_set) = load_candidate(&candidate) {
                return Ok((cache_set, candidate.matched_tokens));
            }
        }
        Err(ColdStoreError::NoMatch)
    }

    fn collect_candidates(
        &self,
        model_id: &str,
        template_sig: &str,
        request_tokens: &[i32],
    ) -> Result<Vec<Candidate>, ColdStoreError> {
        let root = self.root_dir();
        if !root.exists() {
            return Err(ColdStoreError::NoMatch);
        }
        let mut scanned = 0usize;
        let mut candidates = Vec::new();
        for identity in fs::read_dir(root)? {
            let identity = identity?;
            if !identity.file_type()?.is_dir() {
                continue;
            }
            let identity_name = match identity.file_name().into_string() {
                Ok(name) => name,
                Err(_) => continue,
            };
            for generation in fs::read_dir(identity.path())? {
                if scanned >= MAX_GENERATIONS_SCANNED {
                    break;
                }
                scanned += 1;
                let generation = generation?;
                if !generation.file_type()?.is_dir() {
                    continue;
                }
                let Some(header) = read_committed_header(&generation.path()) else {
                    continue;
                };
                if header.runtime_fingerprint != self.runtime_fingerprint
                    || header.model_id != model_id
                    || header.template_sig != template_sig
                    || identity_name != hex_digest(&header.identity_sha256)
                    || header.identity_sha256
                        != identity_hash(
                            &header.runtime_fingerprint,
                            &header.model_id,
                            &header.template_sig,
                            &header.layout_fingerprint,
                            &header.tokens,
                        )
                {
                    continue;
                }
                let matched_tokens = header
                    .tokens
                    .iter()
                    .zip(request_tokens)
                    .take_while(|(stored, request)| stored == request)
                    .count();
                if matched_tokens == 0 {
                    continue;
                }
                candidates.push(Candidate {
                    header,
                    generation_dir: generation.path(),
                    matched_tokens,
                });
                retain_best_bounded_candidates(&mut candidates);
            }
            if scanned >= MAX_GENERATIONS_SCANNED {
                break;
            }
        }
        if candidates.is_empty() {
            return Err(ColdStoreError::NoMatch);
        }
        Ok(candidates)
    }
}

/// Hash a canonical runtime manifest supplied by the fidelity harness.
///
/// The manifest must include model weights/configuration, tokenizer/template,
/// load-time surgery, backend dtype policy, and MLX/Metal build identity. This
/// helper hashes bytes; it does not claim an incomplete manifest is complete.
pub fn runtime_fingerprint_from_manifest(manifest: &[u8]) -> [u8; 32] {
    sha256(manifest)
}

fn validate_cache_covered_tokens(
    tokens: &[i32],
    cache_set: &DetachedCacheSet,
) -> Result<usize, ColdStoreError> {
    if cache_set.caches.is_empty() || !cache_set.has_consistent_seq_len() {
        return Err(invalid_data(
            "cache-covered layer offsets are empty or inconsistent".into(),
        ));
    }
    let cache_covered_tokens = usize::try_from(cache_set.seq_len())
        .map_err(|_| invalid_data("cache-covered layer offset is negative".into()))?;
    if cache_covered_tokens == 0
        || cache_covered_tokens > tokens.len()
        || cache_covered_tokens > MAX_TOKENS
    {
        return Err(invalid_data(format!(
            "cache-covered length {cache_covered_tokens} is invalid for {} available tokens",
            tokens.len()
        )));
    }
    Ok(cache_covered_tokens)
}

fn load_candidate(candidate: &Candidate) -> Result<DetachedCacheSet, ColdStoreError> {
    let header = &candidate.header;
    let mut caches = Vec::with_capacity(header.layers.len());
    for (expected_index, metadata) in header.layers.iter().enumerate() {
        if metadata.index as usize != expected_index || metadata.byte_len > MAX_LAYER_BYTES {
            return Err(invalid_data("invalid layer metadata index or length".into()));
        }
        let path = candidate
            .generation_dir
            .join(layer_filename(expected_index));
        let actual_len = fs::metadata(&path)?.len();
        if actual_len != metadata.byte_len {
            return Err(invalid_data(format!(
                "layer {expected_index} length mismatch"
            )));
        }
        let actual_len = usize::try_from(actual_len)
            .map_err(|_| invalid_data("layer length does not fit usize".into()))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(actual_len)
            .map_err(|error| invalid_data(format!("cannot allocate layer: {error}")))?;
        File::open(&path)?.read_to_end(&mut bytes)?;
        if sha256(&bytes) != metadata.sha256 {
            return Err(invalid_data(format!(
                "layer {expected_index} checksum mismatch"
            )));
        }
        let mut cursor = Cursor::new(bytes.as_slice());
        let cache = read_kv_cache(&mut cursor, expected_index)?;
        if cursor.position() != actual_len as u64 {
            return Err(invalid_data(format!(
                "layer {expected_index} contains trailing bytes"
            )));
        }
        caches.push(cache);
    }

    let cache_set = DetachedCacheSet {
        caches,
        backend: super::super::SequenceStateBackend::DenseKvCache,
        prompt_len: header.prompt_len,
        current_offset: header.cache_covered_tokens as i32,
        created_at: std::time::Instant::now(),
        detached_at: std::time::Instant::now(),
        origin_seq_id: super::super::SequenceId(0),
    };
    if !cache_set.has_consistent_seq_len()
        || usize::try_from(cache_set.seq_len()).ok() != Some(header.cache_covered_tokens)
        || header.tokens.len() != header.cache_covered_tokens
        || header.prompt_len > header.cache_covered_tokens
        || layout_fingerprint(&cache_set) != header.layout_fingerprint
    {
        return Err(invalid_data(
            "restored state does not match committed header".into(),
        ));
    }
    Ok(cache_set)
}

fn read_committed_header(generation_dir: &Path) -> Option<V3Header> {
    let mut commit = Vec::new();
    File::open(generation_dir.join("COMMITTED"))
        .ok()?
        .read_to_end(&mut commit)
        .ok()?;
    if commit.len() != COMMIT_MAGIC.len() + 32 || &commit[..COMMIT_MAGIC.len()] != COMMIT_MAGIC {
        return None;
    }
    let expected_header_sha: [u8; 32] = commit[COMMIT_MAGIC.len()..].try_into().ok()?;
    let header_path = generation_dir.join("header.bin");
    let header_len = fs::metadata(&header_path).ok()?.len();
    if header_len > MAX_HEADER_BYTES {
        return None;
    }
    let header_len = usize::try_from(header_len).ok()?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(header_len).ok()?;
    File::open(header_path).ok()?.read_to_end(&mut bytes).ok()?;
    if sha256(&bytes) != expected_header_sha {
        return None;
    }
    decode_header(&bytes).ok()
}

fn create_generation_dir(identity_dir: &Path) -> Result<PathBuf, ColdStoreError> {
    for _ in 0..32 {
        let generation = format!(
            "gen-{:x}-{:x}-{:x}",
            std::process::id(),
            now_nanos(),
            GENERATION_COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let path = identity_dir.join(generation);
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(invalid_data(
        "could not create a unique generation directory".into(),
    ))
}

fn encode_header(header: &V3Header) -> Result<Vec<u8>, ColdStoreError> {
    if header.model_id.len() > MAX_STRING_BYTES
        || header.template_sig.len() > MAX_STRING_BYTES
        || header.tokens.len() > MAX_TOKENS
        || header.layers.len() > MAX_LAYERS
    {
        return Err(invalid_data("header field exceeds configured limit".into()));
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&V3_FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&header.identity_sha256);
    bytes.extend_from_slice(&header.runtime_fingerprint);
    bytes.extend_from_slice(&header.layout_fingerprint);
    write_bounded_string(&mut bytes, &header.model_id)?;
    write_bounded_string(&mut bytes, &header.template_sig)?;
    bytes.extend_from_slice(&(header.prompt_len as u64).to_le_bytes());
    bytes.extend_from_slice(&(header.cache_covered_tokens as u64).to_le_bytes());
    bytes.extend_from_slice(&header.timestamp_nanos.to_le_bytes());
    bytes.extend_from_slice(&(header.tokens.len() as u64).to_le_bytes());
    for token in &header.tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    bytes.extend_from_slice(&(header.layers.len() as u32).to_le_bytes());
    for layer in &header.layers {
        bytes.extend_from_slice(&layer.index.to_le_bytes());
        bytes.extend_from_slice(&layer.byte_len.to_le_bytes());
        bytes.extend_from_slice(&layer.sha256);
    }
    Ok(bytes)
}

fn decode_header(bytes: &[u8]) -> Result<V3Header, ColdStoreError> {
    let mut reader = BufReader::new(Cursor::new(bytes));
    let format_version = read_u32(&mut reader)?;
    if format_version != V3_FORMAT_VERSION {
        return Err(ColdStoreError::FormatVersionMismatch {
            stored: format_version,
            current: V3_FORMAT_VERSION,
        });
    }
    let identity_sha256 = read_digest(&mut reader)?;
    let runtime_fingerprint = read_digest(&mut reader)?;
    let layout_fingerprint = read_digest(&mut reader)?;
    let model_id = read_bounded_string(&mut reader)?;
    let template_sig = read_bounded_string(&mut reader)?;
    let prompt_len = bounded_usize(read_u64(&mut reader)?, MAX_TOKENS, "prompt length")?;
    let cache_covered_tokens = bounded_usize(
        read_u64(&mut reader)?,
        MAX_TOKENS,
        "cache-covered token count",
    )?;
    let timestamp_nanos = read_u128(&mut reader)?;
    let token_count = bounded_usize(read_u64(&mut reader)?, MAX_TOKENS, "token count")?;
    let mut tokens = Vec::new();
    tokens
        .try_reserve_exact(token_count)
        .map_err(|error| invalid_data(format!("cannot allocate tokens: {error}")))?;
    for _ in 0..token_count {
        tokens.push(read_i32(&mut reader)?);
    }
    let layer_count = bounded_usize(read_u32(&mut reader)? as u64, MAX_LAYERS, "layer count")?;
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(layer_count)
        .map_err(|error| invalid_data(format!("cannot allocate layer metadata: {error}")))?;
    for _ in 0..layer_count {
        let index = read_u32(&mut reader)?;
        let byte_len = read_u64(&mut reader)?;
        if byte_len > MAX_LAYER_BYTES {
            return Err(invalid_data(format!(
                "layer length {byte_len} exceeds limit {MAX_LAYER_BYTES}"
            )));
        }
        layers.push(LayerMetadata {
            index,
            byte_len,
            sha256: read_digest(&mut reader)?,
        });
    }
    if reader.stream_position()? != bytes.len() as u64 {
        return Err(invalid_data("header contains trailing bytes".into()));
    }
    if tokens.len() != cache_covered_tokens
        || prompt_len > cache_covered_tokens
        || layers.is_empty()
    {
        return Err(invalid_data("header causal metadata is inconsistent".into()));
    }
    for (expected, layer) in layers.iter().enumerate() {
        if layer.index as usize != expected {
            return Err(invalid_data("header layer indices are not canonical".into()));
        }
    }
    Ok(V3Header {
        identity_sha256,
        runtime_fingerprint,
        layout_fingerprint,
        model_id,
        template_sig,
        prompt_len,
        cache_covered_tokens,
        timestamp_nanos,
        tokens,
        layers,
    })
}

fn identity_hash(
    runtime_fingerprint: &[u8; 32],
    model_id: &str,
    template_sig: &str,
    layout_fingerprint: &[u8; 32],
    tokens: &[i32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(V3_FORMAT_VERSION.to_le_bytes());
    update_len_delimited(&mut hasher, runtime_fingerprint);
    update_len_delimited(&mut hasher, model_id.as_bytes());
    update_len_delimited(&mut hasher, template_sig.as_bytes());
    update_len_delimited(&mut hasher, layout_fingerprint);
    hasher.update((tokens.len() as u64).to_le_bytes());
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    hasher.finalize().into()
}

fn layout_fingerprint(cache_set: &DetachedCacheSet) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update((cache_set.caches.len() as u64).to_le_bytes());
    for cache in &cache_set.caches {
        hash_cache_layout(&mut hasher, cache);
    }
    hasher.finalize().into()
}

fn hash_cache_layout(hasher: &mut Sha256, cache: &DetachedKVCache) {
    hasher.update([cache.mode as u8]);
    hasher.update(cache.step.to_le_bytes());
    hasher.update(cache.turbo_seed.to_le_bytes());
    hasher.update(cache.hot_threshold.to_le_bytes());
    hasher.update([cache.delegated_fp16_fast_path as u8]);
    hasher.update([cache.delegated_fp16_sidecar_policy as u8]);
    hasher.update([cache.kvarn_v_bits]);
    if cache.mode == KVCacheMode::KVarN8 {
        hasher.update([8]); // KVarN8 K width is fixed at 8 bits.
        hasher.update((KVARN_SINKHORN_ITERS as u64).to_le_bytes());
        hasher.update(KVARN_TILE_TOKENS.to_le_bytes());
        hasher.update(KVARN_V4_GROUP_SIZE.to_le_bytes());
    } else {
        hasher.update([0]);
        hasher.update(0u64.to_le_bytes());
        hasher.update(0i32.to_le_bytes());
        hasher.update(0i32.to_le_bytes());
    }
    for array in [
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
    ] {
        match array {
            Some(array) => {
                hasher.update([1]);
                hasher.update(ffi::array_dtype(array).to_le_bytes());
                let shape = ffi::array_shape(array);
                hasher.update((shape.len() as u64).to_le_bytes());
                for dim in shape {
                    hasher.update(dim.to_le_bytes());
                }
            }
            None => hasher.update([0]),
        }
    }
}

fn update_len_delimited(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn retain_best_bounded_candidates(candidates: &mut Vec<Candidate>) {
    candidates.sort_by(|left, right| {
        right
            .matched_tokens
            .cmp(&left.matched_tokens)
            .then_with(|| right.header.timestamp_nanos.cmp(&left.header.timestamp_nanos))
    });
    while candidates.len() > MAX_CANDIDATES
        || candidates.iter().map(candidate_heap_bytes).sum::<usize>()
            > MAX_RETAINED_CANDIDATE_BYTES
    {
        candidates.pop();
    }
}

fn candidate_heap_bytes(candidate: &Candidate) -> usize {
    std::mem::size_of::<Candidate>()
        .saturating_add(candidate.header.model_id.len())
        .saturating_add(candidate.header.template_sig.len())
        .saturating_add(
            candidate
                .header
                .tokens
                .len()
                .saturating_mul(std::mem::size_of::<i32>()),
        )
        .saturating_add(
            candidate
                .header
                .layers
                .len()
                .saturating_mul(std::mem::size_of::<LayerMetadata>()),
        )
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn hex_digest(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn layer_filename(index: usize) -> String {
    format!("layer_{index:04}.bin")
}

fn now_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn write_file(path: &Path, bytes: &[u8], create_new: bool) -> io::Result<()> {
    let file = if create_new {
        OpenOptions::new().write(true).create_new(true).open(path)?
    } else {
        File::create(path)?
    };
    let mut writer = BufWriter::new(file);
    writer.write_all(bytes)?;
    writer.flush()
}

fn write_bounded_string(writer: &mut impl Write, value: &str) -> io::Result<()> {
    let len = u32::try_from(value.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "string exceeds u32"))?;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(value.as_bytes())
}

fn read_bounded_string(reader: &mut impl Read) -> Result<String, ColdStoreError> {
    let len = bounded_usize(read_u32(reader)? as u64, MAX_STRING_BYTES, "string length")?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|error| invalid_data(format!("cannot allocate string: {error}")))?;
    bytes.resize(len, 0);
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|error| invalid_data(format!("invalid UTF-8: {error}")))
}

fn read_digest(reader: &mut impl Read) -> io::Result<[u8; 32]> {
    let mut digest = [0u8; 32];
    reader.read_exact(&mut digest)?;
    Ok(digest)
}

fn read_i32(reader: &mut impl Read) -> io::Result<i32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_u128(reader: &mut impl Read) -> io::Result<u128> {
    let mut bytes = [0u8; 16];
    reader.read_exact(&mut bytes)?;
    Ok(u128::from_le_bytes(bytes))
}

fn bounded_usize(value: u64, maximum: usize, field: &str) -> Result<usize, ColdStoreError> {
    let value = usize::try_from(value)
        .map_err(|_| invalid_data(format!("{field} does not fit usize")))?;
    if value > maximum {
        return Err(invalid_data(format!(
            "{field} {value} exceeds limit {maximum}"
        )));
    }
    Ok(value)
}

fn invalid_data(detail: String) -> ColdStoreError {
    ColdStoreError::Io(io::Error::new(io::ErrorKind::InvalidData, detail))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::cold_store::tests::make_test_cache_set;

    #[test]
    fn v3_identity_hashes_tokens_beyond_4096() {
        let runtime = runtime_fingerprint_from_manifest(b"runtime");
        let layout = sha256(b"layout");
        let mut first = vec![7; 4097];
        let mut second = first.clone();
        first[4096] = 8;
        second[4096] = 9;

        assert_ne!(
            identity_hash(&runtime, "m", "t", &layout, &first),
            identity_hash(&runtime, "m", "t", &layout, &second)
        );
    }

    #[test]
    fn reference_round_trip_publishes_committed_generation() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReferenceColdStore::new(
            dir.path().to_path_buf(),
            runtime_fingerprint_from_manifest(b"runtime"),
        );
        let tokens = vec![1, 2, 3, 4, 5, 6];
        let cache = make_test_cache_set(2, 5, 16);

        let snapshot = store.persist("m3", "tmpl", &tokens, &cache).unwrap();

        assert!(snapshot.path.join("COMMITTED").is_file());
        assert_eq!(snapshot.identity_hex.len(), 64);
        let (loaded, matched) = store.load_prefix("m3", "tmpl", &tokens).unwrap();
        assert_eq!(matched, 5);
        assert_eq!(loaded.seq_len(), 5);
    }

    #[test]
    fn uncommitted_generation_is_invisible() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReferenceColdStore::new(
            dir.path().to_path_buf(),
            runtime_fingerprint_from_manifest(b"runtime"),
        );
        let generation = store.root_dir().join("identity").join("generation");
        fs::create_dir_all(&generation).unwrap();
        fs::write(generation.join("header.bin"), b"partial").unwrap();

        assert!(matches!(
            store.load_prefix("m3", "tmpl", &[1, 2, 3]),
            Err(ColdStoreError::NoMatch)
        ));
    }

    #[test]
    fn corrupt_longest_candidate_falls_back_to_shorter_generation() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReferenceColdStore::new(
            dir.path().to_path_buf(),
            runtime_fingerprint_from_manifest(b"runtime"),
        );
        let short_tokens = vec![1, 2, 3];
        let long_tokens = vec![1, 2, 3, 4, 5];
        store
            .persist("m3", "tmpl", &short_tokens, &make_test_cache_set(1, 3, 16))
            .unwrap();
        let long = store
            .persist("m3", "tmpl", &long_tokens, &make_test_cache_set(1, 5, 16))
            .unwrap();
        let layer = long.path.join(layer_filename(0));
        let mut bytes = fs::read(&layer).unwrap();
        bytes[0] ^= 1;
        fs::write(layer, bytes).unwrap();

        let (loaded, matched) = store.load_prefix("m3", "tmpl", &long_tokens).unwrap();

        assert_eq!(matched, 3);
        assert_eq!(loaded.seq_len(), 3);
    }

    #[test]
    fn exclusive_generation_creation_is_collision_safe() {
        let dir = tempfile::tempdir().unwrap();
        let identity = dir.path().join("identity");
        fs::create_dir_all(&identity).unwrap();
        let mut threads = Vec::new();
        for _ in 0..8 {
            let identity = identity.clone();
            threads.push(std::thread::spawn(move || create_generation_dir(&identity).unwrap()));
        }
        let mut paths: Vec<_> = threads.into_iter().map(|thread| thread.join().unwrap()).collect();
        paths.sort();
        paths.dedup();
        assert_eq!(paths.len(), 8);
    }

    #[test]
    fn candidate_retention_is_bounded_and_keeps_longest_prefixes() {
        let mut candidates = Vec::new();
        for matched_tokens in 1..=MAX_CANDIDATES + 1 {
            candidates.push(Candidate {
                header: V3Header {
                    identity_sha256: [0; 32],
                    runtime_fingerprint: [0; 32],
                    layout_fingerprint: [0; 32],
                    model_id: "m".into(),
                    template_sig: "t".into(),
                    prompt_len: 1,
                    cache_covered_tokens: 1,
                    timestamp_nanos: matched_tokens as u128,
                    tokens: vec![1],
                    layers: vec![LayerMetadata {
                        index: 0,
                        byte_len: 1,
                        sha256: [0; 32],
                    }],
                },
                generation_dir: PathBuf::new(),
                matched_tokens,
            });
            retain_best_bounded_candidates(&mut candidates);
        }

        assert_eq!(candidates.len(), MAX_CANDIDATES);
        assert_eq!(candidates.first().unwrap().matched_tokens, MAX_CANDIDATES + 1);
        assert_eq!(candidates.last().unwrap().matched_tokens, 2);
    }
}
