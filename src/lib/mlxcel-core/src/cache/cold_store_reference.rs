//! Synchronous v3 cold-store implementation: the correctness oracle for
//! fresh-process append-clean validation, and — since Gate 2 closed — the
//! PRODUCTION persist/restore machinery behind `ColdStore`.
//!
//! The async production `ColdStore` splits work across the Q3 thread-affinity
//! boundary: the inference thread serializes ([`serialize_cache_set_layers`] +
//! [`layout_fingerprint`]), and its background writer publishes via
//! [`ReferenceColdStore::persist_serialized`]. Loads delegate to
//! [`ReferenceColdStore::load_prefix`] (checksum-verified, COMMITTED-gated).

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

/// Cold-store prune mode for ancestor snapshot cleanup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PruneMode {
    /// Prune disabled — no scanning, no logging, no deletion.
    Off,
    /// Observe mode — log what WOULD be pruned, do not delete.
    /// Default: validate on real data before enabling deletion.
    #[default]
    Observe,
    /// Delete mode — log and delete unreachable ancestors.
    Delete,
}

/// Synchronous, one-layer-at-a-time v3 writer and reader used only by tests,
/// benchmarks, and the fresh-process fidelity harness.
pub struct ReferenceColdStore {
    base_dir: PathBuf,
    runtime_fingerprint: [u8; 32],
    persist_lock: Mutex<()>,
    prune_mode: PruneMode,
}

impl ReferenceColdStore {
    pub fn new(base_dir: PathBuf, runtime_fingerprint: [u8; 32]) -> Self {
        let prune_mode = match std::env::var("MLXCEL_COLD_STORE_PRUNE_MODE")
            .as_deref()
            .unwrap_or("observe")
        {
            "off" => PruneMode::Off,
            "observe" => PruneMode::Observe,
            "delete" => PruneMode::Delete,
            _ => PruneMode::Observe,
        };
        Self {
            base_dir,
            runtime_fingerprint,
            persist_lock: Mutex::new(()),
            prune_mode,
        }
    }

    pub fn with_prune_mode(mut self, mode: PruneMode) -> Self {
        self.prune_mode = mode;
        self
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
        let cache_covered_tokens = validate_cache_covered_tokens(tokens, cache_set)?;
        let covered = &tokens[..cache_covered_tokens];
        if cache_set.caches.len() > MAX_LAYERS {
            return Err(invalid_data(format!(
                "layer count {} exceeds limit {MAX_LAYERS}",
                cache_set.caches.len()
            )));
        }
        let layout_fingerprint = layout_fingerprint(cache_set);
        let layer_bytes = serialize_cache_set_layers(cache_set)?;
        self.persist_serialized(
            model_id,
            template_sig,
            covered,
            cache_set.prompt_len,
            layout_fingerprint,
            &layer_bytes,
        )
    }

    /// Publish PRE-SERIALIZED layer bytes as one immutable, per-layer-checksummed,
    /// atomically-committed generation.
    ///
    /// This is the write half of the async production split: the inference thread
    /// serializes the cache (via [`serialize_cache_set_layers`]) and computes the
    /// [`layout_fingerprint`] — because `MlxArray` -> raw bytes is thread-affine
    /// under Metal (see the concurrency verdict) — then a background writer thread
    /// calls this to publish. Only `Vec<u8>` crosses the thread boundary. The
    /// caller MUST have serialized every layer on the thread that owns the arrays
    /// and computed `layout_fingerprint` from the same cache set.
    pub fn persist_serialized(
        &self,
        model_id: &str,
        template_sig: &str,
        covered_tokens: &[i32],
        prompt_len: usize,
        layout_fingerprint: [u8; 32],
        layer_bytes: &[Vec<u8>],
    ) -> Result<ReferenceSnapshot, ColdStoreError> {
        let _guard = self.persist_lock.lock().map_err(|_| {
            ColdStoreError::Io(io::Error::other("reference cold-store persist lock poisoned"))
        })?;
        let cache_covered_tokens = covered_tokens.len();
        if cache_covered_tokens == 0 || cache_covered_tokens > MAX_TOKENS {
            return Err(invalid_data(format!(
                "cache-covered length {cache_covered_tokens} is invalid"
            )));
        }
        if layer_bytes.is_empty() || layer_bytes.len() > MAX_LAYERS {
            return Err(invalid_data(format!(
                "layer count {} is invalid (limit {MAX_LAYERS})",
                layer_bytes.len()
            )));
        }

        let identity_sha256 = identity_hash(
            &self.runtime_fingerprint,
            model_id,
            template_sig,
            &layout_fingerprint,
            covered_tokens,
        );
        let identity_hex = hex_digest(&identity_sha256);
        let identity_dir = self.root_dir().join(&identity_hex);
        fs::create_dir_all(&identity_dir)?;
        let generation_dir = create_generation_dir(&identity_dir)?;

        let mut layers = Vec::with_capacity(layer_bytes.len());
        for (index, bytes) in layer_bytes.iter().enumerate() {
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
                sha256: sha256(bytes),
            };
            write_file(&generation_dir.join(layer_filename(index)), bytes, false)?;
            layers.push(metadata);
        }

        let header = V3Header {
            identity_sha256,
            runtime_fingerprint: self.runtime_fingerprint,
            layout_fingerprint,
            model_id: model_id.to_string(),
            template_sig: template_sig.to_string(),
            prompt_len: prompt_len.min(cache_covered_tokens),
            cache_covered_tokens,
            timestamp_nanos: now_nanos(),
            tokens: covered_tokens.to_vec(),
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
            "COLD-STORE v3 published a committed generation"
        );

        // Prune ancestors: entries whose tokens are a strict prefix of the
        // newly persisted entry. These are unreachable by load_prefix (the
        // longest-match invariant guarantees the new entry always wins).
        self.prune_ancestor_snapshots(
            model_id,
            template_sig,
            &layout_fingerprint,
            covered_tokens,
            &identity_hex,
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

    /// Delete cold-store entries whose tokens are a strict prefix of
    /// `new_tokens`. Best-effort: failures are logged, not propagated.
    ///
    /// **Invariant:** `load_prefix` sorts by `matched_tokens` DESC, then
    /// `timestamp_nanos` DESC (lines 257-262). If A's tokens are a strict
    /// prefix of B's tokens, B always has `matched_tokens >= A`. When equal
    /// (request diverges exactly at `len(A)`), the timestamp tiebreak keeps
    /// B ahead (B is newest). This invariant depends on the sort order.
    ///
    /// **Fallback tradeoff:** Pruning A removes the shorter-prefix fallback
    /// that `load_prefix` uses if B fails to load (lines 246-247, 265-269).
    /// The cold-store is a CACHE — worst case is re-prefill, not data loss.
    ///
    /// **Concurrent read safety:** `load_prefix`/`collect_candidates` do NOT
    /// take `persist_lock`. A concurrent read during `remove_dir_all` sees a
    /// vanished file → returns `None` → skips. Fail-safe.
    fn prune_ancestor_snapshots(
        &self,
        new_model_id: &str,
        new_template_sig: &str,
        new_layout_fingerprint: &[u8; 32],
        new_tokens: &[i32],
        new_identity_hex: &str,
    ) {
        if self.prune_mode == PruneMode::Off {
            return;
        }
        let is_observe = self.prune_mode == PruneMode::Observe;
        let root = self.root_dir();
        if !root.exists() {
            return;
        }
        let Ok(entries) = fs::read_dir(&root) else {
            return;
        };
        for entry in entries {
            let Ok(entry) = entry else { continue };
            let Ok(ft) = entry.file_type() else { continue };
            // Symlink guard: skip non-directory entries and symlinks
            if !ft.is_dir() || entry.path().is_symlink() {
                continue;
            }
            let identity_name = match entry.file_name().into_string() {
                Ok(name) => name,
                Err(_) => continue,
            };
            if identity_name == new_identity_hex {
                continue; // don't delete self
            }
            // Find the latest COMMITTED generation
            let mut latest: Option<V3Header> = None;
            let Ok(gen_entries) = fs::read_dir(entry.path()) else {
                continue;
            };
            for gen_entry in gen_entries {
                let Ok(gen_entry) = gen_entry else { continue };
                let Ok(gen_ft) = gen_entry.file_type() else { continue };
                if !gen_ft.is_dir() {
                    continue;
                }
                if let Some(header) = read_committed_header(&gen_entry.path()) {
                    match &latest {
                        Some(prev) if header.timestamp_nanos <= prev.timestamp_nanos => {}
                        _ => latest = Some(header),
                    }
                }
            }
            let Some(header) = latest else {
                continue;
            };
            // Filter: same identity EXCEPT tokens
            if header.model_id != new_model_id
                || header.template_sig != new_template_sig
                || header.runtime_fingerprint != self.runtime_fingerprint
                || header.layout_fingerprint != *new_layout_fingerprint
            {
                continue;
            }
            // Filter: strict prefix (candidate tokens are a prefix of new tokens)
            if header.tokens.len() >= new_tokens.len() {
                continue;
            }
            if header.tokens[..] != new_tokens[..header.tokens.len()] {
                continue;
            }
            // This entry's tokens are a strict prefix of the new entry.
            // load_prefix will never select it.
            if is_observe {
                tracing::info!(
                    would_prune = %identity_name,
                    old_token_count = header.tokens.len(),
                    new_token_count = new_tokens.len(),
                    "COLD-STORE observe: would prune ancestor snapshot"
                );
            } else {
                match fs::remove_dir_all(entry.path()) {
                    Ok(()) => {
                        tracing::info!(
                            pruned = %identity_name,
                            old_token_count = header.tokens.len(),
                            new_token_count = new_tokens.len(),
                            "COLD-STORE pruned ancestor snapshot"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            pruned = %identity_name,
                            error = %e,
                            "COLD-STORE failed to prune ancestor snapshot"
                        );
                    }
                }
            }
        }
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

/// Serialize every layer of a cache set to raw bytes ON THE CALLING THREAD.
///
/// `MlxArray` -> raw bytes is thread-affine under Metal (`array_to_raw_bytes`
/// runs `contiguous()`+`eval()` on the calling thread's Metal command context),
/// so this MUST run on the thread that owns the arrays (the inference/generation
/// thread). The resulting per-layer `Vec<u8>` is the ONLY thing safe to hand to
/// a background writer thread for publication via [`ReferenceColdStore::persist_serialized`].
pub(crate) fn serialize_cache_set_layers(
    cache_set: &DetachedCacheSet,
) -> Result<Vec<Vec<u8>>, ColdStoreError> {
    let mut all_ptrs = Vec::new();
    for cache in &cache_set.caches {
        collect_array_ptrs(cache, &mut all_ptrs);
    }
    if !all_ptrs.is_empty() {
        unsafe { ffi::eval_all(&all_ptrs) };
    }
    let mut layer_bytes = Vec::with_capacity(cache_set.caches.len());
    for cache in &cache_set.caches {
        let mut bytes = Vec::new();
        write_kv_cache(&mut bytes, cache)?;
        layer_bytes.push(bytes);
    }
    Ok(layer_bytes)
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
        super::validate_consistency(&cache, expected_index)?;
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

pub(crate) fn layout_fingerprint(cache_set: &DetachedCacheSet) -> [u8; 32] {
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

    #[test]
    fn prune_deletes_strict_prefix_ancestor() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReferenceColdStore::new(
            dir.path().to_path_buf(),
            runtime_fingerprint_from_manifest(b"runtime"),
        )
        .with_prune_mode(PruneMode::Delete);
        let short_tokens = vec![1, 2, 3];
        let long_tokens = vec![1, 2, 3, 4, 5];
        // Use the same seq_len for both cache sets so layout_fingerprint matches.
        // The short entry covers fewer tokens but the tensors have the same shape.
        let cache_long = make_test_cache_set(1, 5, 16);
        let mut cache_short = make_test_cache_set(1, 5, 16);
        // Trim the short cache to only cover 3 tokens (trim_to adjusts offset)
        for c in &mut cache_short.caches {
            c.offset = 3;
        }
        let short_snap = store
            .persist("m3", "tmpl", &short_tokens, &cache_short)
            .unwrap();
        let long_snap = store
            .persist("m3", "tmpl", &long_tokens, &cache_long)
            .unwrap();

        // Short entry's identity dir should be deleted
        let short_dir = dir.path().join("cold-storage-v3").join(&short_snap.identity_hex);
        assert!(!short_dir.exists(), "strict-prefix ancestor should be pruned");
        // Long entry's identity dir should still exist
        let long_dir = dir.path().join("cold-storage-v3").join(&long_snap.identity_hex);
        assert!(long_dir.exists(), "new entry should not be pruned");
        // Only one entry remains
        let entries: Vec<_> = fs::read_dir(store.root_dir()).unwrap().collect();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn prune_keeps_different_model() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReferenceColdStore::new(
            dir.path().to_path_buf(),
            runtime_fingerprint_from_manifest(b"runtime"),
        )
        .with_prune_mode(PruneMode::Delete);
        let tokens_a = vec![1, 2, 3];
        let tokens_b = vec![1, 2, 3, 4, 5];
        // Same layout for both (seq_len=5, trim short to 3)
        let cache_b = make_test_cache_set(1, 5, 16);
        let mut cache_a = make_test_cache_set(1, 5, 16);
        for c in &mut cache_a.caches {
            c.offset = 3;
        }
        let snap_a = store
            .persist("model-a", "tmpl", &tokens_a, &cache_a)
            .unwrap();
        store
            .persist("model-b", "tmpl", &tokens_b, &cache_b)
            .unwrap();

        // Entry from model-a should NOT be pruned (different model_id)
        let dir_a = dir.path().join("cold-storage-v3").join(&snap_a.identity_hex);
        assert!(dir_a.exists(), "different-model entry should not be pruned");
        let entries: Vec<_> = fs::read_dir(store.root_dir()).unwrap().collect();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn prune_keeps_same_length_different_content() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReferenceColdStore::new(
            dir.path().to_path_buf(),
            runtime_fingerprint_from_manifest(b"runtime"),
        )
        .with_prune_mode(PruneMode::Delete);
        let tokens_a = vec![1, 2, 3];
        let tokens_b = vec![4, 5, 6];
        let snap_a = store
            .persist("m3", "tmpl", &tokens_a, &make_test_cache_set(1, 3, 16))
            .unwrap();
        store
            .persist("m3", "tmpl", &tokens_b, &make_test_cache_set(1, 3, 16))
            .unwrap();

        // Neither should be pruned (same length, different content)
        let dir_a = dir.path().join("cold-storage-v3").join(&snap_a.identity_hex);
        assert!(dir_a.exists(), "same-length entry should not be pruned");
        let entries: Vec<_> = fs::read_dir(store.root_dir()).unwrap().collect();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn prune_keeps_shorter_but_not_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReferenceColdStore::new(
            dir.path().to_path_buf(),
            runtime_fingerprint_from_manifest(b"runtime"),
        )
        .with_prune_mode(PruneMode::Delete);
        // [1,2,9] is shorter than [1,2,3,4,5] but NOT a prefix (diverges at index 2)
        let short_tokens = vec![1, 2, 9];
        let long_tokens = vec![1, 2, 3, 4, 5];
        let cache_long = make_test_cache_set(1, 5, 16);
        let mut cache_short = make_test_cache_set(1, 5, 16);
        for c in &mut cache_short.caches {
            c.offset = 3;
        }
        let short_snap = store
            .persist("m3", "tmpl", &short_tokens, &cache_short)
            .unwrap();
        store
            .persist("m3", "tmpl", &long_tokens, &cache_long)
            .unwrap();

        // Short entry should NOT be pruned (not a prefix — diverges at index 2)
        let short_dir = dir.path().join("cold-storage-v3").join(&short_snap.identity_hex);
        assert!(short_dir.exists(), "non-prefix shorter entry should not be pruned");
        let entries: Vec<_> = fs::read_dir(store.root_dir()).unwrap().collect();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn prune_keeps_different_layout_fingerprint() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReferenceColdStore::new(
            dir.path().to_path_buf(),
            runtime_fingerprint_from_manifest(b"runtime"),
        )
        .with_prune_mode(PruneMode::Delete);
        let short_tokens = vec![1, 2, 3];
        let long_tokens = vec![1, 2, 3, 4, 5];
        // Create two cache sets with different layouts (different layer counts)
        let cache_3 = make_test_cache_set(1, 3, 16);
        let cache_5_diff = make_test_cache_set(2, 5, 16);
        let snap_a = store
            .persist("m3", "tmpl", &short_tokens, &cache_3)
            .unwrap();
        store
            .persist("m3", "tmpl", &long_tokens, &cache_5_diff)
            .unwrap();

        // Entry should NOT be pruned (different layout_fingerprint)
        let dir_a = dir.path().join("cold-storage-v3").join(&snap_a.identity_hex);
        assert!(dir_a.exists(), "different-layout entry should not be pruned");
        let entries: Vec<_> = fs::read_dir(store.root_dir()).unwrap().collect();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn prune_skips_corrupt_entry() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReferenceColdStore::new(
            dir.path().to_path_buf(),
            runtime_fingerprint_from_manifest(b"runtime"),
        )
        .with_prune_mode(PruneMode::Delete);
        let short_tokens = vec![1, 2, 3];
        let long_tokens = vec![1, 2, 3, 4, 5];
        // Same layout for both (seq_len=5, trim short to 3)
        let cache_long = make_test_cache_set(1, 5, 16);
        let mut cache_short = make_test_cache_set(1, 5, 16);
        for c in &mut cache_short.caches {
            c.offset = 3;
        }
        let short_snap = store
            .persist("m3", "tmpl", &short_tokens, &cache_short)
            .unwrap();
        // Corrupt the COMMITTED file
        let committed = dir
            .path()
            .join("cold-storage-v3")
            .join(&short_snap.identity_hex)
            .join(&short_snap.generation)
            .join("COMMITTED");
        fs::write(&committed, b"corrupt").unwrap();
        store
            .persist("m3", "tmpl", &long_tokens, &cache_long)
            .unwrap();

        // Corrupt entry should be skipped (not deleted, not crashing)
        let short_dir = dir.path().join("cold-storage-v3").join(&short_snap.identity_hex);
        assert!(short_dir.exists(), "corrupt entry should be skipped, not deleted");
    }

    #[test]
    fn prune_observe_mode_does_not_delete() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReferenceColdStore::new(
            dir.path().to_path_buf(),
            runtime_fingerprint_from_manifest(b"runtime"),
        )
        .with_prune_mode(PruneMode::Observe);
        let short_tokens = vec![1, 2, 3];
        let long_tokens = vec![1, 2, 3, 4, 5];
        let cache_long = make_test_cache_set(1, 5, 16);
        let mut cache_short = make_test_cache_set(1, 5, 16);
        for c in &mut cache_short.caches {
            c.offset = 3;
        }
        let short_snap = store
            .persist("m3", "tmpl", &short_tokens, &cache_short)
            .unwrap();
        store
            .persist("m3", "tmpl", &long_tokens, &cache_long)
            .unwrap();

        // In observe mode, the ancestor should NOT be deleted
        let short_dir = dir.path().join("cold-storage-v3").join(&short_snap.identity_hex);
        assert!(short_dir.exists(), "observe mode should not delete");
        let entries: Vec<_> = fs::read_dir(store.root_dir()).unwrap().collect();
        assert_eq!(entries.len(), 2, "both entries should survive in observe mode");
    }
}
