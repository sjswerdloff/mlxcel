// Cold-store v4: block-based content-addressed storage.
//
// Splits KV caches into fixed-size blocks. Blocks are content-addressed
// by Merkle-chain hash. A manifest describes which blocks make up a
// full conversation. Shared prefix blocks are stored once and
// reference-counted.
//
// See docs/DESIGN_cold_store_block_storage_v4_2026-07-26.md for the
// full design.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use sha2::{Digest, Sha256};

use super::cold_store::{ColdStoreError, PruneMode, serialize_cache_set_layers, read_kv_cache};
use super::{DetachedCacheSet, DetachedKVCache, KVCacheMode, SequenceId, SequenceStateBackend};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const V4_FORMAT_VERSION: u32 = 4;
const V4_ROOT: &str = "cold-storage-v4";
const BLOCKS_DIR: &str = "blocks";
const MANIFESTS_DIR: &str = "manifests";
const COMMIT_MAGIC: &[u8; 16] = b"MLXCEL-COMMIT-V4";

const DEFAULT_BLOCK_SIZE: usize = 2048;
const MIN_BLOCK_SIZE: usize = 2048;
const MAX_BLOCK_SIZE: usize = 8192;
const TILE_ALIGNMENT: usize = 128; // KVarN8 tile size

const MAX_HEADER_BYTES: u64 = 32 * 1024 * 1024;
const MAX_STRING_BYTES: usize = 1024 * 1024;
const MAX_TOKENS: usize = 2_000_000;
const MAX_LAYERS: usize = 1024;
const MAX_BLOCK_SIZE_BYTES: u64 = 64 * 1024 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Block hash (Merkle-chain)
// ---------------------------------------------------------------------------

/// Compute the Merkle-chain block hash.
///
/// `prev_hash` is `block_hash[N-1]` for N>=1, or 32 zero bytes for N=0.
/// `block_size` is the configured block size.
/// `kv_mode_config` is a string encoding the KV cache mode and quantization params.
/// `own_tokens` is the block's own token slice `[start..end)`.
///
/// block_hash[N] = SHA-256(prev_hash || block_size || kv_mode_config || token_count || own_tokens)
pub fn block_hash_merkle(
    prev_hash: &[u8; 32],
    block_size: usize,
    kv_mode_config: &str,
    own_tokens: &[i32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(prev_hash);
    hasher.update((block_size as u64).to_le_bytes());
    hasher.update(kv_mode_config.as_bytes());
    hasher.update((own_tokens.len() as u64).to_le_bytes());
    for token in own_tokens {
        hasher.update(token.to_le_bytes());
    }
    hasher.finalize().into()
}

/// Build the kv_mode_config string for a given KV cache mode.
///
/// This string is included in the block hash to prevent collisions between
/// blocks from different KV modes (e.g., KVarN8 vs fp16).
pub fn kv_mode_config_string(mode: super::KVCacheMode) -> String {
    // Include mode tag and key quantization parameters.
    // This must capture everything that affects the KV data bytes.
    format!("{:?}", mode)
}

/// Validate that a block size is 2048 or a power-of-2 multiple, and a
/// multiple of TILE_ALIGNMENT (128).
pub fn validate_block_size(block_size: usize) -> Result<(), ColdStoreError> {
    if block_size < MIN_BLOCK_SIZE {
        return Err(invalid_data(format!(
            "block_size {block_size} is below minimum {MIN_BLOCK_SIZE}"
        )));
    }
    if block_size > MAX_BLOCK_SIZE {
        return Err(invalid_data(format!(
            "block_size {block_size} exceeds maximum {MAX_BLOCK_SIZE}"
        )));
    }
    if !block_size.is_power_of_two() {
        return Err(invalid_data(format!(
            "block_size {block_size} is not a power of two"
        )));
    }
    if block_size % TILE_ALIGNMENT != 0 {
        return Err(invalid_data(format!(
            "block_size {block_size} is not a multiple of {TILE_ALIGNMENT} (KVarN8 tile alignment)"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

/// A manifest describes which blocks make up a full conversation.
#[derive(Clone, Debug)]
pub struct Manifest {
    pub runtime_fingerprint: [u8; 32],
    pub model_id: String,
    pub template_sig: String,
    pub block_size: usize,
    pub block_hashes: Vec<[u8; 32]>,
    pub prompt_len: usize,
    pub total_tokens: usize,
    pub timestamp_nanos: u128,
}

impl Manifest {
    /// Compute the manifest hash.
    ///
    /// manifest_hash = SHA-256(runtime_fingerprint || model_id || template_sig || block_hashes[0] || ... || block_hashes[N-1])
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.runtime_fingerprint);
        hasher.update(self.model_id.as_bytes());
        hasher.update(self.template_sig.as_bytes());
        for bh in &self.block_hashes {
            hasher.update(bh);
        }
        hasher.finalize().into()
    }

    /// Hex digest of the manifest hash.
    pub fn hash_hex(&self) -> String {
        hex_digest(&self.hash())
    }
}

// ---------------------------------------------------------------------------
// BlockColdStore
// ///

/// A v4 block-based cold store.
pub struct BlockColdStore {
    base_dir: PathBuf,
    runtime_fingerprint: [u8; 32],
    block_size: usize,
    prune_mode: PruneMode,
    persist_lock: Mutex<()>,
}

impl BlockColdStore {
    pub fn new(base_dir: PathBuf, runtime_fingerprint: [u8; 32]) -> Self {
        Self {
            base_dir,
            runtime_fingerprint,
            block_size: DEFAULT_BLOCK_SIZE,
            prune_mode: PruneMode::default(),
            persist_lock: Mutex::new(()),
        }
    }

    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    pub fn with_prune_mode(mut self, mode: PruneMode) -> Self {
        self.prune_mode = mode;
        self
    }

    pub fn blocks_dir(&self) -> PathBuf {
        self.base_dir.join(V4_ROOT).join(BLOCKS_DIR)
    }

    pub fn manifests_dir(&self) -> PathBuf {
        self.base_dir.join(V4_ROOT).join(MANIFESTS_DIR)
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    // -----------------------------------------------------------------------
    // Block I/O
    // -----------------------------------------------------------------------

    /// Write a block to disk using temp+rename for atomicity.
    pub fn write_block(
        &self,
        block_hash: &[u8; 32],
        tokens: &[i32],
        cache_set: &DetachedCacheSet,
    ) -> Result<(), ColdStoreError> {
        let block_dir = self.blocks_dir().join(hex_digest(block_hash));
        if block_dir.exists() {
            return Ok(()); // Already written
        }

        // Serialize on calling thread (Metal thread affinity)
        let layer_bytes = serialize_cache_set_layers(cache_set)?;

        // Write to temp dir, then atomic rename
        let tmp_dir = self.blocks_dir().join(format!(".tmp.{}", hex_digest(block_hash)));
        fs::create_dir_all(&tmp_dir)?;

        // Write block header
        let header = BlockHeader {
            format_version: V4_FORMAT_VERSION,
            block_hash: *block_hash,
            token_count: tokens.len(),
            tokens: tokens.to_vec(),
            layer_count: layer_bytes.len(),
            layers: layer_bytes
                .iter()
                .enumerate()
                .map(|(i, bytes)| LayerMetadata {
                    index: i as u32,
                    byte_len: bytes.len() as u64,
                    sha256: sha256(bytes),
                })
                .collect(),
        };
        let header_bytes = encode_block_header(&header)?;
        write_file(&tmp_dir.join("header.bin"), &header_bytes)?;

        // Write layers
        for (index, bytes) in layer_bytes.iter().enumerate() {
            write_file(
                &tmp_dir.join(format!("layer_{index:04}.bin")),
                bytes,
            )?;
        }

        // Atomic rename
        fs::rename(&tmp_dir, &block_dir)?;

        tracing::info!(
            block_hash = %hex_digest(block_hash),
            token_count = tokens.len(),
            layers = header.layer_count,
            "COLD-STORE v4 block written"
        );

        Ok(())
    }

    /// Read a block from disk.
    pub fn read_block(
        &self,
        block_hash: &[u8; 32],
    ) -> Result<(Vec<i32>, DetachedCacheSet), ColdStoreError> {
        let block_dir = self.blocks_dir().join(hex_digest(block_hash));
        if !block_dir.exists() {
            return Err(ColdStoreError::NoMatch);
        }

        let header_path = block_dir.join("header.bin");
        let header_bytes = fs::read(&header_path)?;
        let header = decode_block_header(&header_bytes)?;

        // Verify block hash
        if header.block_hash != *block_hash {
            return Err(invalid_data("block hash mismatch".into()));
        }

        // Read layers
        let mut caches = Vec::with_capacity(header.layer_count);
        for (expected_index, metadata) in header.layers.iter().enumerate() {
            let layer_path = block_dir.join(format!("layer_{expected_index:04}.bin"));
            let bytes = fs::read(&layer_path)?;
            if bytes.len() as u64 != metadata.byte_len {
                return Err(invalid_data(format!(
                    "layer {expected_index} length mismatch"
                )));
            }
            if sha256(&bytes) != metadata.sha256 {
                return Err(invalid_data(format!(
                    "layer {expected_index} checksum mismatch"
                )));
            }
            let mut cursor = Cursor::new(bytes.as_slice());
            let cache = read_kv_cache(&mut cursor, expected_index)?;
            caches.push(cache);
        }

        let cache_set = DetachedCacheSet {
            caches,
            backend: SequenceStateBackend::DenseKvCache,
            prompt_len: header.token_count,
            current_offset: header.token_count as i32,
            created_at: std::time::Instant::now(),
            detached_at: std::time::Instant::now(),
            origin_seq_id: SequenceId(0),
        };

        Ok((header.tokens, cache_set))
    }

    /// Count the number of blocks in the store.
    pub fn count_blocks(&self) -> Result<usize, ColdStoreError> {
        let dir = self.blocks_dir();
        if !dir.exists() {
            return Ok(0);
        }
        let count = fs::read_dir(&dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
            .filter(|e| {
                let name = e.file_name();
                let name = name.to_string_lossy();
                !name.starts_with(".tmp.")
            })
            .count();
        Ok(count)
    }

    // -----------------------------------------------------------------------
    // Manifest I/O
    // -----------------------------------------------------------------------

    /// Write a manifest to disk.
    pub fn write_manifest(&self, manifest: &Manifest) -> Result<(), ColdStoreError> {
        let manifest_hash = manifest.hash();
        let manifest_dir = self.manifests_dir().join(hex_digest(&manifest_hash));

        if manifest_dir.exists() {
            return Ok(()); // Already written
        }

        // Increment refcounts for all blocks BEFORE committing manifest
        for bh in &manifest.block_hashes {
            self.increment_refcount(bh)?;
        }

        // Write to temp dir, then atomic rename
        let tmp_dir = self
            .manifests_dir()
            .join(format!(".tmp.{}", hex_digest(&manifest_hash)));
        fs::create_dir_all(&tmp_dir)?;

        let manifest_bytes = encode_manifest(manifest)?;
        write_file(&tmp_dir.join("manifest.bin"), &manifest_bytes)?;

        // Atomic rename
        fs::rename(&tmp_dir, &manifest_dir)?;

        tracing::info!(
            manifest_hash = %hex_digest(&manifest_hash),
            block_count = manifest.block_hashes.len(),
            total_tokens = manifest.total_tokens,
            "COLD-STORE v4 manifest written"
        );

        Ok(())
    }

    /// Read a manifest from disk.
    pub fn read_manifest(
        &self,
        manifest_hash: &[u8; 32],
    ) -> Result<Manifest, ColdStoreError> {
        let manifest_dir = self.manifests_dir().join(hex_digest(manifest_hash));
        if !manifest_dir.exists() {
            return Err(ColdStoreError::NoMatch);
        }

        let manifest_path = manifest_dir.join("manifest.bin");
        let bytes = fs::read(&manifest_path)?;
        let manifest = decode_manifest(&bytes)?;

        // Verify manifest hash
        if manifest.hash() != *manifest_hash {
            return Err(invalid_data("manifest hash mismatch".into()));
        }

        Ok(manifest)
    }

    // -----------------------------------------------------------------------
    // Reference counting
    // -----------------------------------------------------------------------

    fn refcount_path(&self, block_hash: &[u8; 32]) -> PathBuf {
        self.blocks_dir()
            .join(format!("{}.refcount", hex_digest(block_hash)))
    }

    pub fn get_refcount(&self, block_hash: &[u8; 32]) -> Result<u64, ColdStoreError> {
        let path = self.refcount_path(block_hash);
        if !path.exists() {
            return Ok(0);
        }
        let bytes = fs::read(&path)?;
        if bytes.len() != 8 {
            return Err(invalid_data("refcount file corrupt".into()));
        }
        Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn set_refcount(&self, block_hash: &[u8; 32], count: u64) -> Result<(), ColdStoreError> {
        let path = self.refcount_path(block_hash);
        fs::write(&path, count.to_le_bytes())?;
        Ok(())
    }

    fn increment_refcount(&self, block_hash: &[u8; 32]) -> Result<(), ColdStoreError> {
        let current = self.get_refcount(block_hash)?;
        self.set_refcount(block_hash, current + 1)
    }

    fn decrement_refcount(&self, block_hash: &[u8; 32]) -> Result<(), ColdStoreError> {
        let current = self.get_refcount(block_hash)?;
        if current > 0 {
            self.set_refcount(block_hash, current - 1)?;
        }
        Ok(())
    }

    /// Delete a block with refcount 0.
    pub fn delete_block(&self, block_hash: &[u8; 32]) -> Result<(), ColdStoreError> {
        let refcount = self.get_refcount(block_hash)?;
        if refcount > 0 {
            return Err(invalid_data(format!(
                "cannot delete block with refcount {refcount}"
            )));
        }
        let block_dir = self.blocks_dir().join(hex_digest(block_hash));
        if block_dir.exists() {
            fs::remove_dir_all(&block_dir)?;
        }
        let refcount_path = self.refcount_path(block_hash);
        if refcount_path.exists() {
            fs::remove_file(&refcount_path)?;
        }
        Ok(())
    }

    /// Delete a manifest and decrement refcounts for all its blocks.
    pub fn delete_manifest(&self, manifest_hash: &[u8; 32]) -> Result<(), ColdStoreError> {
        let manifest = self.read_manifest(manifest_hash)?;

        // Decrement refcounts for all blocks
        for bh in &manifest.block_hashes {
            self.decrement_refcount(bh)?;
        }

        // Delete manifest directory
        let manifest_dir = self.manifests_dir().join(hex_digest(manifest_hash));
        if manifest_dir.exists() {
            fs::remove_dir_all(&manifest_dir)?;
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Prune-on-persist (manifest prefix)
    // -----------------------------------------------------------------------

    /// Delete manifests whose block hashes are a prefix of the new manifest's
    /// block hashes. This handles the normal case (conversation grows).
    pub fn prune_prefix_manifests(&self, new_manifest: &Manifest) -> Result<(), ColdStoreError> {
        let manifests_dir = self.manifests_dir();
        if !manifests_dir.exists() {
            return Ok(());
        }

        let entries: Vec<_> = fs::read_dir(&manifests_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
            .filter(|e| {
                let name = e.file_name();
                let name = name.to_string_lossy();
                !name.starts_with(".tmp.")
            })
            .collect();

        for entry in entries {
            let manifest_path = entry.path().join("manifest.bin");
            if !manifest_path.exists() {
                continue;
            }
            let bytes = match fs::read(&manifest_path) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let old_manifest = match decode_manifest(&bytes) {
                Ok(m) => m,
                Err(_) => continue,
            };

            // Check if old manifest's block hashes are a prefix of new manifest's
            if old_manifest.block_hashes.len() >= new_manifest.block_hashes.len() {
                continue;
            }
            if old_manifest.block_hashes[..] != new_manifest.block_hashes[..old_manifest.block_hashes.len()] {
                continue;
            }

            // Same identity check
            if old_manifest.runtime_fingerprint != new_manifest.runtime_fingerprint
                || old_manifest.model_id != new_manifest.model_id
                || old_manifest.template_sig != new_manifest.template_sig
            {
                continue;
            }

            // Prune
            match self.delete_manifest(&old_manifest.hash()) {
                Ok(()) => {
                    tracing::info!(
                        pruned = %old_manifest.hash_hex(),
                        old_blocks = old_manifest.block_hashes.len(),
                        new_blocks = new_manifest.block_hashes.len(),
                        "COLD-STORE v4 pruned prefix manifest"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        pruned = %old_manifest.hash_hex(),
                        error = %e,
                        "COLD-STORE v4 failed to prune prefix manifest"
                    );
                }
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Garbage collection
    // -----------------------------------------------------------------------

    /// Garbage-collect blocks with refcount 0.
    /// In observe mode, logs what WOULD be deleted without deleting.
    pub fn gc_blocks(&self) -> Result<(), ColdStoreError> {
        let blocks_dir = self.blocks_dir();
        if !blocks_dir.exists() {
            return Ok(());
        }

        let entries: Vec<_> = fs::read_dir(&blocks_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
            .filter(|e| {
                let name = e.file_name();
                let name = name.to_string_lossy();
                !name.starts_with(".tmp.")
            })
            .collect();

        for entry in entries {
            let name = entry.file_name();
            let name = name.to_string_lossy().to_string();

            // Parse block hash from directory name
            let block_hash = match parse_hex_digest(&name) {
                Some(h) => h,
                None => continue,
            };

            let refcount = self.get_refcount(&block_hash)?;
            if refcount > 0 {
                continue;
            }

            if self.prune_mode == PruneMode::Observe {
                tracing::info!(
                    would_gc = %name,
                    "COLD-STORE v4 observe: would GC unreferenced block"
                );
            } else {
                match self.delete_block(&block_hash) {
                    Ok(()) => {
                        tracing::info!(
                            gc = %name,
                            "COLD-STORE v4 GC'd unreferenced block"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            gc = %name,
                            error = %e,
                            "COLD-STORE v4 failed to GC block"
                        );
                    }
                }
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Load
    // -----------------------------------------------------------------------

    /// Load the best matching manifest and assemble its blocks into a cache.
    pub fn load_prefix(
        &self,
        model_id: &str,
        template_sig: &str,
        tokens: &[i32],
    ) -> Result<(DetachedCacheSet, usize), ColdStoreError> {
        let manifests_dir = self.manifests_dir();
        if !manifests_dir.exists() {
            return Err(ColdStoreError::NoMatch);
        }

        // Collect candidates
        let mut candidates = Vec::new();
        for entry in fs::read_dir(&manifests_dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let manifest_path = entry.path().join("manifest.bin");
            if !manifest_path.exists() {
                continue;
            }
            let bytes = match fs::read(&manifest_path) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let manifest = match decode_manifest(&bytes) {
                Ok(m) => m,
                Err(_) => continue,
            };

            // Filter: same identity
            if manifest.runtime_fingerprint != self.runtime_fingerprint
                || manifest.model_id != model_id
                || manifest.template_sig != template_sig
            {
                continue;
            }

            // Compute matched tokens by comparing block hashes
            let matched_blocks = manifest
                .block_hashes
                .iter()
                .zip(compute_block_hashes(tokens, self.block_size, &kv_mode_config_string(super::KVCacheMode::Fp16)))
                .take_while(|(a, b)| a == &b)
                .count();

            if matched_blocks == 0 {
                continue;
            }

            let matched_tokens = matched_blocks * self.block_size;
            candidates.push((manifest, matched_tokens));
        }

        // Sort by matched_tokens DESC, then timestamp_nanos DESC
        candidates.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| b.0.timestamp_nanos.cmp(&a.0.timestamp_nanos))
        });

        // Try to load each candidate
        for (manifest, matched_tokens) in candidates {
            match self.assemble_blocks(&manifest) {
                Ok(cache_set) => {
                    let matched = matched_tokens.min(tokens.len());
                    return Ok((cache_set, matched));
                }
                Err(e) => {
                    tracing::warn!(
                        manifest_hash = %manifest.hash_hex(),
                        error = %e,
                        "COLD-STORE v4 failed to load manifest, trying next"
                    );
                    continue;
                }
            }
        }

        Err(ColdStoreError::NoMatch)
    }

    /// Assemble blocks from a manifest into a DetachedCacheSet.
    fn assemble_blocks(&self, manifest: &Manifest) -> Result<DetachedCacheSet, ColdStoreError> {
        let mut all_caches: Vec<DetachedKVCache> = Vec::new();

        for (i, block_hash) in manifest.block_hashes.iter().enumerate() {
            let (tokens, cache_set) = self.read_block(block_hash)?;
            let token_count = tokens.len();

            // For each layer in this block, adjust the offset to reflect
            // the block's position in the full sequence
            for cache in cache_set.caches {
                // The block's cache has offset = token_count (block-local).
                // We don't need to adjust it — the assembled cache's
                // current_offset will be set from the manifest's total_tokens.
                all_caches.push(cache);
            }

            let _ = (i, token_count); // suppress unused warnings
        }

        if all_caches.is_empty() {
            return Err(invalid_data("manifest has no blocks".into()));
        }

        let cache_set = DetachedCacheSet {
            caches: all_caches,
            backend: SequenceStateBackend::DenseKvCache,
            prompt_len: manifest.prompt_len,
            current_offset: manifest.total_tokens as i32,
            created_at: std::time::Instant::now(),
            detached_at: std::time::Instant::now(),
            origin_seq_id: SequenceId(0),
        };

        Ok(cache_set)
    }
}

// ---------------------------------------------------------------------------
// Block header
// ---------------------------------------------------------------------------

struct BlockHeader {
    format_version: u32,
    block_hash: [u8; 32],
    token_count: usize,
    tokens: Vec<i32>,
    layer_count: usize,
    layers: Vec<LayerMetadata>,
}

struct LayerMetadata {
    index: u32,
    byte_len: u64,
    sha256: [u8; 32],
}

fn encode_block_header(header: &BlockHeader) -> Result<Vec<u8>, ColdStoreError> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&header.format_version.to_le_bytes());
    bytes.extend_from_slice(&header.block_hash);
    bytes.extend_from_slice(&(header.token_count as u64).to_le_bytes());
    for token in &header.tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    bytes.extend_from_slice(&(header.layer_count as u32).to_le_bytes());
    for layer in &header.layers {
        bytes.extend_from_slice(&layer.index.to_le_bytes());
        bytes.extend_from_slice(&layer.byte_len.to_le_bytes());
        bytes.extend_from_slice(&layer.sha256);
    }
    Ok(bytes)
}

fn decode_block_header(bytes: &[u8]) -> Result<BlockHeader, ColdStoreError> {
    let mut reader = BufReader::new(Cursor::new(bytes));
    let format_version = read_u32(&mut reader)?;
    if format_version != V4_FORMAT_VERSION {
        return Err(ColdStoreError::FormatVersionMismatch {
            stored: format_version,
            current: V4_FORMAT_VERSION,
        });
    }
    let block_hash = read_digest(&mut reader)?;
    let token_count = bounded_usize(read_u64(&mut reader)?, MAX_TOKENS, "token count")?;
    let mut tokens = Vec::with_capacity(token_count);
    for _ in 0..token_count {
        tokens.push(read_i32(&mut reader)?);
    }
    let layer_count = bounded_usize(read_u32(&mut reader)? as u64, MAX_LAYERS, "layer count")?;
    let mut layers = Vec::with_capacity(layer_count);
    for _ in 0..layer_count {
        let index = read_u32(&mut reader)?;
        let byte_len = read_u64(&mut reader)?;
        let sha256 = read_digest(&mut reader)?;
        layers.push(LayerMetadata {
            index,
            byte_len,
            sha256,
        });
    }
    Ok(BlockHeader {
        format_version,
        block_hash,
        token_count,
        tokens,
        layer_count,
        layers,
    })
}

// ---------------------------------------------------------------------------
// Manifest encoding
// ---------------------------------------------------------------------------

fn encode_manifest(manifest: &Manifest) -> Result<Vec<u8>, ColdStoreError> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&V4_FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&manifest.runtime_fingerprint);
    write_bounded_string(&mut bytes, &manifest.model_id)?;
    write_bounded_string(&mut bytes, &manifest.template_sig)?;
    bytes.extend_from_slice(&(manifest.block_size as u64).to_le_bytes());
    bytes.extend_from_slice(&(manifest.block_hashes.len() as u64).to_le_bytes());
    for bh in &manifest.block_hashes {
        bytes.extend_from_slice(bh);
    }
    bytes.extend_from_slice(&(manifest.prompt_len as u64).to_le_bytes());
    bytes.extend_from_slice(&(manifest.total_tokens as u64).to_le_bytes());
    bytes.extend_from_slice(&manifest.timestamp_nanos.to_le_bytes());
    Ok(bytes)
}

fn decode_manifest(bytes: &[u8]) -> Result<Manifest, ColdStoreError> {
    let mut reader = BufReader::new(Cursor::new(bytes));
    let format_version = read_u32(&mut reader)?;
    if format_version != V4_FORMAT_VERSION {
        return Err(ColdStoreError::FormatVersionMismatch {
            stored: format_version,
            current: V4_FORMAT_VERSION,
        });
    }
    let runtime_fingerprint = read_digest(&mut reader)?;
    let model_id = read_bounded_string(&mut reader)?;
    let template_sig = read_bounded_string(&mut reader)?;
    let block_size = bounded_usize(read_u64(&mut reader)?, MAX_BLOCK_SIZE, "block size")?;
    let block_count = bounded_usize(read_u64(&mut reader)?, MAX_TOKENS, "block count")?;
    let mut block_hashes = Vec::with_capacity(block_count);
    for _ in 0..block_count {
        block_hashes.push(read_digest(&mut reader)?);
    }
    let prompt_len = bounded_usize(read_u64(&mut reader)?, MAX_TOKENS, "prompt length")?;
    let total_tokens = bounded_usize(read_u64(&mut reader)?, MAX_TOKENS, "total tokens")?;
    let timestamp_nanos = read_u128(&mut reader)?;
    Ok(Manifest {
        runtime_fingerprint,
        model_id,
        template_sig,
        block_size,
        block_hashes,
        prompt_len,
        total_tokens,
        timestamp_nanos,
    })
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn hex_digest(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn parse_hex_digest(hex: &str) -> Option<[u8; 32]> {
    if hex.len() != 64 {
        return None;
    }
    let mut bytes = [0u8; 32];
    for i in 0..32 {
        bytes[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(bytes)
}

fn write_file(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(bytes)?;
    writer.flush()
}

fn write_bounded_string(writer: &mut impl Write, value: &str) -> Result<(), ColdStoreError> {
    let len = u32::try_from(value.len())
        .map_err(|_| invalid_data("string exceeds u32".into()))?;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(value.as_bytes())?;
    Ok(())
}

fn read_bounded_string(reader: &mut impl Read) -> Result<String, ColdStoreError> {
    let len = bounded_usize(read_u32(reader)? as u64, MAX_STRING_BYTES, "string length")?;
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|e| invalid_data(format!("invalid UTF-8: {e}")))
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

/// Compute block hashes for a token sequence using the Merkle-chain.
pub fn compute_block_hashes(
    tokens: &[i32],
    block_size: usize,
    kv_mode_config: &str,
) -> Vec<[u8; 32]> {
    let mut hashes = Vec::new();
    let mut prev_hash = [0u8; 32];
    for chunk in tokens.chunks(block_size) {
        let hash = block_hash_merkle(&prev_hash, block_size, kv_mode_config, chunk);
        hashes.push(hash);
        prev_hash = hash;
    }
    hashes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::cold_store::tests::make_test_cache_set;
    use crate::cache::cold_store::runtime_fingerprint_from_manifest;

    #[test]
    fn block_hash_merkle_chain_prefix_dependent() {
        let tokens_a = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let tokens_b = vec![9, 10, 11, 12, 5, 6, 7, 8];
        let kv_mode = kv_mode_config_string(KVCacheMode::Fp16);
        let block_size = 4;

        let hash_a0 = block_hash_merkle(&[0u8; 32], block_size, &kv_mode, &tokens_a[0..4]);
        let hash_b0 = block_hash_merkle(&[0u8; 32], block_size, &kv_mode, &tokens_b[0..4]);
        assert_ne!(hash_a0, hash_b0, "block 0 hashes must differ");

        let hash_a1 = block_hash_merkle(&hash_a0, block_size, &kv_mode, &tokens_a[4..8]);
        let hash_b1 = block_hash_merkle(&hash_b0, block_size, &kv_mode, &tokens_b[4..8]);
        assert_ne!(
            hash_a1, hash_b1,
            "block 1 hashes must differ despite same own tokens"
        );
    }

    #[test]
    fn block_hash_merkle_chain_same_prefix_same_hash() {
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let kv_mode = kv_mode_config_string(KVCacheMode::Fp16);
        let block_size = 4;

        let hash_a0 = block_hash_merkle(&[0u8; 32], block_size, &kv_mode, &tokens[0..4]);
        let hash_b0 = block_hash_merkle(&[0u8; 32], block_size, &kv_mode, &tokens[0..4]);
        assert_eq!(hash_a0, hash_b0);

        let hash_a1 = block_hash_merkle(&hash_a0, block_size, &kv_mode, &tokens[4..8]);
        let hash_b1 = block_hash_merkle(&hash_b0, block_size, &kv_mode, &tokens[4..8]);
        assert_eq!(hash_a1, hash_b1);
    }

    #[test]
    fn block_hash_includes_kv_mode() {
        let tokens = vec![1, 2, 3, 4];
        let block_size = 4;

        let kv_fp16 = kv_mode_config_string(super::super::KVCacheMode::Fp16);
        let kv_kvarn8 = kv_mode_config_string(KVCacheMode::KVarN8);

        let hash_fp16 = block_hash_merkle(&[0u8; 32], block_size, &kv_fp16, &tokens);
        let hash_kvarn8 = block_hash_merkle(&[0u8; 32], block_size, &kv_kvarn8, &tokens);
        assert_ne!(hash_fp16, hash_kvarn8);
    }

    #[test]
    fn block_hash_includes_block_size() {
        let tokens = vec![1, 2, 3, 4];
        let kv_mode = kv_mode_config_string(KVCacheMode::Fp16);

        let hash_4 = block_hash_merkle(&[0u8; 32], 4, &kv_mode, &tokens);
        let hash_8 = block_hash_merkle(&[0u8; 32], 8, &kv_mode, &tokens);
        assert_ne!(hash_4, hash_8);
    }

    #[test]
    fn validate_block_size_valid() {
        assert!(validate_block_size(2048).is_ok());
        assert!(validate_block_size(4096).is_ok());
        assert!(validate_block_size(8192).is_ok());
    }

    #[test]
    fn validate_block_size_invalid() {
        assert!(validate_block_size(1024).is_err()); // too small
        assert!(validate_block_size(3000).is_err()); // not power of 2
        assert!(validate_block_size(2049).is_err()); // not multiple of 128
    }

    #[test]
    fn compute_block_hashes_basic() {
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let kv_mode = kv_mode_config_string(KVCacheMode::Fp16);
        let hashes = compute_block_hashes(&tokens, 4, &kv_mode);
        assert_eq!(hashes.len(), 2);
        assert_ne!(hashes[0], hashes[1]);
    }
}
