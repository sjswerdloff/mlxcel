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
use crate::ffi::MlxArray;
use crate::utils::slice_axis;
use cxx::UniquePtr;

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
    // Persist
    // -----------------------------------------------------------------------

    /// Persist a DetachedCacheSet to the block cold-store.
    ///
    /// Splits the token sequence into blocks, extracts block KV data,
    /// writes blocks (skipping existing ones), creates a manifest,
    /// and prunes old manifests whose block hashes are a prefix.
    ///
    /// Must be called on the inference thread (Metal thread affinity)
    /// because extract_block slices MLX arrays.
    pub fn persist(
        &self,
        model_id: &str,
        template_sig: &str,
        tokens: &[i32],
        cache_set: &DetachedCacheSet,
    ) -> Result<Manifest, ColdStoreError> {
        let kv_mode = kv_mode_config_string(cache_set.caches[0].mode);
        let block_hashes = compute_block_hashes(tokens, self.block_size, &kv_mode);

        // Extract and write each block
        for (i, chunk) in tokens.chunks(self.block_size).enumerate() {
            let block_hash = block_hashes[i];
            let start = i * self.block_size;
            let end = (start + chunk.len()).min(tokens.len());

            // Check if block already exists
            if self.blocks_dir().join(hex_digest(&block_hash)).exists() {
                continue;
            }

            // Extract block from cache set
            let block_cache = extract_block(cache_set, start, end)?;
            self.write_block(&block_hash, chunk, &block_cache)?;
        }

        // Create manifest
        let manifest = Manifest {
            runtime_fingerprint: self.runtime_fingerprint,
            model_id: model_id.to_string(),
            template_sig: template_sig.to_string(),
            block_size: self.block_size,
            block_hashes,
            prompt_len: cache_set.prompt_len,
            total_tokens: tokens.len(),
            timestamp_nanos: now_nanos(),
        };

        self.write_manifest(&manifest)?;

        // Prune old manifests whose block hashes are a prefix
        if self.prune_mode != PruneMode::Off {
            self.prune_prefix_manifests(&manifest)?;
        }

        Ok(manifest)
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
    ///
    /// The result must have ONE cache per LAYER, each carrying that layer's
    /// state concatenated along the token axis across every block — NOT one
    /// cache per (block, layer) pair.
    ///
    /// The previous implementation flat-appended each block's per-layer caches
    /// onto a single `Vec`, so a B-block, L-layer manifest assembled to `B*L`
    /// entries instead of `L`. `read_block` (`:277-294`) iterates
    /// `header.layers.iter().enumerate()` and pushes one cache per layer, so
    /// `caches` IS layer-indexed and the bug was real on every multi-block
    /// load — which is every sequence longer than `block_size` (2048). It was
    /// invisible to single-block manifests, and `load_prefix` (`:672`) is the
    /// production caller, so a wrong assembly here is a wrong adopted prefix.
    fn assemble_blocks(&self, manifest: &Manifest) -> Result<DetachedCacheSet, ColdStoreError> {
        if manifest.block_hashes.is_empty() {
            return Err(invalid_data("manifest has no blocks".into()));
        }

        // Read every block up front: merging is per LAYER across ALL blocks,
        // so no layer can be finished until every block has been read.
        let mut blocks: Vec<DetachedCacheSet> = Vec::with_capacity(manifest.block_hashes.len());
        for block_hash in &manifest.block_hashes {
            let (_tokens, cache_set) = self.read_block(block_hash)?;
            blocks.push(cache_set);
        }

        let layer_count = blocks[0].caches.len();
        for (i, b) in blocks.iter().enumerate() {
            if b.caches.len() != layer_count {
                return Err(invalid_data(format!(
                    "assemble_blocks: block {i} has {} layers but block 0 has {layer_count} — \
                     the manifest mixes structurally incompatible blocks",
                    b.caches.len()
                )));
            }
        }

        let mut all_caches: Vec<DetachedKVCache> = Vec::with_capacity(layer_count);
        for layer_idx in 0..layer_count {
            let layers: Vec<&DetachedKVCache> =
                blocks.iter().map(|b| &b.caches[layer_idx]).collect();
            all_caches.push(merge_layer_across_blocks(
                &layers,
                layer_idx,
                manifest.total_tokens as i32,
            )?);
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
// Block assembly (per-layer merge across blocks)
// ---------------------------------------------------------------------------

/// Selector for one optional array field of a layer.
type LayerField = fn(&DetachedKVCache) -> &Option<UniquePtr<MlxArray>>;

/// Concatenate one field along axis 2 (the token axis) across blocks, in
/// manifest order, skipping blocks where it is absent.
///
/// Absence is meaningful and uniform here: a mid-sequence block has no sink and
/// no tail, so those fields concatenate to exactly the one block that carries
/// them. That is why sink/tail need no special case — only the placement
/// ASSERTIONS in [`merge_layer_across_blocks`], which catch a sink or tail
/// appearing where the sequence layout says it cannot.
fn concat_across(layers: &[&DetachedKVCache], sel: LayerField) -> Option<UniquePtr<MlxArray>> {
    let present: Vec<&MlxArray> = layers
        .iter()
        .filter_map(|l| sel(l).as_ref().map(|a| a.as_ref().unwrap()))
        .collect();
    let (first, rest) = present.split_first()?;
    if rest.is_empty() {
        // Single contributor: full-range slice to obtain an owned array.
        let len = crate::ffi::array_shape(first)[2];
        return Some(slice_axis(first, 2, 0, len));
    }
    let mut acc = crate::utils::concatenate(first, rest[0], 2);
    for a in &rest[1..] {
        acc = crate::utils::concatenate(acc.as_ref().unwrap(), a, 2);
    }
    Some(acc)
}

/// axis-2 extent of an optional field, or 0 when absent.
fn axis2_of(o: &Option<UniquePtr<MlxArray>>) -> i32 {
    o.as_ref()
        .map_or(0, |a| crate::ffi::array_shape(a.as_ref().unwrap())[2])
}

/// Per-token history fields: axis-2 extent is the history token count.
const HIST_PER_TOKEN_FIELDS: &[(&str, LayerField)] = &[
    ("kvarn_hist_k", |c| &c.kvarn_hist_k),
    ("kvarn_hist_v", |c| &c.kvarn_hist_v),
    ("kvarn_k_scale", |c| &c.kvarn_k_scale),
    ("kvarn_k_zp", |c| &c.kvarn_k_zp),
    ("kvarn_k_s_row", |c| &c.kvarn_k_s_row),
    ("kvarn_v_scale", |c| &c.kvarn_v_scale),
    ("kvarn_v_zp", |c| &c.kvarn_v_zp),
    ("kvarn_v_s_row", |c| &c.kvarn_v_s_row),
];

/// Per-TILE column scales: axis-2 extent is the tile count, not the token count.
const HIST_PER_TILE_FIELDS: &[(&str, LayerField)] = &[
    ("kvarn_k_s_col", |c| &c.kvarn_k_s_col),
    ("kvarn_v_s_col", |c| &c.kvarn_v_s_col),
];

/// Prove an assembled layer is internally coherent before any extent claim is
/// made about it.
///
/// ## Why this exists (Violet, review of 2eedf84)
///
/// [`concat_across`] SKIPS blocks where a field is absent. Absence is a
/// legitimate layout fact for exactly two things — a mid-sequence block has no
/// sink and no tail. For every other field, absence in a block that covers
/// history is a BUG (an extract fault, a partial write, a future refactor), and
/// skipping it silently yields a field SHORTER than the history it describes:
/// per-token zero-points or scales misaligned against the tokens they apply to.
/// That is silent wrong inference, and it is exactly the class the mode and
/// `v_bits` consistency checks already guard against — same failure, different
/// field, previously no guard.
///
/// The extent check that follows this one used to sum `sink_k + hist_k + tail_k`
/// and report "layer {n} assembled to N tokens" — measuring three K fields and
/// certifying the LAYER. A dropped or short `hist_v`, `sink_v` or `tail_v`
/// passed clean, and ten fields were never measured at all. K and V are
/// symmetric in the layout and were asymmetric in the check. This runs in the
/// production path (`load_prefix`), so unlike the end-to-end test it is what
/// guards real conversations.
///
/// ## Residual, stated rather than implied
///
/// A field absent from EVERY block passes here, because uniform absence is
/// legitimate for at least one field: v4 folds `v_s_row` into `s_col` and leaves
/// it `None`.
///
/// It would be wrong to say this is because the assembler lacks the information
/// — it has `kvarn_v_bits` right here, and mode consistency is enforced above.
/// The accurate statement is that **the contract is missing from the codebase,
/// not from this function.** Nothing anywhere declares "under `v_bits = 4`,
/// `v_s_row` is absent and these nine fields are mandatory." `trim_to`'s KVarN8
/// branch slices `v_s_row` unconditionally (`slice_seq` no-ops on `None`) and
/// does not branch on `v_bits` either. That knowledge lives only in the
/// quantizer's behaviour, and three places — this assembler, `trim_to`, and the
/// extractor — each trust it independently. The fix is therefore a declared
/// `mandatory_fields(mode, v_bits)` giving all three one source of truth, NOT a
/// cleverer inference here. (Violet, review of `fa50cd6`.)
///
/// Deferring that is a calibration, not an oversight: a PARTIAL skip is
/// SILENT-wrong — misaligned scales against the tokens they apply to, computing
/// a plausible wrong answer — while UNIFORM absence is LOUD-wrong, a missing
/// input that panics or visibly fails downstream. This guard catches the silent
/// class, which is the class that needed catching. The loud one can wait for a
/// contract, and it should be a contract rather than a fourth local check.
fn check_region_coherence(
    merged: &DetachedKVCache,
    layer_idx: usize,
    block_count: usize,
) -> Result<(), ColdStoreError> {
    if merged.mode != KVCacheMode::KVarN8 {
        // Dense: K and V must describe the same tokens.
        let (k, v) = (axis2_of(&merged.keys), axis2_of(&merged.values));
        if k != v {
            return Err(invalid_data(format!(
                "assemble_blocks: layer {layer_idx} assembled keys ({k} tokens) and values \
                 ({v} tokens) disagree — a block was skipped on one side ({block_count} blocks)"
            )));
        }
        return Ok(());
    }

    let sink_len = axis2_of(&merged.kvarn_sink_k);
    let hist_len = axis2_of(&merged.kvarn_hist_k);
    let tail_len = axis2_of(&merged.kvarn_tail_k);

    // K/V symmetry on the regions that carry both sides.
    for (name, k_len, v_len) in [
        ("sink", sink_len, axis2_of(&merged.kvarn_sink_v)),
        ("tail", tail_len, axis2_of(&merged.kvarn_tail_v)),
    ] {
        if k_len != v_len {
            return Err(invalid_data(format!(
                "assemble_blocks: layer {layer_idx} {name} K side is {k_len} tokens but the V \
                 side is {v_len} — a block was skipped on one side only ({block_count} blocks \
                 merged). K and V describe the same tokens; a mismatch means the V side is \
                 misaligned against the tokens it applies to."
            )));
        }
    }

    // Per-token history fields: present ⇒ exactly the history length.
    for &(name, sel) in HIST_PER_TOKEN_FIELDS {
        let n = axis2_of(sel(merged));
        if n != 0 && n != hist_len {
            return Err(invalid_data(format!(
                "assemble_blocks: layer {layer_idx} field `{name}` has {n} history entries but \
                 the history is {hist_len} tokens — a block was silently skipped for this field \
                 ({block_count} blocks merged). A short per-token field is misaligned against \
                 the tokens it describes: silent wrong inference, not a crash."
            )));
        }
    }

    // Per-TILE column scales: present ⇒ exactly the tile count.
    if hist_len % crate::cache::kvarn::KVARN_TILE_TOKENS != 0 {
        return Err(invalid_data(format!(
            "assemble_blocks: layer {layer_idx} assembled history is {hist_len} tokens, not a \
             multiple of the tile size {} — blocks were merged across a partial tile",
            crate::cache::kvarn::KVARN_TILE_TOKENS
        )));
    }
    let n_tiles = hist_len / crate::cache::kvarn::KVARN_TILE_TOKENS;
    for &(name, sel) in HIST_PER_TILE_FIELDS {
        let n = axis2_of(sel(merged));
        if n != 0 && n != n_tiles {
            return Err(invalid_data(format!(
                "assemble_blocks: layer {layer_idx} field `{name}` has {n} entries but the \
                 history is {n_tiles} tiles ({hist_len} tokens) — these are PER-TILE scales, and \
                 a count that is neither 0 nor the tile count means a block was skipped \
                 ({block_count} blocks merged)"
            )));
        }
    }

    Ok(())
}

/// Merge one layer's state across every block of a manifest.
///
/// Produces ONE cache carrying that layer concatenated along the token axis —
/// the thing `assemble_blocks` needs and previously did not do.
fn merge_layer_across_blocks(
    layers: &[&DetachedKVCache],
    layer_idx: usize,
    total_tokens: i32,
) -> Result<DetachedKVCache, ColdStoreError> {
    let first = layers[0];

    // Config consistency. A mismatch means the manifest mixes blocks quantized
    // under different schemes; assembling them would produce a cache whose
    // bytes are then interpreted under the wrong one — silent wrong inference,
    // not a crash.
    for (i, l) in layers.iter().enumerate() {
        if l.mode != first.mode {
            return Err(invalid_data(format!(
                "assemble_blocks: layer {layer_idx} block {i} has mode {:?}, block 0 has {:?} — \
                 manifest mixes blocks from different KV modes",
                l.mode, first.mode
            )));
        }
        if l.kvarn_v_bits != first.kvarn_v_bits {
            return Err(invalid_data(format!(
                "assemble_blocks: layer {layer_idx} block {i} has kvarn_v_bits {}, block 0 has {} \
                 — manifest mixes blocks quantized at different V widths",
                l.kvarn_v_bits, first.kvarn_v_bits
            )));
        }
    }

    // Region placement. The sink is the head of the sequence and the tail is
    // its end, so under KVarN8 they can appear only in the first and last
    // block. A sink in block 3 means the manifest is mis-ordered or its blocks
    // come from different sequences — either way, concatenating would produce a
    // plausible-looking cache with the wrong tokens in it.
    if first.mode == KVCacheMode::KVarN8 {
        let last = layers.len() - 1;
        for (i, l) in layers.iter().enumerate() {
            if i != 0 && (l.kvarn_sink_k.is_some() || l.kvarn_sink_v.is_some()) {
                return Err(invalid_data(format!(
                    "assemble_blocks: layer {layer_idx} block {i} carries a sink, but only the \
                     first block may — manifest order is wrong, or these blocks are from \
                     different sequences"
                )));
            }
            if i != last && (l.kvarn_tail_k.is_some() || l.kvarn_tail_v.is_some()) {
                return Err(invalid_data(format!(
                    "assemble_blocks: layer {layer_idx} block {i} carries a tail, but only the \
                     last block ({last}) may — see above"
                )));
            }
        }
    }

    let merged = DetachedKVCache {
        keys: concat_across(layers, |c| &c.keys),
        values: concat_across(layers, |c| &c.values),
        offset: total_tokens,
        step: first.step,
        mode: first.mode,
        key_scales: concat_across(layers, |c| &c.key_scales),
        val_scales: concat_across(layers, |c| &c.val_scales),
        v_packed: None,
        v_norms: None,
        v_rescale: None,
        k_packed: None,
        k_norms: None,
        turbo_seed: first.turbo_seed,
        cold_offset: 0,
        hot_threshold: first.hot_threshold,
        delegated_fp16_fast_path: first.delegated_fp16_fast_path,
        delegated_fp16_sidecar_policy: first.delegated_fp16_sidecar_policy,
        m3_idx_k: concat_across(layers, |c| &c.m3_idx_k),
        m3_idx_offset: total_tokens,
        kvarn_sink_k: concat_across(layers, |c| &c.kvarn_sink_k),
        kvarn_sink_v: concat_across(layers, |c| &c.kvarn_sink_v),
        kvarn_tail_k: concat_across(layers, |c| &c.kvarn_tail_k),
        kvarn_tail_v: concat_across(layers, |c| &c.kvarn_tail_v),
        kvarn_hist_k: concat_across(layers, |c| &c.kvarn_hist_k),
        kvarn_hist_v: concat_across(layers, |c| &c.kvarn_hist_v),
        kvarn_k_scale: concat_across(layers, |c| &c.kvarn_k_scale),
        kvarn_k_zp: concat_across(layers, |c| &c.kvarn_k_zp),
        kvarn_k_s_row: concat_across(layers, |c| &c.kvarn_k_s_row),
        kvarn_k_s_col: concat_across(layers, |c| &c.kvarn_k_s_col),
        kvarn_v_scale: concat_across(layers, |c| &c.kvarn_v_scale),
        kvarn_v_zp: concat_across(layers, |c| &c.kvarn_v_zp),
        kvarn_v_s_row: concat_across(layers, |c| &c.kvarn_v_s_row),
        kvarn_v_s_col: concat_across(layers, |c| &c.kvarn_v_s_col),
        kvarn_v_bits: first.kvarn_v_bits,
    };

    // Region coherence, then extent. Both are DERIVED FROM THE DATA rather than
    // read from a stored header length — a stored length is a second source of
    // truth that can drift from the bytes it describes; a measured one cannot.
    check_region_coherence(&merged, layer_idx, layers.len())?;

    // Extent must now be measured on a layer already proven internally
    // coherent, so summing the K-side regions is a statement about the LAYER
    // rather than about three of its fields.
    let assembled = if merged.mode == KVCacheMode::KVarN8 {
        axis2_of(&merged.kvarn_sink_k) + axis2_of(&merged.kvarn_hist_k) + axis2_of(&merged.kvarn_tail_k)
    } else {
        axis2_of(&merged.keys)
    };
    if assembled != total_tokens {
        return Err(invalid_data(format!(
            "assemble_blocks: layer {layer_idx} assembled to {assembled} tokens but the manifest \
             says {total_tokens} — a block was dropped, duplicated, or is the wrong size \
             ({} blocks merged)",
            layers.len()
        )));
    }

    Ok(merged)
}

// ---------------------------------------------------------------------------
// KVarN8 region-aware block extraction
// ---------------------------------------------------------------------------

/// The KVarN8 tile state carried by one extracted block.
///
/// `v_s_row` is carried even though v4 folds it into `s_col` and leaves it
/// `None`: `trim_to` slices it when present (`detach.rs:560`), so this must
/// too, or a k8v8 layer would silently lose it.
#[derive(Default)]
struct Kvarn8Regions {
    sink_k: Option<UniquePtr<MlxArray>>,
    sink_v: Option<UniquePtr<MlxArray>>,
    hist_k: Option<UniquePtr<MlxArray>>,
    hist_v: Option<UniquePtr<MlxArray>>,
    k_scale: Option<UniquePtr<MlxArray>>,
    k_zp: Option<UniquePtr<MlxArray>>,
    k_s_row: Option<UniquePtr<MlxArray>>,
    k_s_col: Option<UniquePtr<MlxArray>>,
    v_scale: Option<UniquePtr<MlxArray>>,
    v_zp: Option<UniquePtr<MlxArray>>,
    v_s_row: Option<UniquePtr<MlxArray>>,
    v_s_col: Option<UniquePtr<MlxArray>>,
    tail_k: Option<UniquePtr<MlxArray>>,
    tail_v: Option<UniquePtr<MlxArray>>,
}

/// Slice one KVarN8 layer's tile state to the token range `[start, end)`.
///
/// KVarN8 lays a sequence out along axis 2 as
/// `[ sink (128) | history tiles (T, a multiple of 128) | tail (< 128) ]`,
/// and the three regions live in SEPARATE tensors — so a block's token range
/// must be split across them, not sliced out of one contiguous buffer. That
/// is why `extract_block`'s dense path (a single `slice_axis` per tensor)
/// cannot serve KVarN8, and why it previously returned `None` for all of it.
///
/// This mirrors `DetachedKVCache::trim_to`'s KVarN8 branch
/// (`detach.rs:507-572`), generalized from a prefix `[0, new_len)` to an
/// arbitrary block `[start, end)`. Three properties are carried over
/// deliberately:
///
/// * **`k_s_col`/`v_s_col` slice by TILE INDEX, never by token count.** Their
///   axis-2 length is `n_tiles`, not `T` — `trim_to` uses
///   `n_tiles_keep = hist_keep / tile` (`detach.rs:552,561-562`). Slicing them
///   by tokens is the corruption trap this whole suite is built around: for a
///   small cache it is an out-of-range slice, and for a large one it silently
///   returns the wrong column scales for the block, which is wrong inference
///   rather than a crash.
/// * **A history boundary that is not tile-aligned FAILS LOUDLY**, rather than
///   approximating — a half tile cannot be re-quantized without the original
///   fp16 data (`detach.rs:505-506`). BOTH boundaries are checked. The START
///   is a distinct code path from the END: it exercises the sink-offset
///   arithmetic rather than the end clamp, and it is equally uncomputable.
/// * **The sink is indivisible.** It is taken whole or not at all; a block
///   that would split it is a caller bug and says so.
fn extract_kvarn8_regions(
    layer: &DetachedKVCache,
    start: usize,
    end: usize,
) -> Result<Kvarn8Regions, ColdStoreError> {
    let tile = crate::cache::kvarn::KVARN_TILE_TOKENS;
    let axis2 = |a: &UniquePtr<MlxArray>| crate::ffi::array_shape(a)[2];

    let sink_len = layer.kvarn_sink_k.as_ref().map_or(0, axis2);
    let hist_len = layer.kvarn_hist_k.as_ref().map_or(0, axis2);
    let tail_len = layer.kvarn_tail_k.as_ref().map_or(0, axis2);
    let total = sink_len + hist_len + tail_len;

    let (start_i, end_i) = (start as i32, end as i32);
    if end_i > total {
        return Err(invalid_data(format!(
            "extract_block: KVarN8 block [{start}, {end}) exceeds the layer's extent \
             {total} (sink {sink_len} + hist {hist_len} + tail {tail_len})"
        )));
    }

    let cut = |t: &Option<UniquePtr<MlxArray>>, lo: i32, hi: i32| {
        t.as_ref().map(|a| slice_axis(a, 2, lo, hi))
    };
    let mut out = Kvarn8Regions::default();

    // --- sink: global [0, sink_len). Indivisible. ---
    let sink_lo = start_i.clamp(0, sink_len);
    let sink_hi = end_i.clamp(0, sink_len);
    if sink_hi > sink_lo {
        if sink_lo != 0 || sink_hi != sink_len {
            return Err(invalid_data(format!(
                "extract_block: KVarN8 block [{start}, {end}) would split the sink \
                 [0, {sink_len}). The sink is indivisible — a block takes it whole \
                 or not at all."
            )));
        }
        out.sink_k = cut(&layer.kvarn_sink_k, 0, sink_len);
        out.sink_v = cut(&layer.kvarn_sink_v, 0, sink_len);
    }

    // --- history: global [sink_len, sink_len + hist_len), in hist-local coords. ---
    let h_lo = (start_i - sink_len).clamp(0, hist_len);
    let h_hi = (end_i - sink_len).clamp(0, hist_len);
    if h_hi > h_lo {
        if h_lo % tile != 0 {
            return Err(invalid_data(format!(
                "extract_block: KVarN8 history START {h_lo} is not tile-aligned \
                 (tile = {tile}); block [{start}, {end}), sink {sink_len}. A half \
                 tile cannot be re-quantized without the original fp16 data."
            )));
        }
        if h_hi % tile != 0 {
            return Err(invalid_data(format!(
                "extract_block: KVarN8 history END {h_hi} is not tile-aligned \
                 (tile = {tile}); block [{start}, {end}), sink {sink_len}. A half \
                 tile cannot be re-quantized without the original fp16 data."
            )));
        }

        // Per-TOKEN history fields.
        out.hist_k = cut(&layer.kvarn_hist_k, h_lo, h_hi);
        out.hist_v = cut(&layer.kvarn_hist_v, h_lo, h_hi);
        out.k_scale = cut(&layer.kvarn_k_scale, h_lo, h_hi);
        out.k_zp = cut(&layer.kvarn_k_zp, h_lo, h_hi);
        out.k_s_row = cut(&layer.kvarn_k_s_row, h_lo, h_hi);
        out.v_scale = cut(&layer.kvarn_v_scale, h_lo, h_hi);
        out.v_zp = cut(&layer.kvarn_v_zp, h_lo, h_hi);
        out.v_s_row = cut(&layer.kvarn_v_s_row, h_lo, h_hi);

        // Per-TILE column scales. TILE INDEX, NOT TOKEN COUNT. See doc above.
        let (n_lo, n_hi) = (h_lo / tile, h_hi / tile);
        out.k_s_col = cut(&layer.kvarn_k_s_col, n_lo, n_hi);
        out.v_s_col = cut(&layer.kvarn_v_s_col, n_lo, n_hi);
    }

    // --- tail: global [sink_len + hist_len, total), in tail-local coords. ---
    let t_base = sink_len + hist_len;
    let t_lo = (start_i - t_base).clamp(0, tail_len);
    let t_hi = (end_i - t_base).clamp(0, tail_len);
    if t_hi > t_lo {
        out.tail_k = cut(&layer.kvarn_tail_k, t_lo, t_hi);
        out.tail_v = cut(&layer.kvarn_tail_v, t_lo, t_hi);
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Block extraction
// ---------------------------------------------------------------------------

/// Extract a block from a DetachedCacheSet.
///
/// For dense (fp16/kvarn8) mode: slices K/V tensors along the seq_len
/// dimension (axis 2) for the token range [start, end).
///
/// For KVarN8 mode: the block boundaries are tile-aligned (2048 = 16 tiles),
/// so the slice produces whole tiles. The resulting block stores whatever
/// tile state actually exists (quantized hist tiles are lossy).
///
/// Must be called on the inference thread (Metal thread affinity).
pub fn extract_block(
    cache_set: &DetachedCacheSet,
    start: usize,
    end: usize,
) -> Result<DetachedCacheSet, ColdStoreError> {
    if start >= end {
        return Err(invalid_data(format!(
            "extract_block: start {start} >= end {end}"
        )));
    }
    let token_count = end - start;

    let mut caches = Vec::with_capacity(cache_set.caches.len());
    for layer in &cache_set.caches {
        let keys = layer.keys.as_ref().map(|k| slice_axis(k, 2, start as i32, end as i32));
        let values = layer.values.as_ref().map(|v| slice_axis(v, 2, start as i32, end as i32));
        let key_scales = layer.key_scales.as_ref().map(|s| slice_axis(s, 2, start as i32, end as i32));
        let val_scales = layer.val_scales.as_ref().map(|s| slice_axis(s, 2, start as i32, end as i32));
        let m3_idx_k = layer.m3_idx_k.as_ref().map(|m| slice_axis(m, 2, start as i32, end as i32));

        // KVarN8 keeps its state in three separate region tensors
        // ([sink | history tiles | tail]), so it cannot be served by the dense
        // path's single slice per tensor. For every other mode these fields
        // are already None, and `Default` reproduces exactly the previous
        // behaviour — so this branch adds the KVarN8 case without touching
        // what dense extraction did.
        let kvarn = if layer.mode == KVCacheMode::KVarN8 {
            extract_kvarn8_regions(layer, start, end)?
        } else {
            Kvarn8Regions::default()
        };

        caches.push(DetachedKVCache {
            keys,
            values,
            offset: token_count as i32,
            step: layer.step,
            mode: layer.mode,
            key_scales,
            val_scales,
            v_packed: None, // Turbo sidecars not extracted for blocks
            v_norms: None,
            v_rescale: None,
            k_packed: None,
            k_norms: None,
            turbo_seed: layer.turbo_seed,
            cold_offset: 0,
            hot_threshold: layer.hot_threshold,
            delegated_fp16_fast_path: layer.delegated_fp16_fast_path,
            delegated_fp16_sidecar_policy: layer.delegated_fp16_sidecar_policy,
            m3_idx_k,
            m3_idx_offset: token_count as i32,
            kvarn_sink_k: kvarn.sink_k,
            kvarn_sink_v: kvarn.sink_v,
            kvarn_tail_k: kvarn.tail_k,
            kvarn_tail_v: kvarn.tail_v,
            kvarn_hist_k: kvarn.hist_k,
            kvarn_hist_v: kvarn.hist_v,
            kvarn_k_scale: kvarn.k_scale,
            kvarn_k_zp: kvarn.k_zp,
            kvarn_k_s_row: kvarn.k_s_row,
            kvarn_k_s_col: kvarn.k_s_col,
            kvarn_v_scale: kvarn.v_scale,
            kvarn_v_zp: kvarn.v_zp,
            kvarn_v_s_row: kvarn.v_s_row,
            kvarn_v_s_col: kvarn.v_s_col,
            kvarn_v_bits: layer.kvarn_v_bits,
        });
    }

    Ok(DetachedCacheSet {
        caches,
        backend: SequenceStateBackend::DenseKvCache,
        prompt_len: token_count,
        current_offset: token_count as i32,
        created_at: std::time::Instant::now(),
        detached_at: std::time::Instant::now(),
        origin_seq_id: SequenceId(0),
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

fn now_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
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
        // Unit test: pure function incorporates prev_hash.
        // NOT a discrimination control — the caller's chain logic is tested below.
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
    fn compute_block_hashes_prefix_dependent() {
        // RED CONTROL: the caller's chunk-and-chain path must produce
        // different block addresses for conversations with different
        // prefixes but identical mid-block tokens.
        // Under own-token hashing (no chain), addrs_a[1] == addrs_b[1].
        // Under Merkle-chain, they differ.
        let tokens_a = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let tokens_b = vec![9, 10, 11, 12, 5, 6, 7, 8];
        let kv_mode = kv_mode_config_string(KVCacheMode::Fp16);
        let block_size = 4;

        let addrs_a = compute_block_hashes(&tokens_a, block_size, &kv_mode);
        let addrs_b = compute_block_hashes(&tokens_b, block_size, &kv_mode);

        assert_ne!(addrs_a[0], addrs_b[0], "block 0 must differ");
        assert_ne!(
            addrs_a[1], addrs_b[1],
            "block 1 must differ — chain carries prefix"
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

    #[test]
    fn extract_block_basic() {
        // Extract a block from a cache set and verify token count.
        let cache = make_test_cache_set(1, 8, 16);
        let block = extract_block(&cache, 0, 4).unwrap();
        assert_eq!(block.caches.len(), 1);
        assert_eq!(block.caches[0].offset, 4);
        assert_eq!(block.prompt_len, 4);
        assert_eq!(block.current_offset, 4);
    }

    #[test]
    fn extract_block_partial() {
        // Extract a partial block (token_count < block_size).
        let cache = make_test_cache_set(1, 10, 16);
        let block = extract_block(&cache, 8, 10).unwrap();
        assert_eq!(block.caches[0].offset, 2);
        assert_eq!(block.prompt_len, 2);
    }

    #[test]
    fn extract_block_boundary() {
        // Extract at block_size boundary.
        let cache = make_test_cache_set(1, 2048, 16);
        let block = extract_block(&cache, 0, 2048).unwrap();
        assert_eq!(block.caches[0].offset, 2048);
    }

    #[test]
    fn extract_block_invalid_range() {
        let cache = make_test_cache_set(1, 8, 16);
        assert!(extract_block(&cache, 4, 4).is_err()); // start >= end
        assert!(extract_block(&cache, 8, 4).is_err()); // start > end
    }
}

#[cfg(test)]
#[path = "block_cold_store_tests.rs"]
mod block_cold_store_tests;
