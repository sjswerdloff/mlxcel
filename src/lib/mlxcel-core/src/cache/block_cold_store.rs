// Cold-store v4: block-based content-addressed storage.
//
// Splits KV caches into fixed-size blocks. Blocks are content-addressed
// by Merkle-chain hash. A manifest describes which blocks make up a
// full conversation. Shared prefix blocks are stored once and
// reference-counted.
//
// See docs/DESIGN_cold_store_block_storage_v4_2026-07-26.md for the
// full design.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use sha2::{Digest, Sha256};

use super::cold_store::{read_kv_cache, serialize_cache_set_layers, ColdStoreError, PruneMode};
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

/// Canonical `(mode, v_bits)` for ONE layer — the unit the block address is
/// built from.
///
/// A block address must commit to what determines its bytes in full, because
/// blocks live in ONE GLOBAL POOL keyed only by hash. Two runtimes that compute
/// different bytes for the same tokens must not be able to reach the same
/// address.
///
/// # Why this replaced `kv_mode_config_string`
///
/// The previous version was `format!("{:?}", mode)` under a doc comment
/// claiming it "must capture everything that affects the KV data bytes". It
/// captured the mode NAME only, and missed two things:
///
/// * **`v_bits`** — k8v4 and k8v8 rendered to the same string "KVarN8", so
///   they produced the SAME address with DIFFERENT payloads. No weight change
///   needed; one model and one mode was enough to collide.
/// * **`runtime_fingerprint`** — absent entirely, though the manifest carries
///   it. Two runtimes with different weights computed IDENTICAL block hashes,
///   so the second one's `write_block` hit its `block_dir.exists()` early
///   return, silently skipped writing, and left its manifest pointing at the
///   FIRST runtime's KV data. A silent cross-model wrong adoption.
///
/// # Why `v1` became `v2`: one pair for the whole set was never enough
///
/// `v1` was `mode:{:?}|vbits:{}` — a SINGLE pair, taken from layer 0 and
/// applied to the entire set. `persist` asserted the set was homogeneous so
/// that layer 0 could legitimately speak for the rest, and refused otherwise.
///
/// That precondition is false in production, and not only for one model. THREE
/// independent mechanisms hand the runtime a heterogeneous set:
///
/// * **Boundary-V** (`turbo::boundary::resolve_layer_modes`) gives boundary
///   layers a different mode from the interior for quality.
/// * **`skip_last_layer`** (`BatchKvQuantConfig::resolve_layer_modes`) forces
///   the final layer to `Fp16` while the rest stay quantized.
/// * **D1 dense-prefix downgrade** (MiniMax-M3): a layer with no index
///   projections can never take the gathered path, so an empty KVarN8 cache is
///   downgraded to `Fp16` at first touch. M3's first three layers are dense, so
///   EVERY M3 cache set is mixed by construction.
///
/// v4 met the third one first, and `persist` correctly refused every M3 set:
/// the store persisted nothing at all. The guard was right; the address was one
/// field too narrow. Widening it to the whole per-layer vector is what lets the
/// guard become a conformance check instead of a blanket refusal.
///
/// The version prefix is the anti-aliasing mechanism, so it moves with the
/// field set: a `v1` address and a `v2` address can never collide.
///
/// # Domain separation
///
/// Fields are tagged and `|`-delimited, and every value renders without a `|`:
/// the fingerprint and the plan digest are hex, the layer count is a `usize`.
/// The plan is digested rather than spelled out so the address stays bounded
/// at any layer count; the digest itself is length-prefixed per layer (below)
/// so no two distinct plans can hash to the same input string.
///
/// # Canonicalisation — why `v_bits` is normalised rather than passed through
///
/// The identity must commit to what DETERMINES the bytes and to nothing else.
/// A field that does not affect the bytes must not affect the address, or the
/// same tokens producing the same data land at different addresses and the
/// cache misses for no reason.
///
/// `v_bits` is exactly that field under `Fp16`: an unquantized cache carries
/// whatever `kvarn_v_bits` happens to hold — the test builder leaves it at 8 —
/// while the bytes do not depend on it at all. Passing it through made an
/// Fp16 persist at v_bits=8 unreachable to a loader that correctly considered
/// v_bits irrelevant and passed 0. Found by exactly that miss.
///
/// This is now load-bearing per LAYER, not just per set. A D1-downgraded layer
/// is left at `(Fp16, v_bits=8)` — `downgrade_kvarn8_to_fp16_if_empty` resets
/// the width to 8, not 0 (`cache.rs`) — while a plan naturally spells the same
/// layer `(Fp16, 0)`. Without canonicalisation those two describe identical
/// bytes and would address differently, which is the write-only failure again.
///
/// So: quantized modes commit to `v_bits`; unquantized modes normalise it to 0.
/// The `match` is exhaustive on purpose — a new mode forces this decision to be
/// made rather than inherited.
pub fn canonical_layer_identity(mode: super::KVCacheMode, v_bits: u8) -> (super::KVCacheMode, u8) {
    let effective_v_bits = match mode {
        // Unquantized: the V width is not part of the data.
        super::KVCacheMode::Fp16 => 0,
        // Quantized: v_bits selects the payload layout, so it IS the identity.
        _ => v_bits,
    };
    (mode, effective_v_bits)
}

/// Complete identity of everything that determines a block's BYTES, over the
/// WHOLE per-layer plan. See `canonical_layer_identity` for the versioning and
/// canonicalisation rationale.
///
/// `plan[i]` is the `(mode, v_bits)` layer `i`'s cache will actually be in when
/// its bytes are written. Both sides of the store must pass the same plan:
/// `persist` derives it from the live cache set and checks it against the
/// caller's, `load_prefix` takes the caller's directly.
pub fn cache_computation_id(
    runtime_fingerprint: &[u8; 32],
    plan: &[(super::KVCacheMode, u8)],
) -> String {
    let mut hasher = Sha256::new();
    // Length-prefix the whole plan, then each layer's rendered mode, so a
    // plan can never hash to the same input as a different plan whose
    // variant names happen to concatenate identically.
    hasher.update((plan.len() as u64).to_le_bytes());
    for &(mode, v_bits) in plan {
        let (mode, effective_v_bits) = canonical_layer_identity(mode, v_bits);
        let name = format!("{mode:?}");
        hasher.update((name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update([effective_v_bits]);
    }
    let plan_digest: [u8; 32] = hasher.finalize().into();
    format!(
        "v2|rt:{}|n:{}|plan:{}",
        hex_digest(runtime_fingerprint),
        plan.len(),
        hex_digest(&plan_digest)
    )
}

/// Human-readable rendering of a per-layer plan, for logs and error messages.
///
/// Deliberately NOT part of the address: `cache_computation_id` has exactly one
/// rendering so there is only one way for two plans to collide. This exists
/// because the `v1` address was greppable in a log and the `v2` digest is not,
/// and losing that would make a mismatch far harder to diagnose.
///
/// Runs of identical layers are collapsed, so M3's real plan reads as
/// `Fp16x3,KVarN8v4x57` rather than sixty entries.
pub fn describe_kv_layer_plan(plan: &[(super::KVCacheMode, u8)]) -> String {
    let mut out = String::new();
    let mut run: Option<((super::KVCacheMode, u8), usize)> = None;
    let mut flush = |out: &mut String, entry: ((super::KVCacheMode, u8), usize)| {
        let ((mode, v_bits), count) = entry;
        if !out.is_empty() {
            out.push(',');
        }
        match mode {
            super::KVCacheMode::Fp16 => out.push_str(&format!("{mode:?}")),
            _ => out.push_str(&format!("{mode:?}v{v_bits}")),
        }
        if count > 1 {
            out.push_str(&format!("x{count}"));
        }
    };
    for &(mode, v_bits) in plan {
        let layer = canonical_layer_identity(mode, v_bits);
        match run {
            Some((current, count)) if current == layer => run = Some((current, count + 1)),
            Some(entry) => {
                flush(&mut out, entry);
                run = Some((layer, 1));
            }
            None => run = Some((layer, 1)),
        }
    }
    if let Some(entry) = run {
        flush(&mut out, entry);
    }
    if out.is_empty() {
        out.push_str("<empty>");
    }
    out
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
/// TEST SEAM — fires inside `gc_blocks` after nomination and BEFORE the store
/// lock is acquired.
///
/// This is the window Alden's tests 1 and 2 require: "writer begins after mark
/// but before sweep and commits a manifest referencing candidate X: X MUST
/// survive." A stress test cannot schedule that interleaving — measured, the
/// concurrency test stayed green with the publication lock removed entirely —
/// so the interleaving has to be constructed rather than hoped for.
///
/// Because the seam runs before GC takes the lock, a publisher invoked from it
/// can acquire the lock normally, which means the whole scenario runs on ONE
/// thread. That matters here: `DetachedCacheSet` holds cxx pointers that are
/// not `Send`, and moving MLX work off-thread is its own hazard.
///
/// `cfg(test)` only — it does not exist in a shipped binary.
#[cfg(test)]
pub(crate) static GC_NOMINATION_SEAM: std::sync::Mutex<Option<Box<dyn Fn() + Send>>> =
    std::sync::Mutex::new(None);

#[cfg(test)]
fn fire_gc_nomination_seam() {
    let seam = GC_NOMINATION_SEAM.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(f) = seam.as_ref() {
        f();
    }
}

#[cfg(not(test))]
#[inline(always)]
fn fire_gc_nomination_seam() {}

/// Process-wide publication/sweep exclusion, paired with the kernel file lock.
///
/// Deliberately COARSE — one mutex for every store in the process rather than
/// one per `base_dir`. Publication and GC are both rare and already do file
/// I/O, so contention is irrelevant, and a per-path map would need its own
/// canonicalization to be correct (two `BlockColdStore`s can name one directory
/// through different paths, and a map keyed on the un-canonicalized `PathBuf`
/// would hand them different mutexes while they share a store — an exclusion
/// bug that looks like it works).
/// A RwLock rather than a Mutex because the file lock it pairs with has two
/// modes. Loaders take `read()` + `LOCK_SH`; publication and the sweep take
/// `write()` + `LOCK_EX`. A Mutex here would serialize concurrent loaders
/// against each other for no reason, and — worse — would make the process-level
/// half disagree with the kernel half about what "shared" means.
static STORE_MUTEX: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// RAII exclusion guard. Releases in the reverse of acquisition: file lock
/// first, then the process mutex when the guard field drops.
///
/// A poisoned mutex is recovered rather than propagated: the invariant this
/// protects lives on disk, not in the `()` payload, so a thread that panicked
/// while holding it has not corrupted anything this guard can see. Refusing to
/// lock forever after one unrelated panic would disable GC for the life of the
/// process.
struct StoreLock<'a> {
    _process: StoreGuardKind<'a>,
    file: File,
}

/// Which half of the RwLock a `StoreLock` is holding. Kept as an enum rather
/// than two types so callers cannot accidentally hold a shared process guard
/// while taking an exclusive file lock, or the reverse — the two halves are
/// acquired together and released together.
enum StoreGuardKind<'a> {
    Exclusive(#[allow(dead_code)] std::sync::RwLockWriteGuard<'a, ()>),
    Shared(#[allow(dead_code)] std::sync::RwLockReadGuard<'a, ()>),
}

impl Drop for StoreLock<'_> {
    fn drop(&mut self) {
        // SAFETY: the fd is valid until `file` drops, which happens after this.
        unsafe {
            libc::flock(
                std::os::unix::io::AsRawFd::as_raw_fd(&self.file),
                libc::LOCK_UN,
            );
        }
        // The kernel would also release on close/exit; unlocking explicitly
        // keeps the release point visible at the end of the critical section
        // rather than implicit in a later drop order.
    }
}

pub struct BlockColdStore {
    base_dir: PathBuf,
    runtime_fingerprint: [u8; 32],
    block_size: usize,
    prune_mode: PruneMode,
    persist_lock: Mutex<()>,
    /// Minimum age before an unreferenced block may even be NOMINATED.
    ///
    /// Defence in depth, explicitly NOT a correctness gate (Alden, finding 4:
    /// "age cannot authorize deletion and never replaces lock+final reverify").
    /// What it buys: less churn, a recovery/forensics window in which a
    /// mistakenly-orphaned block is still on disk, and protection against
    /// implementation mistakes around freshly-staged artifacts that are not yet
    /// referenced because their manifest has not been published.
    ///
    /// What it does NOT buy: safety against the publication race. A block older
    /// than any threshold can be referenced by a manifest published one
    /// microsecond from now.
    min_gc_age: std::time::Duration,
}

impl BlockColdStore {
    pub fn new(base_dir: PathBuf, runtime_fingerprint: [u8; 32]) -> Self {
        Self {
            base_dir,
            runtime_fingerprint,
            block_size: DEFAULT_BLOCK_SIZE,
            prune_mode: PruneMode::default(),
            persist_lock: Mutex::new(()),
            // Conservative by default. Zero-age deletion is NOT advertised as
            // safe for production; tests that need immediate collection opt in
            // explicitly via `with_min_gc_age`.
            min_gc_age: std::time::Duration::from_secs(300),
        }
    }

    /// Override the nomination age floor. Intended for tests, which cannot wait
    /// out the production default. Setting this to zero does not make deletion
    /// safe — the lock and the under-lock re-verification are what make it
    /// safe; this only removes a cushion.
    pub fn with_min_gc_age(mut self, age: std::time::Duration) -> Self {
        self.min_gc_age = age;
        self
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

    /// Path of the persistent advisory-lock file. The FILE persists; the LOCK
    /// is kernel-held and released automatically when the holder exits or
    /// crashes — which is the whole reason this is `flock` and not an
    /// `O_EXCL` sentinel. An `O_EXCL` lockfile left behind by a killed process
    /// needs a staleness heuristic to clear, and every such heuristic fails
    /// toward either permanent deadlock or unsafe stealing.
    fn lock_path(&self) -> PathBuf {
        self.base_dir.join("store.lock")
    }

    /// Acquire publication/sweep exclusion: process mutex FIRST, then the
    /// kernel file lock (Alden, finding 4: "Acquire process Mutex then file
    /// lock in one documented order everywhere. Do not assume flock semantics
    /// provide same-process thread exclusion.").
    ///
    /// The two layers cover different things. `flock` on most platforms is
    /// per-open-file-description, so two threads in one process can both hold
    /// it and neither is excluded; the mutex covers threads, the file lock
    /// covers processes. Neither alone is sufficient.
    ///
    /// FAILS CLOSED. If the lock cannot be taken, the caller must not sweep.
    fn acquire_store_lock(&self) -> Result<StoreLock<'static>, ColdStoreError> {
        // Order is load-bearing and identical at every call site.
        let process_guard = STORE_MUTEX.write().unwrap_or_else(|p| p.into_inner());

        fs::create_dir_all(&self.base_dir)?;
        let file = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(self.lock_path())?;

        // SAFETY: `file` owns a valid fd for the duration of this call, and the
        // returned guard keeps it alive until the lock is released in Drop.
        let rc =
            unsafe { libc::flock(std::os::unix::io::AsRawFd::as_raw_fd(&file), libc::LOCK_EX) };
        if rc != 0 {
            let err = io::Error::last_os_error();
            tracing::error!(
                error = %err,
                path = %self.lock_path().display(),
                "COLD-STORE v4: could not acquire the advisory store lock — FAILING \
                 CLOSED. Delete-mode GC must not sweep without exclusion; observe may \
                 report but must not act."
            );
            return Err(ColdStoreError::Io(err));
        }
        Ok(StoreLock {
            _process: StoreGuardKind::Exclusive(process_guard),
            file,
        })
    }

    /// ACTIVE-LOAD LEASE (Alden finding 4: "Active loads need a short
    /// lease/read lock that also counts as a root during GC"; his test 5).
    ///
    /// Shared, so concurrent loaders do not block each other, but mutually
    /// exclusive with the sweep's `LOCK_EX` — which is the point. Without it a
    /// load is a read of manifest-then-blocks with no atomicity: GC can prove a
    /// block unreferenced, tombstone it, and unlink it in the gap between a
    /// loader reading the manifest that names it and the loader opening it. The
    /// loader then fails on a block its own manifest promised.
    ///
    /// Held for the whole read — manifest AND blocks — because the hazard lives
    /// in the gap between them, not in either half.
    ///
    /// Deliberately NOT the same thing as fail-soft re-prefill. Re-prefilling is
    /// a fallback for a store that is legitimately cold; it is not a substitute
    /// for a store that tears under concurrent GC, and treating it as one would
    /// make an incoherence look like a cache miss.
    fn acquire_read_lease(&self) -> Result<StoreLock<'static>, ColdStoreError> {
        let process_guard = STORE_MUTEX.read().unwrap_or_else(|p| p.into_inner());
        fs::create_dir_all(&self.base_dir)?;
        let file = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(self.lock_path())?;
        // SAFETY: fd valid for the duration; the guard keeps it alive.
        let rc =
            unsafe { libc::flock(std::os::unix::io::AsRawFd::as_raw_fd(&file), libc::LOCK_SH) };
        if rc != 0 {
            let err = io::Error::last_os_error();
            tracing::warn!(
                error = %err,
                "COLD-STORE v4: could not take a read lease; a concurrent sweep could \
                 collect a block this load needs"
            );
            return Err(ColdStoreError::Io(err));
        }
        Ok(StoreLock {
            _process: StoreGuardKind::Shared(process_guard),
            file,
        })
    }

    /// Non-blocking variant. Returns `Ok(None)` when another holder has it.
    ///
    /// Exists so a test can PROVE exclusion rather than assume it: a blocking
    /// acquire against a live holder hangs, and "it hung" is not an assertion.
    /// Not used by the sweep — GC wants to wait for the lock, not skip its pass
    /// because a publisher happened to hold it for a millisecond.
    #[cfg(test)]
    fn try_acquire_store_lock(&self) -> Result<Option<StoreLock<'static>>, ColdStoreError> {
        // `try_write`, not `write`. A blocking acquire inside a function named
        // "try" is a latent deadlock for every caller, and it deadlocked the
        // very first one: a test holding a read lease on this thread probed for
        // the exclusive lock, and `write()` blocked on the guard that same
        // thread was holding. The suite HUNG rather than failed, which is the
        // worse failure — a hang carries no message. Non-blocking at BOTH
        // layers, or the probe is not a probe.
        //
        // A held read lease therefore reports `None`, which is the honest
        // answer: a real sweep would block on exactly that guard.
        let Ok(process_guard) = STORE_MUTEX.try_write() else {
            return Ok(None);
        };
        fs::create_dir_all(&self.base_dir)?;
        let file = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(self.lock_path())?;
        // SAFETY: fd valid for the duration; the guard keeps it alive.
        let rc = unsafe {
            libc::flock(
                std::os::unix::io::AsRawFd::as_raw_fd(&file),
                libc::LOCK_EX | libc::LOCK_NB,
            )
        };
        if rc != 0 {
            let err = io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
                return Ok(None);
            }
            return Err(ColdStoreError::Io(err));
        }
        Ok(Some(StoreLock {
            _process: StoreGuardKind::Exclusive(process_guard),
            file,
        }))
    }

    /// Monotonic count of manifest PUBLICATIONS (Alden finding 4, 2026-07-27).
    ///
    /// GC's mark phase is a point-in-time snapshot. The unsafe interleaving is:
    /// GC marks block X unreferenced, a writer commits a manifest referencing X,
    /// GC deletes X on its stale mark, and a live manifest now points at missing
    /// data. Refcount increments do not close this, because the delete decision
    /// came from the earlier scan.
    ///
    /// This counter makes the staleness DETECTABLE: read it before marking, read
    /// it again before deleting, and a change means some manifest was published
    /// since the mark, so the mark cannot be trusted. "With an epoch design, any
    /// manifest/root publication since mark invalidates the mark and requires a
    /// rescan."
    fn epoch_path(&self) -> PathBuf {
        self.base_dir.join("publication.epoch")
    }

    fn read_publication_epoch(&self) -> u64 {
        match fs::read(self.epoch_path()) {
            Ok(b) if b.len() == 8 => u64::from_le_bytes(b.try_into().unwrap()),
            // Missing or malformed reads as a SENTINEL that never compares equal
            // to a later read, so an unreadable epoch degrades to "assume
            // published" rather than to "assume quiet".
            _ => u64::MAX,
        }
    }

    /// Bumped BEFORE a manifest becomes visible — see the ordering argument at
    /// the call site in `write_manifest`. A bump for a publication that then
    /// fails costs a wasted rescan; a visible manifest behind an un-bumped
    /// epoch costs a live block.
    fn bump_publication_epoch(&self) -> Result<(), ColdStoreError> {
        let cur = self.read_publication_epoch();
        let next = if cur == u64::MAX {
            1
        } else {
            cur.wrapping_add(1)
        };
        let tmp = self.base_dir.join(".tmp.publication.epoch");
        write_file(&tmp, &next.to_le_bytes())?;
        fs::rename(&tmp, self.epoch_path())?;
        Ok(())
    }

    pub fn manifests_dir(&self) -> PathBuf {
        self.base_dir.join(V4_ROOT).join(MANIFESTS_DIR)
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// The configured prune mode. Exposed so a test can pin the DEFAULT rather
    /// than only the modes it sets explicitly — the default is the one an
    /// operator gets without choosing.
    pub fn prune_mode(&self) -> PruneMode {
        self.prune_mode
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
        let tmp_dir = self
            .blocks_dir()
            .join(format!(".tmp.{}", hex_digest(block_hash)));
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
            write_file(&tmp_dir.join(format!("layer_{index:04}.bin")), bytes)?;
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
            // `metadata.index` was WRITTEN and never READ: the path below is built
            // from the enumerate position, so a corrupted stored index changed
            // nothing and went unchallenged. Found 2026-07-28 by flipping every
            // header byte in turn — after the token region was covered by address
            // re-derivation, these 8 bytes (two `index` fields) were the only ones
            // left that could be corrupted with the load still succeeding.
            //
            // Harmless today precisely BECAUSE nothing trusts it, which is the
            // argument for checking rather than deleting: a stored field that is
            // allowed to lie is inherited by whoever reads it next.
            if metadata.index as usize != expected_index {
                return Err(invalid_data(format!(
                    "block header layer slot {expected_index} declares index {} — the \
                     layer table is out of order or corrupted, and the payload read below \
                     would be attributed to the wrong layer",
                    metadata.index
                )));
            }
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

        // PUBLICATION CRITICAL SECTION (Alden finding 4, publication protocol
        // steps 2-5). Block bytes and the temp manifest above were written
        // outside it; from here to the rename we hold the same exclusion GC's
        // sweep takes, which is what makes GC's tombstone rename a genuine
        // linearization point. Without this half, GC could tombstone a block
        // between a publisher's check and its publish, and the tombstone
        // protocol would be decoration.
        let _pub_lock = self.acquire_store_lock()?;

        // Revalidate under the lock. "Existing means fully committed and
        // integrity-valid, not path-exists" — a block observed before the lock
        // may have been tombstoned since, and publishing a manifest that
        // references it would commit a reference to something GC is about to
        // unlink.
        for bh in &manifest.block_hashes {
            let dir = self.blocks_dir().join(hex_digest(bh));
            if !dir.exists() {
                return Err(invalid_data(format!(
                    "write_manifest: block {} is absent at publication time — it was \
                     collected or never installed. The manifest is NOT published; \
                     reinstall the block and retry rather than committing a reference \
                     to missing data.",
                    hex_digest(bh)
                )));
            }
        }

        // Advance the epoch BEFORE the manifest becomes visible (Alden, finding
        // 4 publication protocol step 4). The two are separate files and there
        // is no transaction across them, so one of the two crash windows is
        // going to exist. This ordering picks the harmless one:
        //
        //   epoch bumped, rename never happens  -> a GC rescan that finds
        //                                          nothing changed. Wasted work.
        //   manifest visible, epoch not bumped  -> a concurrent GC sees a QUIET
        //                                          epoch, trusts its stale mark,
        //                                          and deletes a block the new
        //                                          manifest references.
        //
        // "An epoch with no manifest causes an unnecessary rescan; a visible
        // manifest can never be hidden behind an old epoch."
        //
        // CORRECTED 2026-07-27: `031163d` bumped AFTER the rename — the unsafe
        // order — while the comment above it argued for this one. The prose was
        // right and the code did the opposite; only Alden's review caught that
        // they disagreed. A comment that states the correct rule is not a
        // control, and I had read past mine twice.
        self.bump_publication_epoch()?;

        // Atomic rename — the manifest becomes VISIBLE here.
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
    ///
    /// Enforces that the decoded content hashes to the directory it was filed
    /// under. Every MANIFEST-directory consumer should reach content through
    /// here rather than decoding `manifest.bin` directly — see
    /// [`committed_manifest_hash`].
    pub fn read_manifest(&self, manifest_hash: &[u8; 32]) -> Result<Manifest, ColdStoreError> {
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

    /// Delete manifests that describe a strict TOKEN prefix of the new
    /// manifest's token sequence. This handles the normal case (conversation
    /// grows).
    ///
    /// # Why this proves a token prefix rather than checking a hash-list prefix
    ///
    /// Alden's finding 2, confirmed against this code 2026-07-27 and pinned by
    /// `growth_from_a_partial_tail_block_still_prunes_the_old_manifest`:
    ///
    /// `persist` chunks with `tokens.chunks(block_size)`, which yields a
    /// PARTIAL final chunk whenever the token count is not an exact multiple of
    /// the block size — and `block_hash_merkle` commits to that chunk's own
    /// token count and tokens. So the moment the next turn fills that tail
    /// block, its hash CHANGES. The old manifest's hash list is therefore not a
    /// prefix of the new one, and a hash-list prefix check skips it: the
    /// superseded manifest survives, holding a refcount on a now-orphaned tail
    /// block. A conversation lands on an exact multiple of `block_size` only by
    /// accident, so that was the ordinary path, not an edge case.
    ///
    /// The fix is to prove the thing we actually mean. We are inside `persist`
    /// and hold the full token sequence, so for each candidate we recompute the
    /// block hashes that `tokens[..old.total_tokens]` WOULD produce under the
    /// current cache-computation identity, and prune only on an exact match.
    /// That is a positive proof that the old manifest describes a prefix of
    /// this same conversation under this same identity — strictly stronger than
    /// the hash-list check it replaces, and it does not care where block
    /// boundaries fall.
    ///
    /// Conservative by construction: any mismatch (divergent branch, different
    /// mode/`v_bits`, different `block_size`, unreadable manifest) leaves the
    /// candidate alone. Failing to prune costs disk; pruning wrongly costs a
    /// live manifest, so every uncertain case must fall the same way.
    pub fn prune_prefix_manifests(
        &self,
        new_manifest: &Manifest,
        tokens: &[i32],
        cache_id: &str,
    ) -> Result<(), ColdStoreError> {
        let manifests_dir = self.manifests_dir();
        if !manifests_dir.exists() {
            return Ok(());
        }

        let entries: Vec<_> = fs::read_dir(&manifests_dir)?
            .filter_map(|e| e.ok())
            .collect();

        for entry in entries {
            // MIGRATED TO THE SHARED PREDICATE — Alden, 2026-07-28, blocker 2.
            //
            // This surface kept its own inline `.tmp.` filter and direct-decoded
            // `manifest.bin`, so the "single audit surface" was still two.
            //
            // ⚠️ NOT A DEMONSTRATED LIVE BUG — and this is recorded because the
            // person who raised the concern WITHDREW it, which is more useful to
            // a future reader than an anonymous worry.
            //
            // The concern: a direct decode lets misfiled bytes at directory Y
            // drive `delete_manifest` against a different directory X. I could
            // not construct that as an observable harm; Alden then confirmed the
            // claim was hypothetical and that he had stated it as demonstrated.
            // His own enumeration, kept here so nobody re-derives it or "fixes"
            // the non-bug: misfiled Y contains X, so it carries X's pruning
            // eligibility — if correctly-filed X exists and is prunable, X's own
            // entry reaches the same deletion; if X is divergent, both entries
            // reject; if X is absent, `delete_manifest(X)` has no target. Every
            // check below is CONTENT-based, so the two entries are judged
            // identically, and the old deletion was content-addressed too.
            //
            // What the migration buys is therefore STRUCTURAL, not a bug fix:
            // one audit surface instead of two, and a deletion target bound to
            // the directory actually examined, so a future direct-decode
            // regression cannot reintroduce the concern. Pinned by
            // `a_misfiled_manifest_is_not_a_prune_input_and_causes_no_deletion`
            // (property that holds either way) and its positive control.
            //
            // FAILS SOFT on both the predicate and the read, which is safe in the
            // pruning direction specifically: skipping a prune leaves EXTRA state
            // behind, never deletes live state. The sweeper cannot make that
            // trade; this function can.
            let manifest_hash = match committed_manifest_hash(&entry) {
                Ok(Some(h)) => h,
                Ok(None) => continue,
                Err(_) => continue,
            };
            // CANONICAL READ — binds the content to the address it is filed
            // under, so `manifest_hash` below is the directory we actually read
            // and not a hash recovered from bytes that may not belong here.
            let old_manifest = match self.read_manifest(&manifest_hash) {
                Ok(m) => m,
                Err(_) => continue,
            };

            // Same identity check — cheapest discriminator, so it runs first.
            if old_manifest.runtime_fingerprint != new_manifest.runtime_fingerprint
                || old_manifest.model_id != new_manifest.model_id
                || old_manifest.template_sig != new_manifest.template_sig
            {
                continue;
            }

            // Never prune the manifest we just wrote, and never prune a
            // manifest that is not STRICTLY shorter — equal length is either
            // the same manifest or a divergence, and longer is not a prefix.
            if old_manifest.total_tokens >= new_manifest.total_tokens {
                continue;
            }

            // Block hashes computed under a different block_size are not
            // comparable to ours at all.
            if old_manifest.block_size != self.block_size {
                continue;
            }

            // Guard the slice below. `total_tokens` is read off disk, so it is
            // untrusted input, not an invariant.
            if old_manifest.total_tokens > tokens.len() {
                continue;
            }

            // TOKEN-PREFIX PROOF. Recompute what this candidate's block hashes
            // would be for the first `old.total_tokens` tokens of the sequence
            // we are persisting. An exact match proves the candidate describes
            // a prefix of THIS conversation under THIS cache identity —
            // including when its final block was partial and has since filled.
            let expected = compute_block_hashes(
                &tokens[..old_manifest.total_tokens],
                self.block_size,
                cache_id,
            );
            if expected != old_manifest.block_hashes {
                continue;
            }

            // OBSERVE MEANS OBSERVE.
            //
            // `PruneMode::Observe` is the DEFAULT and is documented as "log
            // what WOULD be pruned, do not delete". `gc_blocks` honours that;
            // this function did not — it called `delete_manifest` in every mode
            // except `Off`, so the default configuration was silently deleting
            // manifests while reporting itself as observe-only. Found
            // 2026-07-27 while testing Alden's finding 2: the prune tests
            // passed under the default mode, which they could only do if
            // deletion was really happening. Pinned by
            // `observe_mode_reports_a_prune_without_performing_it`.
            if self.prune_mode == PruneMode::Observe {
                tracing::info!(
                    would_prune = %old_manifest.hash_hex(),
                    old_tokens = old_manifest.total_tokens,
                    new_tokens = new_manifest.total_tokens,
                    "COLD-STORE v4 observe: would prune token-prefix manifest"
                );
                continue;
            }

            // Prune BY THE DIRECTORY ADDRESS we read, not by a hash recovered
            // from the bytes. `read_manifest` has already proved the two agree;
            // using the address makes the deletion target structurally the thing
            // that was examined, so a future direct-decode regression cannot
            // reintroduce "delete X because Y contained X's bytes".
            match self.delete_manifest(&manifest_hash) {
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
        plan: &[(super::KVCacheMode, u8)],
    ) -> Result<Manifest, ColdStoreError> {
        // WRITE side of the cache-computation identity. Must stay symmetric
        // with the READ side in load_prefix or the store becomes write-only.
        //
        // ASYMMETRY IS THE HAZARD, AND IT IS SILENT. Only `persist` can see the
        // real per-layer state; `load_prefix` runs before any layer is touched
        // and must be told the plan. If the two disagree the computed addresses
        // simply never match: `matched_blocks` is 0, every candidate is dropped,
        // the caller sees `NoMatch` — indistinguishable from a legitimately cold
        // cache. That is exactly how the hardcoded-Fp16 bug hid (handoff §7.5),
        // so the plan is CHECKED against reality here rather than trusted.
        //
        // This replaced a homogeneity assertion. That guard was correct about
        // its own premise — a single per-set pair means layer 0 speaks for every
        // layer, which is the m3_idx_offset shape (116924f) — but the premise
        // itself was false: Boundary-V, `skip_last_layer` and M3's D1
        // dense-prefix downgrade each produce genuinely mixed sets. Refusing
        // them meant v4 persisted NOTHING on M3. Now the address carries the
        // whole vector, so a mixed set is storable and what must be asserted is
        // that the caller's plan matches the bytes about to be written.
        // ONE derivation of "what this set actually is" — `layer_plan`, which
        // is also what every caller outside this module must use, so the check
        // here and the plan a caller builds cannot come from two definitions
        // that drift apart.
        let actual_plan = cache_set.layer_plan();
        if actual_plan.len() != plan.len() {
            return Err(invalid_data(format!(
                "persist: cache set has {} layers but the caller's plan describes {}. \
                 The plan addresses the ENTIRE set, so a length disagreement means \
                 the address would describe a different set than the one being \
                 written. Refusing rather than mislabelling.",
                actual_plan.len(),
                plan.len()
            )));
        }
        if let Some((i, actual, expected)) = actual_plan
            .iter()
            .zip(plan.iter())
            .enumerate()
            .map(|(i, (&(a_mode, a_bits), &(e_mode, e_bits)))| {
                (
                    i,
                    canonical_layer_identity(a_mode, a_bits),
                    canonical_layer_identity(e_mode, e_bits),
                )
            })
            .find(|(_, actual, expected)| actual != expected)
        {
            return Err(invalid_data(format!(
                "persist: layer {i} is ({:?}, v_bits={}) but the caller's plan says \
                 ({:?}, v_bits={}). Plan for the whole set: {}. The plan IS the block \
                 address, so writing under it would store bytes that do not match the \
                 address describing them — and the READ side, which can only use the \
                 plan, would then miss its own data silently. Refusing rather than \
                 mislabelling.",
                actual.0,
                actual.1,
                expected.0,
                expected.1,
                describe_kv_layer_plan(plan)
            )));
        }

        let cache_id = cache_computation_id(&self.runtime_fingerprint, plan);
        let block_hashes = compute_block_hashes(tokens, self.block_size, &cache_id);

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

        // Prune manifests proven to describe a strict token prefix of `tokens`.
        // `cache_id` is passed through so the proof is computed under the same
        // identity the hashes above were — see prune_prefix_manifests.
        if self.prune_mode != PruneMode::Off {
            self.prune_prefix_manifests(&manifest, tokens, &cache_id)?;
        }

        Ok(manifest)
    }

    // -----------------------------------------------------------------------
    // Garbage collection
    // -----------------------------------------------------------------------

    /// Garbage-collect blocks with refcount 0.
    /// In observe mode, logs what WOULD be deleted without deleting.
    /// MARK PHASE — the authoritative live set.
    ///
    /// Alden's finding 4 (2026-07-27): "Mark phase validates every committed
    /// manifest plus roots, rollback pins, and active-load leases. Refcounts
    /// are nomination/observability hints only, never authority."
    ///
    /// Reachability is a property of the committed manifests, so it is read
    /// from them. A refcount file is a derived cache of that fact and can be
    /// stale, racy, or simply wrong; deleting a block because a *hint* says
    /// zero is how a live manifest ends up pointing at missing data.
    ///
    /// An unreadable manifest ABORTS the caller rather than being skipped:
    /// "an unreadable committed manifest or authoritative root means zero
    /// references are NOT proved. Do not interpret corruption as an empty
    /// reference set." Skipping one is precisely how corruption becomes
    /// deletion — the block it referenced would look unreachable.
    ///
    /// NOT YET IMPLEMENTED, and this function is not safe for delete mode
    /// without them (finding 4, remaining bullets): publication/sweep
    /// exclusion via a shared lock or generation epoch, re-verification of
    /// every candidate after the grace interval, and active-load leases. The
    /// mark below is a point-in-time snapshot; a writer may commit a manifest
    /// referencing a candidate the instant after it is taken.
    fn mark_reachable_blocks(&self) -> Result<HashSet<[u8; 32]>, ColdStoreError> {
        let mut live: HashSet<[u8; 32]> = HashSet::new();
        let manifests_dir = self.manifests_dir();
        if !manifests_dir.exists() {
            return Ok(live);
        }
        for entry in fs::read_dir(&manifests_dir)? {
            let entry = entry?;
            // PROPAGATES on error. An I/O failure here must not silently shrink
            // the root set — a manifest missing from it is a manifest whose
            // blocks this pass would authorize deleting.
            let Some(manifest_hash) = committed_manifest_hash(&entry)? else {
                continue;
            };
            // Deliberately propagates — NOT the fail-soft `continue` that
            // `load_prefix` uses on the same helper. The asymmetry is the point:
            // a manifest this function cannot read is not a manifest with no
            // references, so the sweep aborts rather than treating its blocks as
            // unreferenced. A reader that skips a bad candidate loses a cache
            // hit; a sweeper that skips one deletes live data.
            let manifest = self.read_manifest(&manifest_hash).map_err(|e| {
                tracing::error!(
                    manifest = %hex_digest(&manifest_hash),
                    error = %e,
                    "COLD-STORE v4 GC: a committed manifest is unreadable, so reachability \
                     cannot be proved — ABORTING this pass rather than treating its blocks \
                     as unreferenced"
                );
                e
            })?;
            for b in &manifest.block_hashes {
                live.insert(*b);
            }
        }
        Ok(live)
    }

    pub fn gc_blocks(&self) -> Result<(), ColdStoreError> {
        let blocks_dir = self.blocks_dir();
        if !blocks_dir.exists() {
            return Ok(());
        }

        // Authority comes from here, not from refcount files.
        //
        // EPOCH-GUARDED MARK. Read the publication counter, mark, then confirm
        // nothing was published while we marked. A change means some manifest
        // became visible after we started reading them, so the live set we just
        // built may be missing its blocks — rescan rather than act on it.
        //
        // Bounded, because an unbounded retry under a busy writer is a hang, and
        // a GC that never runs is safer than one that acts on a stale mark. On
        // exhaustion this returns Ok WITHOUT deleting: declining to collect is
        // always safe, and the next pass will try again.
        const MAX_MARK_ATTEMPTS: usize = 4;
        let mut live: HashSet<[u8; 32]> = HashSet::new();
        let mut stable = false;
        for attempt in 0..MAX_MARK_ATTEMPTS {
            let before = self.read_publication_epoch();
            let marked = self.mark_reachable_blocks()?;
            let after = self.read_publication_epoch();
            if before == after && before != u64::MAX {
                live = marked;
                stable = true;
                break;
            }
            tracing::info!(
                attempt = attempt + 1,
                epoch_before = before,
                epoch_after = after,
                "COLD-STORE v4 GC: a manifest was published during the mark phase — \
                 the mark is stale, rescanning"
            );
        }
        if !stable {
            tracing::warn!(
                attempts = MAX_MARK_ATTEMPTS,
                "COLD-STORE v4 GC: could not obtain a stable mark under concurrent \
                 publication; collecting NOTHING this pass. Declining to collect is \
                 safe; acting on a stale mark is not."
            );
            return Ok(());
        }

        // The epoch as observed under a stable mark. Re-checked under the lock
        // before anything is tombstoned.
        let marked_epoch = self.read_publication_epoch();

        // Nominations. Nothing is destroyed from this unlocked scan.
        let mut candidates: Vec<([u8; 32], String)> = Vec::new();

        let entries: Vec<_> = fs::read_dir(&blocks_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
            .filter(|e| {
                let name = e.file_name();
                let name = name.to_string_lossy();
                // `.tombstone.` dirs are GC's own in-flight artifacts, not
                // blocks. A crash between the rename and the unlink leaves one
                // behind; it must never be mistaken for live state, nor
                // re-nominated as though it were a block in its own right.
                //
                // BELT AND BRACES, measured 2026-07-28: removing this clause
                // does NOT redden
                // `a_tombstone_left_by_a_crash_is_inert_and_the_final_path_can_be_reoccupied`,
                // because `parse_hex_digest` already rejects a dotted prefix and
                // the entry is skipped a few lines below. The real guard is
                // there. This clause is kept because it states the intent at the
                // point where someone would otherwise wonder, and because a
                // future change to the naming scheme could make hex parsing
                // accept something it currently rejects — but do not mistake it
                // for the control.
                !name.starts_with(".tmp.") && !name.starts_with(".tombstone.")
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

            // REACHABILITY decides. A block named by any committed manifest
            // survives regardless of what its refcount file claims.
            if live.contains(&block_hash) {
                continue;
            }

            // The refcount is now a HINT. Where it disagrees with reachability
            // it is the refcount that is wrong, and that disagreement is worth
            // seeing: it means the increment/decrement ordering leaked
            // somewhere. Reported, never obeyed.
            match self.get_refcount(&block_hash) {
                Ok(rc) if rc > 0 => tracing::warn!(
                    block = %name,
                    refcount = rc,
                    "COLD-STORE v4 GC: refcount hint disagrees with reachability — no \
                     committed manifest references this block but its refcount is {rc}. \
                     Proceeding on reachability; the hint is stale.",
                    rc = rc
                ),
                Ok(_) => {}
                Err(e) => tracing::warn!(
                    block = %name,
                    error = %e,
                    "COLD-STORE v4 GC: refcount hint unreadable; reachability already \
                     decided this block is unreferenced"
                ),
            }

            if self.prune_mode == PruneMode::Observe {
                tracing::info!(
                    would_gc = %name,
                    "COLD-STORE v4 observe: would GC unreferenced block"
                );
            } else {
                // AGE FLOOR — defence in depth, never authorization. A block
                // younger than the floor is skipped even though it is provably
                // unreferenced right now, because the most likely reason for a
                // brand-new unreferenced block is a publication in flight.
                //
                // Missing, unreadable, or FUTURE timestamps count as fresh and
                // are skipped: an unknown age is not an old age, and a clock
                // that jumped backwards must not be able to authorize a sweep.
                let age = entry
                    .metadata()
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| std::time::SystemTime::now().duration_since(t).ok());
                match age {
                    Some(a) if a >= self.min_gc_age => {}
                    _ => {
                        tracing::debug!(
                            block = %name,
                            age_secs = age.map(|a| a.as_secs_f64()),
                            floor_secs = self.min_gc_age.as_secs_f64(),
                            "COLD-STORE v4 GC: unreferenced but below the age floor (or \
                             its timestamp is unknown/future) — not nominated"
                        );
                        continue;
                    }
                }

                // NOMINATION ONLY. Nothing is deleted from an unlocked scan —
                // every decision so far came from a snapshot taken while
                // publishers were free to run.
                candidates.push((block_hash, name.clone()));
            }
        }

        if candidates.is_empty() {
            return Ok(());
        }

        // Candidates are chosen; the lock is not yet held. This is exactly the
        // window a concurrent publisher occupies, so it is where the test seam
        // fires.
        fire_gc_nomination_seam();

        // ---------------------------------------------------------------
        // Under exclusion (Alden finding 4, GC protocol steps 3-6).
        // ---------------------------------------------------------------
        let lock = self.acquire_store_lock()?; // fails closed

        // Step 4: cheap staleness hint first.
        let epoch_now = self.read_publication_epoch();
        if epoch_now != marked_epoch {
            tracing::info!(
                epoch_at_mark = marked_epoch,
                epoch_now = epoch_now,
                candidates = candidates.len(),
                "COLD-STORE v4 GC: published during nomination — collecting nothing \
                 this pass, rescanning next"
            );
            return Ok(());
        }

        // Step 5: AUTHORITATIVE re-verification while holding the lock, even
        // though the epoch was unchanged. Alden: "epoch and manifest are two
        // files, so a prior process can crash after one operation; unchanged
        // epoch is not proof by itself unless publication is transactionally
        // atomic, which it is not." The epoch is an optimization; this is the
        // proof.
        let live_final = self.mark_reachable_blocks()?;

        // Step 6: rename to a tombstone. THIS IS THE LINEARIZATION POINT. No
        // publisher can have observed the block and then published across it,
        // because a publisher's revalidation and its manifest rename happen
        // under this same lock.
        let mut tombstones: Vec<PathBuf> = Vec::new();
        for (block_hash, name) in candidates {
            if live_final.contains(&block_hash) {
                tracing::warn!(
                    block = %name,
                    "COLD-STORE v4 GC: candidate became REACHABLE between nomination \
                     and the locked re-verify — not collected. This is the publication \
                     race being caught rather than lost."
                );
                continue;
            }
            let from = self.blocks_dir().join(&name);
            let to = self.blocks_dir().join(format!(".tombstone.{name}"));
            match fs::rename(&from, &to) {
                Ok(()) => tombstones.push(to),
                Err(e) => tracing::warn!(
                    block = %name,
                    error = %e,
                    "COLD-STORE v4 GC: could not tombstone block; leaving it in place"
                ),
            }
        }

        // Step 7: the slow physical unlink happens OUTSIDE the critical
        // section. A writer that now looks for the final path finds it absent
        // and installs a fresh immutable copy; we only ever unlink our own
        // tombstone, never that replacement.
        drop(lock);

        for t in tombstones {
            match fs::remove_dir_all(&t) {
                Ok(()) => tracing::info!(gc = %t.display(), "COLD-STORE v4 GC'd block"),
                // Step 8: a crash here leaves a tombstone artifact, not a
                // missing live block. Aged tombstones are collectable later
                // under the same conservative rules.
                Err(e) => tracing::warn!(
                    tombstone = %t.display(),
                    error = %e,
                    "COLD-STORE v4 GC: tombstone left behind; it is inert and \
                     collectable later"
                ),
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Load
    // -----------------------------------------------------------------------

    /// Load the best matching manifest and assemble its blocks into a cache.
    ///
    /// `plan` MUST be the per-layer `(mode, v_bits)` vector the CALLING RUNTIME
    /// will actually produce — including any structural downgrade the model
    /// applies later (M3's D1 dense-prefix layers), because `persist` addresses
    /// with the vector it observes AFTER those downgrades have happened. Load
    /// runs before any layer is touched, so it cannot observe them and must be
    /// told. `LanguageModel::kv_cache_plan` is what tells it; `persist` checks
    /// the same plan against reality, which is what keeps the two sides honest.
    ///
    /// This parameter replaces a hardcoded `KVCacheMode::Fp16` at the hashing
    /// site. Under KVarN8 that hardcode made the store WRITE-ONLY: computed
    /// addresses could never equal the ones in its own manifest, `matched_blocks`
    /// was always 0, every candidate was dropped, and the caller saw `NoMatch` —
    /// indistinguishable from a legitimately cold cache, so it never surfaced
    /// as a failure. Handoff §7.5.
    ///
    /// Passing a mode that does not match the persisted one is SAFE by
    /// construction: the addresses simply will not match and the candidate is
    /// skipped, so a KVarN8 cache can never be adopted into an Fp16 runtime.
    pub fn load_prefix(
        &self,
        model_id: &str,
        template_sig: &str,
        tokens: &[i32],
        plan: &[(super::KVCacheMode, u8)],
    ) -> Result<(DetachedCacheSet, usize), ColdStoreError> {
        // READ side of the cache-computation identity — must mirror persist's
        // WRITE side exactly. Built once here so the symmetry is visible in
        // one place rather than inline at the hashing site.
        let load_cache_id = cache_computation_id(&self.runtime_fingerprint, plan);

        // ACTIVE-LOAD LEASE, held for the whole of this call. The hazard is the
        // gap between reading a manifest and opening the blocks it names: a
        // sweep can prove one unreferenced, tombstone it, and unlink it in
        // between, leaving this load to fail on a block its own manifest just
        // promised. Shared, so loads do not block each other; exclusive against
        // the sweep, which is what makes it a lease rather than a formality.
        let _lease = self.acquire_read_lease()?;

        let manifests_dir = self.manifests_dir();
        if !manifests_dir.exists() {
            return Err(ColdStoreError::NoMatch);
        }

        // Collect candidates
        let mut candidates = Vec::new();
        for entry in fs::read_dir(&manifests_dir)? {
            let entry = entry?;
            // ONE PREDICATE, shared with `mark_reachable_blocks`. See
            // `committed_manifest_hash` for what went wrong when this site had
            // its own inline rules.
            // FAILS SOFT, deliberately: a reader that skips a candidate loses a
            // cache hit. Contrast `mark_reachable_blocks`, which propagates,
            // because a sweeper that skips one deletes live data.
            let manifest_hash = match committed_manifest_hash(&entry) {
                Ok(Some(h)) => h,
                Ok(None) => continue,
                Err(_) => continue,
            };
            // CANONICAL READ, not a direct decode of `manifest.bin`.
            //
            // Alden, 2026-07-28: the `.tmp.` filter alone is necessary but NOT
            // sufficient. This site used to `fs::read` + `decode_manifest`
            // straight from the path, which also bypassed `read_manifest`'s
            // check that `manifest.hash()` equals the directory name. So a
            // manifest whose CONTENT does not match the address it is filed
            // under was adoptable here and nowhere else.
            //
            // Going through `read_manifest` makes the rename/name the commit
            // marker BY CONSTRUCTION: `.tmp.` prefixes, malformed names, and
            // content/name mismatches all fail soft as non-candidates through
            // one path rather than three hand-written ones.
            let manifest = match self.read_manifest(&manifest_hash) {
                Ok(m) => m,
                Err(_) => continue, // not a candidate; never fatal to the scan
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
                .zip(compute_block_hashes(
                    tokens,
                    self.block_size,
                    &load_cache_id,
                ))
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
            // `assemble_blocks` is fail-closed: what it returns is already
            // identity-verified. `load_prefix` keeps CANDIDATE POLICY — a
            // manifest whose tokens do not re-derive its addresses simply is not
            // a candidate, and falls through to the next-shorter one by the same
            // path a damaged payload takes.
            match self.assemble_blocks(&manifest, &load_cache_id) {
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
    /// load — any sequence longer than `block_size` (2048). It was invisible to
    /// single-block manifests.
    ///
    /// **Scope, corrected:** an earlier revision of this comment called that a
    /// *production* defect. It was not. `BlockColdStore` has no non-test callers
    /// — the scheduler holds `cold_store::ColdStore`, which delegates to the v3
    /// `ReferenceColdStore` — so v4 is complete but UNWIRED. A wrong assembly
    /// here would become a wrong adopted prefix the moment v4 is wired in; it
    /// never was one. The within-module fact (`load_prefix` calls this) does not
    /// establish reachability from the server, and the earlier wording conflated
    /// the two.
    /// Merge a manifest's blocks into one cache set, and return the token list
    /// the blocks themselves carry alongside it.
    ///
    /// **The tokens are returned, not discarded.** They are the input to
    /// [`Self::verify_manifest_addresses`], and discarding them is what left them
    /// unverified. This function stays a mechanical merge — it does not decide
    /// whether the manifest is the one the caller asked for. That is a candidate
    /// ACCEPTANCE question and belongs to `load_prefix`, which is also the only
    /// caller that knows the `kv_mode_config` the addresses were computed under.
    fn assemble_blocks(
        &self,
        manifest: &Manifest,
        kv_mode_config: &str,
    ) -> Result<DetachedCacheSet, ColdStoreError> {
        if manifest.block_hashes.is_empty() {
            return Err(invalid_data("manifest has no blocks".into()));
        }

        // Read every block up front: merging is per LAYER across ALL blocks,
        // so no layer can be finished until every block has been read.
        let mut blocks: Vec<DetachedCacheSet> = Vec::with_capacity(manifest.block_hashes.len());
        let mut stored_tokens: Vec<i32> = Vec::with_capacity(manifest.total_tokens);
        for block_hash in &manifest.block_hashes {
            let (tokens, cache_set) = self.read_block(block_hash)?;
            stored_tokens.extend_from_slice(&tokens);
            blocks.push(cache_set);
        }

        // FAIL-CLOSED, BEFORE THE MERGE (Alden, 2026-07-28).
        //
        // This check first lived in `load_prefix`, after the merge. I moved it
        // there because putting it here reddened two structural fixtures, and I
        // read that as "wrong layer". Alden read the same red correctly: those
        // fixtures use hand-written hashes and partitions that do not satisfy the
        // fixed-chunk address contract, so they were calling a production
        // ACCEPTANCE boundary with deliberately invalid input. The tests were
        // right to fail; my conclusion from their failing was wrong.
        //
        // The invariant, in his words: "no non-test function returns
        // DetachedCacheSet from disk bytes before token-chain verification."
        // Verifying in the caller left `assemble_blocks` returning directly
        // adoptable state on an unchecked identity claim — a fail-OPEN seam that
        // any second caller would inherit by omission. Here it is fail-closed by
        // construction.
        //
        // BEFORE the merge, not after: a candidate that fails cannot be adopted,
        // so merging its payload first is work spent to reach a refusal.
        self.verify_manifest_addresses(manifest, &stored_tokens, kv_mode_config)?;

        self.merge_read_blocks(&blocks, manifest)
    }

    /// Merge ALREADY-READ blocks into one cache set. Pure in-memory mechanics:
    /// per-layer concatenation and the structural checks around it.
    ///
    /// **Deliberately NOT an unverified `assemble_blocks`.** It does not touch
    /// disk and so cannot return state derived from unchecked disk bytes —
    /// Alden's invariant ("no non-test function returns `DetachedCacheSet` from
    /// disk bytes before token-chain verification") is satisfied by its shape
    /// rather than by its callers remembering.
    ///
    /// It exists because two structural fixtures need block partitions the
    /// address contract cannot express — sink and tail placed across tile
    /// boundaries — to exercise the merge itself. Those fixtures read their own
    /// blocks and call this. Exposing a disk-reading unverified variant for them
    /// was the alternative, and it is exactly the seam this refactor removes.
    fn merge_read_blocks(
        &self,
        blocks: &[DetachedCacheSet],
        manifest: &Manifest,
    ) -> Result<DetachedCacheSet, ColdStoreError> {
        if blocks.is_empty() {
            return Err(invalid_data("merge_read_blocks: no blocks".into()));
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

    /// Re-derive this manifest's block addresses from the tokens the blocks
    /// themselves carry, and require the chain to match.
    ///
    /// **This is the check the store's correctness argument assumed and did not
    /// perform.** `read_block` compares `header.block_hash` against the hash it
    /// was ASKED for — a stored field against an argument — and checksums each
    /// layer PAYLOAD. Nothing digested the header, and the block's token list
    /// lives in the header. Measured 2026-07-28 by flipping every byte in turn:
    /// **8200 of 8328 header bytes** could be corrupted with `read_block` still
    /// returning `Ok`, and 8192 of those are the 2048 stored tokens
    /// (`encode_block_header`: version 4 ‖ hash 32 ‖ count 8 ‖ tokens 4·N ‖ … ).
    ///
    /// Why it mattered: the whole argument for this store is that the address is
    /// a hash over the tokens, so a matching address means matching tokens.
    /// `block_hash_merkle` *does* commit to `own_tokens` — but nothing recomputed
    /// it on read, so corrupted tokens yielded KV state attributed to a prefix it
    /// did not come from. Intact payload, wrong interpretation: the `116924f`
    /// failure class, which did not crash — it computed the wrong thing. v3
    /// (`cold_store_reference.rs`) digests its header and cross-checks the
    /// identity on load, so v4 was **less** protected than the store it
    /// supersedes.
    ///
    /// The data to verify against was already on disk. This is a verification
    /// that was skipped, not new state.
    ///
    /// Chunking the concatenation reproduces the original partition because
    /// `persist` chunks with `tokens.chunks(block_size)`, so every block but the
    /// last is exactly `block_size` long.
    fn verify_manifest_addresses(
        &self,
        manifest: &Manifest,
        stored_tokens: &[i32],
        kv_mode_config: &str,
    ) -> Result<(), ColdStoreError> {
        let rederived = compute_block_hashes(stored_tokens, manifest.block_size, kv_mode_config);
        if rederived == manifest.block_hashes {
            return Ok(());
        }
        let first_bad = rederived
            .iter()
            .zip(&manifest.block_hashes)
            .position(|(a, b)| a != b);
        Err(invalid_data(format!(
            "the stored token list does not re-derive this manifest's block addresses \
             (manifest {} blocks / {} tokens, re-derived {} blocks from {} stored tokens; \
             first divergence at block {}). The payload bytes may be perfectly intact — what \
             failed is the claim that they belong to THESE tokens. Adopting this state would \
             attach a KV prefix to a conversation it did not come from.",
            manifest.block_hashes.len(),
            manifest.total_tokens,
            rederived.len(),
            stored_tokens.len(),
            first_bad
                .map(|i| i.to_string())
                .unwrap_or_else(|| "length".into()),
        )))
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

    // `m3_idx_offset` is the LOGICAL LENGTH of `m3_idx_k` (detach.rs:142-145),
    // not a copy of the token count. Those coincide on MSA layers and DO NOT on
    // M3's dense layers 0-2, which have no indexer at all (detach.rs:136) — the
    // live cache only advances the counter inside `m3_idx_k_update_and_fetch`
    // (cache.rs:5333), which those layers never call, so they sit at 0 while
    // `offset` grows.
    //
    // Setting it to `total_tokens` unconditionally handed every dense layer back
    // declaring a multi-thousand-token indexer behind a `None` tensor. Bytes
    // perfect, interpretation wrong — handoff §7.3's exact failure class, and the
    // mirror of the cycle-79 desync that detach.rs:139 documents as crashing the
    // asymmetric reshape. Derive it from the tensor that actually arrived: right
    // when the two agree, honest when they do not.
    let m3_idx_k = concat_across(layers, |c| &c.m3_idx_k);
    let m3_idx_offset = m3_idx_k
        .as_ref()
        .map_or(0, |a| crate::ffi::array_shape(a.as_ref().unwrap())[2]);

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
        m3_idx_k,
        m3_idx_offset,
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
        axis2_of(&merged.kvarn_sink_k)
            + axis2_of(&merged.kvarn_hist_k)
            + axis2_of(&merged.kvarn_tail_k)
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
        let keys = layer
            .keys
            .as_ref()
            .map(|k| slice_axis(k, 2, start as i32, end as i32));
        let values = layer
            .values
            .as_ref()
            .map(|v| slice_axis(v, 2, start as i32, end as i32));
        let key_scales = layer
            .key_scales
            .as_ref()
            .map(|s| slice_axis(s, 2, start as i32, end as i32));
        let val_scales = layer
            .val_scales
            .as_ref()
            .map(|s| slice_axis(s, 2, start as i32, end as i32));
        let m3_idx_k = layer
            .m3_idx_k
            .as_ref()
            .map(|m| slice_axis(m, 2, start as i32, end as i32));

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

/// THE predicate for "this directory entry is a COMMITTED manifest", and the
/// only one. Returns the address it is filed under, or `None` if it is not a
/// committed manifest directory at all.
///
/// WHY THIS IS ONE FUNCTION AND NOT AN INLINE RULE PER CALLER. Five places
/// enumerate store directories and each had its own hand-written idea of what
/// to skip. On 2026-07-28 `load_prefix` turned out to be missing the `.tmp.`
/// rule the other four had, so it would ADOPT AS A PREFIX a manifest staged by
/// a publication that crashed before its rename — while `mark_reachable_blocks`
/// correctly refused to root that same manifest, leaving GC free to collect the
/// very blocks `load_prefix` was offering. Two functions, one on-disk state,
/// opposite answers about what is live.
///
/// Convergence is a property, not a step: when two places assert the same thing
/// they drift, and the drift is silent because each site reads correctly on its
/// own. One surface, so there is nothing left to disagree.
///
/// Callers must still reach content through [`BlockColdStore::read_manifest`],
/// which enforces content-hash == directory-name. This function decides
/// CANDIDACY; that one decides INTEGRITY.
///
/// Deliberately scoped to MANIFEST directories. Block and tombstone enumeration
/// keep their own rules — their artifact policy differs (`.tombstone.` is a live
/// stage of a delete, not a discard), and collapsing them would be the opposite
/// error to the one this fixes.
/// RETURNS `Result`, NOT BARE `Option` — Alden, 2026-07-28, and the distinction
/// has a deletion consequence.
///
/// The first draft mapped a `file_type()` I/O error to `None`. That is fine for
/// a reader: skip the candidate, lose a cache hit. It is **unsafe for
/// `mark_reachable_blocks`**, where a transient I/O error would make a COMMITTED
/// manifest silently vanish from the root set — and a manifest missing from the
/// root set is a manifest whose blocks the sweep is authorized to delete.
///
/// So the error channel is separated from the verdict, and each caller states
/// its own policy: readers and pruners may `continue` on `Err`, the sweeper must
/// propagate. Same asymmetry as `read_manifest`, one layer earlier.
fn committed_manifest_hash(entry: &fs::DirEntry) -> Result<Option<[u8; 32]>, ColdStoreError> {
    if !entry.file_type()?.is_dir() {
        return Ok(None);
    }
    let name = entry.file_name();
    let name = name.to_string_lossy();
    // Staged, not published: `write_manifest` fills `.tmp.<hash>` BEFORE taking
    // the lock, and only the final rename commits. The name IS the commit marker.
    //
    // BELT-AND-BRACES, NOT INDEPENDENTLY NECESSARY — measured, not assumed.
    // Disabling this clause alone leaves all three tests GREEN, because
    // `parse_hex_digest` rejects `.tmp.<64 hex>` on length (69 != 64) anyway.
    // Retained to state intent where a reader looks, and because the redundancy
    // is a coincidence of the current naming scheme rather than a guarantee.
    // Do NOT cite it as a proven guard; see the handoff for what actually holds.
    if name.starts_with(".tmp.") {
        return Ok(None);
    }
    Ok(parse_hex_digest(&name))
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
    let len = u32::try_from(value.len()).map_err(|_| invalid_data("string exceeds u32".into()))?;
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
    let value =
        usize::try_from(value).map_err(|_| invalid_data(format!("{field} does not fit usize")))?;
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
    use crate::cache::cold_store::runtime_fingerprint_from_manifest;
    use crate::cache::cold_store::tests::make_test_cache_set;

    #[test]
    fn block_hash_merkle_chain_prefix_dependent() {
        // Unit test: pure function incorporates prev_hash.
        // NOT a discrimination control — the caller's chain logic is tested below.
        let tokens_a = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let tokens_b = vec![9, 10, 11, 12, 5, 6, 7, 8];
        let kv_mode = cache_computation_id(&[0u8; 32], &[(KVCacheMode::Fp16, 0)]);
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
        let kv_mode = cache_computation_id(&[0u8; 32], &[(KVCacheMode::Fp16, 0)]);
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
        let kv_mode = cache_computation_id(&[0u8; 32], &[(KVCacheMode::Fp16, 0)]);
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

        let kv_fp16 = cache_computation_id(&[0u8; 32], &[(super::super::KVCacheMode::Fp16, 0)]);
        let kv_kvarn8 = cache_computation_id(&[0u8; 32], &[(KVCacheMode::KVarN8, 4)]);

        let hash_fp16 = block_hash_merkle(&[0u8; 32], block_size, &kv_fp16, &tokens);
        let hash_kvarn8 = block_hash_merkle(&[0u8; 32], block_size, &kv_kvarn8, &tokens);
        assert_ne!(hash_fp16, hash_kvarn8);
    }

    #[test]
    fn block_hash_includes_block_size() {
        let tokens = vec![1, 2, 3, 4];
        let kv_mode = cache_computation_id(&[0u8; 32], &[(KVCacheMode::Fp16, 0)]);

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
        let kv_mode = cache_computation_id(&[0u8; 32], &[(KVCacheMode::Fp16, 0)]);
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
