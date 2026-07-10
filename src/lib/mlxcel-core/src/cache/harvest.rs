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

//! Env-gated KVarN real-tile harvest instrumentation
//! (SPEC_kvarn4_realtile_harvest_2026-07-10).
//!
//! `MLXCEL_KVARN_HARVEST=<dir>` dumps SAMPLED quantization-pipeline inputs
//! during a live session so the KVarN4 re-screens can run on real
//! activations instead of synthetic proxies:
//!   * `k_rot_f32` / `v_rot_f32` — pre-quantization ROTATED tile batches,
//!     exactly the tensors `kvarn_quantize` receives, ≤ [`TILES_PER_EVENT`]
//!     evenly-spread tiles per finalization event (a 300K prefill finalizes
//!     thousands of tiles in ONE event — spreading samples inside the event
//!     gives depth stratification and stops the firehose);
//!   * `idx_k` — m3_idx block samples at depth-threshold crossings;
//!   * `idx_q` / `sel` — model-side real index queries + their selected
//!     sets (the near-tie statistic needs real queries). Dtypes live in
//!     the sidecar, not the role name; only `*_rot_f32` roles carry a
//!     suffix because that dtype is structurally guaranteed (astype).
//!
//! Sidecar `cache` field is OVERLOADED by role: an address key for
//! cache-side dumps (`k_rot_f32`/`v_rot_f32`/`idx_k` — stable within a
//! process, groups dumps per layer-cache) and `layer_idx` for the
//! model-side pairs (`idx_q`/`sel`). Analysis scripts read it per role.
//!
//! Format: raw little-endian device bytes + a JSON sidecar per dump
//! (`harvest_NNNNNN_<role>.bin` / `.json` — shape, dtype code, offset,
//! cache key, n_full where applicable). numpy reads it in three lines.
//!
//! SAFETY PROPERTY (the one that matters): with the env unset this module
//! is one relaxed `OnceLock` read returning `None` — zero behavior change,
//! pinned by the entire existing suite running env-unset. With it set,
//! filesystem operations are best-effort (WARN and return). Precision on
//! the panic claim (Violet's review): the FFI evals here
//! (`array_to_raw_bytes`, the sampling gather) share the HOST path's risk
//! class — same graph, same tensors, no NEW panic class — while all
//! harvest-specific failure modes (fs) are non-panicking.
//! A failed harvest is an empty directory (fail-loud at the operator
//! console), never a corrupted cache.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use cxx::UniquePtr;

use crate::dtype;
use crate::ffi::{self, MlxArray};

/// Max tiles dumped per finalization event, evenly spread across the
/// batch. 8 keeps a 300K-prefill event at ~2 MB instead of ~1.2 GB.
pub const TILES_PER_EVENT: i32 = 8;
/// Runaway stop across the whole process (not a target; a full harvest
/// session lands well under it).
const MAX_DUMPS: u64 = 32_768;
/// m3_idx snapshot cadence in absolute positions.
pub const IDX_STRIDE: i32 = 32_768;
/// K/V tile-dump cadence in absolute positions. Chunked prefill finalizes
/// tiles PER CHUNK (a 300K prefill at 2048 = ~146 events × 57 layers × 2
/// roles) — dumping every event floods the budget by mid-depth and starves
/// the DEEP strata, exactly the depth-drift coverage the spec's clause (e)
/// exists for. Stride-gating keeps coverage EVEN across depth.
pub const KV_STRIDE: i32 = 8_192;

/// True when `old..new` crosses a multiple of `stride` — the shared
/// depth-cadence test for K/V and idx dumps.
pub fn stride_crossed(old: i32, new: i32, stride: i32) -> bool {
    old / stride != new / stride
}

static SEQ: AtomicU64 = AtomicU64::new(0);

/// The harvest directory, read once per process. `None` (unset) is the
/// production state: every call site early-returns on it.
pub fn harvest_dir() -> Option<&'static Path> {
    static D: OnceLock<Option<PathBuf>> = OnceLock::new();
    D.get_or_init(|| {
        let dir = std::env::var("MLXCEL_KVARN_HARVEST").ok().map(PathBuf::from);
        if let Some(d) = &dir {
            tracing::warn!(
                dir = %d.display(),
                "MLXCEL_KVARN_HARVEST active: sampled KVarN tile/idx dumps enabled \
                 (harvest instrumentation, not a production mode)"
            );
        }
        dir
    })
    .as_deref()
}

/// `k` evenly-spread indices over `0..n` (all of them when `n <= k`).
/// Ascending, unique, always in-range — the sampling that keeps one giant
/// prefill finalization from flooding the disk while still covering the
/// batch's depth span.
pub fn spread_indices(n: i32, k: i32) -> Vec<i32> {
    if n <= 0 {
        return Vec::new();
    }
    if n <= k {
        return (0..n).collect();
    }
    (0..k).map(|i| (i as i64 * n as i64 / k as i64) as i32).collect()
}

/// Gather `indices` rows of a `[N, R, C]` batch → `[len, R, C]`.
fn take_rows(batch: &MlxArray, indices: &[i32]) -> UniquePtr<MlxArray> {
    let shape = ffi::array_shape(batch);
    let (r, c) = (shape[1], shape[2]);
    let idx_f: Vec<f32> = indices.iter().map(|&i| i as f32).collect();
    let idx = ffi::astype(
        &ffi::from_slice_f32(&idx_f, &[indices.len() as i32, 1, 1]),
        dtype::INT32,
    );
    let idx_b = ffi::broadcast_to(&idx, &[indices.len() as i32, r, c]);
    ffi::take_along_axis(batch, &idx_b, 0)
}

/// Best-effort dump of `a` as raw bytes + JSON sidecar. Fallible steps
/// warn-and-return; never panics into the caller.
pub fn dump(role: &str, cache_key: usize, offset: i32, n_full: i32, a: &MlxArray) {
    let Some(dir) = harvest_dir() else { return };
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    if seq >= MAX_DUMPS {
        if seq == MAX_DUMPS {
            tracing::warn!("harvest: MAX_DUMPS ({MAX_DUMPS}) reached — further dumps dropped");
        }
        return;
    }
    let shape = ffi::array_shape(a);
    let dt = ffi::array_dtype(a);
    let bytes = ffi::array_to_raw_bytes(a);
    let base = dir.join(format!("harvest_{seq:06}_{role}"));
    let sidecar = format!(
        "{{\"role\":\"{role}\",\"shape\":{shape:?},\"dtype\":{dt},\"offset\":{offset},\
         \"n_full\":{n_full},\"cache\":\"{cache_key:x}\",\"seq\":{seq}}}"
    );
    if let Err(e) = std::fs::write(base.with_extension("bin"), &bytes)
        .and_then(|()| std::fs::write(base.with_extension("json"), &sidecar))
    {
        tracing::warn!(role, seq, error = %e, "harvest dump failed (best-effort, continuing)");
    }
}

/// Sampled dump of a `[N, R, C]` tile batch: ≤ [`TILES_PER_EVENT`] spread
/// rows. The gather runs on the same lazy graph the caller is about to
/// evaluate anyway; `dump` forces only the sampled slice.
pub fn dump_tiles(role: &str, cache_key: usize, offset: i32, batch: &MlxArray) {
    if harvest_dir().is_none() {
        return;
    }
    let n = ffi::array_shape(batch)[0];
    let idxs = spread_indices(n, TILES_PER_EVENT);
    if idxs.is_empty() {
        return;
    }
    let sampled = take_rows(batch, &idxs);
    dump(role, cache_key, offset, n, &sampled);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride_crossed_contract() {
        assert!(stride_crossed(8191, 8192, 8192), "exact boundary fires");
        assert!(stride_crossed(7000, 9000, 8192), "spanning crossing fires");
        assert!(!stride_crossed(8192, 9000, 8192), "within one stripe is quiet");
        assert!(stride_crossed(0, 40_000, 8192), "large first event fires");
    }

    #[test]
    fn spread_indices_contract() {
        assert_eq!(spread_indices(3, 8), vec![0, 1, 2]); // n <= k: all
        assert_eq!(spread_indices(0, 8), Vec::<i32>::new());
        let s = spread_indices(2340, 8); // one 300K-prefill event
        assert_eq!(s.len(), 8);
        assert!(s.windows(2).all(|w| w[0] < w[1]), "ascending unique");
        assert_eq!(s[0], 0);
        assert!(*s.last().unwrap() < 2340, "in range");
        assert!(*s.last().unwrap() > 2000, "spread reaches the deep end");
    }

    #[test]
    fn take_rows_gathers_the_requested_tiles() {
        // [4, 2, 3] batch with row-identifying values; pick rows 1 and 3.
        let vals: Vec<f32> = (0..24).map(|i| (i / 6) as f32).collect();
        let batch = ffi::from_slice_f32(&vals, &[4, 2, 3]);
        let picked = take_rows(&batch, &[1, 3]);
        assert_eq!(ffi::array_shape(&picked), vec![2, 2, 3]);
        let bytes = ffi::array_to_raw_bytes(&picked);
        let f: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert!(f[..6].iter().all(|&x| x == 1.0), "first picked row is batch row 1");
        assert!(f[6..].iter().all(|&x| x == 3.0), "second picked row is batch row 3");
    }
}
