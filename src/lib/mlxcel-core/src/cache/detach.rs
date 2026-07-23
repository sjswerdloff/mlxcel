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

//! Cross-sequence KV cache reuse: trim / detach / adopt primitives.
//!
//! This module extends [`super::KVCache`] and [`super::CachePool`] with the
//! primitives required by the cross-request prompt prefix cache:
//!
//! * [`KVCache::trim_to`] — shrink the logical cache length to an exact value
//!   while keeping the pre-allocated backing buffer around.
//! * [`KVCache::clone_handle`] — move the underlying `MlxArray` ownership out
//!   into an inert [`DetachedKVCache`] that can outlive the original sequence.
//! * [`CachePool::detach`] — lift a whole [`super::SequenceCacheSet`] off the
//!   active HashMap and return it as an owned [`DetachedCacheSet`] without
//!   freeing the MLX buffers.
//! * [`CachePool::adopt`] — install a previously-detached cache set under a
//!   fresh [`super::SequenceId`] and prime the model's sidecar state for it.
//!
//! Only the **dense** KV cache backend (`SequenceStateBackend::DenseKvCache`)
//! is handled directly by this file. Paged sequences are handled by the
//! parallel API surface in [`super::paged_detach`]; this
//! module delegates through the shared `DetachedHandle` namespace so parking
//! remains a single pool-level abstraction.
//!
//! ## Memory accounting
//!
//! While a detached cache set is in-flight — e.g. inside a scheduler that has
//! taken it out of `CachePool` but is about to re-adopt it — callers can
//! [`park`] the set so that the pool's
//! [`CachePool::memory_usage_bytes`] keeps including the bytes. Parking is
//! optional; `detach` + `adopt` work end-to-end without it.
//!
//! ## INT8 preservation
//!
//! Both the INT8 key/value tensors and the per-token FP16 scale tensors are
//! moved through detach/adopt unchanged, so `KVCache::mode == Int8` sequences
//! round-trip losslessly.
//!
//! ## Aliasing with `MLXCEL_ENABLE_DIRECT_PREFILL_CACHE_STORE`
//!
//! The direct-prefill-store fast path in [`super::KVCache::update`] installs
//! the incoming FP16 tensor directly as the cache buffer (with a
//! `contiguous` call) when the cache is empty and the env var is set. Detach
//! simply moves that buffer out via [`UniquePtr::take`]; no aliasing survives
//! because `MlxArray` buffers are functional — every operation produces a
//! fresh array, and the move semantics of `UniquePtr` prevent concurrent
//! access. Adopting that same buffer into a new sequence is therefore safe.
//!
//! [`park`]: CachePool::park_detached

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Instant;

use cxx::UniquePtr;

use crate::ffi;
use crate::ffi::MlxArray;

use super::{
    CachePool, KVCache, KVCacheMode, RotatingKVCache, SequenceCacheSet, SequenceId,
    SequenceStateBackend,
};

// ---------------------------------------------------------------------------
// DetachedKVCache
// ---------------------------------------------------------------------------

/// Inert, model-agnostic snapshot of a single [`KVCache`] that can outlive the
/// sequence which produced it.
///
/// `DetachedKVCache` owns the underlying MLX `UniquePtr<MlxArray>` buffers
/// directly, so detach/adopt never allocates a new tensor or copies data.
/// INT8-mode caches carry their per-token scale tensors alongside the INT8
/// key/value buffers so dequantization behavior is bit-identical after adopt.
/// Turbo4Asym-mode caches additionally carry the packed-V buffer, the
/// per-token V norms, and the deterministic Turbo4 seed so the adopted cache
/// can rebuild [`crate::cache::turbo::TurboQuantParams`] without referring
/// back to the originating `KVCache`.
/// Turbo4 (symmetric) caches also carry packed-K + per-token K norms.
/// Turbo4Delegated-mode caches (K unified) carry
/// the unified FP16 K buffer in `keys` (same shape contract as `Fp16` mode),
/// the V-side cold/hot split state (`cold_offset`), and the configured
/// hot-V fold threshold so the adopted cache resumes decoding from the same
/// V cold/hot boundary it left at detach time. There is no separate
/// `cold_keys` field — K is unified. When the opt-in FP16 fast path is active,
/// the detached `values` tensor is the unified FP16 V buffer rather than a hot
/// ring; `delegated_fp16_fast_path` preserves that interpretation on adopt and
/// `delegated_fp16_sidecar_policy` preserves the foreground compaction policy.
/// Turbo3Asym-mode caches use the same `(keys, v_packed, v_norms)` triple
/// as `Turbo4Asym` but the V buffer carries the 24-bit-grouped 3-bit indices
/// The `mode` field on the handle preserves the bit-width
/// distinction so adopt rebuilds the right `TurboQuantParams3` instance.
impl DetachedKVCache {
    /// Test-only accessor for the detached indexer cache (contract tests
    /// pin the exact-length slice-to-fill behavior of clone_handle).
    #[cfg(test)]
    pub(crate) fn m3_idx_k_for_tests(&self) -> Option<&MlxArray> {
        self.m3_idx_k.as_deref()
    }
}

pub struct DetachedKVCache {
    pub(super) keys: Option<UniquePtr<MlxArray>>,
    pub(super) values: Option<UniquePtr<MlxArray>>,
    pub(super) offset: i32,
    pub(super) step: i32,
    pub(super) mode: KVCacheMode,
    pub(super) key_scales: Option<UniquePtr<MlxArray>>,
    pub(super) val_scales: Option<UniquePtr<MlxArray>>,
    pub(super) v_packed: Option<UniquePtr<MlxArray>>,
    pub(super) v_norms: Option<UniquePtr<MlxArray>>,
    /// Turbo4-V Sparse-V kernel rescale sidecar. Lockstep with
    /// `v_norms`; round-trips through detach/adopt so paged + prefix-cache
    /// reuse paths preserve the precomputed rescale.
    pub(super) v_rescale: Option<UniquePtr<MlxArray>>,
    pub(super) k_packed: Option<UniquePtr<MlxArray>>,
    pub(super) k_norms: Option<UniquePtr<MlxArray>>,
    pub(super) turbo_seed: u32,
    pub(super) cold_offset: i32,
    pub(super) hot_threshold: i32,
    pub(super) delegated_fp16_fast_path: bool,
    pub(super) delegated_fp16_sidecar_policy: super::turbo::DelegatedFp16SidecarPolicy,
    /// MiniMax-M3 indexer K cache. `None` for non-M3 models (and for M3's
    /// dense layers 0-2). Shape when `Some`: `[b, 1, m3_idx_offset, index_dim]`.
    /// Round-trips through detach/adopt so prompt-cache reuse preserves the
    /// indexer state alongside main K/V; otherwise an adopted MSA path would
    /// see `m3_idx_offset == 0` while `offset == matched_len > 0` and crash on
    /// the asymmetric reshape (the cycle-79 desync).
    pub(super) m3_idx_k: Option<UniquePtr<MlxArray>>,
    /// Logical length of `m3_idx_k`. Lockstep with `offset` under normal
    /// operation. Preserved through detach/adopt and trimmed by `trim_to`
    /// in parallel with the main K seq axis.
    pub(super) m3_idx_offset: i32,

    // KVarN8 tile state (all None for other modes). Same rotated-frame
    // layout as the live cache fields — see the `kvarn_*` docs on KVCache.
    // Round-trips through detach/adopt; trimmed tile-aligned by `trim_to`.
    pub(super) kvarn_sink_k: Option<UniquePtr<MlxArray>>,
    pub(super) kvarn_sink_v: Option<UniquePtr<MlxArray>>,
    pub(super) kvarn_tail_k: Option<UniquePtr<MlxArray>>,
    pub(super) kvarn_tail_v: Option<UniquePtr<MlxArray>>,
    pub(super) kvarn_hist_k: Option<UniquePtr<MlxArray>>,
    pub(super) kvarn_hist_v: Option<UniquePtr<MlxArray>>,
    pub(super) kvarn_k_scale: Option<UniquePtr<MlxArray>>,
    pub(super) kvarn_k_zp: Option<UniquePtr<MlxArray>>,
    pub(super) kvarn_k_s_row: Option<UniquePtr<MlxArray>>,
    pub(super) kvarn_k_s_col: Option<UniquePtr<MlxArray>>,
    pub(super) kvarn_v_scale: Option<UniquePtr<MlxArray>>,
    pub(super) kvarn_v_zp: Option<UniquePtr<MlxArray>>,
    pub(super) kvarn_v_s_row: Option<UniquePtr<MlxArray>>,
    pub(super) kvarn_v_s_col: Option<UniquePtr<MlxArray>>,
    /// V width of the kvarn tile state (see `KVCache::kvarn_v_bits`). MUST
    /// round-trip: `mode` alone cannot distinguish k8v8 from k8v4 (both are
    /// `KVarN8`), and a v4 donation re-installed into a fresh slot without
    /// this field would resurrect as v_bits=8 — packed-u32 V codes silently
    /// mislabeled as u8, passing the reader guards they exist to trip.
    pub(super) kvarn_v_bits: u8,
}

impl DetachedKVCache {
    /// Logical length of the stored cache (matches the live
    /// [`KVCache::seq_len`] at detach time).
    pub fn seq_len(&self) -> i32 {
        self.offset
    }

    /// Metadata-only copy of this handle for a non-consuming paged prefix
    /// clone (#227), re-anchored at `offset_override`.
    ///
    /// Only valid for a handle that carries no tensors (the pool-backed Fp16
    /// shape `clone_handle` produces: the live K/V lives in the shared block
    /// pool, the handle only carries offset bookkeeping). Returns `None` when
    /// any tensor field is populated, so a dense-compat set cannot be
    /// shallow-cloned into aliased buffers.
    pub(super) fn pool_backed_handle_clone(&self, offset_override: i32) -> Option<DetachedKVCache> {
        if self.keys.is_some()
            || self.values.is_some()
            || self.key_scales.is_some()
            || self.val_scales.is_some()
            || self.v_packed.is_some()
            || self.v_norms.is_some()
            || self.v_rescale.is_some()
            || self.k_packed.is_some()
            || self.k_norms.is_some()
            || self.m3_idx_k.is_some()
            || self.kvarn_sink_k.is_some()
            || self.kvarn_tail_k.is_some()
            || self.kvarn_hist_k.is_some()
        {
            return None;
        }
        Some(DetachedKVCache {
            keys: None,
            values: None,
            offset: offset_override,
            step: self.step,
            mode: self.mode,
            key_scales: None,
            val_scales: None,
            v_packed: None,
            v_norms: None,
            v_rescale: None,
            k_packed: None,
            k_norms: None,
            turbo_seed: self.turbo_seed,
            cold_offset: 0,
            hot_threshold: self.hot_threshold,
            delegated_fp16_fast_path: self.delegated_fp16_fast_path,
            delegated_fp16_sidecar_policy: self.delegated_fp16_sidecar_policy,
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
            kvarn_v_bits: self.kvarn_v_bits,
        })
    }

    /// Quantization mode of the detached cache.
    pub fn mode(&self) -> KVCacheMode {
        self.mode
    }

    /// Total byte footprint of the detached tensors (keys + values + INT8
    /// scales + Turbo4Asym v_packed/v_norms + Turbo4 symmetric
    /// k_packed/k_norms). Turbo4Delegated caches no longer carry
    /// a separate `cold_keys` tensor — the unified K buffer is already
    /// counted under `keys`.
    pub fn nbytes(&self) -> usize {
        let k = self.keys.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let v = self.values.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let ks = self.key_scales.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let vs = self.val_scales.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let vp = self.v_packed.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let vn = self.v_norms.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let vr = self.v_rescale.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let kp = self.k_packed.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let kn = self.k_norms.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let im3 = self.m3_idx_k.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let kvarn: usize = [
            &self.kvarn_sink_k,
            &self.kvarn_sink_v,
            &self.kvarn_tail_k,
            &self.kvarn_tail_v,
            &self.kvarn_hist_k,
            &self.kvarn_hist_v,
            &self.kvarn_k_scale,
            &self.kvarn_k_zp,
            &self.kvarn_k_s_row,
            &self.kvarn_k_s_col,
            &self.kvarn_v_scale,
            &self.kvarn_v_zp,
            &self.kvarn_v_s_row,
            &self.kvarn_v_s_col,
        ]
        .iter()
        .map(|t| t.as_ref().map_or(0, |a| ffi::array_nbytes(a)))
        .sum();
        k + v + ks + vs + vp + vn + vr + kp + kn + im3 + kvarn
    }

    /// Whether the detached handle carries no data (all tensors were `None`).
    /// KVarN8 caches store nothing in `keys`/`k_packed` — their earliest
    /// state lives in the sink pool, so it must be checked too.
    pub fn is_empty(&self) -> bool {
        self.keys.is_none() && self.k_packed.is_none() && self.kvarn_sink_k.is_none()
    }

    /// Read-only access to the detached keys tensor.
    pub fn keys(&self) -> Option<&MlxArray> {
        self.keys.as_deref()
    }

    /// Read-only access to the detached values tensor.
    pub fn values(&self) -> Option<&MlxArray> {
        self.values.as_deref()
    }

    /// Shrink the inert detached cache to exactly `new_len` tokens along the
    /// sequence-length axis (axis 2 of the per-layer 4-D tensors).
    ///
    /// Mirrors [`KVCache::trim`] semantics on the post-detach handle:
    ///
    /// * `new_len == 0` drops every backing buffer (keys, values, INT8 scale
    ///   sidecars, Turbo4 packed sidecars / norms / V-rescale, K-side packed
    ///   sidecars and norms when in symmetric Turbo4) and resets `offset`.
    ///   The mode tag and `step` are preserved so a subsequent
    ///   [`KVCache::install_detached`] round-trip stays consistent.
    /// * `0 < new_len < self.offset` re-slices each tensor's seq-len axis to
    ///   `new_len` via the same `ffi::slice` primitive `KVCache::trim` uses.
    ///   For `Turbo4Delegated` the K side is unified so the shape contract is
    ///   identical to `Fp16`; the V-side cold/hot split is rebuilt from
    ///   `cold_offset.min(new_len)` and `delegated_fp16_fast_path` decides
    ///   whether `values` is a unified FP16 buffer or a hot ring.
    /// * `new_len == self.offset` is a no-op (zero allocations).
    /// * `new_len < 0` or `new_len > self.offset` returns `Err` and leaves
    ///   the handle untouched.
    ///
    /// This is the inert-side counterpart of [`KVCache::trim_to`] and is the
    /// load-bearing primitive that lets the scheduler adopt only the first
    /// `matched_len` tokens' worth of KV state when an Automatic Prefix
    /// Caching (APC) lookup returns a block-aligned matched
    /// length shorter than the full cached entry. See server-side
    /// `try_adopt_cached_prefix` in `src/server/batch/scheduler.rs`.
    ///
    /// INT8 scale tensors and the Turbo4* per-token sidecars (`v_packed`,
    /// `v_norms`, `v_rescale`, `k_packed`, `k_norms`) are sliced in lockstep
    /// so a subsequent install + dequantize stays bit-identical to the
    /// already-trimmed live cache.
    ///
    /// Used by: [`DetachedCacheSet::truncate_to`] (— APC block-level partial cache adoption in the scheduler).
    pub fn trim_to(&mut self, new_len: i32) -> Result<(), String> {
        if new_len < 0 {
            return Err(format!(
                "DetachedKVCache::trim_to: new_len must be non-negative, got {new_len}"
            ));
        }
        if new_len > self.offset {
            return Err(format!(
                "DetachedKVCache::trim_to: new_len ({new_len}) exceeds current offset ({})",
                self.offset
            ));
        }
        if new_len == self.offset {
            return Ok(());
        }

        let prev_offset = self.offset;

        if new_len == 0 {
            // Drop every backing buffer and reset offsets / cold-V state.
            self.keys = None;
            self.values = None;
            self.key_scales = None;
            self.val_scales = None;
            self.v_packed = None;
            self.v_norms = None;
            self.v_rescale = None;
            self.k_packed = None;
            self.kvarn_sink_k = None;
            self.kvarn_sink_v = None;
            self.kvarn_tail_k = None;
            self.kvarn_tail_v = None;
            self.kvarn_hist_k = None;
            self.kvarn_hist_v = None;
            self.kvarn_k_scale = None;
            self.kvarn_k_zp = None;
            self.kvarn_k_s_row = None;
            self.kvarn_k_s_col = None;
            self.kvarn_v_scale = None;
            self.kvarn_v_zp = None;
            self.kvarn_v_s_row = None;
            self.kvarn_v_s_col = None;
            self.k_norms = None;
            self.cold_offset = 0;
            self.offset = 0;
            // MiniMax-M3 indexer K cache trims in lockstep with main K.
            self.m3_idx_k = None;
            self.m3_idx_offset = 0;
            return Ok(());
        }

        // Slice axis 2 (seq-len) of each tensor to the new length. Width axes
        // (B, H, head_dim / packed_dim) come from the existing shape.
        let trim_axis_seq =
            |a: &Option<UniquePtr<MlxArray>>, tail_axis: i32| -> Option<UniquePtr<MlxArray>> {
                a.as_ref().map(|arr| {
                    let shape = ffi::array_shape(arr);
                    let last = if tail_axis == 0 { shape[3] } else { tail_axis };
                    ffi::slice(arr, &[0, 0, 0, 0], &[shape[0], shape[1], new_len, last])
                })
            };

        if self.mode == KVCacheMode::Turbo4Delegated {
            // K is unified — same shape contract as Fp16. Slice to `new_len`.
            self.keys = trim_axis_seq(&self.keys, 0);

            let new_cold = self.cold_offset.min(new_len);
            let new_hot_len = new_len - new_cold;

            // V buffer interpretation depends on the FP16 fast path. When the
            // fast path is on, V is a unified FP16 working set sliced to the
            // total length; when off, V is the hot ring sliced to the new hot
            // tail length.
            if let Some(ref v) = self.values {
                let v_shape = ffi::array_shape(v);
                let visible_v_len = if self.delegated_fp16_fast_path {
                    new_len
                } else {
                    new_hot_len
                };
                // NOTE: when `visible_v_len == 0` we explicitly drop the V
                // buffer (set to `None`). This diverges intentionally from
                // `KVCache::trim`, which leaves the backing buffer in place
                // in the same scenario (its `if visible_v_len > 0` guard
                // simply skips the slice call). In the live cache the
                // retained stale buffer is harmless because `update_and_fetch`
                // checks `offset == 0` before reading V; in the detached
                // handle the stale buffer would be re-installed via
                // `install_detached` and could be decoded against, so we zero
                // it out defensively here.
                self.values = if visible_v_len > 0 {
                    Some(ffi::slice(
                        v,
                        &[0, 0, 0, 0],
                        &[v_shape[0], v_shape[1], visible_v_len, v_shape[3]],
                    ))
                } else {
                    None
                };
            }

            // Cold-V sidecars only need re-slicing when cold actually shrank.
            // Re-slice unconditionally to `new_cold` so we never carry cold
            // state past the new logical boundary.
            if new_cold > 0 {
                if let Some(ref vp) = self.v_packed {
                    let vp_shape = ffi::array_shape(vp);
                    self.v_packed = Some(ffi::slice(
                        vp,
                        &[0, 0, 0, 0],
                        &[vp_shape[0], vp_shape[1], new_cold, vp_shape[3]],
                    ));
                }
                if let Some(ref vn) = self.v_norms {
                    let vn_shape = ffi::array_shape(vn);
                    self.v_norms = Some(ffi::slice(
                        vn,
                        &[0, 0, 0, 0],
                        &[vn_shape[0], vn_shape[1], new_cold, 1],
                    ));
                }
                if let Some(ref vr) = self.v_rescale {
                    let vr_shape = ffi::array_shape(vr);
                    self.v_rescale = Some(ffi::slice(
                        vr,
                        &[0, 0, 0, 0],
                        &[vr_shape[0], vr_shape[1], new_cold, 1],
                    ));
                }
            } else {
                // Cold portion fully erased.
                self.v_packed = None;
                self.v_norms = None;
                self.v_rescale = None;
            }

            self.cold_offset = new_cold;
            self.offset = new_len;
        } else {
            // Non-delegated modes: simple buffer-prefix trim along axis 2.
            self.keys = trim_axis_seq(&self.keys, 0);
            self.values = trim_axis_seq(&self.values, 0);

            // INT8 scale sidecars carry a width-1 head-axis (axis 3).
            if self.mode == KVCacheMode::Int8 {
                self.key_scales = trim_axis_seq(&self.key_scales, 1);
                self.val_scales = trim_axis_seq(&self.val_scales, 1);
            }

            // Turbo4* V sidecars (per-token packed + norms + rescale).
            if matches!(
                self.mode,
                KVCacheMode::Turbo4Asym | KVCacheMode::Turbo4 | KVCacheMode::Turbo3Asym
            ) {
                self.v_packed = trim_axis_seq(&self.v_packed, 0);
                self.v_norms = trim_axis_seq(&self.v_norms, 1);
                self.v_rescale = trim_axis_seq(&self.v_rescale, 1);
            }

            // K-side sidecars exist only in symmetric Turbo4.
            if self.mode == KVCacheMode::Turbo4 {
                self.k_packed = trim_axis_seq(&self.k_packed, 0);
                self.k_norms = trim_axis_seq(&self.k_norms, 1);
            }

            // KVarN8: truncate across the [sink | history tiles | tail]
            // layout. History truncation must land on a tile boundary —
            // the adoption floor (prefill_alignment = tile size = 128)
            // guarantees this for every real adopt, so a misaligned
            // new_len here is a caller bug and fails loudly rather than
            // approximating (a half-tile cannot be re-quantized without
            // the original fp16 data).
            if self.mode == KVCacheMode::KVarN8 {
                let tile = crate::cache::kvarn::KVARN_TILE_TOKENS;
                let slice_seq = |t: &Option<UniquePtr<MlxArray>>, keep: i32| {
                    t.as_ref().map(|a| {
                        let s = ffi::array_shape(a);
                        ffi::slice(a, &[0, 0, 0, 0], &[s[0], s[1], keep, s[3]])
                    })
                };
                let sink_len = self
                    .kvarn_sink_k
                    .as_ref()
                    .map_or(0, |s| ffi::array_shape(s)[2]);
                let hist_len = self
                    .kvarn_hist_k
                    .as_ref()
                    .map_or(0, |h| ffi::array_shape(h)[2]);

                if new_len <= sink_len {
                    self.kvarn_sink_k = slice_seq(&self.kvarn_sink_k, new_len);
                    self.kvarn_sink_v = slice_seq(&self.kvarn_sink_v, new_len);
                    self.kvarn_hist_k = None;
                    self.kvarn_hist_v = None;
                    self.kvarn_k_scale = None;
                    self.kvarn_k_zp = None;
                    self.kvarn_k_s_row = None;
                    self.kvarn_k_s_col = None;
                    self.kvarn_v_scale = None;
                    self.kvarn_v_zp = None;
                    self.kvarn_v_s_row = None;
                    self.kvarn_v_s_col = None;
                    self.kvarn_tail_k = None;
                    self.kvarn_tail_v = None;
                } else {
                    let hist_keep = (new_len - sink_len).min(hist_len);
                    let tail_keep = new_len - sink_len - hist_keep;
                    if hist_keep % tile != 0 {
                        return Err(format!(
                            "DetachedKVCache::trim_to: KVarN8 history \
                             truncation to {hist_keep} tokens is not \
                             tile-aligned (tile = {tile}); adoption floors \
                             to the alignment quantum so this indicates a \
                             caller bug"
                        ));
                    }
                    if hist_keep < hist_len {
                        let n_tiles_keep = hist_keep / tile;
                        self.kvarn_hist_k = slice_seq(&self.kvarn_hist_k, hist_keep);
                        self.kvarn_hist_v = slice_seq(&self.kvarn_hist_v, hist_keep);
                        self.kvarn_k_scale = slice_seq(&self.kvarn_k_scale, hist_keep);
                        self.kvarn_k_zp = slice_seq(&self.kvarn_k_zp, hist_keep);
                        self.kvarn_k_s_row = slice_seq(&self.kvarn_k_s_row, hist_keep);
                        self.kvarn_v_scale = slice_seq(&self.kvarn_v_scale, hist_keep);
                        self.kvarn_v_zp = slice_seq(&self.kvarn_v_zp, hist_keep);
                        self.kvarn_v_s_row = slice_seq(&self.kvarn_v_s_row, hist_keep);
                        self.kvarn_k_s_col = slice_seq(&self.kvarn_k_s_col, n_tiles_keep);
                        self.kvarn_v_s_col = slice_seq(&self.kvarn_v_s_col, n_tiles_keep);
                    }
                    if tail_keep > 0 {
                        self.kvarn_tail_k = slice_seq(&self.kvarn_tail_k, tail_keep);
                        self.kvarn_tail_v = slice_seq(&self.kvarn_tail_v, tail_keep);
                    } else {
                        self.kvarn_tail_k = None;
                        self.kvarn_tail_v = None;
                    }
                }
            }

            self.offset = new_len;
        }

        // MiniMax-M3 indexer K cache. Mode-agnostic: M3 layers 3-59 populate
        // it regardless of which KV mode is in play, dense layers leave it
        // None. Shape: `[b, 1, m3_idx_offset, index_dim]`. Slice axis 2 in
        // lockstep with main K so partial APC adoption preserves the
        // (cache_offset == m3_idx_offset) invariant the MSA dispatch relies on.
        if let Some(ref idx_k) = self.m3_idx_k {
            let shape = ffi::array_shape(idx_k);
            self.m3_idx_k = Some(ffi::slice(
                idx_k,
                &[0, 0, 0, 0],
                &[shape[0], shape[1], new_len, shape[3]],
            ));
            self.m3_idx_offset = new_len;
        }

        debug_assert!(
            self.offset == new_len,
            "DetachedKVCache::trim_to: post-condition failed: offset {} != new_len {} (was {prev_offset})",
            self.offset,
            new_len
        );
        Ok(())
    }
}

impl std::fmt::Debug for DetachedKVCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DetachedKVCache")
            .field("offset", &self.offset)
            .field("step", &self.step)
            .field("mode", &self.mode)
            .field("has_keys", &self.keys.is_some())
            .field("has_values", &self.values.is_some())
            .field("has_key_scales", &self.key_scales.is_some())
            .field("has_val_scales", &self.val_scales.is_some())
            .field("has_v_packed", &self.v_packed.is_some())
            .field("has_v_norms", &self.v_norms.is_some())
            .field("has_v_rescale", &self.v_rescale.is_some())
            .field("has_k_packed", &self.k_packed.is_some())
            .field("has_k_norms", &self.k_norms.is_some())
            .field("turbo_seed", &self.turbo_seed)
            .field("cold_offset", &self.cold_offset)
            .field("hot_threshold", &self.hot_threshold)
            .field("delegated_fp16_fast_path", &self.delegated_fp16_fast_path)
            .field(
                "delegated_fp16_sidecar_policy",
                &self.delegated_fp16_sidecar_policy,
            )
            .field("has_m3_idx_k", &self.m3_idx_k.is_some())
            .field("m3_idx_offset", &self.m3_idx_offset)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Adopt-time deep validation (issue-4: cold-store adopt defense-in-depth)
// ---------------------------------------------------------------------------

/// The live model's expected dense per-layer KV cache geometry, used to
/// anchor a deserialized [`DetachedKVCache`] to the real model before
/// [`CachePool::adopt`] installs it.
///
/// Core-side sibling of the server's `ExpectedBlockGeometry`
/// (`src/distributed/kv_cache_serde/types.rs`). That struct cannot be
/// reused here: it lives in the server crate (dependency points the other
/// way) and carries paged-pool fields (`num_layers`, `block_size`) that a
/// per-layer dense check does not consult. K and V head dims are separate
/// fields because `update_fp16` tracks them separately (MLA-class models
/// project them differently); on most architectures they are equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpectedCacheGeometry {
    /// Key/value head count (axis 1 of every per-layer 4-D tensor).
    pub n_kv_heads: i32,
    /// K per-head dimension (axis 3 of `keys` / K-side sidecars).
    pub k_head_dim: i32,
    /// V per-head dimension (axis 3 of `values` / V-side sidecars).
    pub v_head_dim: i32,
    /// MLX dtype code of the model's runtime activation dtype
    /// (e.g. `dtype::FLOAT16` = 9, `dtype::BFLOAT16` = 12). This is what
    /// `Fp16`-mode keys/values are stored as — `update_fp16` allocates its
    /// buffers with the incoming activation dtype rather than forcing
    /// float16. Quantized modes pin their own storage dtypes and do not
    /// consult this field for the quantized tensors.
    pub dtype: i32,
}

/// Human-readable name for an MLX dtype code (error messages only).
fn dtype_code_name(code: i32) -> &'static str {
    match code {
        crate::dtype::BOOL => "bool",
        crate::dtype::UINT8 => "uint8",
        crate::dtype::UINT16 => "uint16",
        crate::dtype::UINT32 => "uint32",
        crate::dtype::UINT64 => "uint64",
        crate::dtype::INT8 => "int8",
        crate::dtype::INT16 => "int16",
        crate::dtype::INT32 => "int32",
        crate::dtype::INT64 => "int64",
        crate::dtype::FLOAT16 => "float16",
        crate::dtype::FLOAT32 => "float32",
        crate::dtype::FLOAT64 => "float64",
        crate::dtype::BFLOAT16 => "bfloat16",
        crate::dtype::COMPLEX64 => "complex64",
        _ => "unknown",
    }
}

/// A tensor the current mode requires. Returns the array or a role-named
/// error.
fn require_present<'a>(
    role: &str,
    t: &'a Option<UniquePtr<MlxArray>>,
) -> Result<&'a MlxArray, String> {
    t.as_deref()
        .ok_or_else(|| format!("{role}: required for this mode but absent"))
}

/// A tensor the current mode must NOT carry. A populated forbidden sidecar
/// means the handle was produced under a different mode than its tag
/// claims (or the file is corrupt) — either way adopting it is unsafe.
fn require_absent(role: &str, t: &Option<UniquePtr<MlxArray>>) -> Result<(), String> {
    if t.is_some() {
        return Err(format!("{role}: must be absent for this mode but is present"));
    }
    Ok(())
}

/// Shared per-tensor check: rank 4, batch axis 1, expected head count,
/// expected trailing dim, expected dtype. Returns the seq-axis length
/// (axis 2) for the caller's lockstep / length checks.
fn check_tensor(
    role: &str,
    arr: &MlxArray,
    dtype_expected: i32,
    n_kv_heads: i32,
    last_dim: i32,
) -> Result<i32, String> {
    let shape = ffi::array_shape(arr);
    if shape.len() != 4 {
        return Err(format!(
            "{role}: expected rank 4, got rank {} (shape {:?})",
            shape.len(),
            shape
        ));
    }
    if shape[0] != 1 {
        return Err(format!(
            "{role}: expected batch axis 1, got {} (shape {:?})",
            shape[0], shape
        ));
    }
    if shape[1] != n_kv_heads {
        return Err(format!(
            "{role}: expected {n_kv_heads} kv heads on axis 1, got {} (shape {:?})",
            shape[1], shape
        ));
    }
    if shape[3] != last_dim {
        return Err(format!(
            "{role}: expected trailing dim {last_dim}, got {} (shape {:?})",
            shape[3], shape
        ));
    }
    let dt = ffi::array_dtype(arr);
    if dt != dtype_expected {
        return Err(format!(
            "{role}: expected dtype {} ({dtype_expected}), got {} ({dt})",
            dtype_code_name(dtype_expected),
            dtype_code_name(dt)
        ));
    }
    Ok(shape[2])
}

impl DetachedKVCache {
    /// Adopt-time deep validation: verify this handle's mode tag, per-mode
    /// tensor presence/absence, per-role dtypes, and per-role shapes
    /// against the live model's configured mode, KVarN V width, and
    /// geometry — BEFORE [`KVCache::install_detached`] wires the buffers
    /// under a reader that will interpret them by mode.
    ///
    /// Defense-in-depth for the cold-store adopt path (issue-4): the
    /// serialized format validates only per-tensor byte_len/dtype
    /// consistency and the set-level offset equality; nothing today stops
    /// a wrong-mode / wrong-width / wrong-model-family entry from being
    /// installed and silently dequantized into garbage attention. The
    /// in-memory handoff paths already enforce geometry
    /// (`ExpectedBlockGeometry` in `handoff_impl.rs`) and layout equality
    /// (`PagedCacheLayout` in `paged_detach.rs`); this is the dense
    /// deserialization-boundary sibling.
    ///
    /// Seq-axis semantics (verified against the writers, NOT symmetric):
    /// * Dense / Turbo tensors carry a step-aligned CAPACITY on axis 2
    ///   (`update_fp16` et al. allocate `n_steps * step` slabs and
    ///   `clone_handle` moves them unsliced), so the check is
    ///   `capacity >= offset` plus lockstep equality across sidecars —
    ///   requiring exact equality would reject virtually every real
    ///   entry.
    /// * KVarN tensors are exact-length (concatenate-grown), so
    ///   `sink_len + hist_len + tail_len == offset` is enforced exactly,
    ///   with `hist_len` tile-aligned.
    ///
    /// `v_bits_expected` is consulted only when `mode_expected` is
    /// [`KVCacheMode::KVarN8`] (the width field is inert for every other
    /// mode, mirroring `KVCache::kvarn_v_bits` semantics).
    ///
    /// Returns `Err(String)` naming the exact mismatch (detach.rs error
    /// convention; cold-store callers wrap it in
    /// `ColdStoreError::CorruptLayer { layer, detail }`).
    pub fn validate_for(
        &self,
        mode_expected: KVCacheMode,
        v_bits_expected: u8,
        geometry: &ExpectedCacheGeometry,
    ) -> Result<(), String> {
        self.validate_for_inner(mode_expected, v_bits_expected, geometry)
            .map_err(|e| format!("DetachedKVCache::validate_for: {e}"))
    }

    fn validate_for_inner(
        &self,
        mode_expected: KVCacheMode,
        v_bits_expected: u8,
        g: &ExpectedCacheGeometry,
    ) -> Result<(), String> {
        use crate::dtype;

        if g.n_kv_heads <= 0 || g.k_head_dim <= 0 || g.v_head_dim <= 0 {
            return Err(format!(
                "expected geometry is degenerate (n_kv_heads={}, k_head_dim={}, v_head_dim={}) — caller bug",
                g.n_kv_heads, g.k_head_dim, g.v_head_dim
            ));
        }
        if self.offset <= 0 {
            return Err(format!(
                "offset {} — an adoptable cache must carry at least one token",
                self.offset
            ));
        }
        // (a) mode-vs-config. This is THE issue-4 gate: `CachePool::adopt`
        // installs `KVCache::new_with_mode(detached.mode)`, so a wrong-mode
        // entry adopts "successfully" and the model then reads tensors laid
        // out for a different quantization scheme.
        if self.mode != mode_expected {
            return Err(format!(
                "mode mismatch: cache is {:?}, model is configured for {:?} — \
                 adopting would install {:?}-layout tensors under a {:?} reader",
                self.mode, mode_expected, self.mode, mode_expected
            ));
        }
        if mode_expected == KVCacheMode::KVarN8 {
            if v_bits_expected != 8 && v_bits_expected != 4 {
                return Err(format!(
                    "expected kvarn_v_bits {v_bits_expected} is not a legal width (8 or 4) — caller bug"
                ));
            }
            if self.kvarn_v_bits != v_bits_expected {
                return Err(format!(
                    "kvarn_v_bits mismatch: cache is v{}, model is configured for v{} — \
                     packed V codes would be silently mislabeled (the exact wrong the \
                     field's round-trip contract exists to kill)",
                    self.kvarn_v_bits, v_bits_expected
                ));
            }
        }

        // MiniMax-M3 indexer lockstep (mode-agnostic; the cycle-79 desync
        // class: adopted MSA dispatch seeing m3_idx_offset != offset).
        match self.m3_idx_k.as_deref() {
            None => {
                if self.m3_idx_offset != 0 {
                    return Err(format!(
                        "m3_idx_offset {} with no m3_idx_k tensor",
                        self.m3_idx_offset
                    ));
                }
            }
            Some(idx) => {
                let shape = ffi::array_shape(idx);
                if shape.len() != 4 {
                    return Err(format!(
                        "m3_idx_k: expected rank 4, got rank {} (shape {:?})",
                        shape.len(),
                        shape
                    ));
                }
                if shape[2] != self.m3_idx_offset {
                    return Err(format!(
                        "m3_idx_k seq axis {} != m3_idx_offset {}",
                        shape[2], self.m3_idx_offset
                    ));
                }
                if self.m3_idx_offset != self.offset {
                    return Err(format!(
                        "m3_idx_offset {} != offset {} — adopted MSA dispatch would desync",
                        self.m3_idx_offset, self.offset
                    ));
                }
            }
        }

        // KVarN sidecars must be empty for every non-KVarN mode; dense /
        // turbo tensors must be empty for KVarN8. Listed once, consumed by
        // the per-mode arms below.
        let kvarn_fields: [(&str, &Option<UniquePtr<MlxArray>>); 14] = [
            ("kvarn_sink_k", &self.kvarn_sink_k),
            ("kvarn_sink_v", &self.kvarn_sink_v),
            ("kvarn_tail_k", &self.kvarn_tail_k),
            ("kvarn_tail_v", &self.kvarn_tail_v),
            ("kvarn_hist_k", &self.kvarn_hist_k),
            ("kvarn_hist_v", &self.kvarn_hist_v),
            ("kvarn_k_scale", &self.kvarn_k_scale),
            ("kvarn_k_zp", &self.kvarn_k_zp),
            ("kvarn_k_s_row", &self.kvarn_k_s_row),
            ("kvarn_k_s_col", &self.kvarn_k_s_col),
            ("kvarn_v_scale", &self.kvarn_v_scale),
            ("kvarn_v_zp", &self.kvarn_v_zp),
            ("kvarn_v_s_row", &self.kvarn_v_s_row),
            ("kvarn_v_s_col", &self.kvarn_v_s_col),
        ];

        // (b) + (c) + (d): per-mode decision table.
        match mode_expected {
            KVCacheMode::Fp16 => {
                for (role, t) in [
                    ("key_scales", &self.key_scales),
                    ("val_scales", &self.val_scales),
                    ("v_packed", &self.v_packed),
                    ("v_norms", &self.v_norms),
                    ("v_rescale", &self.v_rescale),
                    ("k_packed", &self.k_packed),
                    ("k_norms", &self.k_norms),
                ] {
                    require_absent(role, t)?;
                }
                for (role, t) in &kvarn_fields {
                    require_absent(role, t)?;
                }
                let k = require_present("keys", &self.keys)?;
                let v = require_present("values", &self.values)?;
                let k_cap = check_tensor("keys", k, g.dtype, g.n_kv_heads, g.k_head_dim)?;
                let v_cap = check_tensor("values", v, g.dtype, g.n_kv_heads, g.v_head_dim)?;
                if k_cap < self.offset {
                    return Err(format!(
                        "offset {} exceeds keys seq capacity {k_cap}",
                        self.offset
                    ));
                }
                if v_cap != k_cap {
                    return Err(format!(
                        "keys/values seq-capacity lockstep broken: keys {k_cap}, values {v_cap}"
                    ));
                }
            }

            KVCacheMode::Int8 => {
                for (role, t) in [
                    ("v_packed", &self.v_packed),
                    ("v_norms", &self.v_norms),
                    ("v_rescale", &self.v_rescale),
                    ("k_packed", &self.k_packed),
                    ("k_norms", &self.k_norms),
                ] {
                    require_absent(role, t)?;
                }
                for (role, t) in &kvarn_fields {
                    require_absent(role, t)?;
                }
                let k = require_present("keys", &self.keys)?;
                let v = require_present("values", &self.values)?;
                let ks = require_present("key_scales", &self.key_scales)?;
                let vs = require_present("val_scales", &self.val_scales)?;
                let k_cap = check_tensor("keys", k, dtype::INT8, g.n_kv_heads, g.k_head_dim)?;
                let v_cap = check_tensor("values", v, dtype::INT8, g.n_kv_heads, g.v_head_dim)?;
                let ks_cap = check_tensor("key_scales", ks, dtype::FLOAT16, g.n_kv_heads, 1)?;
                let vs_cap = check_tensor("val_scales", vs, dtype::FLOAT16, g.n_kv_heads, 1)?;
                if k_cap < self.offset {
                    return Err(format!(
                        "offset {} exceeds keys seq capacity {k_cap}",
                        self.offset
                    ));
                }
                if v_cap != k_cap || ks_cap != k_cap || vs_cap != k_cap {
                    return Err(format!(
                        "INT8 seq-capacity lockstep broken: keys {k_cap}, values {v_cap}, \
                         key_scales {ks_cap}, val_scales {vs_cap} — dequantization would \
                         pair codes with the wrong per-token scales"
                    ));
                }
            }

            KVCacheMode::Turbo4Asym | KVCacheMode::Turbo3Asym => {
                for (role, t) in [
                    ("values", &self.values),
                    ("key_scales", &self.key_scales),
                    ("val_scales", &self.val_scales),
                    ("k_packed", &self.k_packed),
                    ("k_norms", &self.k_norms),
                ] {
                    require_absent(role, t)?;
                }
                for (role, t) in &kvarn_fields {
                    require_absent(role, t)?;
                }
                let packed_dim = if mode_expected == KVCacheMode::Turbo4Asym {
                    if g.v_head_dim % 2 != 0 {
                        return Err(format!(
                            "Turbo4Asym requires even v_head_dim, geometry says {}",
                            g.v_head_dim
                        ));
                    }
                    g.v_head_dim / 2
                } else {
                    if g.v_head_dim % 8 != 0 {
                        return Err(format!(
                            "Turbo3Asym requires v_head_dim divisible by 8, geometry says {}",
                            g.v_head_dim
                        ));
                    }
                    g.v_head_dim * 3 / 8
                };
                let k = require_present("keys", &self.keys)?;
                let vp = require_present("v_packed", &self.v_packed)?;
                let vn = require_present("v_norms", &self.v_norms)?;
                let k_cap = check_tensor("keys", k, dtype::FLOAT16, g.n_kv_heads, g.k_head_dim)?;
                let vp_cap = check_tensor("v_packed", vp, dtype::UINT8, g.n_kv_heads, packed_dim)?;
                let vn_cap = check_tensor("v_norms", vn, dtype::FLOAT16, g.n_kv_heads, 1)?;
                if k_cap < self.offset {
                    return Err(format!(
                        "offset {} exceeds keys seq capacity {k_cap}",
                        self.offset
                    ));
                }
                if vp_cap != k_cap || vn_cap != k_cap {
                    return Err(format!(
                        "Turbo V-sidecar seq-capacity lockstep broken: keys {k_cap}, \
                         v_packed {vp_cap}, v_norms {vn_cap}"
                    ));
                }
                // v_rescale is a precomputed optimization sidecar; lockstep
                // when present, but legitimately absent (e.g. entries built
                // by paths that predate it).
                if let Some(vr) = self.v_rescale.as_deref() {
                    let vr_cap = check_tensor("v_rescale", vr, dtype::FLOAT16, g.n_kv_heads, 1)?;
                    if vr_cap != k_cap {
                        return Err(format!(
                            "v_rescale seq capacity {vr_cap} != keys capacity {k_cap}"
                        ));
                    }
                }
            }

            KVCacheMode::Turbo4 => {
                for (role, t) in [
                    ("keys", &self.keys),
                    ("values", &self.values),
                    ("key_scales", &self.key_scales),
                    ("val_scales", &self.val_scales),
                ] {
                    require_absent(role, t)?;
                }
                for (role, t) in &kvarn_fields {
                    require_absent(role, t)?;
                }
                if g.k_head_dim % 2 != 0 || g.v_head_dim % 2 != 0 {
                    return Err(format!(
                        "Turbo4 requires even head dims, geometry says k={} v={}",
                        g.k_head_dim, g.v_head_dim
                    ));
                }
                let kp = require_present("k_packed", &self.k_packed)?;
                let kn = require_present("k_norms", &self.k_norms)?;
                let vp = require_present("v_packed", &self.v_packed)?;
                let vn = require_present("v_norms", &self.v_norms)?;
                let kp_cap =
                    check_tensor("k_packed", kp, dtype::UINT8, g.n_kv_heads, g.k_head_dim / 2)?;
                let kn_cap = check_tensor("k_norms", kn, dtype::FLOAT16, g.n_kv_heads, 1)?;
                let vp_cap =
                    check_tensor("v_packed", vp, dtype::UINT8, g.n_kv_heads, g.v_head_dim / 2)?;
                let vn_cap = check_tensor("v_norms", vn, dtype::FLOAT16, g.n_kv_heads, 1)?;
                if kp_cap < self.offset {
                    return Err(format!(
                        "offset {} exceeds k_packed seq capacity {kp_cap}",
                        self.offset
                    ));
                }
                if kn_cap != kp_cap || vp_cap != kp_cap || vn_cap != kp_cap {
                    return Err(format!(
                        "Turbo4 packed/norms seq-capacity lockstep broken: k_packed {kp_cap}, \
                         k_norms {kn_cap}, v_packed {vp_cap}, v_norms {vn_cap}"
                    ));
                }
                if let Some(vr) = self.v_rescale.as_deref() {
                    let vr_cap = check_tensor("v_rescale", vr, dtype::FLOAT16, g.n_kv_heads, 1)?;
                    if vr_cap != kp_cap {
                        return Err(format!(
                            "v_rescale seq capacity {vr_cap} != k_packed capacity {kp_cap}"
                        ));
                    }
                }
            }

            KVCacheMode::Turbo4Delegated => {
                for (role, t) in [
                    ("key_scales", &self.key_scales),
                    ("val_scales", &self.val_scales),
                    ("k_packed", &self.k_packed),
                    ("k_norms", &self.k_norms),
                ] {
                    require_absent(role, t)?;
                }
                for (role, t) in &kvarn_fields {
                    require_absent(role, t)?;
                }
                if g.v_head_dim % 2 != 0 {
                    return Err(format!(
                        "Turbo4Delegated requires even v_head_dim, geometry says {}",
                        g.v_head_dim
                    ));
                }
                if self.cold_offset < 0 || self.cold_offset > self.offset {
                    return Err(format!(
                        "cold_offset {} outside [0, offset={}]",
                        self.cold_offset, self.offset
                    ));
                }
                // Unified K, same shape contract as Fp16 but always cast
                // to float16 by the delegated update path.
                let k = require_present("keys", &self.keys)?;
                let k_cap = check_tensor("keys", k, dtype::FLOAT16, g.n_kv_heads, g.k_head_dim)?;
                if k_cap < self.offset {
                    return Err(format!(
                        "offset {} exceeds unified-K seq capacity {k_cap}",
                        self.offset
                    ));
                }
                // Cold-V sidecars must cover the cold span when one exists.
                if self.cold_offset > 0 {
                    let vp = require_present("v_packed (cold V)", &self.v_packed)?;
                    let vn = require_present("v_norms (cold V)", &self.v_norms)?;
                    let vp_cap = check_tensor(
                        "v_packed",
                        vp,
                        dtype::UINT8,
                        g.n_kv_heads,
                        g.v_head_dim / 2,
                    )?;
                    let vn_cap = check_tensor("v_norms", vn, dtype::FLOAT16, g.n_kv_heads, 1)?;
                    if vp_cap < self.cold_offset || vn_cap < self.cold_offset {
                        return Err(format!(
                            "cold-V sidecars shorter than cold_offset {}: v_packed {vp_cap}, \
                             v_norms {vn_cap}",
                            self.cold_offset
                        ));
                    }
                } else {
                    // Predecode-policy sidecars may exist ahead of the cold
                    // boundary; validate their layout when present.
                    if let Some(vp) = self.v_packed.as_deref() {
                        check_tensor("v_packed", vp, dtype::UINT8, g.n_kv_heads, g.v_head_dim / 2)?;
                    }
                    if let Some(vn) = self.v_norms.as_deref() {
                        check_tensor("v_norms", vn, dtype::FLOAT16, g.n_kv_heads, 1)?;
                    }
                }
                if let Some(vr) = self.v_rescale.as_deref() {
                    check_tensor("v_rescale", vr, dtype::FLOAT16, g.n_kv_heads, 1)?;
                }
                // V working set: unified FP16 (fast path) or hot ring.
                let visible_v = if self.delegated_fp16_fast_path {
                    self.offset
                } else {
                    self.offset - self.cold_offset
                };
                match self.values.as_deref() {
                    Some(v) => {
                        let v_cap =
                            check_tensor("values", v, dtype::FLOAT16, g.n_kv_heads, g.v_head_dim)?;
                        if v_cap < visible_v {
                            return Err(format!(
                                "visible V length {visible_v} exceeds values seq capacity {v_cap}"
                            ));
                        }
                    }
                    // A populated visible-V span with no buffer cannot be
                    // decoded against. (`values == None` with
                    // `visible_v == 0` is a real post-fold state.)
                    None if visible_v > 0 => {
                        return Err(format!(
                            "values: required (visible V span {visible_v} > 0) but absent"
                        ));
                    }
                    None => {}
                }
            }

            KVCacheMode::KVarN8 => {
                use crate::cache::kvarn::{KVARN_TILE_TOKENS, KVARN_V4_GROUP_SIZE};
                for (role, t) in [
                    ("keys", &self.keys),
                    ("values", &self.values),
                    ("key_scales", &self.key_scales),
                    ("val_scales", &self.val_scales),
                    ("v_packed", &self.v_packed),
                    ("v_norms", &self.v_norms),
                    ("v_rescale", &self.v_rescale),
                    ("k_packed", &self.k_packed),
                    ("k_norms", &self.k_norms),
                ] {
                    require_absent(role, t)?;
                }

                // Sink: first tile, FP16, never quantized. `is_empty()`
                // keys off kvarn_sink_k, so a non-empty KVarN cache MUST
                // carry it.
                let sink_k = require_present("kvarn_sink_k", &self.kvarn_sink_k)?;
                let sink_v = require_present("kvarn_sink_v", &self.kvarn_sink_v)?;
                let sink_k_len =
                    check_tensor("kvarn_sink_k", sink_k, dtype::FLOAT16, g.n_kv_heads, g.k_head_dim)?;
                let sink_v_len =
                    check_tensor("kvarn_sink_v", sink_v, dtype::FLOAT16, g.n_kv_heads, g.v_head_dim)?;
                if sink_v_len != sink_k_len {
                    return Err(format!(
                        "sink K/V lockstep broken: sink_k {sink_k_len}, sink_v {sink_v_len}"
                    ));
                }
                if sink_k_len <= 0 || sink_k_len > KVARN_TILE_TOKENS {
                    return Err(format!(
                        "sink length {sink_k_len} outside (0, {KVARN_TILE_TOKENS}]"
                    ));
                }

                // History group: all-or-none presence (writer appends every
                // member of the group per finalized tile batch).
                let hist_group: [(&str, bool); 9] = [
                    ("kvarn_hist_k", self.kvarn_hist_k.is_some()),
                    ("kvarn_hist_v", self.kvarn_hist_v.is_some()),
                    ("kvarn_k_scale", self.kvarn_k_scale.is_some()),
                    ("kvarn_k_zp", self.kvarn_k_zp.is_some()),
                    ("kvarn_k_s_row", self.kvarn_k_s_row.is_some()),
                    ("kvarn_k_s_col", self.kvarn_k_s_col.is_some()),
                    ("kvarn_v_scale", self.kvarn_v_scale.is_some()),
                    ("kvarn_v_zp", self.kvarn_v_zp.is_some()),
                    ("kvarn_v_s_col", self.kvarn_v_s_col.is_some()),
                ];
                let hist_present = self.kvarn_hist_k.is_some();
                for (role, present) in hist_group {
                    if present != hist_present {
                        return Err(format!(
                            "history tile group presence broken: kvarn_hist_k is {} but {role} is {}",
                            if hist_present { "present" } else { "absent" },
                            if present { "present" } else { "absent" }
                        ));
                    }
                }
                // v_s_row: k8v8 stores it per token; k8v4 folds it into
                // scale/zp at write time — "the fold IS its storage" — so a
                // populated v_s_row under v4 means the entry was written by
                // a different width than its tag claims.
                match self.kvarn_v_bits {
                    8 => {
                        if self.kvarn_v_s_row.is_some() != hist_present {
                            return Err(format!(
                                "kvarn_v_s_row presence ({}) must match history presence ({}) at v_bits=8",
                                self.kvarn_v_s_row.is_some(),
                                hist_present
                            ));
                        }
                    }
                    4 => {
                        require_absent("kvarn_v_s_row (folded at v_bits=4)", &self.kvarn_v_s_row)?;
                    }
                    other => {
                        return Err(format!("stored kvarn_v_bits {other} is not a legal width"));
                    }
                }

                let mut hist_len = 0i32;
                if hist_present {
                    let hist_k = require_present("kvarn_hist_k", &self.kvarn_hist_k)?;
                    hist_len = check_tensor(
                        "kvarn_hist_k",
                        hist_k,
                        dtype::UINT8,
                        g.n_kv_heads,
                        g.k_head_dim,
                    )?;
                    if hist_len <= 0 || hist_len % KVARN_TILE_TOKENS != 0 {
                        return Err(format!(
                            "history length {hist_len} is not a positive multiple of the \
                             tile size {KVARN_TILE_TOKENS}"
                        ));
                    }
                    let n_tiles = hist_len / KVARN_TILE_TOKENS;
                    for (role, t) in [
                        ("kvarn_k_scale", &self.kvarn_k_scale),
                        ("kvarn_k_zp", &self.kvarn_k_zp),
                        ("kvarn_k_s_row", &self.kvarn_k_s_row),
                    ] {
                        let arr = require_present(role, t)?;
                        let len = check_tensor(role, arr, dtype::FLOAT32, g.n_kv_heads, 1)?;
                        if len != hist_len {
                            return Err(format!(
                                "{role} seq length {len} != history length {hist_len}"
                            ));
                        }
                    }
                    let ksc = require_present("kvarn_k_s_col", &self.kvarn_k_s_col)?;
                    let ksc_tiles = check_tensor(
                        "kvarn_k_s_col",
                        ksc,
                        dtype::FLOAT32,
                        g.n_kv_heads,
                        g.k_head_dim,
                    )?;
                    if ksc_tiles != n_tiles {
                        return Err(format!(
                            "kvarn_k_s_col tile axis {ksc_tiles} != n_tiles {n_tiles}"
                        ));
                    }
                    let vsc = require_present("kvarn_v_s_col", &self.kvarn_v_s_col)?;
                    let vsc_tiles = check_tensor(
                        "kvarn_v_s_col",
                        vsc,
                        dtype::FLOAT32,
                        g.n_kv_heads,
                        g.v_head_dim,
                    )?;
                    if vsc_tiles != n_tiles {
                        return Err(format!(
                            "kvarn_v_s_col tile axis {vsc_tiles} != n_tiles {n_tiles}"
                        ));
                    }

                    let hist_v = require_present("kvarn_hist_v", &self.kvarn_hist_v)?;
                    match self.kvarn_v_bits {
                        8 => {
                            let hv_len = check_tensor(
                                "kvarn_hist_v",
                                hist_v,
                                dtype::UINT8,
                                g.n_kv_heads,
                                g.v_head_dim,
                            )?;
                            if hv_len != hist_len {
                                return Err(format!(
                                    "kvarn_hist_v seq length {hv_len} != history length {hist_len}"
                                ));
                            }
                            for (role, t) in [
                                ("kvarn_v_scale", &self.kvarn_v_scale),
                                ("kvarn_v_zp", &self.kvarn_v_zp),
                                ("kvarn_v_s_row", &self.kvarn_v_s_row),
                            ] {
                                let arr = require_present(role, t)?;
                                let len =
                                    check_tensor(role, arr, dtype::FLOAT32, g.n_kv_heads, 1)?;
                                if len != hist_len {
                                    return Err(format!(
                                        "{role} seq length {len} != history length {hist_len}"
                                    ));
                                }
                            }
                        }
                        4 => {
                            if g.v_head_dim % 8 != 0 || g.v_head_dim % KVARN_V4_GROUP_SIZE != 0 {
                                return Err(format!(
                                    "k8v4 requires v_head_dim divisible by 8 and by \
                                     gs={KVARN_V4_GROUP_SIZE}, geometry says {}",
                                    g.v_head_dim
                                ));
                            }
                            let hv_len = check_tensor(
                                "kvarn_hist_v",
                                hist_v,
                                dtype::UINT32,
                                g.n_kv_heads,
                                g.v_head_dim / 8,
                            )?;
                            if hv_len != hist_len {
                                return Err(format!(
                                    "kvarn_hist_v seq length {hv_len} != history length {hist_len}"
                                ));
                            }
                            for (role, t) in [
                                ("kvarn_v_scale", &self.kvarn_v_scale),
                                ("kvarn_v_zp", &self.kvarn_v_zp),
                            ] {
                                let arr = require_present(role, t)?;
                                let len = check_tensor(
                                    role,
                                    arr,
                                    dtype::FLOAT32,
                                    g.n_kv_heads,
                                    g.v_head_dim / KVARN_V4_GROUP_SIZE,
                                )?;
                                if len != hist_len {
                                    return Err(format!(
                                        "{role} seq length {len} != history length {hist_len}"
                                    ));
                                }
                            }
                        }
                        _ => unreachable!("width validated above"),
                    }
                }

                // Tail: FP16 partial-tile accumulation, K/V lockstep.
                let mut tail_len = 0i32;
                match (self.kvarn_tail_k.as_deref(), self.kvarn_tail_v.as_deref()) {
                    (Some(tk), Some(tv)) => {
                        let tk_len = check_tensor(
                            "kvarn_tail_k",
                            tk,
                            dtype::FLOAT16,
                            g.n_kv_heads,
                            g.k_head_dim,
                        )?;
                        let tv_len = check_tensor(
                            "kvarn_tail_v",
                            tv,
                            dtype::FLOAT16,
                            g.n_kv_heads,
                            g.v_head_dim,
                        )?;
                        if tv_len != tk_len {
                            return Err(format!(
                                "tail K/V lockstep broken: tail_k {tk_len}, tail_v {tv_len}"
                            ));
                        }
                        if tk_len <= 0 {
                            return Err(format!("tail present with non-positive length {tk_len}"));
                        }
                        tail_len = tk_len;
                    }
                    (None, None) => {}
                    (k, v) => {
                        return Err(format!(
                            "tail K/V presence broken: tail_k {}, tail_v {}",
                            if k.is_some() { "present" } else { "absent" },
                            if v.is_some() { "present" } else { "absent" }
                        ));
                    }
                }

                // The writer fills the sink completely before anything
                // reaches history or tail.
                if (hist_present || tail_len > 0) && sink_k_len != KVARN_TILE_TOKENS {
                    return Err(format!(
                        "sink length {sink_k_len} < {KVARN_TILE_TOKENS} with history/tail \
                         present — sink must fill before spill"
                    ));
                }
                // KVarN tensors are exact-length: the three spans must
                // account for every logical token.
                let total = sink_k_len + hist_len + tail_len;
                if total != self.offset {
                    return Err(format!(
                        "KVarN length equation broken: sink {sink_k_len} + hist {hist_len} + \
                         tail {tail_len} = {total} != offset {}",
                        self.offset
                    ));
                }
            }
        }
        Ok(())
    }
}

impl DetachedCacheSet {
    /// Set-level adopt-time validation: layer count vs the expected
    /// per-layer mode table, cross-layer seq-len consistency, then
    /// [`DetachedKVCache::validate_for`] on every layer.
    ///
    /// `layer_modes_expected` must be the SAME table the scheduler applies
    /// to freshly allocated sequences (`apply_kv_cache_mode_to`'s
    /// resolution — Boundary-V / `skip_last_layer` keep some layers Fp16),
    /// so the adopt expectation can never drift from allocation behavior.
    /// Assumes uniform head geometry across layers (true for every dense
    /// transformer this pool serves; a future mixed-geometry architecture
    /// needs a per-layer geometry table here).
    pub fn validate_for(
        &self,
        layer_modes_expected: &[KVCacheMode],
        v_bits_expected: u8,
        geometry: &ExpectedCacheGeometry,
    ) -> Result<(), String> {
        if self.caches.is_empty() {
            return Err("DetachedCacheSet::validate_for: set carries no layer caches".into());
        }
        if self.caches.len() != layer_modes_expected.len() {
            return Err(format!(
                "DetachedCacheSet::validate_for: set has {} layers, model expects {}",
                self.caches.len(),
                layer_modes_expected.len()
            ));
        }
        if !self.has_consistent_seq_len() {
            return Err(format!(
                "DetachedCacheSet::validate_for: layers disagree on seq_len: {:?}",
                self.caches.iter().map(|c| c.offset).collect::<Vec<_>>()
            ));
        }
        for (i, (cache, &mode)) in self.caches.iter().zip(layer_modes_expected).enumerate() {
            cache
                .validate_for(mode, v_bits_expected, geometry)
                .map_err(|e| format!("layer {i}: {e}"))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// KVCache extensions
// ---------------------------------------------------------------------------

impl KVCache {
    /// Shrink the logical cache length to exactly `new_len`.
    ///
    /// Semantics:
    /// * `new_len == 0` fully rewinds the cache (equivalent to
    ///   `trim(self.offset)`) and drops all backing buffers.
    /// * `0 < new_len < self.offset` keeps the pre-allocated buffer but
    ///   re-slices its visible region to `new_len`. This mirrors
    ///   [`KVCache::trim`] but takes an absolute target instead of a delta.
    /// * `new_len == self.offset` is a no-op.
    /// * `new_len < 0` or `new_len > self.offset` returns `Err`.
    ///
    /// INT8 mode: the per-token scale buffers are trimmed in lock-step so
    /// subsequent `update_and_fetch` dequantization stays consistent.
    ///
    /// Used by: prompt prefix cache reuse, speculative decode rewinds,
    /// server scheduler trim-to-exact-prefix paths.
    pub fn trim_to(&mut self, new_len: i32) -> Result<(), String> {
        if new_len < 0 {
            return Err(format!(
                "KVCache::trim_to: new_len must be non-negative, got {new_len}"
            ));
        }
        if new_len > self.offset {
            return Err(format!(
                "KVCache::trim_to: new_len ({new_len}) exceeds current offset ({})",
                self.offset
            ));
        }

        let delta = self.offset - new_len;
        if delta == 0 {
            return Ok(());
        }

        let trimmed = self.trim(delta);
        debug_assert_eq!(
            trimmed, delta,
            "KVCache::trim returned {trimmed} but trim_to computed delta {delta}"
        );
        Ok(())
    }

    /// Move the underlying MLX buffers out of this cache into a
    /// [`DetachedKVCache`] handle.
    ///
    /// After this call the source `KVCache` is empty (`is_empty() == true`,
    /// `offset == 0`) but retains its quantization mode and step size so it
    /// can be reused for a new sequence. The returned `DetachedKVCache`
    /// carries the original tensors unchanged — including INT8 scale buffers
    /// when `mode == Int8` — so adopt is a zero-copy operation. For the
    /// delegated FP16 fast path this first compacts any missing packed V
    /// sidecars so lazy generation can skip foreground folds without donating
    /// a sidecar-incomplete prompt-cache entry.
    ///
    /// Used by: prompt prefix cache detach/adopt, cross-request reuse
    /// handoff inside `CachePool::detach`.
    pub fn clone_handle(&mut self) -> DetachedKVCache {
        self.compact_turbo4_delegated_fp16_sidecars();

        // The live indexer cache is a CAPACITY buffer (grow-by-doubling,
        // see m3_idx_k_update_and_fetch) whose logical length is
        // m3_idx_offset. The detached contract documents exact-length
        // shape `[b, 1, m3_idx_offset, index_dim]` — trim_to and any
        // future serialization rely on it — so slice-to-fill here. One
        // O(fill) copy per detach (per turn, not per token).
        let m3_idx_k_exact = self.m3_idx_k.take().map(|buf| {
            let s = ffi::array_shape(&buf);
            if s[2] == self.m3_idx_offset {
                buf
            } else {
                ffi::slice(&buf, &[0, 0, 0, 0], &[s[0], s[1], self.m3_idx_offset, s[3]])
            }
        });

        let handle = DetachedKVCache {
            keys: self.keys.take(),
            values: self.values.take(),
            offset: std::mem::replace(&mut self.offset, 0),
            step: self.step,
            mode: self.mode,
            key_scales: self.key_scales.take(),
            val_scales: self.val_scales.take(),
            v_packed: self.v_packed.take(),
            v_norms: self.v_norms.take(),
            v_rescale: self.v_rescale.take(),
            k_packed: self.k_packed.take(),
            k_norms: self.k_norms.take(),
            turbo_seed: self.turbo_seed,
            cold_offset: std::mem::replace(&mut self.cold_offset, 0),
            hot_threshold: self.hot_threshold,
            delegated_fp16_fast_path: self.delegated_fp16_fast_path,
            delegated_fp16_sidecar_policy: self.delegated_fp16_sidecar_policy,
            m3_idx_k: m3_idx_k_exact,
            m3_idx_offset: std::mem::replace(&mut self.m3_idx_offset, 0),
            kvarn_sink_k: self.kvarn_sink_k.take(),
            kvarn_sink_v: self.kvarn_sink_v.take(),
            kvarn_tail_k: self.kvarn_tail_k.take(),
            kvarn_tail_v: self.kvarn_tail_v.take(),
            kvarn_hist_k: self.kvarn_hist_k.take(),
            kvarn_hist_v: self.kvarn_hist_v.take(),
            kvarn_k_scale: self.kvarn_k_scale.take(),
            kvarn_k_zp: self.kvarn_k_zp.take(),
            kvarn_k_s_row: self.kvarn_k_s_row.take(),
            kvarn_k_s_col: self.kvarn_k_s_col.take(),
            kvarn_v_scale: self.kvarn_v_scale.take(),
            kvarn_v_zp: self.kvarn_v_zp.take(),
            kvarn_v_s_row: self.kvarn_v_s_row.take(),
            kvarn_v_s_col: self.kvarn_v_s_col.take(),
            // Copied, not reset: like `mode`, the source slot keeps its
            // width for reuse under the same boot construction.
            kvarn_v_bits: self.kvarn_v_bits,
        };
        // Clear turbo_params on the source so the next quantize call rebuilds
        // it from scratch (required if the slot is reused with a different
        // head_dim after detach). LOW-1 fix. The 3-bit
        // `turbo3_params` follows the same contract.
        self.turbo_params = None;
        self.turbo3_params = None;
        // retired the cold-V dequant memo — nothing to drop on
        // the source.
        handle
    }

    /// Re-install a previously detached cache into this `KVCache` slot.
    ///
    /// This is the inverse of [`KVCache::clone_handle`]. The receiver must be
    /// empty (`is_empty() == true`) to guarantee no live buffer is silently
    /// dropped; callers that need to overwrite a populated cache should
    /// `trim_to(0)` first.
    ///
    /// Used by: `CachePool::adopt` when re-hydrating per-layer caches for a
    /// freshly allocated sequence id.
    pub fn install_detached(&mut self, detached: DetachedKVCache) -> Result<(), String> {
        if !self.is_empty() {
            return Err(
                "KVCache::install_detached: target cache is not empty; trim_to(0) first".into(),
            );
        }
        self.keys = detached.keys;
        self.values = detached.values;
        self.offset = detached.offset;
        self.step = detached.step;
        self.mode = detached.mode;
        self.key_scales = detached.key_scales;
        self.val_scales = detached.val_scales;
        self.v_packed = detached.v_packed;
        self.v_norms = detached.v_norms;
        self.v_rescale = detached.v_rescale;
        self.k_packed = detached.k_packed;
        self.k_norms = detached.k_norms;
        self.turbo_seed = detached.turbo_seed;
        self.cold_offset = detached.cold_offset;
        self.hot_threshold = detached.hot_threshold;
        self.delegated_fp16_fast_path = detached.delegated_fp16_fast_path;
        self.delegated_fp16_sidecar_policy = detached.delegated_fp16_sidecar_policy;
        // MiniMax-M3 indexer K cache + offset. Restored alongside main K/V so
        // the asymmetric MSA dispatch (`sparse_sdpa` with `cache_offset > 0`)
        // sees idx_k spanning the same logical length as the main K cache.
        // Without this restoration the dense-fallback band-aid in
        // `models/minimax_m3.rs` would fire permanently on adopted sessions.
        self.m3_idx_k = detached.m3_idx_k;
        self.m3_idx_offset = detached.m3_idx_offset;
        self.kvarn_sink_k = detached.kvarn_sink_k;
        self.kvarn_sink_v = detached.kvarn_sink_v;
        self.kvarn_tail_k = detached.kvarn_tail_k;
        self.kvarn_tail_v = detached.kvarn_tail_v;
        self.kvarn_hist_k = detached.kvarn_hist_k;
        self.kvarn_hist_v = detached.kvarn_hist_v;
        self.kvarn_k_scale = detached.kvarn_k_scale;
        self.kvarn_k_zp = detached.kvarn_k_zp;
        self.kvarn_k_s_row = detached.kvarn_k_s_row;
        self.kvarn_k_s_col = detached.kvarn_k_s_col;
        self.kvarn_v_scale = detached.kvarn_v_scale;
        self.kvarn_v_zp = detached.kvarn_v_zp;
        self.kvarn_v_s_row = detached.kvarn_v_s_row;
        self.kvarn_v_s_col = detached.kvarn_v_s_col;
        self.kvarn_v_bits = detached.kvarn_v_bits;
        // turbo_params is rebuilt lazily on the next quantize call, but if we
        // can already see the V head_dim from v_packed we may as well prebuild
        // so dequantize-only consumers (which don't go through update_*) still
        // work. Detect it from v_packed (or k_packed for symmetric Turbo4)
        // shape: [B, H, T, head_dim/2].
        if matches!(
            self.mode,
            KVCacheMode::Turbo4Asym | KVCacheMode::Turbo4 | KVCacheMode::Turbo4Delegated
        ) {
            let probe = self.v_packed.as_ref().or(self.k_packed.as_ref());
            if let Some(p) = probe {
                let shape = ffi::array_shape(p);
                if shape.len() == 4 && shape[3] > 0 {
                    let head_dim = (shape[3] as u32) * 2;
                    self.turbo_params = Some(super::turbo::TurboQuantParams::new(
                        head_dim,
                        self.turbo_seed,
                    ));
                }
            }
        }
        // Turbo3Asym: rebuild the 3-bit params from v_packed
        // shape. Inverse of `head_dim * 3 / 8`: head_dim = packed_dim * 8 / 3.
        // Mirrors the Turbo4 prebuild above so dequantize-only consumers see
        // a ready-to-go cache after install.
        if self.mode == KVCacheMode::Turbo3Asym {
            if let Some(p) = self.v_packed.as_ref() {
                let shape = ffi::array_shape(p);
                if shape.len() == 4 && shape[3] > 0 {
                    let head_dim = (shape[3] as u32) * 8 / 3;
                    self.turbo3_params = Some(super::turbo::quant3::TurboQuantParams3::new(
                        head_dim,
                        self.turbo_seed,
                    ));
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DetachedRotatingKVCache (B9)
// ---------------------------------------------------------------------------

/// Inert snapshot of a single [`RotatingKVCache`] (sliding-window cache) that
/// can outlive the sequence which produced it.
///
/// Mirrors [`DetachedKVCache`] for the rotating cache backend, including the
/// `Turbo4Asym` packed sidecar buffers (`v_packed`, `v_norms`) and the
/// deterministic seed so the adopted cache can rebuild
/// [`crate::cache::turbo::TurboQuantParams`] without consulting the originating
/// cache. Adds the rotating-specific `max_size` and `idx` fields so the ring
/// position is preserved across the round-trip — without `idx`, a wrap-around
/// state would silently fall back to "no wraparound yet" semantics.
///
/// Used by: prompt prefix cache reuse for sliding-window models (Gemma 3/4,
/// Ministral 3, GPT-OSS, RecurrentGemma, Exaone) under the same
/// architecture as the dense `DetachedKVCache`.
pub struct DetachedRotatingKVCache {
    pub(super) keys: Option<UniquePtr<MlxArray>>,
    pub(super) values: Option<UniquePtr<MlxArray>>,
    pub(super) max_size: i32,
    pub(super) offset: i32,
    pub(super) idx: i32,
    pub(super) step: i32,
    pub(super) mode: KVCacheMode,
    pub(super) key_scales: Option<UniquePtr<MlxArray>>,
    pub(super) val_scales: Option<UniquePtr<MlxArray>>,
    pub(super) v_packed: Option<UniquePtr<MlxArray>>,
    pub(super) v_norms: Option<UniquePtr<MlxArray>>,
    /// Sparse-V rescale sidecar for rotating Turbo4Asym caches.
    pub(super) v_rescale: Option<UniquePtr<MlxArray>>,
    pub(super) turbo_seed: u32,
}

impl DetachedRotatingKVCache {
    /// Logical sequence length at detach time (matches `RotatingKVCache::offset`).
    pub fn seq_len(&self) -> i32 {
        self.offset
    }

    /// Quantization mode of the detached cache.
    pub fn mode(&self) -> KVCacheMode {
        self.mode
    }

    /// Sliding window upper bound preserved across the round-trip.
    pub fn max_size(&self) -> i32 {
        self.max_size
    }

    /// Total byte footprint of the detached tensors.
    pub fn nbytes(&self) -> usize {
        let k = self.keys.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let v = self.values.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let ks = self.key_scales.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let vs = self.val_scales.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let vp = self.v_packed.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let vn = self.v_norms.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let vr = self.v_rescale.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        k + v + ks + vs + vp + vn + vr
    }

    /// Whether the detached handle carries no data (all tensors were `None`).
    pub fn is_empty(&self) -> bool {
        self.keys.is_none()
    }
}

impl std::fmt::Debug for DetachedRotatingKVCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DetachedRotatingKVCache")
            .field("max_size", &self.max_size)
            .field("offset", &self.offset)
            .field("idx", &self.idx)
            .field("step", &self.step)
            .field("mode", &self.mode)
            .field("has_keys", &self.keys.is_some())
            .field("has_values", &self.values.is_some())
            .field("has_v_packed", &self.v_packed.is_some())
            .field("has_v_norms", &self.v_norms.is_some())
            .field("has_v_rescale", &self.v_rescale.is_some())
            .field("turbo_seed", &self.turbo_seed)
            .finish()
    }
}

impl RotatingKVCache {
    /// Move the underlying MLX buffers out of this rotating cache into a
    /// [`DetachedRotatingKVCache`] handle.
    ///
    /// After this call the source `RotatingKVCache` is empty
    /// (`is_empty() == true`, `offset == 0`, `idx == 0`) but retains its
    /// `max_size`, quantization mode, step, and `turbo_seed` so it can be
    /// reused for a new sequence. The returned handle carries the original
    /// tensors unchanged — including `v_packed` / `v_norms` for `Turbo4Asym`
    /// — so adopt is a zero-copy operation.
    ///
    /// Used by: sliding-window prompt prefix cache detach/adopt (B9; dense counterpart is [`KVCache::clone_handle`]).
    pub fn clone_handle(&mut self) -> DetachedRotatingKVCache {
        let handle = DetachedRotatingKVCache {
            keys: self.keys.take(),
            values: self.values.take(),
            max_size: self.max_size,
            offset: std::mem::replace(&mut self.offset, 0),
            idx: std::mem::replace(&mut self.idx, 0),
            step: self.step,
            mode: self.mode,
            key_scales: self.key_scales.take(),
            val_scales: self.val_scales.take(),
            v_packed: self.v_packed.take(),
            v_norms: self.v_norms.take(),
            v_rescale: self.v_rescale.take(),
            turbo_seed: self.turbo_seed,
        };
        // Mirror `KVCache::clone_handle` (LOW-1): clear cached
        // turbo_params on the source so the next quantize call rebuilds them
        // from scratch (slot may be reused with a different head_dim).
        self.turbo_params = None;
        handle
    }

    /// Re-install a previously detached rotating cache into this slot.
    ///
    /// Inverse of [`RotatingKVCache::clone_handle`]. The receiver must be
    /// empty (`is_empty() == true`) so no live buffer is silently dropped;
    /// callers that need to overwrite a populated cache should construct a
    /// fresh `RotatingKVCache::new_with_mode_and_seed` and install into that.
    ///
    /// Block alignment: the adopted state is bit-identical to the source's,
    /// including `idx`. Because per-token Turbo4 quantization is independent
    /// across slots, no alignment-recovery work is needed at install time.
    pub fn install_detached(&mut self, detached: DetachedRotatingKVCache) -> Result<(), String> {
        if !self.is_empty() {
            return Err("RotatingKVCache::install_detached: target cache is not empty".into());
        }
        self.keys = detached.keys;
        self.values = detached.values;
        self.max_size = detached.max_size;
        self.offset = detached.offset;
        self.idx = detached.idx;
        self.step = detached.step;
        self.mode = detached.mode;
        self.key_scales = detached.key_scales;
        self.val_scales = detached.val_scales;
        self.v_packed = detached.v_packed;
        self.v_norms = detached.v_norms;
        self.v_rescale = detached.v_rescale;
        self.turbo_seed = detached.turbo_seed;
        // Pre-build turbo_params from v_packed shape if available so the
        // first dequantize-only consumer doesn't need to wait for an update
        // call (mirrors `KVCache::install_detached`).
        if self.mode == KVCacheMode::Turbo4Asym {
            if let Some(ref vp) = self.v_packed {
                let shape = ffi::array_shape(vp);
                if shape.len() == 4 && shape[3] > 0 {
                    let head_dim = (shape[3] as u32) * 2;
                    self.turbo_params = Some(super::turbo::TurboQuantParams::new(
                        head_dim,
                        self.turbo_seed,
                    ));
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DetachedCacheSet
// ---------------------------------------------------------------------------

/// Inert snapshot of a whole sequence's per-layer KV caches.
///
/// Produced by [`CachePool::detach`] and consumed by [`CachePool::adopt`].
/// Only the dense backend is supported; paged sequences produce `None` on
/// detach.
pub struct DetachedCacheSet {
    /// Per-layer detached caches, one per model layer.
    pub caches: Vec<DetachedKVCache>,
    /// Logical backend tag (always `DenseKvCache` in this API surface).
    pub backend: SequenceStateBackend,
    /// Prompt length as recorded on the originating [`SequenceCacheSet`].
    pub prompt_len: usize,
    /// Last known decode offset at detach time.
    pub current_offset: i32,
    /// Timestamp of the originating allocation (preserved across handoffs).
    pub created_at: Instant,
    /// Timestamp of the most recent detach.
    pub detached_at: Instant,
    /// Original sequence id this cache set was last installed under (for
    /// logging / observability; the adopt path always assigns a fresh id).
    pub origin_seq_id: SequenceId,
}

impl DetachedCacheSet {
    /// Summed tensor bytes across all layer caches.
    pub fn nbytes(&self) -> usize {
        self.caches.iter().map(|c| c.nbytes()).sum()
    }

    /// Number of layer caches carried by this set.
    pub fn num_layers(&self) -> usize {
        self.caches.len()
    }

    /// Logical token length of the first non-empty layer (or 0).
    ///
    /// All transformer layers share a common prefix length by construction,
    /// so the first layer's `offset` is a faithful summary of the set.
    pub fn seq_len(&self) -> i32 {
        self.caches.first().map(|c| c.offset).unwrap_or(0)
    }

    /// Whether every layer carries the same logical token length.
    pub fn has_consistent_seq_len(&self) -> bool {
        let Some(first) = self.caches.first() else {
            return false;
        };
        self.caches.iter().all(|cache| cache.offset == first.offset)
    }

    /// Shrink every per-layer detached cache to exactly `new_len` tokens.
    ///
    /// Walks each [`DetachedKVCache`] and calls [`DetachedKVCache::trim_to`]
    /// in lockstep, then updates the set-wide `prompt_len` and
    /// `current_offset` so accounting downstream of `CachePool::adopt`
    /// matches the new logical length. An empty cache set or `new_len`
    /// equal to the existing length is a no-op.
    ///
    /// The intended caller is the scheduler's `try_adopt_cached_prefix` when
    /// an APC lookup returns a block-aligned `matched_len`
    /// shorter than the candidate entry's full token length — i.e. the
    /// request and the cached entry agree on the first N blocks but diverge
    /// at block N+1. Truncating the detached set to `matched_len` before
    /// adoption gives the model worker a KV cache whose logical length
    /// exactly matches the prefix the prefill loop will skip, so the next
    /// `update_and_fetch` writes at the correct seq-len offset.
    ///
    /// Returns `Err(_)` if any layer's `trim_to` rejects the request (e.g.
    /// `new_len > seq_len`). On error, layers already truncated stay at
    /// the new length — the caller should drop the set rather than retry.
    ///
    /// Used by: [`crate::cache::CachePool::adopt`] callers that need
    /// per-block partial adoption.
    #[must_use = "truncate_to returns Err on partial failure; on error some layers are already trimmed and the set must be dropped, not retried"]
    pub fn truncate_to(&mut self, new_len: i32) -> Result<(), String> {
        if new_len < 0 {
            return Err(format!(
                "DetachedCacheSet::truncate_to: new_len must be non-negative, got {new_len}"
            ));
        }
        if self.caches.is_empty() {
            // Empty set is a degenerate value; truncation is vacuously OK.
            self.current_offset = new_len;
            self.prompt_len = (new_len as usize).min(self.prompt_len);
            return Ok(());
        }
        // Sanity: every layer must agree on the pre-truncate seq length so
        // we never silently divergent-trim a set produced by a different
        // sequence layout.
        let head = self.caches[0].offset;
        debug_assert!(
            self.has_consistent_seq_len(),
            "DetachedCacheSet::truncate_to: layers disagree on seq_len: {:?}",
            self.caches.iter().map(|c| c.offset).collect::<Vec<_>>()
        );
        if new_len == head {
            return Ok(());
        }
        if new_len > head {
            return Err(format!(
                "DetachedCacheSet::truncate_to: new_len ({new_len}) exceeds current seq_len ({head})"
            ));
        }

        for (i, cache) in self.caches.iter_mut().enumerate() {
            cache.trim_to(new_len).map_err(|e| {
                format!("DetachedCacheSet::truncate_to: layer {i} trim_to failed: {e}")
            })?;
        }

        self.current_offset = new_len;
        // `prompt_len` is the originating prompt size at detach time. After a
        // partial adoption the request shares only `new_len` of those tokens,
        // so clamp to that — never grow beyond.
        let new_prompt = new_len.max(0) as usize;
        if new_prompt < self.prompt_len {
            self.prompt_len = new_prompt;
        }
        Ok(())
    }
}

impl std::fmt::Debug for DetachedCacheSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DetachedCacheSet")
            .field("backend", &self.backend)
            .field("num_layers", &self.num_layers())
            .field("seq_len", &self.seq_len())
            .field("prompt_len", &self.prompt_len)
            .field("current_offset", &self.current_offset)
            .field("origin_seq_id", &self.origin_seq_id)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Parking (in-flight holding)
// ---------------------------------------------------------------------------

/// Opaque handle returned by [`CachePool::park_detached`].
///
/// Parking is an optional escape hatch: a scheduler can hand a
/// [`DetachedCacheSet`] back to the pool for the duration of a cross-request
/// lookup so that [`CachePool::memory_usage_bytes`] keeps accounting for the
/// tensors that the pool logically still holds in-flight. The normal
/// `detach` → store in external cache → `adopt` flow does not require
/// parking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DetachedHandle(u64);

impl DetachedHandle {
    /// Raw numeric representation of this handle, useful for logging and
    /// metric labels.
    pub fn as_u64(self) -> u64 {
        self.0
    }

    /// Construct a handle from a raw id. Provided for cross-module builders
    /// (e.g. the paged detach surface in [`super::paged_detach`]) that mint
    /// handles out of the same `CachePool::next_id` space.
    pub(super) fn from_raw(id: u64) -> Self {
        Self(id)
    }
}

impl std::fmt::Display for DetachedHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "detached-{}", self.0)
    }
}

/// Internal map of parked detached cache sets, keyed by handle. This map is
/// attached to [`CachePool`] as `detached: DetachedMap` via the
/// `detached` field declared in the parent module. The map stores a
/// [`super::paged_detach::ParkedCache`] enum so dense and paged variants
/// share the same handle namespace.
pub(super) type DetachedMap = HashMap<DetachedHandle, super::paged_detach::ParkedCache>;

// ---------------------------------------------------------------------------
// CachePool extensions
// ---------------------------------------------------------------------------

impl CachePool {
    /// Remove `seq_id` from the active set and return its per-layer caches as
    /// a `DetachedCacheSet` without freeing the MLX buffers.
    ///
    /// Returns `None` if:
    /// * `seq_id` is not currently active, or
    /// * the sequence uses the paged backend (paged detach is's responsibility — this method deliberately rejects it), or
    /// * any dense layer has front-trimmed state that the detached format
    ///   cannot represent without its monotonic `live_start`.
    ///
    /// The caller is responsible for re-homing the detached set, either by
    /// passing it to [`CachePool::adopt`] or by parking it via
    /// [`CachePool::park_detached`]. Dropping the returned set releases the
    /// underlying MLX memory normally.
    ///
    /// Used by: prompt prefix cache store, scheduler request-boundary
    /// handoff.
    pub fn detach(&mut self, seq_id: SequenceId) -> Option<DetachedCacheSet> {
        // Peek first so we can refuse non-dense backends without destructive
        // side effects.
        {
            let sequence = self.active.get(&seq_id)?;
            if sequence.backend != SequenceStateBackend::DenseKvCache {
                return None;
            }
            if sequence.caches.iter().any(KVCache::is_front_trimmed) {
                return None;
            }
        }

        let mut sequence = self.active.remove(&seq_id)?;
        let detached_caches: Vec<DetachedKVCache> = sequence
            .caches
            .iter_mut()
            .map(|cache| cache.clone_handle())
            .collect();

        Some(DetachedCacheSet {
            caches: detached_caches,
            backend: sequence.backend,
            prompt_len: sequence.prompt_len,
            current_offset: sequence.current_offset,
            created_at: sequence.created_at,
            detached_at: Instant::now(),
            origin_seq_id: sequence.seq_id,
        })
    }

    /// Install a previously-detached cache set under a fresh `SequenceId`.
    ///
    /// Capacity is checked against `max_sequences` before allocation. On
    /// success the model's
    /// [`prepare_sequence_state`](crate::generate::LanguageModel::prepare_sequence_state)
    /// hook is invoked with the new id so any per-model sidecar maps
    /// (mixed-cache models, quantized sidecars, etc.) are initialized
    /// consistently with a freshly allocated sequence.
    ///
    /// Only `DenseKvCache` sets are supported; attempting to adopt a paged
    /// set returns an error and the original set is dropped (its tensors
    /// freed) to avoid leaks. Use [`CachePool::adopt_preserving`] when the
    /// caller wants the set back on failure.
    ///
    /// Used by: prompt prefix cache re-entry, scheduler fast-path
    /// when a new request reuses an existing prefix.
    pub fn adopt(
        &mut self,
        model: &dyn crate::generate::LanguageModel,
        detached: DetachedCacheSet,
    ) -> Result<SequenceId, String> {
        self.adopt_preserving(model, detached)
            .map_err(|(err, _)| err)
    }

    /// Like [`CachePool::adopt`] but returns the original [`DetachedCacheSet`]
    /// back to the caller on failure so it can be retried or routed
    /// elsewhere.
    pub fn adopt_preserving(
        &mut self,
        model: &dyn crate::generate::LanguageModel,
        detached: DetachedCacheSet,
    ) -> Result<SequenceId, (String, DetachedCacheSet)> {
        if detached.backend != SequenceStateBackend::DenseKvCache {
            return Err((
                format!(
                    "CachePool::adopt: backend {:?} is not supported (paged adopt is tracked)",
                    detached.backend
                ),
                detached,
            ));
        }
        if self.active.len() >= self.max_sequences {
            return Err((
                format!(
                    "CachePool::adopt: max capacity ({}) reached, cannot adopt new sequence",
                    self.max_sequences
                ),
                detached,
            ));
        }

        let id = SequenceId::from_raw(self.next_id.fetch_add(1, Ordering::Relaxed));

        // Reconstruct the live per-layer caches from the detached handles.
        // `KVCache::install_detached` demands an empty target, which
        // `KVCache::new()` trivially satisfies.
        let DetachedCacheSet {
            caches,
            backend,
            prompt_len,
            current_offset,
            created_at,
            detached_at: _,
            origin_seq_id: _,
        } = detached;

        let mut live: Vec<KVCache> = Vec::with_capacity(caches.len());
        for detached_cache in caches {
            let mut cache = KVCache::new_with_mode(detached_cache.mode);
            cache
                .install_detached(detached_cache)
                .expect("freshly constructed KVCache is empty");
            live.push(cache);
        }

        let mut entry = SequenceCacheSet::dense_external(id, live);
        // Preserve the originating metadata across the handoff so scheduler
        // stats and reuse bookkeeping stay coherent.
        entry.backend = backend;
        entry.prompt_len = prompt_len;
        entry.current_offset = current_offset;
        entry.created_at = created_at;
        self.active.insert(id, entry);

        // Hook in the model-side sidecar state for the new id, matching the
        // normal `allocate` -> `prepare_sequence_state` sequencing that the
        // batch scheduler uses today.
        model.prepare_sequence_state(id);

        Ok(id)
    }

    /// Park a detached set inside the pool so its bytes remain visible to
    /// [`CachePool::memory_usage_bytes`].
    ///
    /// Returns an opaque [`DetachedHandle`] that can later be consumed by
    /// [`CachePool::take_parked`] or [`CachePool::adopt_parked`]. Parked
    /// caches do **not** count toward `active_count()` and do not consume
    /// an `allocate()` slot — they only contribute to memory accounting.
    pub fn park_detached(&mut self, detached: DetachedCacheSet) -> DetachedHandle {
        let handle = DetachedHandle(self.next_id.fetch_add(1, Ordering::Relaxed));
        self.detached
            .insert(handle, super::paged_detach::ParkedCache::Dense(detached));
        handle
    }

    /// Retrieve a previously parked dense set, leaving the pool.
    ///
    /// Returns `None` if the handle was never parked, already taken, or
    /// points to a paged set (use [`CachePool::take_parked_paged`] for
    /// paged sets).
    pub fn take_parked(&mut self, handle: DetachedHandle) -> Option<DetachedCacheSet> {
        match self.detached.remove(&handle) {
            Some(super::paged_detach::ParkedCache::Dense(set)) => Some(set),
            Some(other) => {
                // Wrong variant — put it back so the caller can dispatch to
                // the paged-side take_parked.
                self.detached.insert(handle, other);
                None
            }
            None => None,
        }
    }

    /// Read-only peek at a parked dense set (for inspection / metrics).
    ///
    /// Returns `None` if the handle points to a paged set.
    pub fn peek_parked(&self, handle: DetachedHandle) -> Option<&DetachedCacheSet> {
        match self.detached.get(&handle) {
            Some(super::paged_detach::ParkedCache::Dense(set)) => Some(set),
            _ => None,
        }
    }

    /// Convenience: consume a parked handle and re-adopt it in one call.
    pub fn adopt_parked(
        &mut self,
        model: &dyn crate::generate::LanguageModel,
        handle: DetachedHandle,
    ) -> Result<SequenceId, String> {
        let detached = self
            .take_parked(handle)
            .ok_or_else(|| format!("CachePool::adopt_parked: unknown handle {handle}"))?;
        self.adopt(model, detached)
    }

    /// Number of currently parked detached sets (dense and paged combined).
    pub fn parked_count(&self) -> usize {
        self.detached.len()
    }

    /// Summed bytes across all parked detached sets (dense and paged).
    pub fn parked_bytes(&self) -> usize {
        self.detached.values().map(|d| d.nbytes()).sum()
    }
}

// Tests live in the companion `detach_tests.rs` so this file stays at a
// comfortable implementation-only size (see `docs/code-guidelines.md`).
#[cfg(test)]
#[path = "detach_tests.rs"]
mod tests;
