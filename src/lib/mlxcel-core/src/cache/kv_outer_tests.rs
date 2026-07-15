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

//! Unit tests for the KV-outer block-sparse decode attention kernel.
//!
//! Tests the two-phase Metal kernel directly via FFI — no model loading
//! required. Reference output is computed using standard MLX matmul + softmax.

use cxx::UniquePtr;

use crate::dtype;
use crate::ffi;
use crate::ffi::MlxArray;
use crate::ops;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic LCG-driven f32 tensor on `[shape]`.
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

/// Build a per-query-head inverted index for decode (L=1).
///
/// All query heads in a GQA group share the same selected blocks.
/// `compact_indices` maps each selection position to its compact slot in k_blocked.
/// Returns (inv_index_arr, counts_arr) with shape [b, hq, n_selected, max_qpb].
fn build_decode_inv_index(
    b: i32,
    hq: i32,
    hkv: i32,
    n_selected: i32,
    top_k: i32,
    offset: i32,
    compact_indices: &[usize],
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let n_rep = hq / hkv;
    let max_qpb = 1i32;
    let mut counts = vec![0i32; (hq * n_selected) as usize];
    let mut inv_index = vec![0i32; (hq * n_selected * max_qpb) as usize];

    for qh in 0..hq as usize {
        for j in 0..top_k as usize {
            let compact = compact_indices[j];
            if compact < n_selected as usize {
                let idx = qh * n_selected as usize + compact;
                counts[idx] = 1;
                inv_index[idx * max_qpb as usize] = offset;
            }
        }
    }

    let inv_arr = ffi::astype(
        &ffi::from_slice_f32(
            &inv_index.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &[b, hq, n_selected, max_qpb],
        ),
        dtype::INT32,
    );
    let counts_arr = ffi::astype(
        &ffi::from_slice_f32(
            &counts.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &[b, hq, n_selected],
        ),
        dtype::INT32,
    );
    (inv_arr, counts_arr)
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

fn rms_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "rms_diff: length mismatch");
    let s: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| ((x - y) as f64).powi(2))
        .sum();
    ((s / a.len() as f64).sqrt()) as f32
}

/// Reference block-sparse attention using standard MLX ops.
///
/// For each query head h:
///   1. Determine the KV head: kv_head = h / n_rep
///   2. Gather the selected blocks from K/V into [1, 1, top_k * block_size, dim]
///   3. Compute scores = Q · Kᵀ · scale
///   4. Apply causal mask (q_pos = offset, so only k_pos <= offset is valid)
///   5. Softmax → weights
///   6. Output = weights · V
///
/// `selected` is the per-KV-head block indices: [Hkv * top_k].
/// All query heads sharing a KV head use the same selection.
fn reference_block_sparse_attention(
    q: &MlxArray,          // [1, Hq, 1, Dim]
    k_blocked: &MlxArray,  // [1, Hkv, n_selected, BlockSize, Dim] fp16
    v_blocked: &MlxArray,  // [1, Hkv, n_selected, BlockSize, Dim] fp16
    selected: &[i32],      // [Hkv * top_k] absolute block indices per KV head
    n_rep: i32,
    top_k: i32,
    block_size: i32,
    _num_key_blocks: i32,
    offset: i32,
    scale: f32,
) -> UniquePtr<MlxArray> {
    let hq = ffi::array_shape(q)[1];
    let dim = ffi::array_shape(q)[3];
    let n_selected = ffi::array_shape(k_blocked)[2];
    let kv_tokens = n_selected * block_size;

    // For each query head, compute attention over its selected blocks.
    let mut head_outputs: Vec<UniquePtr<MlxArray>> = Vec::new();

    for h in 0..hq {
        let kv_head = h / n_rep;

        // Get this KV head's selected block indices.
        let head_sel = &selected[(kv_head * top_k) as usize..((kv_head + 1) * top_k) as usize];

        // Gather K and V for this head's selected blocks.
        // k_blocked is [1, Hkv, n_selected, BlockSize, Dim].
        // We need to slice out kv_head's portion and reshape to [1, 1, kv_tokens, Dim].
        let k_slice = ffi::slice(
            k_blocked,
            &[0, kv_head, 0, 0, 0],
            &[1, kv_head + 1, n_selected, block_size, dim],
        );
        let v_slice = ffi::slice(
            v_blocked,
            &[0, kv_head, 0, 0, 0],
            &[1, kv_head + 1, n_selected, block_size, dim],
        );
        let k_flat = ffi::reshape(&k_slice, &[1, 1, kv_tokens, dim]);
        let v_flat = ffi::reshape(&v_slice, &[1, 1, kv_tokens, dim]);

        // Extract this head's query: q[:, h:h+1, :, :]
        let q_h = ffi::slice(q, &[0, h, 0, 0], &[1, h + 1, 1, dim]);

        // Scores = Q · Kᵀ · scale  →  [1, 1, 1, kv_tokens]
        let k_t = ffi::transpose_axes(&k_flat, &[0, 1, 3, 2]);
        let scores = ffi::matmul(&q_h, &k_t);
        let scores = ops::multiply_scalar(&scores, scale);

        // Causal mask: only k positions <= offset are valid.
        // For each selected block, compute the absolute start position.
        // A block b covers positions [b*block_size, (b+1)*block_size).
        // Valid if start_pos <= offset.
        let mut mask_data = vec![f32::NEG_INFINITY; kv_tokens as usize];
        for (i, &block_idx) in head_sel.iter().enumerate() {
            let block_start = block_idx * block_size;
            for t in 0..block_size {
                let abs_pos = block_start + t;
                if abs_pos <= offset {
                    mask_data[i * block_size as usize + t as usize] = 0.0;
                }
            }
        }
        let mask = ffi::from_slice_f32(&mask_data, &[1, 1, 1, kv_tokens]);
        let scores = ffi::add(&scores, &mask);

        // Softmax → weights  [1, 1, 1, kv_tokens]
        let weights = ffi::softmax(&scores, -1);

        // Output = weights · V  [1, 1, 1, Dim]
        let out = ffi::matmul(&weights, &v_flat);
        head_outputs.push(out);
    }

    // Concatenate all heads: [1, Hq, 1, Dim]
    let mut result = ffi::astype(&head_outputs[0], dtype::FLOAT32);
    for i in 1..hq as usize {
        let ptrs: [*const MlxArray; 2] = [&*result as *const _, &*head_outputs[i] as *const _];
        result = unsafe { ffi::concatenate(&ptrs, 1) };
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Basic smoke test: KV-outer kernel produces correct shape and finite values.
#[test]
fn kv_outer_smoke_test_shape_and_finite() {
    let b = 1i32;
    let hq = 4i32;
    let hkv = 2i32;
    let dim = 32i32;
    let block_size = 4i32;
    let top_k = 3i32;
    let total_kv_len = 12i32; // 3 blocks
    let num_key_blocks = total_kv_len / block_size;
    let _n_rep = hq / hkv;
    let offset = 11i32; // last position
    let scale = 1.0 / (dim as f32).sqrt();

    // Synthetic data.
    let q = synth_tensor(&[b, hq, 1, dim], 100);
    let k = synth_tensor(&[b, hkv, total_kv_len, dim], 200);
    let v = synth_tensor(&[b, hkv, total_kv_len, dim], 300);

    // Selection: each head selects top_k blocks (0-indexed).
    // Head 0,1 (kv_head 0): blocks [0, 1, 2]
    // Head 2,3 (kv_head 1): blocks [0, 1, 2]
    let selected = ffi::from_slice_f32(
        &[0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        &[b, hkv, 1, top_k],
    );
    let _selected = ffi::astype(&selected, dtype::INT32);

    // Build blocked KV: [b, hkv, num_key_blocks, block_size, dim]
    // Slice to n_selected blocks (compact).
    let n_selected = top_k;
    let k_all = ffi::reshape(&k, &[b, hkv, num_key_blocks, block_size, dim]);
    let v_all = ffi::reshape(&v, &[b, hkv, num_key_blocks, block_size, dim]);
    let k_blocked = ffi::slice(&k_all, &[0, 0, 0, 0, 0], &[b, hkv, n_selected, block_size, dim]);
    let v_blocked = ffi::slice(&v_all, &[0, 0, 0, 0, 0], &[b, hkv, n_selected, block_size, dim]);
    ffi::eval(&k_blocked);
    ffi::eval(&v_blocked);

    // Per-query-head inverted index.
    let compact_indices: Vec<usize> = (0..top_k as usize).collect();
    let (inv_index_arr, counts_arr) = build_decode_inv_index(
        b, hq, hkv, n_selected, top_k, offset, &compact_indices,
    );

    // Block IDs: compact → absolute (same in this test).
    let block_ids: Vec<i32> = (0..n_selected).collect();
    let block_ids_arr = ffi::from_slice_f32(
        &block_ids.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        &[n_selected],
    );
    let block_ids_arr = ffi::astype(&block_ids_arr, dtype::INT32);

    // Run Phase 1.
    let mut partials = ffi::turbo_minimax_sparse_kv_outer_sdpa(
        &q,
        &k_blocked,
        &v_blocked,
        &inv_index_arr,
        &counts_arr,
        &block_ids_arr,
        scale,
        block_size,
        1,  // max_qpb = 1 for decode
    );
    let partial_m = ffi::kv_outer_partials_take_m(partials.pin_mut());
    let partial_l = ffi::kv_outer_partials_take_l(partials.pin_mut());
    let partial_v = ffi::kv_outer_partials_take_v(partials.pin_mut());

    // Verify partial shapes.
    let m_shape = ffi::array_shape(&partial_m);
    let l_shape = ffi::array_shape(&partial_l);
    let v_shape = ffi::array_shape(&partial_v);
    assert_eq!(m_shape, vec![b, hq, 1, n_selected], "partial_m shape");
    assert_eq!(l_shape, vec![b, hq, 1, n_selected], "partial_l shape");
    assert_eq!(v_shape, vec![b, hq, 1, n_selected, dim], "partial_v shape");

    // Run Phase 2.
    let out = ffi::turbo_minimax_sparse_kv_outer_reduction(
        &q,
        &partial_m,
        &partial_l,
        &partial_v,
        n_selected,
    );

    // Verify output shape and finiteness.
    let out_shape = ffi::array_shape(&out);
    assert_eq!(out_shape, vec![b, hq, 1, dim], "output shape");

    let out_flat = flatten_fp32(&out);
    for (i, &v) in out_flat.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] = {v} is not finite");
    }
}

/// Accuracy test: KV-outer kernel matches reference block-sparse attention.
#[test]
fn kv_outer_matches_reference_attention() {
    let b = 1i32;
    let hq = 4i32;
    let hkv = 2i32;
    let dim = 32i32;
    let block_size = 4i32;
    let top_k = 3i32;
    let total_kv_len = 16i32; // 4 blocks
    let num_key_blocks = total_kv_len / block_size;
    let n_rep = hq / hkv;
    let offset = 15i32; // last position
    let scale = 1.0 / (dim as f32).sqrt();

    // Synthetic data with controlled magnitudes.
    let q = synth_tensor(&[b, hq, 1, dim], 42);
    let k = synth_tensor(&[b, hkv, total_kv_len, dim], 43);
    let v = synth_tensor(&[b, hkv, total_kv_len, dim], 44);

    // Selection: each head selects blocks [0, 2, 3].
    let sel_blocks = [0i32, 2, 3];
    let mut selected_data = Vec::new();
    for _ in 0..hq {
        for &blk in &sel_blocks {
            selected_data.push(blk as f32);
        }
    }
    let _selected = ffi::astype(
        &ffi::from_slice_f32(&selected_data, &[b, hkv, 1, top_k]),
        dtype::INT32,
    );

    // Build blocked KV: [b, hkv, num_key_blocks, block_size, dim]
    // Slice to selected blocks only (compact).
    let n_selected = top_k;
    let k_all = ffi::reshape(&k, &[b, hkv, num_key_blocks, block_size, dim]);
    let v_all = ffi::reshape(&v, &[b, hkv, num_key_blocks, block_size, dim]);
    let k_blocked = ffi::slice(&k_all, &[0, 0, 0, 0, 0], &[b, hkv, n_selected, block_size, dim]);
    let v_blocked = ffi::slice(&v_all, &[0, 0, 0, 0, 0], &[b, hkv, n_selected, block_size, dim]);
    ffi::eval(&k_blocked);
    ffi::eval(&v_blocked);

    // Per-query-head inverted index. Compact indices are [0, 1, 2] (positions in the slice).
    let compact_indices: Vec<usize> = (0..top_k as usize).collect();
    let (inv_index_arr, counts_arr) = build_decode_inv_index(
        b, hq, hkv, n_selected, top_k, offset, &compact_indices,
    );

    // Block IDs: compact index → absolute block ID. Selected blocks are [0, 2, 3].
    let block_ids_arr = ffi::astype(
        &ffi::from_slice_f32(
            &sel_blocks.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &[n_selected],
        ),
        dtype::INT32,
    );

    // Run KV-outer kernel (Phase 1 + Phase 2).
    let mut partials = ffi::turbo_minimax_sparse_kv_outer_sdpa(
        &q,
        &k_blocked,
        &v_blocked,
        &inv_index_arr,
        &counts_arr,
        &block_ids_arr,
        scale,
        block_size,
        1,  // max_qpb = 1 for decode
    );
    let partial_m = ffi::kv_outer_partials_take_m(partials.pin_mut());
    let partial_l = ffi::kv_outer_partials_take_l(partials.pin_mut());
    let partial_v = ffi::kv_outer_partials_take_v(partials.pin_mut());
    let out_kernel = ffi::turbo_minimax_sparse_kv_outer_reduction(
        &q,
        &partial_m,
        &partial_l,
        &partial_v,
        n_selected,
    );

    // Compute reference attention.
    // selected is per-KV-head: [Hkv * top_k]. Both KV heads select [0, 2, 3].
    let mut sel_raw = Vec::new();
    for _ in 0..hkv {
        for &blk in &sel_blocks {
            sel_raw.push(blk);
        }
    }
    let out_ref = reference_block_sparse_attention(
        &q,
        &k_blocked,
        &v_blocked,
        &sel_raw,
        n_rep,
        top_k,
        block_size,
        num_key_blocks,
        offset,
        scale,
    );

    // Compare.
    let flat_kernel = flatten_fp32(&out_kernel);
    let flat_ref = flatten_fp32(&out_ref);
    let rms = rms_diff(&flat_kernel, &flat_ref);

    eprintln!("kv_outer_matches_reference: RMS = {rms:.6e}");
    assert!(
        rms < 0.1,
        "KV-outer kernel RMS {rms:.6e} exceeds threshold 0.1"
    );
}

/// GQA diagnostic: check if kernel produces non-zero output with hkv=1.
#[test]
fn kv_outer_gqa_nonzero_debug() {
    let b = 1i32;
    let hq = 4i32;
    let hkv = 1i32;
    let dim = 32i32;
    let block_size = 4i32;
    let top_k = 2i32;
    let total_kv_len = 8i32;
    let num_key_blocks = total_kv_len / block_size;
    let offset = 7i32;
    let scale = 1.0 / (dim as f32).sqrt();

    let q = synth_tensor(&[b, hq, 1, dim], 500);
    let k = synth_tensor(&[b, hkv, total_kv_len, dim], 600);
    let v = synth_tensor(&[b, hkv, total_kv_len, dim], 700);

    let k_blocked = ffi::reshape(&k, &[b, hkv, num_key_blocks, block_size, dim]);
    let v_blocked = ffi::reshape(&v, &[b, hkv, num_key_blocks, block_size, dim]);

    let n_selected = top_k;
    let compact_indices: Vec<usize> = (0..top_k as usize).collect();
    let (inv_index_arr, counts_arr) = build_decode_inv_index(
        b, hq, hkv, n_selected, top_k, offset, &compact_indices,
    );

    let block_ids: Vec<i32> = (0..n_selected).collect();
    let block_ids_arr = ffi::astype(
        &ffi::from_slice_f32(
            &block_ids.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &[n_selected],
        ),
        dtype::INT32,
    );

    let mut partials = ffi::turbo_minimax_sparse_kv_outer_sdpa(
        &q, &k_blocked, &v_blocked, &inv_index_arr, &counts_arr, &block_ids_arr,
        scale, block_size, 1,  // max_qpb = 1 for decode
    );
    let partial_m = ffi::kv_outer_partials_take_m(partials.pin_mut());
    let partial_l = ffi::kv_outer_partials_take_l(partials.pin_mut());
    let partial_v = ffi::kv_outer_partials_take_v(partials.pin_mut());

    let m_flat = flatten_fp32(&partial_m);
    let l_flat = flatten_fp32(&partial_l);
    let v_flat = flatten_fp32(&partial_v);

    eprintln!("hkv=1 partial_m = {:?}", &m_flat);
    eprintln!("hkv=1 partial_l = {:?}", &l_flat);
    eprintln!("hkv=1 partial_v first20 = {:?}", &v_flat[..20]);

    // At minimum, partials should be non-zero.
    let has_nonzero = m_flat.iter().any(|&x| x != 0.0);
    assert!(has_nonzero, "partial_m should have non-zero values");
}

/// GQA test: different query heads sharing a KV head produce different outputs.
#[test]
fn kv_outer_gqa_heads_differ() {
    let b = 1i32;
    let hq = 4i32;
    let hkv = 1i32; // all 4 query heads share 1 KV head
    let dim = 32i32;
    let block_size = 4i32;
    let top_k = 2i32;
    let total_kv_len = 8i32; // 2 blocks
    let num_key_blocks = total_kv_len / block_size;
    let _n_rep = hq / hkv;
    let offset = 7i32;
    let scale = 1.0 / (dim as f32).sqrt();

    // Q has different values per head.
    let q = synth_tensor(&[b, hq, 1, dim], 500);
    let k = synth_tensor(&[b, hkv, total_kv_len, dim], 600);
    let v = synth_tensor(&[b, hkv, total_kv_len, dim], 700);

    // All heads select blocks [0, 1].
    let _selected = ffi::astype(
        &ffi::from_slice_f32(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], &[b, hkv, 1, top_k]),
        dtype::INT32,
    );

    let k_blocked = ffi::reshape(&k, &[b, hkv, num_key_blocks, block_size, dim]);
    let v_blocked = ffi::reshape(&v, &[b, hkv, num_key_blocks, block_size, dim]);

    let n_selected = top_k;
    let compact_indices: Vec<usize> = (0..top_k as usize).collect();
    let (inv_index_arr, counts_arr) = build_decode_inv_index(
        b, hq, hkv, n_selected, top_k, offset, &compact_indices,
    );

    let block_ids: Vec<i32> = (0..n_selected).collect();
    let block_ids_arr = ffi::astype(
        &ffi::from_slice_f32(
            &block_ids.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &[n_selected],
        ),
        dtype::INT32,
    );

    let mut partials = ffi::turbo_minimax_sparse_kv_outer_sdpa(
        &q, &k_blocked, &v_blocked, &inv_index_arr, &counts_arr, &block_ids_arr,
        scale, block_size, 1,  // max_qpb = 1 for decode
    );
    let partial_m = ffi::kv_outer_partials_take_m(partials.pin_mut());
    let partial_l = ffi::kv_outer_partials_take_l(partials.pin_mut());
    let partial_v = ffi::kv_outer_partials_take_v(partials.pin_mut());

    let out = ffi::turbo_minimax_sparse_kv_outer_reduction(
        &q, &partial_m, &partial_l, &partial_v, n_selected,
    );

    let flat = flatten_fp32(&out);

    // Each head should produce a different output (different Q, same K/V).
    let head_size = dim as usize;
    for h1 in 0..hq as usize {
        for h2 in (h1 + 1)..hq as usize {
            let a = &flat[h1 * head_size..(h1 + 1) * head_size];
            let b_sl = &flat[h2 * head_size..(h2 + 1) * head_size];
            let rms = rms_diff(a, b_sl);
            assert!(
                rms > 1e-6,
                "GQA heads {h1} and {h2} should differ but RMS = {rms:.6e}"
            );
        }
    }
}

/// Causal mask test: tokens beyond offset must not contribute.
#[test]
fn kv_outer_causal_mask_blocks_future_tokens() {
    let b = 1i32;
    let hq = 1i32;
    let hkv = 1i32;
    let dim = 32i32;
    let block_size = 4i32;
    let top_k = 2i32;
    let total_kv_len = 8i32; // 2 blocks: [0..3] and [4..7]
    let num_key_blocks = total_kv_len / block_size;
    let _n_rep = hq / hkv;
    let scale = 1.0 / (dim as f32).sqrt();

    // Query at offset 3 — only block 0 [0..3] is fully visible.
    // Block 1 [4..7] has all tokens beyond offset → should be masked.
    let offset = 3i32;

    let q = synth_tensor(&[b, hq, 1, dim], 800);
    let k = synth_tensor(&[b, hkv, total_kv_len, dim], 801);
    let v = synth_tensor(&[b, hkv, total_kv_len, dim], 802);

    // Select both blocks.
    let _selected = ffi::astype(
        &ffi::from_slice_f32(&[0.0, 1.0], &[b, hkv, 1, top_k]),
        dtype::INT32,
    );

    let k_blocked = ffi::reshape(&k, &[b, hkv, num_key_blocks, block_size, dim]);
    let v_blocked = ffi::reshape(&v, &[b, hkv, num_key_blocks, block_size, dim]);

    let n_selected = top_k;
    let compact_indices: Vec<usize> = (0..top_k as usize).collect();
    let (inv_index_arr, counts_arr) = build_decode_inv_index(
        b, hq, hkv, n_selected, top_k, offset, &compact_indices,
    );

    let block_ids: Vec<i32> = (0..n_selected).collect();
    let block_ids_arr = ffi::astype(
        &ffi::from_slice_f32(
            &block_ids.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &[n_selected],
        ),
        dtype::INT32,
    );

    let mut partials = ffi::turbo_minimax_sparse_kv_outer_sdpa(
        &q, &k_blocked, &v_blocked, &inv_index_arr, &counts_arr, &block_ids_arr,
        scale, block_size, 1,  // max_qpb = 1 for decode
    );
    let partial_m = ffi::kv_outer_partials_take_m(partials.pin_mut());
    let partial_l = ffi::kv_outer_partials_take_l(partials.pin_mut());
    let partial_v = ffi::kv_outer_partials_take_v(partials.pin_mut());
    let out_with_both = ffi::turbo_minimax_sparse_kv_outer_reduction(
        &q, &partial_m, &partial_l, &partial_v, n_selected,
    );

    // Now run with only block 0 (the visible one).
    let _selected_0only = ffi::astype(
        &ffi::from_slice_f32(&[0.0], &[b, hkv, 1, 1]),
        dtype::INT32,
    );
    let n_sel_0 = 1i32;
    let inv_0 = ffi::astype(
        &ffi::from_slice_f32(&[offset as f32], &[b, hkv, n_sel_0, 1]),
        dtype::INT32,
    );
    let cnt_0 = ffi::astype(
        &ffi::from_slice_f32(&[1.0], &[b, hkv, n_sel_0]),
        dtype::INT32,
    );
    let block_ids_0 = ffi::astype(
        &ffi::from_slice_f32(&[0.0], &[n_sel_0]),
        dtype::INT32,
    );

    let mut partials_0 = ffi::turbo_minimax_sparse_kv_outer_sdpa(
        &q, &k_blocked, &v_blocked, &inv_0, &cnt_0, &block_ids_0,
        scale, block_size, 1,  // max_qpb = 1 for decode
    );
    let pm0 = ffi::kv_outer_partials_take_m(partials_0.pin_mut());
    let pl0 = ffi::kv_outer_partials_take_l(partials_0.pin_mut());
    let pv0 = ffi::kv_outer_partials_take_v(partials_0.pin_mut());
    let out_with_0only = ffi::turbo_minimax_sparse_kv_outer_reduction(
        &q, &pm0, &pl0, &pv0, n_sel_0,
    );

    // The outputs should be nearly identical because block 1 is fully masked.
    let flat_both = flatten_fp32(&out_with_both);
    let flat_0 = flatten_fp32(&out_with_0only);
    let rms = rms_diff(&flat_both, &flat_0);

    eprintln!("kv_outer_causal_mask: RMS between 2-block and 1-block = {rms:.6e}");
    assert!(
        rms < 0.15,
        "Causal mask: future block should contribute nothing, RMS = {rms:.6e}"
    );
}

/// Large-context test: closer to real server parameters.
/// 8192 tokens (64 blocks), top_k=32 — exercises larger inverted index.
#[test]
fn kv_outer_large_context() {
    let b = 1i32;
    let hq = 64i32;
    let hkv = 4i32;
    let dim = 128i32;
    let block_size = 128i32;
    let top_k = 32i32; // real top_k
    let total_kv_len = 8192i32; // 64 blocks
    let num_key_blocks = total_kv_len / block_size;
    let offset = 8191i32;
    let scale = 1.0 / (dim as f32).sqrt();
    let n_rep = hq / hkv;

    let q = synth_tensor(&[b, hq, 1, dim], 100);
    let k = synth_tensor(&[b, hkv, total_kv_len, dim], 200);
    let v = synth_tensor(&[b, hkv, total_kv_len, dim], 300);

    // Selection: all heads select blocks 0..top_k.
    let sel_blocks_per_head: Vec<i32> = (0..top_k).collect();
    let mut selected_data = Vec::new();
    for _ in 0..hkv {
        for &blk in &sel_blocks_per_head {
            selected_data.push(blk as f32);
        }
    }
    let _selected = ffi::astype(
        &ffi::from_slice_f32(&selected_data, &[b, hkv, 1, top_k]),
        dtype::INT32,
    );

    // Fetch only selected blocks (compact), not all blocks.
    // The real dispatch does: fetch_msa_blocks(&union) → reshape to [b, hkv, n_selected, bs, d].
    // Here we slice the first n_selected blocks from the full tensor.
    // IMPORTANT: eval the slice to force materialization — MLX lazy evaluation
    // means k_shape[2] in the C++ launcher reads the pre-slice shape otherwise.
    let n_selected = top_k;
    let k_all = ffi::reshape(&k, &[b, hkv, num_key_blocks, block_size, dim]);
    let v_all = ffi::reshape(&v, &[b, hkv, num_key_blocks, block_size, dim]);
    let k_blocked = ffi::slice(&k_all, &[0, 0, 0, 0, 0], &[b, hkv, n_selected, block_size, dim]);
    let v_blocked = ffi::slice(&v_all, &[0, 0, 0, 0, 0], &[b, hkv, n_selected, block_size, dim]);
    ffi::eval(&k_blocked);
    ffi::eval(&v_blocked);

    let compact_indices: Vec<usize> = (0..top_k as usize).collect();
    let (inv_index_arr, counts_arr) = build_decode_inv_index(
        b, hq, hkv, n_selected, top_k, offset, &compact_indices,
    );
    let block_ids_arr = ffi::astype(
        &ffi::from_slice_f32(
            &sel_blocks_per_head.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &[n_selected],
        ),
        dtype::INT32,
    );

    // Phase 1 + Phase 2.
    let mut partials = ffi::turbo_minimax_sparse_kv_outer_sdpa(
        &q, &k_blocked, &v_blocked, &inv_index_arr, &counts_arr, &block_ids_arr,
        scale, block_size, 1,  // max_qpb = 1 for decode
    );
    let partial_m = ffi::kv_outer_partials_take_m(partials.pin_mut());
    let partial_l = ffi::kv_outer_partials_take_l(partials.pin_mut());
    let partial_v = ffi::kv_outer_partials_take_v(partials.pin_mut());

    // Check partials are finite.
    let m_flat = flatten_fp32(&partial_m);
    let l_flat = flatten_fp32(&partial_l);
    let m_shape = ffi::array_shape(&partial_m);
    eprintln!("partial_m shape: {m_shape:?}, len: {}", m_flat.len());
    // Find first non-finite value for debugging.
    for (i, &v) in m_flat.iter().enumerate() {
        if !v.is_finite() {
            eprintln!("partial_m[{i}] = {v}");
            break;
        }
    }
    let all_m_finite = m_flat.iter().all(|&x| x.is_finite());
    let all_l_finite = l_flat.iter().all(|&x| x.is_finite());
    assert!(all_m_finite, "partial_m should be finite at large context");
    assert!(all_l_finite, "partial_l should be finite at large context");

    let out = ffi::turbo_minimax_sparse_kv_outer_reduction(
        &q, &partial_m, &partial_l, &partial_v, n_selected,
    );

    let out_flat = flatten_fp32(&out);
    let all_finite = out_flat.iter().all(|&x| x.is_finite());
    let has_nonzero = out_flat.iter().any(|&x| x != 0.0);
    assert!(all_finite, "large-context output should be finite");
    assert!(has_nonzero, "large-context output should be non-zero");

    // Compare head 0 against reference.
    let sel_full: Vec<i32> = sel_blocks_per_head.iter().cycle().take((hkv * top_k) as usize).cloned().collect();
    let out_ref = reference_block_sparse_attention(
        &q, &k_blocked, &v_blocked, &sel_full,
        n_rep, top_k, block_size, num_key_blocks, offset, scale,
    );
    let ref_flat = flatten_fp32(&out_ref);
    let head_size = dim as usize;
    let rms = rms_diff(&out_flat[..head_size], &ref_flat[..head_size]);
    eprintln!("large_context: head 0 RMS = {rms:.6e}");
    assert!(rms < 0.1, "large-context RMS {rms:.6e} exceeds 0.1");
}
/// This matches the actual MiniMax-M3 server configuration and exercises
/// multiple chunk iterations (chunk_size=16, 8 iterations per block).
#[test]
fn kv_outer_real_dimensions() {
    let b = 1i32;
    let hq = 64i32;
    let hkv = 4i32;
    let dim = 128i32;
    let block_size = 128i32;
    let top_k = 16i32; // real top_k for M3
    let total_kv_len = 2048i32; // 16 blocks
    let num_key_blocks = total_kv_len / block_size;
    let offset = 2047i32; // last position
    let scale = 1.0 / (dim as f32).sqrt();
    let n_rep = hq / hkv;

    let q = synth_tensor(&[b, hq, 1, dim], 42);
    let k = synth_tensor(&[b, hkv, total_kv_len, dim], 43);
    let v = synth_tensor(&[b, hkv, total_kv_len, dim], 44);

    // Selection: all heads select the same top_k blocks.
    let sel_blocks_per_head: Vec<i32> = (0..top_k).collect();
    let mut selected_data = Vec::new();
    for _ in 0..hkv {
        for &blk in &sel_blocks_per_head {
            selected_data.push(blk as f32);
        }
    }
    let _selected = ffi::astype(
        &ffi::from_slice_f32(&selected_data, &[b, hkv, 1, top_k]),
        dtype::INT32,
    );

    // Fetch only selected blocks (compact), not all blocks.
    let n_selected = top_k;
    let k_all = ffi::reshape(&k, &[b, hkv, num_key_blocks, block_size, dim]);
    let v_all = ffi::reshape(&v, &[b, hkv, num_key_blocks, block_size, dim]);
    let k_blocked = ffi::slice(&k_all, &[0, 0, 0, 0, 0], &[b, hkv, n_selected, block_size, dim]);
    let v_blocked = ffi::slice(&v_all, &[0, 0, 0, 0, 0], &[b, hkv, n_selected, block_size, dim]);
    ffi::eval(&k_blocked);
    ffi::eval(&v_blocked);

    let compact_indices: Vec<usize> = (0..top_k as usize).collect();
    let (inv_index_arr, counts_arr) = build_decode_inv_index(
        b, hq, hkv, n_selected, top_k, offset, &compact_indices,
    );

    let block_ids_arr = ffi::astype(
        &ffi::from_slice_f32(
            &sel_blocks_per_head.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &[n_selected],
        ),
        dtype::INT32,
    );

    // Run Phase 1 + Phase 2.
    let mut partials = ffi::turbo_minimax_sparse_kv_outer_sdpa(
        &q, &k_blocked, &v_blocked, &inv_index_arr, &counts_arr, &block_ids_arr,
        scale, block_size, 1,  // max_qpb = 1 for decode
    );
    let partial_m = ffi::kv_outer_partials_take_m(partials.pin_mut());
    let partial_l = ffi::kv_outer_partials_take_l(partials.pin_mut());
    let partial_v = ffi::kv_outer_partials_take_v(partials.pin_mut());

    let out = ffi::turbo_minimax_sparse_kv_outer_reduction(
        &q, &partial_m, &partial_l, &partial_v, n_selected,
    );

    // Verify output is finite and non-zero.
    let out_flat = flatten_fp32(&out);
    assert_eq!(out_flat.len(), (hq * dim) as usize);
    let has_nonzero = out_flat.iter().any(|&x| x != 0.0);
    assert!(has_nonzero, "real-dimension output should be non-zero");
    let all_finite = out_flat.iter().all(|&x| x.is_finite());
    assert!(all_finite, "real-dimension output should be finite");

    // Compare against reference for first few heads.
    let k_blocked_ref = ffi::reshape(&k, &[b, hkv, num_key_blocks, block_size, dim]);
    let v_blocked_ref = ffi::reshape(&v, &[b, hkv, num_key_blocks, block_size, dim]);
    // Reference needs [Hkv * top_k] entries — all heads select the same blocks.
    let sel_full: Vec<i32> = sel_blocks_per_head.iter().cycle().take((hkv * top_k) as usize).cloned().collect();
    let out_ref = reference_block_sparse_attention(
        &q, &k_blocked_ref, &v_blocked_ref, &sel_full,
        n_rep, top_k, block_size, num_key_blocks, offset, scale,
    );
    let ref_flat = flatten_fp32(&out_ref);

    // Compare first head only (full comparison is expensive).
    let head_size = dim as usize;
    let rms = rms_diff(&out_flat[..head_size], &ref_flat[..head_size]);
    eprintln!("real_dimensions: head 0 RMS = {rms:.6e}");
    assert!(
        rms < 0.1,
        "real-dimension RMS {rms:.6e} exceeds 0.1"
    );
}

// ---------------------------------------------------------------------------
// Focused tests for all-heads-per-block kernel structure (no model load)
// ---------------------------------------------------------------------------

/// Hq > NumSims: exercises heads_per_simd > 1.
/// With NumSims=4 and Hq=8, each SIMD group processes 2 query heads.
/// This tests that the per-head accumulators don't interfere.
#[test]
fn kv_outer_heads_per_simd_gt1() {
    let b = 1i32;
    let hq = 8i32;
    let hkv = 2i32; // nrep = 4
    let dim = 32i32;
    let block_size = 4i32;
    let top_k = 2i32;
    let total_kv_len = 8i32;
    let num_key_blocks = total_kv_len / block_size;
    let n_rep = hq / hkv;
    let offset = 7i32;
    let scale = 1.0 / (dim as f32).sqrt();

    let q = synth_tensor(&[b, hq, 1, dim], 42);
    let k = synth_tensor(&[b, hkv, total_kv_len, dim], 43);
    let v = synth_tensor(&[b, hkv, total_kv_len, dim], 44);

    let k_blocked = ffi::reshape(&k, &[b, hkv, num_key_blocks, block_size, dim]);
    let v_blocked = ffi::reshape(&v, &[b, hkv, num_key_blocks, block_size, dim]);
    ffi::eval(&k_blocked);
    ffi::eval(&v_blocked);

    let n_selected = top_k;
    let compact_indices: Vec<usize> = (0..top_k as usize).collect();
    let (inv_index_arr, counts_arr) = build_decode_inv_index(
        b, hq, hkv, n_selected, top_k, offset, &compact_indices,
    );
    let block_ids_arr = ffi::astype(
        &ffi::from_slice_f32(&[0.0, 1.0], &[n_selected]),
        dtype::INT32,
    );

    let mut partials = ffi::turbo_minimax_sparse_kv_outer_sdpa(
        &q, &k_blocked, &v_blocked, &inv_index_arr, &counts_arr, &block_ids_arr,
        scale, block_size, 1,
    );
    let partial_m = ffi::kv_outer_partials_take_m(partials.pin_mut());
    let partial_l = ffi::kv_outer_partials_take_l(partials.pin_mut());
    let partial_v = ffi::kv_outer_partials_take_v(partials.pin_mut());

    let out = ffi::turbo_minimax_sparse_kv_outer_reduction(
        &q, &partial_m, &partial_l, &partial_v, n_selected,
    );

    let out_flat = flatten_fp32(&out);
    assert_eq!(out_flat.len(), (hq * dim) as usize);
    let all_finite = out_flat.iter().all(|&x| x.is_finite());
    let has_nonzero = out_flat.iter().any(|&x| x != 0.0);
    assert!(all_finite, "heads_per_simd>1: output should be finite");
    assert!(has_nonzero, "heads_per_simd>1: output should be non-zero");

    // Verify against reference.
    let sel_full: Vec<i32> = (0..hkv).flat_map(|_| vec![0i32, 1]).collect();
    let out_ref = reference_block_sparse_attention(
        &q, &k_blocked, &v_blocked, &sel_full,
        n_rep, top_k, block_size, num_key_blocks, offset, scale,
    );
    let ref_flat = flatten_fp32(&out_ref);
    let rms = rms_diff(&out_flat, &ref_flat);
    eprintln!("heads_per_simd_gt1: RMS = {rms:.6e}");
    assert!(rms < 0.15, "heads_per_simd>1 RMS {rms:.6e} exceeds 0.15");
}

/// Non-uniform head distribution: Hq=5, NumSims=4 → [2, 2, 1, 0].
/// The 4th SIMD group gets 0 heads. Tests that the zero-head path
/// writes neutral partials without crashing.
#[test]
fn kv_outer_non_uniform_head_distribution() {
    let b = 1i32;
    let hq = 5i32;
    let hkv = 1i32; // nrep = 5
    let dim = 32i32;
    let block_size = 4i32;
    let top_k = 2i32;
    let total_kv_len = 8i32;
    let num_key_blocks = total_kv_len / block_size;
    let n_rep = hq / hkv;
    let offset = 7i32;
    let scale = 1.0 / (dim as f32).sqrt();

    let q = synth_tensor(&[b, hq, 1, dim], 100);
    let k = synth_tensor(&[b, hkv, total_kv_len, dim], 200);
    let v = synth_tensor(&[b, hkv, total_kv_len, dim], 300);

    let k_blocked = ffi::reshape(&k, &[b, hkv, num_key_blocks, block_size, dim]);
    let v_blocked = ffi::reshape(&v, &[b, hkv, num_key_blocks, block_size, dim]);
    ffi::eval(&k_blocked);
    ffi::eval(&v_blocked);

    let n_selected = top_k;
    let compact_indices: Vec<usize> = (0..top_k as usize).collect();
    let (inv_index_arr, counts_arr) = build_decode_inv_index(
        b, hq, hkv, n_selected, top_k, offset, &compact_indices,
    );
    let block_ids_arr = ffi::astype(
        &ffi::from_slice_f32(&[0.0, 1.0], &[n_selected]),
        dtype::INT32,
    );

    let mut partials = ffi::turbo_minimax_sparse_kv_outer_sdpa(
        &q, &k_blocked, &v_blocked, &inv_index_arr, &counts_arr, &block_ids_arr,
        scale, block_size, 1,
    );
    let partial_m = ffi::kv_outer_partials_take_m(partials.pin_mut());
    let partial_l = ffi::kv_outer_partials_take_l(partials.pin_mut());
    let partial_v = ffi::kv_outer_partials_take_v(partials.pin_mut());

    let out = ffi::turbo_minimax_sparse_kv_outer_reduction(
        &q, &partial_m, &partial_l, &partial_v, n_selected,
    );

    let out_flat = flatten_fp32(&out);
    assert_eq!(out_flat.len(), (hq * dim) as usize);
    let all_finite = out_flat.iter().all(|&x| x.is_finite());
    let has_nonzero = out_flat.iter().any(|&x| x != 0.0);
    assert!(all_finite, "non-uniform: output should be finite");
    assert!(has_nonzero, "non-uniform: output should be non-zero");

    // Verify against reference.
    let sel_full: Vec<i32> = (0..hkv).flat_map(|_| vec![0i32, 1]).collect();
    let out_ref = reference_block_sparse_attention(
        &q, &k_blocked, &v_blocked, &sel_full,
        n_rep, top_k, block_size, num_key_blocks, offset, scale,
    );
    let ref_flat = flatten_fp32(&out_ref);
    let rms = rms_diff(&out_flat, &ref_flat);
    eprintln!("non_uniform: RMS = {rms:.6e}");
    assert!(rms < 0.1, "non-uniform RMS {rms:.6e} exceeds 0.1");
}

/// M3-realistic GQA ratio: Hq=64, Hkv=4, nrep=16.
/// This matches the actual MiniMax-M3 configuration.
/// With NumSims=4, each SIMD group processes 16 query heads.
#[test]
fn kv_outer_m3_gqa_ratio() {
    let b = 1i32;
    let hq = 64i32;
    let hkv = 4i32;
    let dim = 128i32;
    let block_size = 128i32;
    let top_k = 4i32; // small k for speed
    let total_kv_len = 512i32; // 4 blocks
    let num_key_blocks = total_kv_len / block_size;
    let n_rep = hq / hkv;
    let offset = 511i32;
    let scale = 1.0 / (dim as f32).sqrt();

    let q = synth_tensor(&[b, hq, 1, dim], 42);
    let k = synth_tensor(&[b, hkv, total_kv_len, dim], 43);
    let v = synth_tensor(&[b, hkv, total_kv_len, dim], 44);

    let k_blocked = ffi::reshape(&k, &[b, hkv, num_key_blocks, block_size, dim]);
    let v_blocked = ffi::reshape(&v, &[b, hkv, num_key_blocks, block_size, dim]);
    ffi::eval(&k_blocked);
    ffi::eval(&v_blocked);

    let n_selected = top_k;
    let compact_indices: Vec<usize> = (0..top_k as usize).collect();
    let (inv_index_arr, counts_arr) = build_decode_inv_index(
        b, hq, hkv, n_selected, top_k, offset, &compact_indices,
    );
    let block_ids_arr = ffi::astype(
        &ffi::from_slice_f32(
            &(0..top_k).map(|x| x as f32).collect::<Vec<_>>(),
            &[n_selected],
        ),
        dtype::INT32,
    );

    let mut partials = ffi::turbo_minimax_sparse_kv_outer_sdpa(
        &q, &k_blocked, &v_blocked, &inv_index_arr, &counts_arr, &block_ids_arr,
        scale, block_size, 1,
    );
    let partial_m = ffi::kv_outer_partials_take_m(partials.pin_mut());
    let partial_l = ffi::kv_outer_partials_take_l(partials.pin_mut());
    let partial_v = ffi::kv_outer_partials_take_v(partials.pin_mut());

    let out = ffi::turbo_minimax_sparse_kv_outer_reduction(
        &q, &partial_m, &partial_l, &partial_v, n_selected,
    );

    let out_flat = flatten_fp32(&out);
    assert_eq!(out_flat.len(), (hq * dim) as usize);
    let all_finite = out_flat.iter().all(|&x| x.is_finite());
    let has_nonzero = out_flat.iter().any(|&x| x != 0.0);
    assert!(all_finite, "M3 GQA: output should be finite");
    assert!(has_nonzero, "M3 GQA: output should be non-zero");

    // Verify against reference (first 4 heads only for speed).
    let sel_full: Vec<i32> = (0..hkv).flat_map(|_| (0..top_k).collect::<Vec<_>>()).collect();
    let out_ref = reference_block_sparse_attention(
        &q, &k_blocked, &v_blocked, &sel_full,
        n_rep, top_k, block_size, num_key_blocks, offset, scale,
    );
    let ref_flat = flatten_fp32(&out_ref);
    let head_size = dim as usize;
    // Compare all heads — the kernel should match the reference for every head.
    let rms = rms_diff(&out_flat, &ref_flat);
    eprintln!("m3_gqa_ratio: RMS = {rms:.6e}");
    assert!(rms < 0.1, "M3 GQA RMS {rms:.6e} exceeds 0.1");
}

/// Causal mask with partial visibility: query at offset 129 with block_size=128.
/// Block 0 (positions 0..127) is fully visible.
/// Block 1 (positions 128..255) has only position 128 visible.
/// The kernel must mask positions 129..255 to -inf.
#[test]
fn kv_outer_causal_mask_partial_block() {
    let b = 1i32;
    let hq = 1i32;
    let hkv = 1i32;
    let dim = 32i32;
    let block_size = 128i32;
    let top_k = 2i32;
    let total_kv_len = 256i32; // 2 blocks
    let num_key_blocks = total_kv_len / block_size;
    let n_rep = hq / hkv;
    let offset = 129i32; // block 0 fully visible, block 1 only pos 128
    let scale = 1.0 / (dim as f32).sqrt();

    let q = synth_tensor(&[b, hq, 1, dim], 42);
    let k = synth_tensor(&[b, hkv, total_kv_len, dim], 43);
    let v = synth_tensor(&[b, hkv, total_kv_len, dim], 44);

    let k_blocked = ffi::reshape(&k, &[b, hkv, num_key_blocks, block_size, dim]);
    let v_blocked = ffi::reshape(&v, &[b, hkv, num_key_blocks, block_size, dim]);
    ffi::eval(&k_blocked);
    ffi::eval(&v_blocked);

    let n_selected = top_k;
    let compact_indices: Vec<usize> = (0..top_k as usize).collect();
    let (inv_index_arr, counts_arr) = build_decode_inv_index(
        b, hq, hkv, n_selected, top_k, offset, &compact_indices,
    );
    let block_ids_arr = ffi::astype(
        &ffi::from_slice_f32(&[0.0, 1.0], &[n_selected]),
        dtype::INT32,
    );

    let mut partials = ffi::turbo_minimax_sparse_kv_outer_sdpa(
        &q, &k_blocked, &v_blocked, &inv_index_arr, &counts_arr, &block_ids_arr,
        scale, block_size, 1,
    );
    let partial_m = ffi::kv_outer_partials_take_m(partials.pin_mut());
    let partial_l = ffi::kv_outer_partials_take_l(partials.pin_mut());
    let partial_v = ffi::kv_outer_partials_take_v(partials.pin_mut());

    let out = ffi::turbo_minimax_sparse_kv_outer_reduction(
        &q, &partial_m, &partial_l, &partial_v, n_selected,
    );

    let out_flat = flatten_fp32(&out);
    let all_finite = out_flat.iter().all(|&x| x.is_finite());
    assert!(all_finite, "partial-block causal: output should be finite");

    // Verify against reference.
    let sel_full: Vec<i32> = vec![0, 1];
    let out_ref = reference_block_sparse_attention(
        &q, &k_blocked, &v_blocked, &sel_full,
        n_rep, top_k, block_size, num_key_blocks, offset, scale,
    );
    let ref_flat = flatten_fp32(&out_ref);
    let rms = rms_diff(&out_flat, &ref_flat);
    eprintln!("causal_mask_partial: RMS = {rms:.6e}");
    assert!(rms < 0.3, "partial-block causal RMS {rms:.6e} exceeds 0.3");
}

/// Zero-query blocks: disjoint selections between GQA groups (production-valid).
///
/// Hq=4, Hkv=2, G=2. Group 0 (heads 0-1) selects block 0 only.
/// Group 1 (heads 2-3) selects block 1 only. Union = [0, 1].
///
/// Tile (kv_head=0, block=1) is inactive: no head in group 0 selected it.
/// Tile (kv_head=1, block=0) is inactive: no head in group 1 selected it.
///
/// Uses zero K and distinct V per (KV head, block) to catch leakage.
#[test]
fn kv_outer_zero_query_blocks() {
    let b = 1i32;
    let hq = 4i32;
    let hkv = 2i32; // nrep = 2
    let dim = 32i32;
    let block_size = 4i32;
    let total_kv_len = 8i32; // 2 blocks
    let num_key_blocks = total_kv_len / block_size;
    let n_rep = hq / hkv;
    let offset = 7i32;
    let scale = 1.0 / (dim as f32).sqrt();

    let q = synth_tensor(&[b, hq, 1, dim], 100);
    // Zero K → uniform softmax weights.
    let k_data = vec![0.0f32; (b * hkv * total_kv_len * dim) as usize];
    let k = ffi::from_slice_f32(&k_data, &[b, hkv, total_kv_len, dim]);

    // V: distinct per (KV head, block).
    let block_elems = (block_size * dim) as usize;
    let mut v_data = vec![0.0f32; (b * hkv * total_kv_len * dim) as usize];
    // kv_head 0, block 0 → 1.0
    for i in 0..block_elems { v_data[i] = 1.0; }
    // kv_head 0, block 1 → 0.25
    for i in block_elems..2 * block_elems { v_data[i] = 0.25; }
    // kv_head 1, block 0 → 0.75
    for i in 2 * block_elems..3 * block_elems { v_data[i] = 0.75; }
    // kv_head 1, block 1 → 0.0 (stays)
    let v = ffi::from_slice_f32(&v_data, &[b, hkv, total_kv_len, dim]);

    let k_blocked = ffi::reshape(&k, &[b, hkv, num_key_blocks, block_size, dim]);
    let v_blocked = ffi::reshape(&v, &[b, hkv, num_key_blocks, block_size, dim]);
    ffi::eval(&k_blocked);
    ffi::eval(&v_blocked);

    // Build per-query-head inverted index with disjoint GQA-group selections.
    // Group 0 (heads 0-1): selects block 0 (compact index 0)
    // Group 1 (heads 2-3): selects block 1 (compact index 1)
    let n_selected = 2i32;
    let max_qpb = 1i32;
    let mut counts = vec![0i32; (hq * n_selected) as usize];
    let mut inv_index = vec![0i32; (hq * n_selected * max_qpb) as usize];

    // Group 0: heads 0-1 → block 0
    for h in 0..(n_rep as usize) {
        counts[h * n_selected as usize + 0] = 1;
        inv_index[(h * n_selected as usize + 0) * max_qpb as usize] = offset;
    }
    // Group 1: heads 2-3 → block 1
    for h in (n_rep as usize)..(hq as usize) {
        counts[h * n_selected as usize + 1] = 1;
        inv_index[(h * n_selected as usize + 1) * max_qpb as usize] = offset;
    }

    let inv_arr = ffi::astype(
        &ffi::from_slice_f32(
            &inv_index.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &[b, hq, n_selected, max_qpb],
        ),
        dtype::INT32,
    );
    let counts_arr = ffi::astype(
        &ffi::from_slice_f32(
            &counts.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &[b, hq, n_selected],
        ),
        dtype::INT32,
    );
    let block_ids_arr = ffi::astype(
        &ffi::from_slice_f32(&[0.0, 1.0], &[n_selected]),
        dtype::INT32,
    );

    let mut partials = ffi::turbo_minimax_sparse_kv_outer_sdpa(
        &q, &k_blocked, &v_blocked, &inv_arr, &counts_arr, &block_ids_arr,
        scale, block_size, 1,
    );
    let partial_m = ffi::kv_outer_partials_take_m(partials.pin_mut());
    let partial_l = ffi::kv_outer_partials_take_l(partials.pin_mut());
    let partial_v = ffi::kv_outer_partials_take_v(partials.pin_mut());

    // --- Assert inactive tiles have neutral partials ---
    let m_flat = flatten_fp32(&partial_m);
    let l_flat = flatten_fp32(&partial_l);

    // Tile (kv_head=0, block=1): inactive for group 0.
    // In the output layout, this corresponds to heads 0-1, block 1.
    for h in 0..(n_rep as usize) {
        let idx = h * n_selected as usize + 1;
        assert!(
            m_flat[idx] == f32::NEG_INFINITY,
            "Inactive tile (kv0, blk1): partial_m should be -INF for head {h}, got {}",
            m_flat[idx]
        );
        assert!(
            l_flat[idx] == 0.0,
            "Inactive tile (kv0, blk1): partial_l should be 0 for head {h}, got {}",
            l_flat[idx]
        );
    }
    // Tile (kv_head=1, block=0): inactive for group 1.
    for h in (n_rep as usize)..(hq as usize) {
        let idx = h * n_selected as usize + 0;
        assert!(
            m_flat[idx] == f32::NEG_INFINITY,
            "Inactive tile (kv1, blk0): partial_m should be -INF for head {h}, got {}",
            m_flat[idx]
        );
        assert!(
            l_flat[idx] == 0.0,
            "Inactive tile (kv1, blk0): partial_l should be 0 for head {h}, got {}",
            l_flat[idx]
        );
    }

    // --- Assert final output ---
    let out = ffi::turbo_minimax_sparse_kv_outer_reduction(
        &q, &partial_m, &partial_l, &partial_v, n_selected,
    );

    let out_flat = flatten_fp32(&out);
    let head_size = dim as usize;

    // Group 0 (heads 0-1): attends block 0, V=1.0 → output ≈ 1.0.
    for h in 0..(n_rep as usize) {
        let head_out = &out_flat[h * head_size..(h + 1) * head_size];
        let mean: f32 = head_out.iter().sum::<f32>() / head_size as f32;
        assert!(
            (mean - 1.0).abs() < 0.01,
            "Group 0 head {h}: mean={mean:.4}, expected 1.0"
        );
    }
    // Group 1 (heads 2-3): attends block 1, V=0.0 → output ≈ 0.0.
    for h in (n_rep as usize)..(hq as usize) {
        let head_out = &out_flat[h * head_size..(h + 1) * head_size];
        let mean: f32 = head_out.iter().sum::<f32>() / head_size as f32;
        assert!(
            mean.abs() < 0.01,
            "Group 1 head {h}: mean={mean:.4}, expected 0.0"
        );
    }
}

/// Adversarial test: distinct KV-head V signatures catch cross-KV mixing.
///
/// Each KV head's V is filled with a distinct constant (kv_head 0 → 1.0,
/// kv_head 1 → 0.0). If the kernel mixes KV heads, heads in group 0 would
/// show contamination from 0.0 and heads in group 1 from 1.0.
///
/// With softmax weights summing to 1, expected output is exactly the V constant.
#[test]
fn kv_outer_adversarial_distinct_kv_signatures() {
    let b = 1i32;
    let hq = 8i32;
    let hkv = 2i32; // nrep = 4
    let dim = 32i32;
    let block_size = 4i32;
    let top_k = 2i32;
    let total_kv_len = 8i32; // 2 blocks
    let num_key_blocks = total_kv_len / block_size;
    let n_rep = hq / hkv;
    let offset = 7i32;
    let scale = 1.0 / (dim as f32).sqrt();

    let q = synth_tensor(&[b, hq, 1, dim], 42);
    let k = synth_tensor(&[b, hkv, total_kv_len, dim], 43);

    // V: kv_head 0 → all 1.0, kv_head 1 → all 0.0.
    let v_tokens_per_head = (total_kv_len * dim) as usize;
    let mut v_data = vec![0.0f32; (b * hkv * total_kv_len * dim) as usize];
    // Fill kv_head 0's tokens with 1.0.
    for i in 0..v_tokens_per_head {
        v_data[i] = 1.0;
    }
    // kv_head 1 stays 0.0.
    let v = ffi::from_slice_f32(&v_data, &[b, hkv, total_kv_len, dim]);

    let k_blocked = ffi::reshape(&k, &[b, hkv, num_key_blocks, block_size, dim]);
    let v_blocked = ffi::reshape(&v, &[b, hkv, num_key_blocks, block_size, dim]);
    ffi::eval(&k_blocked);
    ffi::eval(&v_blocked);

    let n_selected = top_k;
    let compact_indices: Vec<usize> = (0..top_k as usize).collect();
    let (inv_index_arr, counts_arr) = build_decode_inv_index(
        b, hq, hkv, n_selected, top_k, offset, &compact_indices,
    );
    let block_ids_arr = ffi::astype(
        &ffi::from_slice_f32(&[0.0, 1.0], &[n_selected]),
        dtype::INT32,
    );

    let mut partials = ffi::turbo_minimax_sparse_kv_outer_sdpa(
        &q, &k_blocked, &v_blocked, &inv_index_arr, &counts_arr, &block_ids_arr,
        scale, block_size, 1,
    );
    let partial_m = ffi::kv_outer_partials_take_m(partials.pin_mut());
    let partial_l = ffi::kv_outer_partials_take_l(partials.pin_mut());
    let partial_v = ffi::kv_outer_partials_take_v(partials.pin_mut());

    let out = ffi::turbo_minimax_sparse_kv_outer_reduction(
        &q, &partial_m, &partial_l, &partial_v, n_selected,
    );

    let out_flat = flatten_fp32(&out);
    let head_size = dim as usize;

    // Heads 0-3 (kv_head 0): output should be ≈ 1.0 (V=1.0, softmax weights sum to 1).
    for h in 0..(n_rep as usize) {
        let head_out = &out_flat[h * head_size..(h + 1) * head_size];
        let mean: f32 = head_out.iter().sum::<f32>() / head_size as f32;
        assert!(
            (mean - 1.0).abs() < 0.01,
            "GQA group 0 head {h}: mean={mean:.6}, expected ≈ 1.0 (cross-KV mixing?)"
        );
        // All values should be close to 1.0.
        let max_dev = head_out.iter().map(|&v| (v - 1.0).abs()).fold(0.0f32, f32::max);
        assert!(
            max_dev < 0.05,
            "GQA group 0 head {h}: max deviation from 1.0 = {max_dev:.6} (cross-KV mixing?)"
        );
    }

    // Heads 4-7 (kv_head 1): output should be ≈ 0.0 (V=0.0).
    for h in (n_rep as usize)..(hq as usize) {
        let head_out = &out_flat[h * head_size..(h + 1) * head_size];
        let mean: f32 = head_out.iter().sum::<f32>() / head_size as f32;
        assert!(
            mean.abs() < 0.01,
            "GQA group 1 head {h}: mean={mean:.6}, expected ≈ 0.0 (cross-KV mixing?)"
        );
        let max_dev = head_out.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            max_dev < 0.05,
            "GQA group 1 head {h}: max deviation from 0.0 = {max_dev:.6} (cross-KV mixing?)"
        );
    }
}

/// Production-valid zero-count test: disjoint block selections between GQA groups.
///
/// Hq=8, Hkv=2, G=4. Group 0 selects block 0 only, group 1 selects block 1 only.
/// Union = [0, 1].
///
/// Uses zero K (all scores = 0 → softmax weights = uniform) and distinct V per
/// (KV head, block) pair:
///   kv_head 0, block 0: V = 1.0   (selected by group 0)
///   kv_head 0, block 1: V = 0.25  (NOT selected by group 0)
///   kv_head 1, block 0: V = 0.75  (NOT selected by group 1)
///   kv_head 1, block 1: V = 0.0   (selected by group 1)
///
/// If query_counts is ignored and both blocks attend:
///   group 0 output = (1.0 + 0.25) / 2 = 0.625  (WRONG, should be 1.0)
///   group 1 output = (0.75 + 0.0) / 2 = 0.375  (WRONG, should be 0.0)
///
/// Also asserts inactive phase-1 partials directly:
///   inactive partial_m == -INFINITY
///   inactive partial_l == 0
///   inactive partial_v == 0
#[test]
fn kv_outer_disjoint_selections_no_leakage() {
    let b = 1i32;
    let hq = 8i32;
    let hkv = 2i32; // nrep = 4
    let dim = 32i32;
    let block_size = 4i32;
    let top_k = 1i32; // each group selects exactly 1 block
    let total_kv_len = 8i32; // 2 blocks
    let num_key_blocks = total_kv_len / block_size;
    let n_rep = hq / hkv;
    let offset = 7i32;
    let scale = 1.0 / (dim as f32).sqrt();

    let q = synth_tensor(&[b, hq, 1, dim], 42);
    // Zero K → all scores = 0 → softmax weights = uniform over visible tokens.
    let k_data = vec![0.0f32; (b * hkv * total_kv_len * dim) as usize];
    let k = ffi::from_slice_f32(&k_data, &[b, hkv, total_kv_len, dim]);

    // V: distinct per (kv_head, block) pair.
    // kv_head 0, block 0 → 1.0;  kv_head 0, block 1 → 0.25
    // kv_head 1, block 0 → 0.75; kv_head 1, block 1 → 0.0
    let block_elems = (block_size * dim) as usize;
    let mut v_data = vec![0.0f32; (b * hkv * total_kv_len * dim) as usize];
    // kv_head 0, block 0 (positions 0..block_elems-1)
    for i in 0..block_elems { v_data[i] = 1.0; }
    // kv_head 0, block 1 (positions block_elems..2*block_elems-1)
    for i in block_elems..2 * block_elems { v_data[i] = 0.25; }
    // kv_head 1, block 0 (positions 2*block_elems..3*block_elems-1)
    for i in 2 * block_elems..3 * block_elems { v_data[i] = 0.75; }
    // kv_head 1, block 1 stays 0.0
    let v = ffi::from_slice_f32(&v_data, &[b, hkv, total_kv_len, dim]);

    // Blocked layout: [B, Hkv, num_key_blocks, BlockSize, Dim]
    let k_blocked = ffi::reshape(&k, &[b, hkv, num_key_blocks, block_size, dim]);
    let v_blocked = ffi::reshape(&v, &[b, hkv, num_key_blocks, block_size, dim]);
    ffi::eval(&k_blocked);
    ffi::eval(&v_blocked);

    // Build per-query-head inverted index with disjoint selections.
    // Group 0 (heads 0-3): selects block 0 (compact index 0)
    // Group 1 (heads 4-7): selects block 1 (compact index 1)
    let n_selected = 2i32; // union has 2 blocks
    let max_qpb = 1i32;
    let mut counts = vec![0i32; (hq * n_selected) as usize];
    let mut inv_index = vec![0i32; (hq * n_selected * max_qpb) as usize];

    // Group 0: heads 0-3 → block 0 (compact index 0)
    for h in 0..(n_rep as usize) {
        counts[h * n_selected as usize + 0] = 1;
        inv_index[(h * n_selected as usize + 0) * max_qpb as usize] = offset;
    }
    // Group 1: heads 4-7 → block 1 (compact index 1)
    for h in (n_rep as usize)..(hq as usize) {
        counts[h * n_selected as usize + 1] = 1;
        inv_index[(h * n_selected as usize + 1) * max_qpb as usize] = offset;
    }

    let inv_arr = ffi::astype(
        &ffi::from_slice_f32(
            &inv_index.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &[b, hq, n_selected, max_qpb],
        ),
        dtype::INT32,
    );
    let counts_arr = ffi::astype(
        &ffi::from_slice_f32(
            &counts.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &[b, hq, n_selected],
        ),
        dtype::INT32,
    );
    let block_ids_arr = ffi::astype(
        &ffi::from_slice_f32(&[0.0, 1.0], &[n_selected]),
        dtype::INT32,
    );

    let mut partials = ffi::turbo_minimax_sparse_kv_outer_sdpa(
        &q, &k_blocked, &v_blocked, &inv_arr, &counts_arr, &block_ids_arr,
        scale, block_size, 1,
    );
    let partial_m = ffi::kv_outer_partials_take_m(partials.pin_mut());
    let partial_l = ffi::kv_outer_partials_take_l(partials.pin_mut());
    let partial_v = ffi::kv_outer_partials_take_v(partials.pin_mut());

    // --- Assert phase-1 partials directly ---
    let m_flat = flatten_fp32(&partial_m);
    let l_flat = flatten_fp32(&partial_l);
    let v_flat = flatten_fp32(&partial_v);

    // Inactive (head, block) pairs: partial_m == -INF, partial_l == 0, partial_v == 0.
    // Group 0 (heads 0-3) did NOT select block 1 (compact index 1).
    for h in 0..(n_rep as usize) {
        let idx = h * n_selected as usize + 1; // block 1
        assert!(
            m_flat[idx] == f32::NEG_INFINITY,
            "Inactive partial_m should be -INF: head {h} block 1 = {}",
            m_flat[idx]
        );
        assert!(
            l_flat[idx] == 0.0,
            "Inactive partial_l should be 0: head {h} block 1 = {}",
            l_flat[idx]
        );
    }
    // Group 1 (heads 4-7) did NOT select block 0 (compact index 0).
    for h in (n_rep as usize)..(hq as usize) {
        let idx = h * n_selected as usize + 0; // block 0
        assert!(
            m_flat[idx] == f32::NEG_INFINITY,
            "Inactive partial_m should be -INF: head {h} block 0 = {}",
            m_flat[idx]
        );
        assert!(
            l_flat[idx] == 0.0,
            "Inactive partial_l should be 0: head {h} block 0 = {}",
            l_flat[idx]
        );
    }
    // Inactive partial_v should be 0 (zero-init).
    let dim_usize = dim as usize;
    for h in 0..(n_rep as usize) {
        let base = (h * n_selected as usize + 1) * dim_usize; // block 1
        for d in 0..dim_usize {
            assert!(
                v_flat[base + d] == 0.0,
                "Inactive partial_v should be 0: head {h} block 1 dim {d} = {}",
                v_flat[base + d]
            );
        }
    }

    // --- Assert final output ---
    let out = ffi::turbo_minimax_sparse_kv_outer_reduction(
        &q, &partial_m, &partial_l, &partial_v, n_selected,
    );

    let out_flat = flatten_fp32(&out);
    let head_size = dim as usize;

    // Group 0 (heads 0-3): attends block 0 where V=1.0 → output ≈ 1.0.
    // If query_counts were ignored and block 1 (V=0.25) also contributed,
    // output would be (1.0 + 0.25) / 2 = 0.625.
    for h in 0..(n_rep as usize) {
        let head_out = &out_flat[h * head_size..(h + 1) * head_size];
        let mean: f32 = head_out.iter().sum::<f32>() / head_size as f32;
        assert!(
            (mean - 1.0).abs() < 0.01,
            "Group 0 head {h}: mean={mean:.4}, expected 1.0 (block 1 leaking in?)"
        );
    }

    // Group 1 (heads 4-7): attends block 1 where V=0.0 → output ≈ 0.0.
    // If query_counts were ignored and block 0 (V=0.75) also contributed,
    // output would be (0.75 + 0.0) / 2 = 0.375.
    for h in (n_rep as usize)..(hq as usize) {
        let head_out = &out_flat[h * head_size..(h + 1) * head_size];
        let mean: f32 = head_out.iter().sum::<f32>() / head_size as f32;
        assert!(
            mean.abs() < 0.01,
            "Group 1 head {h}: mean={mean:.4}, expected 0.0 (block 0 leaking in?)"
        );
    }
}
