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

//! C step-1 micro-bench: `gather_qmm` at MSA decode shapes
//! (DESIGN_c_qmm_union_sketch_2026-07-10.md, recommended sequence #1).
//!
//! Question: is ONE gather_qmm dispatch over a 2344-tile pool with ~128
//! selected (kv_head × top_k) rows cheap enough to replace the gathered
//! flow's fetch+core (~3.2 ms/layer serialized) — i.e. is Shape 2 viable?
//! Decode-shape caveat the MoE precedent doesn't cover: tiny M (128 rows
//! of lhs), large pool, per-row rhs indexing.
//!
//! Times the SCORES-side call (transpose=true) at production geometry:
//! pool [2344 tiles, 128 tokens, 128 d] u8 codes as bits=8/gs=128 affine,
//! lhs [128, 128] fp16 (4 kv-heads × top_k 32 s_col-scaled query rows),
//! rhs_indices [128] i32. Reports µs/dispatch serialized (eval per call)
//! and pipelined (eval every 57 calls ≈ one token's MSA layers).

use std::time::Instant;

use mlxcel_core::{MlxArray, UniquePtr, dtype};

fn main() {
    let (n_tiles, bs, d) = (2344i32, 128i32, 128i32);
    let (kv_heads, top_k) = (4i32, 32i32);
    let rows = kv_heads * top_k; // 128 lhs rows
    mlxcel_core::random_seed(42);
    let null = std::ptr::null();

    // Pool: codes packed 4/u32 along d → [n_tiles, bs, d/4] u32 view.
    // Random u32 IS random packed u8 codes (fold verification: layout
    // identity). scales/biases fp32 per row, one group at gs=128.
    let w = unsafe {
        mlxcel_core::random_randint(0, i32::MAX, &[n_tiles, bs, d / 4], dtype::UINT32, null)
    };
    let scales =
        unsafe { mlxcel_core::random_uniform(1e-3, 2e-2, &[n_tiles, bs, 1], dtype::FLOAT32, null) };
    let biases =
        unsafe { mlxcel_core::random_uniform(-1.5, 0.0, &[n_tiles, bs, 1], dtype::FLOAT32, null) };
    let x = unsafe { mlxcel_core::random_normal(&[rows, 1, d], dtype::FLOAT16, null) };
    let lhs_idx = {
        let v: Vec<f32> = (0..rows).map(|i| i as f32).collect();
        mlxcel_core::astype(&mlxcel_core::from_slice_f32(&v, &[rows]), dtype::INT32)
    };
    for a in [&w, &scales, &biases, &x] {
        mlxcel_core::eval(a);
    }
    mlxcel_core::eval(&lhs_idx);
    println!(
        "gather_qmm micro: pool [{n_tiles},{bs},{d}] u8 (as u32 packed), lhs [{rows},1,{d}] fp16, \
         rhs per-row from a fresh random tile set per iteration, gs=128 bits=8 transpose=T"
    );

    let call = |rhs_idx: &MlxArray| -> UniquePtr<MlxArray> {
        unsafe {
            mlxcel_core::gather_qmm(
                &x,
                &w,
                &scales,
                biases.as_ref().unwrap() as *const MlxArray,
                lhs_idx.as_ref().unwrap() as *const MlxArray,
                rhs_idx as *const MlxArray,
                true,
                128,
                8,
                false,
                "affine",
            )
        }
    };

    // Fresh indices per call (defeats any graph caching), built host-side
    // outside the timed region.
    let mk_idx = |seed: u64| -> UniquePtr<MlxArray> {
        mlxcel_core::random_seed(seed);
        let idx = unsafe {
            mlxcel_core::random_randint(0, n_tiles, &[rows], dtype::INT32, std::ptr::null())
        };
        mlxcel_core::eval(&idx);
        idx
    };
    let idx_pool: Vec<UniquePtr<MlxArray>> = (0..640).map(|i| mk_idx(1000 + i as u64)).collect();

    // Warmup.
    for idx in idx_pool.iter().take(16) {
        let o = call(idx);
        mlxcel_core::eval(&o);
    }

    // Serialized: eval per dispatch — the exclusive per-call ceiling.
    let t0 = Instant::now();
    let n_ser = 256;
    for idx in idx_pool.iter().skip(16).take(n_ser) {
        let o = call(idx);
        mlxcel_core::eval(&o);
    }
    let ser_us = t0.elapsed().as_secs_f64() * 1e6 / n_ser as f64;

    // Pipelined: 57 dispatches per eval ≈ one decode token's MSA layers.
    let t1 = Instant::now();
    let (mut done, mut batches) = (0usize, 0usize);
    let mut acc: Option<UniquePtr<MlxArray>> = None;
    for idx in idx_pool.iter().skip(16 + n_ser) {
        let o = call(idx);
        acc = Some(match acc {
            Some(a) => mlxcel_core::add(&a, &o),
            None => o,
        });
        done += 1;
        if done % 57 == 0 {
            mlxcel_core::eval(acc.as_ref().unwrap());
            acc = None;
            batches += 1;
        }
    }
    if let Some(a) = acc.as_ref() {
        mlxcel_core::eval(a);
    }
    let pip_us = t1.elapsed().as_secs_f64() * 1e6 / done as f64;

    println!(
        "RESULT: serialized {ser_us:.0} µs/dispatch ({n_ser} calls); pipelined {pip_us:.0} \
         µs/dispatch ({done} calls, {batches} token-shaped evals) → per-token 57-layer scores-side \
         cost ≈ {:.2} ms serialized / {:.2} ms pipelined",
        ser_us * 57.0 / 1e3,
        pip_us * 57.0 / 1e3
    );
    println!(
        "verdict guide: fetch+core budget to beat is ~3.2 ms/layer serialized (~1.9 fetch + 1.3 \
         core); scores-side is roughly half the work → viable if pipelined per-token cost ≲ 90 ms \
         for BOTH sides combined (2× the number above)."
    );
}
