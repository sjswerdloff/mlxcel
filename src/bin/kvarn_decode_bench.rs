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

//! H0 synthetic-state decode bench for the KVarN8 × MSA decode paths.
//!
//! Phase H0 of `DESIGN_decode_experiment_harness_2026-07-10.md`: instead of
//! paying a multi-minute deep prefill per data point, this binary writes
//! random codes/scales DIRECTLY into KVarN8 cache fields
//! (`KVCache::synth_kvarn8_state`) — an instant 100K/300K/500K state — and
//! drives the real per-token decode loop across all layers with
//! production-geometry `SparseAttention` layers built from random weights.
//! No model load, no tokenizer, no server.
//!
//! What it answers (and what it cannot):
//! - RANKS decode-path candidates and profiles per-stage cost at depth
//!   (`--profile` reuses the MLXCEL_K1_PROFILE serialized-ceiling spans).
//! - The absolute tok/s here is an ATTENTION-ONLY ceiling: no MoE/MLP, no
//!   sampling, fp16 projections instead of the live MXFP8 quantized ones.
//!   The bench RANKS; the live server CONFIRMS — never promote a path on
//!   bench numbers alone.
//!
//! Layer mix is production-true by default: M3's first 3 layers are dense
//! (no index projections → `update_and_fetch`, O(history) full dequant +
//! dense attention), the remaining 57 are MSA (gathered decode path when
//! eligible). `--dense-prefix 0` isolates the MSA path; `--layers 3
//! --dense-prefix 3` isolates the dense O(T) floor.
//!
//! ```bash
//! # depth profile at 300K, production layer mix, K1 spans on:
//! kvarn-decode-bench --depth 300000 --steps 512 --profile
//! # MSA-only ranking run:
//! kvarn-decode-bench --depth 300000 --steps 512 --dense-prefix 0 --profile
//! ```

use std::time::Instant;

use clap::Parser;
use mlxcel::models::minimax_m3::SparseAttention;
use mlxcel_core::cache::KVCache;
use mlxcel_core::layers::{GemmaRMSNorm, Linear, UnifiedLinear};
use mlxcel_core::{MlxArray, UniquePtr};
use mlxcel_core::{dtype, multiply_scalar};

/// H0 synthetic-state decode bench (no model load; layout-true cache state).
#[derive(Parser, Debug)]
#[command(name = "kvarn-decode-bench")]
struct Args {
    /// Cache depth (tokens) to synthesize before the first decode step.
    #[arg(long, default_value_t = 300_000)]
    depth: i32,

    /// Measured decode steps (after warmup).
    #[arg(long, default_value_t = 512)]
    steps: usize,

    /// Warmup decode steps (kernel compilation, cache shakedown; excluded
    /// from the report).
    #[arg(long, default_value_t = 8)]
    warmup: usize,

    /// Total layers to drive per token.
    #[arg(long, default_value_t = 60)]
    layers: usize,

    /// Leading layers WITHOUT index projections (M3 production: 3). These
    /// dispatch dense every step.
    #[arg(long, default_value_t = 3)]
    dense_prefix: usize,

    /// Cache mode synthesized for the dense-prefix layers: "fp16" (the D1
    /// layer-selective state a live session reaches via the first-touch
    /// downgrade) or "kvarn8" (pre-D1 behavior: the O(T) full-window
    /// dequant floor, for A/B).
    #[arg(long, default_value = "fp16")]
    dense_cache: String,

    /// Cache mode synthesized for the MSA layers:
    ///   "kvarn8"        — gathered decode path, dequant-after-gather;
    ///   "fp16"          — fp16 KV, full-window flow (the v1-shape
    ///                     baseline: O(T), fastest below ~250K);
    ///   "fp16-gathered" — fp16 KV through the GATHERED flow (sets
    ///                     MLXCEL_FP16_GATHERED=1): selection → block
    ///                     gather off the fp16 buffer, zero dequant. The
    ///                     missing cell of the {full,gathered}×{fp16,
    ///                     kvarn8} rank matrix. With --profile, k1.profile
    ///                     lines firing is the proof the gathered flow
    ///                     engaged (they exist only on that path).
    /// NOTE: fp16 modes at 500K are a ~70-100 GB state — honor the RAM
    /// protocol before launching.
    #[arg(long, default_value = "kvarn8")]
    cache_mode: String,

    /// RNG seed for weights and cache state.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Enable the MLXCEL_K1_PROFILE serialized-ceiling stage spans
    /// (KNOWN upward bias: forced eval at stage boundaries — exclusive
    /// per-stage cost, not end-to-end truth).
    #[arg(long, default_value_t = false)]
    profile: bool,

    // ---- geometry (MiniMax-M3 defaults, from config.json) ----
    #[arg(long, default_value_t = 6144)]
    hidden: i32,
    #[arg(long, default_value_t = 64)]
    heads: i32,
    #[arg(long, default_value_t = 4)]
    kv_heads: i32,
    #[arg(long, default_value_t = 128)]
    head_dim: i32,
    #[arg(long, default_value_t = 128)]
    index_dim: i32,
    #[arg(long, default_value_t = 32)]
    top_k: i32,
    #[arg(long, default_value_t = 128)]
    block_size: i32,
    #[arg(long, default_value_t = 64)]
    rope_dims: i32,
    #[arg(long, default_value_t = 5_000_000.0)]
    rope_base: f32,
}

fn rand_fp16(shape: &[i32]) -> UniquePtr<MlxArray> {
    unsafe { mlxcel_core::random_normal(shape, dtype::FLOAT16, std::ptr::null()) }
}

/// Random linear with production-like activation scale (weights ~
/// N(0, 1/in_dim) so activations stay O(1) through the stack — fp16 range
/// safety, not numerics fidelity).
fn rand_linear(out_dim: i32, in_dim: i32) -> UnifiedLinear {
    let w = rand_fp16(&[out_dim, in_dim]);
    let w = multiply_scalar(&w, 1.0 / (in_dim as f32).sqrt());
    mlxcel_core::eval(&w);
    UnifiedLinear::Regular(Linear::new(w, None))
}

/// Gemma-style norm weights are DELTAS near zero (the norm scales by
/// 1 + weight — cycle-65 lesson).
fn rand_norm(dim: i32) -> GemmaRMSNorm {
    let w =
        unsafe { mlxcel_core::random_uniform(-0.1, 0.1, &[dim], dtype::FLOAT16, std::ptr::null()) };
    mlxcel_core::eval(&w);
    GemmaRMSNorm::new(w, 1e-6)
}

fn main() {
    let args = Args::parse();
    if args.profile {
        // Must be set before the first forward touches the OnceLock.
        unsafe { std::env::set_var("MLXCEL_K1_PROFILE", "1") };
    }
    // k1.profile lines emit at INFO; default to info so the spans are
    // visible without extra flags (RUST_LOG still wins if set).
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(env_filter).init();

    let dense_fp16 = match args.dense_cache.as_str() {
        "fp16" => true,
        "kvarn8" => false,
        other => panic!("--dense-cache must be 'fp16' or 'kvarn8'; got '{other}'"),
    };
    // (msa_fp16, msa_gathered): which STATE the MSA layers carry and which
    // FLOW the dispatch should take on it.
    let (msa_fp16, msa_gathered) = match args.cache_mode.as_str() {
        "fp16" => (true, false),
        "fp16-gathered" => (true, true),
        "kvarn8" => (false, true),
        other => panic!("--cache-mode must be 'kvarn8', 'fp16' or 'fp16-gathered'; got '{other}'"),
    };
    if args.cache_mode == "fp16-gathered" {
        // Must be set before the first supports_block_fetch touches its
        // OnceLock.
        unsafe { std::env::set_var("MLXCEL_FP16_GATHERED", "1") };
    }

    // Fail-loud boot artifact: every parameter that shapes the measurement,
    // so no captured number can be mis-attributed to the wrong config.
    println!(
        "kvarn-decode-bench BOOT: depth={} steps={} warmup={} layers={} dense_prefix={} \
         dense_cache={} cache_mode={} seed={} profile={}",
        args.depth,
        args.steps,
        args.warmup,
        args.layers,
        args.dense_prefix,
        args.dense_cache,
        args.cache_mode,
        args.seed,
        args.profile
    );
    println!(
        "geometry: hidden={} heads={} kv_heads={} head_dim={} index_dim={} top_k={} \
         block_size={} rope_dims={} rope_base={}",
        args.hidden,
        args.heads,
        args.kv_heads,
        args.head_dim,
        args.index_dim,
        args.top_k,
        args.block_size,
        args.rope_dims,
        args.rope_base
    );
    // Self-labeling for the msa fetch/core axes (env-gated modes): a
    // capture whose first lines don't carry these cannot be trusted to a
    // rank cell. NOTE: this echoes the REQUEST — the model's first-dispatch
    // INFO line ("C qmm-fetch fused core active") is the witness that C
    // actually ran; the rank runner requires both.
    println!(
        "msa modes: fetch={} core={}",
        std::env::var("MLXCEL_MSA_FETCH").unwrap_or_else(|_| "default".into()),
        std::env::var("MLXCEL_MSA_CORE").unwrap_or_else(|_| "blocked".into()),
    );

    // Rough state-size estimate up front (fail-loud awareness before a
    // multi-GB allocation, not a guard). Per-mode: kvarn8 stores u8 codes
    // (K+V) + six f32 per-row scalars; fp16 stores 2-byte K+V. m3_idx
    // (fp16, capacity rounds up) exists only on MSA layers.
    let kv_tokens = (args.kv_heads as i64) * (args.depth as i64);
    let kvarn8_layer = kv_tokens * (args.head_dim as i64) * 2 + kv_tokens * 6 * 4;
    let fp16_layer = kv_tokens * (args.head_dim as i64) * 2 * 2;
    let idx_layer = (args.depth as i64) * (args.index_dim as i64) * 2;
    let msa_count = (args.layers - args.dense_prefix) as i64;
    let dense_count = args.dense_prefix as i64;
    let state_bytes = msa_count * (if msa_fp16 { fp16_layer } else { kvarn8_layer } + idx_layer)
        + dense_count * (if dense_fp16 { fp16_layer } else { kvarn8_layer });
    let weights_bytes = (args.layers as i64)
        * ((args.hidden as i64) * (args.heads as i64) * (args.head_dim as i64) * 2 * 2 // q,o
            + (args.hidden as i64) * (args.kv_heads as i64) * (args.head_dim as i64) * 2 * 2); // k,v
    println!(
        "estimated allocation: state ~{:.1} GB + weights ~{:.1} GB",
        state_bytes as f64 / 1e9,
        weights_bytes as f64 / 1e9
    );

    mlxcel_core::random_seed(args.seed);
    let t_setup = Instant::now();
    let mut layers: Vec<SparseAttention> = Vec::with_capacity(args.layers);
    let mut caches: Vec<KVCache> = Vec::with_capacity(args.layers);
    for i in 0..args.layers {
        let is_msa = i >= args.dense_prefix;
        layers.push(SparseAttention {
            q_proj: rand_linear(args.heads * args.head_dim, args.hidden),
            k_proj: rand_linear(args.kv_heads * args.head_dim, args.hidden),
            v_proj: rand_linear(args.kv_heads * args.head_dim, args.hidden),
            o_proj: rand_linear(args.hidden, args.heads * args.head_dim),
            q_norm: Some(rand_norm(args.head_dim)),
            k_norm: Some(rand_norm(args.head_dim)),
            index_q_proj: is_msa.then(|| rand_linear(args.kv_heads * args.index_dim, args.hidden)),
            index_k_proj: is_msa.then(|| rand_linear(args.index_dim, args.hidden)),
            index_q_norm: is_msa.then(|| rand_norm(args.index_dim)),
            index_k_norm: is_msa.then(|| rand_norm(args.index_dim)),
            num_heads: args.heads,
            num_kv_heads: args.kv_heads,
            head_dim: args.head_dim,
            scale: 1.0 / (args.head_dim as f32).sqrt(),
            rope_dims: args.rope_dims,
            rope_base: args.rope_base,
            block_size: args.block_size,
            top_k: args.top_k,
            index_dim: args.index_dim,
            sparse_local_block: 1,
            layer_idx: i,
        });
        let layer_seed = args.seed.wrapping_add(i as u64 * 7919);
        let fp16_cache = if is_msa { msa_fp16 } else { dense_fp16 };
        caches.push(if fp16_cache {
            // fp16 states: dense-prefix layers carry no m3_idx (the D1 end
            // state a live session reaches via the first-touch downgrade);
            // MSA layers on fp16 keep m3_idx — same sparse selection, fp16
            // windows, no dequant (the fp16-baseline comparison).
            let index_dim = if is_msa { args.index_dim } else { 0 };
            KVCache::synth_fp16_state(
                1,
                args.kv_heads,
                args.head_dim,
                args.depth,
                index_dim,
                layer_seed,
            )
        } else {
            KVCache::synth_kvarn8_state(
                1,
                args.kv_heads,
                args.head_dim,
                args.depth,
                args.index_dim,
                layer_seed,
            )
        });
        if (i + 1) % 10 == 0 {
            eprintln!("setup: layer {}/{}", i + 1, args.layers);
        }
    }
    println!("setup: {:.1}s", t_setup.elapsed().as_secs_f64());

    // Fail-loud BEFORE any step: every MSA cache's block-fetch support must
    // match the requested flow — kvarn8 and fp16-gathered take the gathered
    // path (true), plain fp16 takes the full-window flow (false). A
    // mismatch here means the env plumbing or the predicate broke, and
    // every number the run produced would be attributed to the wrong path.
    for (i, cache) in caches.iter().enumerate().skip(args.dense_prefix) {
        assert_eq!(
            cache.supports_block_fetch(),
            msa_gathered,
            "layer {i}: supports_block_fetch != expected flow for cache_mode={}",
            args.cache_mode
        );
    }

    // One shared decode input; reused every step. Timing is data-independent
    // on the hot path, and reuse keeps setup out of the measured loop.
    let x = rand_fp16(&[1, 1, args.hidden]);
    mlxcel_core::eval(&x);

    let mut times: Vec<f64> = Vec::with_capacity(args.steps);
    for step in 0..(args.warmup + args.steps) {
        let t0 = Instant::now();
        let mut acc: Option<UniquePtr<MlxArray>> = None;
        for (layer, cache) in layers.iter().zip(caches.iter_mut()) {
            let o = layer.forward(&x, cache, None);
            // Sum outputs so ONE eval forces every layer's graph — mirrors
            // the per-token materialization the real decode loop performs at
            // sampling. An unconsumed lazy output would never compute.
            acc = Some(match acc {
                Some(a) => mlxcel_core::add(&a, &o),
                None => o,
            });
        }
        mlxcel_core::eval(acc.as_ref().expect("at least one layer"));
        let dt = t0.elapsed().as_secs_f64();

        if step == 0 {
            // Dispatch verification, fail-loud: on an eligible MSA layer the
            // forward must have advanced the indexer cache in lockstep (the
            // gathered path's precondition and side effect); a dense-prefix
            // layer must NOT have (no index projections → no idx update;
            // fp16 dense caches carry no m3_idx state at all).
            for (i, cache) in caches.iter().enumerate() {
                let expect = if i >= args.dense_prefix {
                    args.depth + 1
                } else if dense_fp16 {
                    0
                } else {
                    args.depth
                };
                assert_eq!(
                    cache.m3_idx_offset(),
                    expect,
                    "layer {i}: m3_idx_offset after step 0 — wrong dispatch path \
                     (expected {} for {} layer)",
                    expect,
                    if i >= args.dense_prefix {
                        "MSA"
                    } else {
                        "dense"
                    }
                );
            }
            println!(
                "dispatch check PASSED: {} dense ({}) + {} MSA ({}) layers on the expected paths",
                args.dense_prefix,
                args.dense_cache,
                args.layers - args.dense_prefix,
                args.cache_mode
            );
        }
        if step >= args.warmup {
            times.push(dt);
        }
        if step < args.warmup || (step - args.warmup) % 32 == 0 {
            eprintln!(
                "step {:>4}{}: {:>8.1} ms",
                step,
                if step < args.warmup { " (warmup)" } else { "" },
                dt * 1e3
            );
        }
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let p50 = times[times.len() / 2];
    println!(
        "RESULT depth={} layers={} dense_prefix={} dense_cache={} cache_mode={}: \
         mean {:.1} ms/token, p50 {:.1} ms, min {:.1} ms, max {:.1} ms over {} steps → \
         attention-only ceiling {:.2} tok/s",
        args.depth,
        args.layers,
        args.dense_prefix,
        args.dense_cache,
        args.cache_mode,
        mean * 1e3,
        p50 * 1e3,
        times[0] * 1e3,
        times[times.len() - 1] * 1e3,
        times.len(),
        1.0 / mean
    );
}
