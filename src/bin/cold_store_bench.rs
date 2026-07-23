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

//! Model-free benchmark for persistent KV cold storage.
//!
//! This synthesizes production-layout MiniMax-M3 cache state without loading
//! weights, then exercises the real serialization, buffered write-drain, and
//! page-cache-warm restore paths. The writer does not fsync, so these timings
//! do not claim physical SSD durability or cold-device read throughput.
//! The required output directory must not exist and is deliberately retained.
//! Wrap the command in `/usr/bin/time -l` to capture peak resident memory.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;
use mlxcel_core::cache::cold_store::ColdStore;
use mlxcel_core::cache::kvarn::{KVARN_TILE_TOKENS, KVARN_V4_GROUP_SIZE};
use mlxcel_core::cache::{DetachedCacheSet, KVCache, SequenceId, SequenceStateBackend};

#[derive(Parser, Debug)]
#[command(name = "cold-store-bench")]
struct Args {
    /// New directory in which the retained benchmark snapshot is written.
    #[arg(long)]
    output_dir: PathBuf,

    /// Logical cache depth in tokens.
    #[arg(long, default_value_t = 10_000)]
    depth: i32,

    /// Total transformer layers (MiniMax-M3 production default: 60).
    #[arg(long, default_value_t = 60)]
    layers: usize,

    /// Leading dense FP16 layers (MiniMax-M3 production default: 3).
    #[arg(long, default_value_t = 3)]
    dense_prefix: usize,

    /// KVarN V width for sparse layers: 8 or 4.
    #[arg(long, default_value_t = 4)]
    v_bits: u8,

    /// Number of page-cache-warm restore passes to time.
    #[arg(long, default_value_t = 2)]
    reads: usize,

    #[arg(long, default_value_t = 4)]
    kv_heads: i32,

    #[arg(long, default_value_t = 128)]
    head_dim: i32,

    #[arg(long, default_value_t = 128)]
    index_dim: i32,
}

fn directory_bytes(path: &Path) -> Result<u64> {
    let mut total = 0u64;
    for entry in fs::read_dir(path).with_context(|| format!("read {}", path.display()))? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        let bytes = if metadata.is_dir() {
            directory_bytes(&entry.path())?
        } else {
            metadata.len()
        };
        total = total
            .checked_add(bytes)
            .context("directory byte count overflow")?;
    }
    Ok(total)
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.output_dir.exists() {
        bail!(
            "--output-dir must not already exist; refusing to overwrite {}",
            args.output_dir.display()
        );
    }
    if args.layers == 0 || args.dense_prefix > args.layers {
        bail!("require 0 < --layers and --dense-prefix <= --layers");
    }
    if args.depth <= 0 {
        bail!("--depth must be positive");
    }
    if args.dense_prefix < args.layers && args.depth < 2 * KVARN_TILE_TOKENS {
        bail!(
            "--depth must be at least {} for a KVarN sink plus one history tile",
            2 * KVARN_TILE_TOKENS
        );
    }
    if args.v_bits != 4 && args.v_bits != 8 {
        bail!("--v-bits must be 4 or 8");
    }
    if args.kv_heads <= 0 || args.head_dim <= 0 || args.index_dim <= 0 {
        bail!("--kv-heads, --head-dim, and --index-dim must be positive");
    }
    if args.v_bits == 4 && args.head_dim % KVARN_V4_GROUP_SIZE != 0 {
        bail!("--head-dim must be divisible by {KVARN_V4_GROUP_SIZE} for 4-bit V");
    }
    if args.reads == 0 {
        bail!("--reads must be at least 1");
    }

    let output_parent = args
        .output_dir
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !output_parent.is_dir() {
        bail!(
            "--output-dir parent must already exist: {}",
            output_parent.display()
        );
    }
    fs::create_dir(&args.output_dir)?;
    let model_dir = args.output_dir.join("empty-model-fingerprint");
    let store_dir = args.output_dir.join("cold-store");
    fs::create_dir(&model_dir)?;

    println!(
        "cold-store-bench BOOT depth={} layers={} dense_prefix={} v_bits={} reads={} kv_heads={} head_dim={} index_dim={} output_dir={}",
        args.depth,
        args.layers,
        args.dense_prefix,
        args.v_bits,
        args.reads,
        args.kv_heads,
        args.head_dim,
        args.index_dim,
        args.output_dir.display()
    );

    let synth_started = Instant::now();
    let mut caches = Vec::with_capacity(args.layers);
    for layer in 0..args.layers {
        let seed = 0xC01D_5700u64 + layer as u64;
        let mut cache = if layer < args.dense_prefix {
            KVCache::synth_fp16_state(1, args.kv_heads, args.head_dim, args.depth, 0, seed)
        } else {
            KVCache::synth_kvarn_state(
                1,
                args.kv_heads,
                args.head_dim,
                args.depth,
                args.index_dim,
                seed,
                args.v_bits,
            )
        };
        caches.push(cache.clone_handle());
    }
    let cache_set = DetachedCacheSet {
        caches,
        backend: SequenceStateBackend::DenseKvCache,
        prompt_len: args.depth as usize,
        current_offset: args.depth,
        created_at: Instant::now(),
        detached_at: Instant::now(),
        origin_seq_id: SequenceId::from_raw(1),
    };
    let synth_elapsed = synth_started.elapsed();
    let logical_bytes = cache_set.nbytes();

    let tokens: Vec<i32> = (0..args.depth).collect();
    let model_path = model_dir.to_str().context("non-UTF-8 model path")?;
    let mut writer = ColdStore::with_base_dir(store_dir.clone(), model_path);
    let persist_started = Instant::now();
    writer.persist(
        "minimax-m3-synthetic",
        "cold-store-bench-v1",
        &tokens,
        &cache_set,
    )?;
    let serialize_enqueue_elapsed = persist_started.elapsed();
    let write_started = Instant::now();
    writer.shutdown();
    let post_enqueue_drain_elapsed = write_started.elapsed();
    let buffered_persist_elapsed = persist_started.elapsed();
    let disk_bytes = directory_bytes(&store_dir)?;

    drop(cache_set);
    mlxcel_core::clear_memory_cache();

    let mut warm_restore_ms = Vec::with_capacity(args.reads);
    let mut reader = ColdStore::with_base_dir(store_dir, model_path);
    for pass in 0..args.reads {
        let started = Instant::now();
        let (loaded, match_len) =
            reader.load_prefix("minimax-m3-synthetic", "cold-store-bench-v1", &tokens)?;
        let elapsed = started.elapsed();
        if match_len != tokens.len() || loaded.nbytes() != logical_bytes {
            bail!(
                "restore {pass} mismatch: match_len={match_len}, loaded_bytes={}, expected_match={}, expected_bytes={logical_bytes}",
                loaded.nbytes(),
                tokens.len()
            );
        }
        warm_restore_ms.push(elapsed.as_secs_f64() * 1000.0);
        drop(loaded);
        mlxcel_core::clear_memory_cache();
    }
    reader.shutdown();

    println!(
        "RESULT logical_bytes={} disk_bytes={} synthesis_ms={:.3} serialize_enqueue_ms={:.3} buffered_persist_total_ms={:.3} post_enqueue_drain_ms={:.3} warm_restore_ms={:?}",
        logical_bytes,
        disk_bytes,
        synth_elapsed.as_secs_f64() * 1000.0,
        serialize_enqueue_elapsed.as_secs_f64() * 1000.0,
        buffered_persist_elapsed.as_secs_f64() * 1000.0,
        post_enqueue_drain_elapsed.as_secs_f64() * 1000.0,
        warm_restore_ms
    );
    println!(
        "NOTE writer drain is buffered filesystem completion without fsync; restores are OS-page-cache warm"
    );
    println!("Artifacts retained at {}", args.output_dir.display());
    Ok(())
}
