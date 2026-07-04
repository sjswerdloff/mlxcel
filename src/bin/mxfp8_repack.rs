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

//! Offline MXFP8 checkpoint converter: HF/vLLM → MLX-native storage.
//!
//! # Purpose
//!
//! Converts a HuggingFace/vLLM-format MXFP8 checkpoint directory (e.g.
//! `morriszjm/MiniMax-M3-MXFP8-64e`) into the MLX-native quantized layout
//! that mlxcel and mlx-lm can mmap-load directly, without any tensor math or
//! MLX at runtime.
//!
//! This is a **pure byte-copy with header rewrites** — the data region of
//! every shard is copied verbatim.
//!
//! # Byte-compatibility facts (verified empirically 2026-07-04)
//!
//! MLX's mxfp8 packed weight representation is byte-identical to the HF
//! safetensors layout:
//!
//! * `{prefix}.weight` (dtype `F8_E4M3`, shape `[out, in]`) → retyped to
//!   `U32` with shape `[out, in/4]`.  The fp8 payload bytes are simply
//!   reinterpreted as 32-bit words (4 bytes per word), identical to what
//!   `mlxcel_core::view(&fp8_bytes, dtype::UINT32)` produces at load time.
//!
//! * `{prefix}.weight_scale_inv` (dtype `U8`, shape `[out, in/32]`) →
//!   renamed to `{prefix}.scales`.  Dtype, shape, and bytes are unchanged —
//!   the U8 E8M0 exponent bytes that HF stores are exactly the scale bytes
//!   that MLX reads.
//!
//! The load-time counterpart is `repack_hf_mxfp8_weights` in
//! `src/models/sanitize.rs`, which performs the same transformation in
//! memory at model-load time. This binary moves that work offline so a
//! converted checkpoint loads without any runtime surgery.
//!
//! # Safety
//!
//! * Shards are written to `<name>.tmp` and renamed into place only when the
//!   shard completes — a partial conversion cannot be mistaken for a complete
//!   one.
//! * The index (`model.safetensors.index.json`) and config are written LAST,
//!   only after all shards succeed.
//! * Malformed inputs (orphaned scales, split pairs, shape mismatches) are
//!   hard errors that name the offending tensor.
//! * Memory usage is bounded by the streaming buffer size regardless of shard
//!   file size.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde_json::{Map, Value};

/// Offline converter: HF/vLLM MXFP8 checkpoint → MLX-native quantized layout.
///
/// Rewrites safetensors headers to change F8_E4M3 weights to U32 (4 fp8 bytes
/// per word) and rename `weight_scale_inv` → `scales`, then bulk-copies the
/// data region unchanged.  No tensor math, no MLX, no model loading.
#[derive(Parser, Debug)]
#[command(name = "mxfp8_repack")]
struct Args {
    /// Input checkpoint directory (HF/vLLM MXFP8 format).
    #[arg(long)]
    input: PathBuf,

    /// Output directory (must not already contain shard files).
    #[arg(long)]
    output: PathBuf,

    /// After conversion, verify output shards for correctness (tensor counts,
    /// dtypes/shapes, and first+last payload byte-equality against input).
    #[arg(long, default_value_t = false)]
    verify: bool,
}

/// Streaming buffer size for bulk-copying shard data regions (64 MiB).
const COPY_BUF_SIZE: usize = 64 * 1024 * 1024;

// ─── safetensors header helpers ──────────────────────────────────────────────

/// Read and parse the safetensors header from an open file.
///
/// Returns `(header_json_bytes, header_map)` where `header_json_bytes` is the
/// raw JSON (without the 8-byte length prefix) and `header_map` is the parsed
/// `serde_json::Map`.  The file cursor is positioned at the start of the data
/// region on return.
fn read_st_header(file: &mut File, path: &Path) -> Result<(Vec<u8>, Map<String, Value>)> {
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .with_context(|| format!("reading safetensors header length from {}", path.display()))?;
    let hdr_len = u64::from_le_bytes(len_buf) as usize;
    let mut hdr_bytes = vec![0u8; hdr_len];
    file.read_exact(&mut hdr_bytes)
        .with_context(|| format!("reading safetensors header from {}", path.display()))?;
    let val: Value = serde_json::from_slice(&hdr_bytes)
        .with_context(|| format!("parsing safetensors header from {}", path.display()))?;
    let map = match val {
        Value::Object(m) => m,
        _ => bail!(
            "safetensors header is not a JSON object: {}",
            path.display()
        ),
    };
    Ok((hdr_bytes, map))
}

/// Write a safetensors header: 8-byte LE length + JSON bytes (no padding added
/// here — the caller must ensure the header is already the correct length if it
/// needs to match the original data offsets, or adjust data_offsets otherwise).
///
/// This converter rewrites headers so data_offsets refer to the same payload
/// positions relative to the data region; only the JSON bytes change size, so
/// we write the new header followed immediately by the verbatim data region.
fn write_st_header(writer: &mut impl Write, header_json: &[u8]) -> Result<()> {
    let len = header_json.len() as u64;
    writer
        .write_all(&len.to_le_bytes())
        .context("writing safetensors header length")?;
    writer
        .write_all(header_json)
        .context("writing safetensors header JSON")?;
    Ok(())
}

// ─── tensor metadata extracted from a safetensors header entry ───────────────

#[derive(Debug, Clone)]
struct TensorMeta {
    dtype: String,
    shape: Vec<i64>,
    data_offsets: [u64; 2],
}

impl TensorMeta {
    fn from_value(v: &Value, name: &str, shard: &Path) -> Result<Self> {
        let obj = v.as_object().with_context(|| {
            format!(
                "tensor entry for {name} is not an object in {}",
                shard.display()
            )
        })?;
        let dtype = obj
            .get("dtype")
            .and_then(Value::as_str)
            .with_context(|| format!("missing dtype for {name} in {}", shard.display()))?
            .to_string();
        let shape: Vec<i64> = obj
            .get("shape")
            .and_then(Value::as_array)
            .with_context(|| format!("missing shape for {name} in {}", shard.display()))?
            .iter()
            .map(|d| {
                d.as_i64().with_context(|| {
                    format!("non-integer shape dim for {name} in {}", shard.display())
                })
            })
            .collect::<Result<_>>()?;
        let offsets_arr = obj
            .get("data_offsets")
            .and_then(Value::as_array)
            .with_context(|| format!("missing data_offsets for {name} in {}", shard.display()))?;
        if offsets_arr.len() != 2 {
            bail!(
                "data_offsets for {name} in {} must have 2 elements",
                shard.display()
            );
        }
        let start = offsets_arr[0].as_u64().with_context(|| {
            format!(
                "non-integer data_offsets[0] for {name} in {}",
                shard.display()
            )
        })?;
        let end = offsets_arr[1].as_u64().with_context(|| {
            format!(
                "non-integer data_offsets[1] for {name} in {}",
                shard.display()
            )
        })?;
        Ok(TensorMeta {
            dtype,
            shape,
            data_offsets: [start, end],
        })
    }

    fn to_value(&self) -> Value {
        serde_json::json!({
            "dtype": self.dtype,
            "shape": self.shape,
            "data_offsets": self.data_offsets,
        })
    }
}

// ─── conversion summary ───────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct ShardSummary {
    repacked: usize,
    renamed: usize,
    passthrough: usize,
    bytes_written: u64,
}

// ─── index loading ────────────────────────────────────────────────────────────

/// Load `model.safetensors.index.json` and return the full `weight_map`
/// (tensor_name → shard_filename) plus all tensors keyed by prefix to detect
/// cross-shard pairs.
fn load_index(input_dir: &Path) -> Result<(Value, HashMap<String, String>)> {
    let index_path = input_dir.join("model.safetensors.index.json");
    let raw = fs::read_to_string(&index_path)
        .with_context(|| format!("reading {}", index_path.display()))?;
    let val: Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", index_path.display()))?;
    let weight_map = val
        .get("weight_map")
        .and_then(Value::as_object)
        .with_context(|| format!("missing weight_map in {}", index_path.display()))?
        .iter()
        .map(|(k, v)| {
            let shard = v
                .as_str()
                .with_context(|| format!("non-string shard name for {k}"))?;
            Ok((k.clone(), shard.to_string()))
        })
        .collect::<Result<HashMap<String, String>>>()?;
    Ok((val, weight_map))
}

// ─── shard-level conversion ───────────────────────────────────────────────────

/// Validate that every `weight_scale_inv` tensor in `weight_map` that maps to
/// `shard_file` has its companion `.weight` in the same shard.
///
/// Cross-shard pairs are not supported because the data regions cannot be
/// merged without re-indexing.
fn check_no_split_pairs(
    weight_map: &HashMap<String, String>,
    shard_file: &str,
    shard_path: &Path,
    header: &Map<String, Value>,
) -> Result<()> {
    for name in header.keys() {
        if name == "__metadata__" {
            continue;
        }
        if !name.ends_with(".weight_scale_inv") {
            continue;
        }
        let prefix = name.strip_suffix(".weight_scale_inv").unwrap();
        let weight_key = format!("{prefix}.weight");
        match weight_map.get(&weight_key) {
            None => bail!(
                "orphaned scale tensor {name} in {} has no companion {weight_key} anywhere in the checkpoint",
                shard_path.display()
            ),
            Some(companion_shard) if companion_shard != shard_file => bail!(
                "split pair: {name} is in {shard_file} but {weight_key} is in {companion_shard}; \
                 cross-shard pairs are not supported"
            ),
            _ => {}
        }
    }
    Ok(())
}

/// Rewrite a single shard: new header + verbatim data region.
///
/// Returns the per-shard summary.  On success the output file is at its final
/// path.  On failure the `.tmp` file is left in place and the error is
/// propagated so the caller can report partial completion.
fn convert_shard(
    input_path: &Path,
    output_path: &Path,
    weight_map: &HashMap<String, String>,
    shard_file: &str,
) -> Result<ShardSummary> {
    let mut input_file = File::open(input_path)
        .with_context(|| format!("opening input shard {}", input_path.display()))?;

    let (_, header) = read_st_header(&mut input_file, input_path)?;

    // Data region starts immediately after the header we just read.
    let data_start_in_input = input_file
        .stream_position()
        .context("getting input data start position")?;

    check_no_split_pairs(weight_map, shard_file, input_path, &header)?;

    // Build a map of weight_key → TensorMeta for F8 weights in this shard so
    // we can validate scale shapes.
    let mut f8_weights: HashMap<String, TensorMeta> = HashMap::new();
    for (name, val) in &header {
        if name == "__metadata__" {
            continue;
        }
        if name.ends_with(".weight") {
            let meta = TensorMeta::from_value(val, name, input_path)?;
            if meta.dtype == "F8_E4M3" {
                f8_weights.insert(name.clone(), meta);
            }
        }
    }

    let mut summary = ShardSummary::default();
    let mut new_header: Map<String, Value> = Map::new();

    // Preserve __metadata__ if present.
    if let Some(m) = header.get("__metadata__") {
        new_header.insert("__metadata__".to_string(), m.clone());
    }

    // Process tensors in original order (serde_json with preserve_order is not
    // guaranteed here, but we iterate the input header to build the output).
    for (name, val) in &header {
        if name == "__metadata__" {
            continue;
        }

        let meta = TensorMeta::from_value(val, name, input_path)?;

        if name.ends_with(".weight_scale_inv") {
            let prefix = name.strip_suffix(".weight_scale_inv").unwrap();
            let weight_key = format!("{prefix}.weight");

            let w_meta = f8_weights.get(&weight_key).ok_or_else(|| {
                anyhow::anyhow!(
                    "scale tensor {name} has no F8_E4M3 companion {weight_key} in this shard"
                )
            })?;

            // Validate shape: weight shape [out, in], scale shape [out, in/32].
            let w_in = w_meta.shape.last().copied().unwrap_or(0);
            let s_last = meta.shape.last().copied().unwrap_or(0);
            if w_in <= 0 || w_in % 32 != 0 {
                bail!(
                    "MXFP8 repack: {weight_key} last dim {w_in} is not a positive multiple of 32"
                );
            }
            if s_last * 32 != w_in {
                bail!(
                    "MXFP8 repack: scale shape {:?} last dim {s_last}*32={} does not match \
                     weight last dim {w_in} for {name}",
                    meta.shape,
                    s_last * 32
                );
            }

            // Rename weight_scale_inv → scales; dtype/shape/bytes unchanged.
            let new_name = format!("{prefix}.scales");
            new_header.insert(new_name, meta.to_value());
            summary.renamed += 1;
        } else if name.ends_with(".weight") && meta.dtype == "F8_E4M3" {
            // Check companion scale exists anywhere in the checkpoint.
            let scale_key = format!("{name}_scale_inv");
            if !weight_map.contains_key(&scale_key) {
                bail!(
                    "MXFP8 repack: F8_E4M3 tensor {name} has no companion {scale_key} \
                     anywhere in the checkpoint — cannot safely repack"
                );
            }

            // Retype to U32, reshape [out, in] → [out, in/4]; bytes unchanged.
            let w_in = meta.shape.last().copied().unwrap_or(0);
            if w_in <= 0 || w_in % 4 != 0 {
                bail!("MXFP8 repack: {name} last dim {w_in} is not a positive multiple of 4");
            }
            let mut new_shape = meta.shape.clone();
            *new_shape.last_mut().unwrap() = w_in / 4;
            let new_meta = TensorMeta {
                dtype: "U32".to_string(),
                shape: new_shape,
                data_offsets: meta.data_offsets,
            };
            new_header.insert(name.clone(), new_meta.to_value());
            summary.repacked += 1;
        } else {
            // Pass through unchanged.
            new_header.insert(name.clone(), meta.to_value());
            summary.passthrough += 1;
        }
    }

    let new_header_json = serde_json::to_vec(&Value::Object(new_header))
        .context("serializing new safetensors header")?;

    // Write to a .tmp file, rename on success.
    let tmp_path = output_path.with_extension("safetensors.tmp");
    {
        let out_file = File::create(&tmp_path)
            .with_context(|| format!("creating output shard tmp {}", tmp_path.display()))?;
        let mut writer = BufWriter::new(out_file);

        write_st_header(&mut writer, &new_header_json)?;

        // Seek input to data start and bulk-copy the data region.
        input_file
            .seek(SeekFrom::Start(data_start_in_input))
            .context("seeking input to data region")?;

        let bytes_copied = copy_streamed(&mut input_file, &mut writer)
            .with_context(|| format!("copying data region from {}", input_path.display()))?;
        summary.bytes_written = new_header_json.len() as u64 + 8 + bytes_copied;

        writer.flush().context("flushing output shard")?;
    }

    // Rename tmp → final path.
    fs::rename(&tmp_path, output_path).with_context(|| {
        format!(
            "renaming {} to {}",
            tmp_path.display(),
            output_path.display()
        )
    })?;

    Ok(summary)
}

/// Stream-copy from `reader` to `writer` using a fixed-size buffer.
/// Returns total bytes copied.
fn copy_streamed(reader: &mut impl Read, writer: &mut impl Write) -> Result<u64> {
    let mut buf = vec![0u8; COPY_BUF_SIZE];
    let mut total = 0u64;
    loop {
        let n = reader.read(&mut buf).context("reading data region")?;
        if n == 0 {
            break;
        }
        writer.write_all(&buf[..n]).context("writing data region")?;
        total += n as u64;
    }
    Ok(total)
}

// ─── index rewrite ────────────────────────────────────────────────────────────

fn rewrite_index(input_dir: &Path, output_dir: &Path) -> Result<()> {
    let index_path = input_dir.join("model.safetensors.index.json");
    let raw = fs::read_to_string(&index_path)
        .with_context(|| format!("reading {}", index_path.display()))?;
    let mut val: Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", index_path.display()))?;

    // Rename weight_scale_inv → scales in weight_map.
    if let Some(wm) = val
        .as_object_mut()
        .and_then(|o| o.get_mut("weight_map"))
        .and_then(Value::as_object_mut)
    {
        let renames: Vec<(String, String, Value)> = wm
            .iter()
            .filter(|(k, _)| k.ends_with(".weight_scale_inv"))
            .map(|(k, v)| {
                let new_key = k
                    .strip_suffix(".weight_scale_inv")
                    .map(|p| format!("{p}.scales"))
                    .unwrap();
                (k.clone(), new_key, v.clone())
            })
            .collect();
        for (old, new, shard) in renames {
            wm.remove(&old);
            wm.insert(new, shard);
        }
    }

    let out_path = output_dir.join("model.safetensors.index.json");
    let out_json = serde_json::to_string_pretty(&val).context("serializing index")?;
    fs::write(&out_path, out_json).with_context(|| format!("writing {}", out_path.display()))?;
    Ok(())
}

// ─── config rewrite ───────────────────────────────────────────────────────────

fn rewrite_config(input_dir: &Path, output_dir: &Path) -> Result<()> {
    let cfg_path = input_dir.join("config.json");
    let raw =
        fs::read_to_string(&cfg_path).with_context(|| format!("reading {}", cfg_path.display()))?;
    let mut val: Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", cfg_path.display()))?;

    if let Some(obj) = val.as_object_mut() {
        // Remove HF quantization_config.
        obj.remove("quantization_config");
        // Add MLX-style quantization.
        obj.insert(
            "quantization".to_string(),
            serde_json::json!({"group_size": 32, "bits": 8, "mode": "mxfp8"}),
        );
    }

    let out_path = output_dir.join("config.json");
    let out_json = serde_json::to_string_pretty(&val).context("serializing config")?;
    fs::write(&out_path, out_json).with_context(|| format!("writing {}", out_path.display()))?;
    Ok(())
}

// ─── ancillary file copy ──────────────────────────────────────────────────────

/// Files to skip when copying ancillary files.
fn should_skip_file(name: &str) -> bool {
    name.starts_with(".git")
        || name == ".gitignore"
        || name.ends_with(".lock")
        || name.ends_with(".index.json")     // we rewrite this explicitly
        || name == "config.json"              // we rewrite this explicitly
        || name.ends_with(".safetensors")     // shards handled separately
        || name == ".cache"
}

fn copy_ancillary_files(input_dir: &Path, output_dir: &Path) -> Result<usize> {
    let mut count = 0;
    for entry in fs::read_dir(input_dir)
        .with_context(|| format!("reading input directory {}", input_dir.display()))?
    {
        let entry = entry.context("reading dir entry")?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if should_skip_file(&name_str) {
            continue;
        }
        let path = entry.path();
        if path.is_file() {
            let dest = output_dir.join(&name);
            fs::copy(&path, &dest)
                .with_context(|| format!("copying {} to {}", path.display(), dest.display()))?;
            count += 1;
        }
    }
    Ok(count)
}

// ─── verification ─────────────────────────────────────────────────────────────

/// Verify a converted shard against its input:
/// - Tensor count matches.
/// - Every repacked entry has dtype U32 and shape [out, in/4].
/// - Every renamed entry is `*.scales` with original dtype/shape.
/// - First and last tensor payload byte-equality (streamed, bounded memory).
fn verify_shard(
    input_path: &Path,
    output_path: &Path,
    weight_map: &HashMap<String, String>,
) -> Result<()> {
    let mut inf = File::open(input_path)
        .with_context(|| format!("verify: opening input {}", input_path.display()))?;
    let mut outf = File::open(output_path)
        .with_context(|| format!("verify: opening output {}", output_path.display()))?;

    let (_, in_header) = read_st_header(&mut inf, input_path)?;
    let in_data_start = inf.stream_position().context("verify: input data start")?;
    let (_, out_header) = read_st_header(&mut outf, output_path)?;
    let out_data_start = outf
        .stream_position()
        .context("verify: output data start")?;

    // Count non-metadata entries.
    let in_count = in_header.keys().filter(|k| *k != "__metadata__").count();
    let out_count = out_header.keys().filter(|k| *k != "__metadata__").count();
    if in_count != out_count {
        bail!(
            "verify: {}: tensor count mismatch: input={in_count} output={out_count}",
            output_path.display()
        );
    }

    // Collect ordered (name, meta) for first/last payload check.
    let mut in_entries: Vec<(String, TensorMeta)> = in_header
        .iter()
        .filter(|(k, _)| *k != "__metadata__")
        .map(|(k, v)| TensorMeta::from_value(v, k, input_path).map(|m| (k.clone(), m)))
        .collect::<Result<Vec<_>>>()?;
    in_entries.sort_by_key(|(_, m)| m.data_offsets[0]);

    // Check renamed/retyped entries.
    for (in_name, in_meta) in &in_entries {
        if in_name.ends_with(".weight_scale_inv") {
            let prefix = in_name.strip_suffix(".weight_scale_inv").unwrap();
            let expected_out_name = format!("{prefix}.scales");
            match out_header.get(&expected_out_name) {
                None => bail!(
                    "verify: {}: expected scales tensor {expected_out_name} missing from output",
                    output_path.display()
                ),
                Some(out_val) => {
                    let out_meta =
                        TensorMeta::from_value(out_val, &expected_out_name, output_path)?;
                    if out_meta.dtype != in_meta.dtype {
                        bail!(
                            "verify: {}: {expected_out_name} dtype changed: \
                             {} → {} (should be unchanged)",
                            output_path.display(),
                            in_meta.dtype,
                            out_meta.dtype
                        );
                    }
                    if out_meta.shape != in_meta.shape {
                        bail!(
                            "verify: {}: {expected_out_name} shape changed: \
                             {:?} → {:?} (should be unchanged)",
                            output_path.display(),
                            in_meta.shape,
                            out_meta.shape
                        );
                    }
                }
            }
        } else if in_name.ends_with(".weight") && in_meta.dtype == "F8_E4M3" {
            let scale_key = format!("{in_name}_scale_inv");
            if !weight_map.contains_key(&scale_key) {
                // Not an MXFP8 weight — skip repack check.
                continue;
            }
            match out_header.get(in_name.as_str()) {
                None => bail!(
                    "verify: {}: repacked weight {in_name} missing from output",
                    output_path.display()
                ),
                Some(out_val) => {
                    let out_meta = TensorMeta::from_value(out_val, in_name, output_path)?;
                    if out_meta.dtype != "U32" {
                        bail!(
                            "verify: {}: {in_name} should be U32 after repack, got {}",
                            output_path.display(),
                            out_meta.dtype
                        );
                    }
                    let expected_in = in_meta.shape.last().copied().unwrap_or(0);
                    let expected_out = out_meta.shape.last().copied().unwrap_or(0);
                    if expected_out * 4 != expected_in {
                        bail!(
                            "verify: {}: {in_name} last dim: in={expected_in} out={expected_out} \
                             (expected out*4==in)",
                            output_path.display()
                        );
                    }
                }
            }
        }
    }

    // Byte-equality check on first and last tensor payloads.
    if let Some(first) = in_entries.first() {
        verify_payload_bytes(&mut inf, in_data_start, &mut outf, out_data_start, first)?;
    }
    // Check the last tensor only when there is more than one (otherwise it is
    // the same tensor as first and was already checked above).
    if let Some(last) = (in_entries.len() > 1).then(|| in_entries.last()).flatten() {
        verify_payload_bytes(&mut inf, in_data_start, &mut outf, out_data_start, last)?;
    }

    Ok(())
}

/// Stream-compare the payload of a specific tensor between input and output
/// files (using the input tensor's data_offsets in both files, since data
/// regions are identical).
fn verify_payload_bytes(
    inf: &mut File,
    in_data_start: u64,
    outf: &mut File,
    out_data_start: u64,
    entry: &(String, TensorMeta),
) -> Result<()> {
    let (name, meta) = entry;
    let [start, end] = meta.data_offsets;
    let len = end - start;
    if len == 0 {
        return Ok(());
    }

    inf.seek(SeekFrom::Start(in_data_start + start))
        .with_context(|| format!("seek input payload for {name}"))?;
    outf.seek(SeekFrom::Start(out_data_start + start))
        .with_context(|| format!("seek output payload for {name}"))?;

    let chunk = len.min(COPY_BUF_SIZE as u64) as usize;
    let mut in_buf = vec![0u8; chunk];
    let mut out_buf = vec![0u8; chunk];
    let mut remaining = len;
    let mut offset = 0u64;

    while remaining > 0 {
        let to_read = remaining.min(chunk as u64) as usize;
        inf.read_exact(&mut in_buf[..to_read])
            .with_context(|| format!("reading input payload for {name} at offset {offset}"))?;
        outf.read_exact(&mut out_buf[..to_read])
            .with_context(|| format!("reading output payload for {name} at offset {offset}"))?;
        if in_buf[..to_read] != out_buf[..to_read] {
            bail!(
                "verify: byte mismatch in tensor {name} payload at data offset {}..{}",
                start + offset,
                start + offset + to_read as u64
            );
        }
        remaining -= to_read as u64;
        offset += to_read as u64;
    }

    Ok(())
}

// ─── guard: no existing shard files in output ─────────────────────────────────

fn check_output_clean(output_dir: &Path) -> Result<()> {
    if !output_dir.exists() {
        return Ok(());
    }
    let has_shards = fs::read_dir(output_dir)
        .with_context(|| format!("checking output dir {}", output_dir.display()))?
        .filter_map(|e| e.ok())
        .any(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"));
    if has_shards {
        bail!(
            "output directory {} already contains .safetensors files; \
             remove them before running mxfp8_repack to prevent overwriting a conversion",
            output_dir.display()
        );
    }
    Ok(())
}

// ─── main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    check_output_clean(&args.output)?;
    fs::create_dir_all(&args.output)
        .with_context(|| format!("creating output directory {}", args.output.display()))?;

    let (_index_val, weight_map) = load_index(&args.input)?;

    // Collect shard filenames in sorted order.
    let shard_files: Vec<String> = {
        let mut seen = std::collections::BTreeSet::new();
        for shard in weight_map.values() {
            seen.insert(shard.clone());
        }
        seen.into_iter().collect()
    };

    if shard_files.is_empty() {
        bail!("no shards found in model.safetensors.index.json weight_map");
    }

    println!(
        "Converting {} shard(s) from {} → {}",
        shard_files.len(),
        args.input.display(),
        args.output.display()
    );

    let mut total = ShardSummary::default();
    let mut completed_shards: Vec<String> = Vec::new();
    let mut failed = false;

    for (i, shard_file) in shard_files.iter().enumerate() {
        let input_shard = args.input.join(shard_file);
        let output_shard = args.output.join(shard_file);
        print!(
            "[{:>3}/{:>3}] {} ... ",
            i + 1,
            shard_files.len(),
            shard_file
        );
        io::stdout().flush().ok();

        match convert_shard(&input_shard, &output_shard, &weight_map, shard_file) {
            Ok(summary) => {
                println!(
                    "repacked={} renamed={} pass={} bytes={:.1} MiB",
                    summary.repacked,
                    summary.renamed,
                    summary.passthrough,
                    summary.bytes_written as f64 / (1024.0 * 1024.0)
                );
                total.repacked += summary.repacked;
                total.renamed += summary.renamed;
                total.passthrough += summary.passthrough;
                total.bytes_written += summary.bytes_written;
                completed_shards.push(shard_file.clone());
            }
            Err(e) => {
                eprintln!("ERROR: {e}");
                eprintln!("Completed shards before failure: {:?}", completed_shards);
                failed = true;
                break;
            }
        }
    }

    if failed {
        bail!(
            "conversion failed; {} of {} shards completed: {:?}",
            completed_shards.len(),
            shard_files.len(),
            completed_shards
        );
    }

    // All shards succeeded — write index, config, and ancillary files.
    rewrite_index(&args.input, &args.output)?;
    rewrite_config(&args.input, &args.output)?;
    let ancillary = copy_ancillary_files(&args.input, &args.output)?;

    println!("\nConversion complete:");
    println!("  Shards:      {}", shard_files.len());
    println!(
        "  Repacked:    {} weight tensors (F8_E4M3 → U32)",
        total.repacked
    );
    println!(
        "  Renamed:     {} scale tensors (weight_scale_inv → scales)",
        total.renamed
    );
    println!("  Passthrough: {} tensors unchanged", total.passthrough);
    println!(
        "  Total bytes: {:.1} MiB",
        total.bytes_written as f64 / (1024.0 * 1024.0)
    );
    println!("  Ancillary:   {} files copied", ancillary);

    // Optional verification pass.
    if args.verify {
        println!("\nRunning verification pass ...");
        let mut all_ok = true;
        for (i, shard_file) in shard_files.iter().enumerate() {
            let input_shard = args.input.join(shard_file);
            let output_shard = args.output.join(shard_file);
            print!(
                "  [{:>3}/{:>3}] verify {} ... ",
                i + 1,
                shard_files.len(),
                shard_file
            );
            io::stdout().flush().ok();
            match verify_shard(&input_shard, &output_shard, &weight_map) {
                Ok(()) => println!("OK"),
                Err(e) => {
                    eprintln!("FAIL: {e}");
                    all_ok = false;
                }
            }
        }
        if !all_ok {
            bail!("verification failed on one or more shards");
        }
        println!("All shards verified OK.");
    }

    Ok(())
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::TempDir;

    // ── synthetic safetensors builder ─────────────────────────────────────────

    /// Build a minimal safetensors file from a list of (name, dtype, shape, data) tuples.
    fn build_safetensors(tensors: &[(&str, &str, &[i64], &[u8])]) -> Vec<u8> {
        // Compute data_offsets.
        let mut offset = 0u64;
        let mut entries: Vec<(String, TensorMeta, Vec<u8>)> = Vec::new();
        for &(name, dtype, shape, data) in tensors {
            let start = offset;
            let end = start + data.len() as u64;
            offset = end;
            entries.push((
                name.to_string(),
                TensorMeta {
                    dtype: dtype.to_string(),
                    shape: shape.to_vec(),
                    data_offsets: [start, end],
                },
                data.to_vec(),
            ));
        }

        // Build header JSON.
        let mut header_map = Map::new();
        for (name, meta, _) in &entries {
            header_map.insert(name.clone(), meta.to_value());
        }
        let hdr_json = serde_json::to_vec(&Value::Object(header_map)).unwrap();

        // Build file bytes: 8-byte LE length + JSON + data.
        let mut out = Vec::new();
        let len = hdr_json.len() as u64;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&hdr_json);
        for (_, _, data) in &entries {
            out.extend_from_slice(data);
        }
        out
    }

    /// Write a safetensors file to `dir/filename`.
    fn write_shard(dir: &Path, filename: &str, tensors: &[(&str, &str, &[i64], &[u8])]) {
        let bytes = build_safetensors(tensors);
        fs::write(dir.join(filename), bytes).unwrap();
    }

    /// Build a minimal `model.safetensors.index.json`.
    fn write_index(dir: &Path, entries: &[(&str, &str)]) {
        let weight_map: Map<String, Value> = entries
            .iter()
            .map(|(k, v)| (k.to_string(), Value::String(v.to_string())))
            .collect();
        let index = serde_json::json!({
            "metadata": {"total_size": 1234},
            "weight_map": weight_map,
        });
        fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_string_pretty(&index).unwrap(),
        )
        .unwrap();
    }

    /// Build a minimal `config.json` with HF quantization_config.
    fn write_config(dir: &Path) {
        let cfg = serde_json::json!({
            "model_type": "test",
            "quantization_config": {
                "quant_method": "mxfp8",
                "weight_block_size": [1, 32],
                "activation_scheme": "dynamic",
                "ignored_layers": [],
            },
            "extra_field": "preserved",
        });
        fs::write(
            dir.join("config.json"),
            serde_json::to_string_pretty(&cfg).unwrap(),
        )
        .unwrap();
    }

    /// Flip one byte in `file` at byte `offset` within the data region (i.e.,
    /// after the header prefix).
    fn flip_byte_in_data(file: &Path, data_byte_offset: u64) {
        let mut bytes = fs::read(file).unwrap();
        // Parse header to find data region start.
        let hdr_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let data_start = 8 + hdr_len;
        let abs = data_start + data_byte_offset as usize;
        bytes[abs] ^= 0xFF;
        fs::write(file, bytes).unwrap();
    }

    // ── synthetic 2-shard setup ───────────────────────────────────────────────

    struct TestCheckpoint {
        input_dir: TempDir,
        output_dir: TempDir,
        weight_map: HashMap<String, String>,
    }

    impl TestCheckpoint {
        /// 2-shard checkpoint:
        /// shard1: `foo.weight` (F8, [4,8]), `foo.weight_scale_inv` (U8, [4,1])
        /// shard2: `bar.weight` (BF16, [4,4]) passthrough
        fn new() -> Self {
            let input_dir = TempDir::new().unwrap();
            let output_dir = TempDir::new().unwrap();

            // Weight [4, 32]: 4*32 = 128 fp8 bytes. Block size is 32, so
            // scale shape is [4, 1]. After repack: weight becomes U32 [4, 8].
            let fp8_data: Vec<u8> = (0u8..128).collect(); // 4*32 = 128 bytes
            let scale_data: Vec<u8> = vec![127u8; 4]; // [4, 1] = 4 bytes

            // BF16 passthrough: 4*4*2 = 32 bytes
            let bf16_data: Vec<u8> = (0u8..32).collect();

            write_shard(
                input_dir.path(),
                "model-00001-of-00002.safetensors",
                &[
                    ("foo.weight", "F8_E4M3", &[4, 32], &fp8_data),
                    ("foo.weight_scale_inv", "U8", &[4, 1], &scale_data),
                ],
            );

            write_shard(
                input_dir.path(),
                "model-00002-of-00002.safetensors",
                &[("bar.weight", "BF16", &[4, 4], &bf16_data)],
            );

            let mut weight_map = HashMap::new();
            weight_map.insert(
                "foo.weight".to_string(),
                "model-00001-of-00002.safetensors".to_string(),
            );
            weight_map.insert(
                "foo.weight_scale_inv".to_string(),
                "model-00001-of-00002.safetensors".to_string(),
            );
            weight_map.insert(
                "bar.weight".to_string(),
                "model-00002-of-00002.safetensors".to_string(),
            );

            write_index(
                input_dir.path(),
                &[
                    ("foo.weight", "model-00001-of-00002.safetensors"),
                    ("foo.weight_scale_inv", "model-00001-of-00002.safetensors"),
                    ("bar.weight", "model-00002-of-00002.safetensors"),
                ],
            );
            write_config(input_dir.path());

            TestCheckpoint {
                input_dir,
                output_dir,
                weight_map,
            }
        }

        fn input(&self) -> &Path {
            self.input_dir.path()
        }

        fn output(&self) -> &Path {
            self.output_dir.path()
        }

        fn convert_shard1(&self) -> Result<ShardSummary> {
            let input_shard = self.input().join("model-00001-of-00002.safetensors");
            let output_shard = self.output().join("model-00001-of-00002.safetensors");
            convert_shard(
                &input_shard,
                &output_shard,
                &self.weight_map,
                "model-00001-of-00002.safetensors",
            )
        }

        fn convert_shard2(&self) -> Result<ShardSummary> {
            let input_shard = self.input().join("model-00002-of-00002.safetensors");
            let output_shard = self.output().join("model-00002-of-00002.safetensors");
            convert_shard(
                &input_shard,
                &output_shard,
                &self.weight_map,
                "model-00002-of-00002.safetensors",
            )
        }
    }

    // ── helpers to read output header ─────────────────────────────────────────

    fn read_output_header(path: &Path) -> Map<String, Value> {
        let mut f = File::open(path).unwrap();
        let (_, map) = read_st_header(&mut f, path).unwrap();
        map
    }

    // ── happy path ────────────────────────────────────────────────────────────

    #[test]
    fn happy_path_names_dtypes_shapes_correct() {
        let tc = TestCheckpoint::new();
        let summary = tc.convert_shard1().expect("shard1 conversion must succeed");
        assert_eq!(summary.repacked, 1, "one weight repacked");
        assert_eq!(summary.renamed, 1, "one scale renamed");
        assert_eq!(summary.passthrough, 0, "no passthrough in shard1");

        let output_path = tc.output().join("model-00001-of-00002.safetensors");
        let hdr = read_output_header(&output_path);

        // foo.weight → U32, shape [4, 8]  (32 fp8 bytes per row / 4 = 8 u32s)
        let w = hdr.get("foo.weight").expect("foo.weight must be in output");
        let wm = TensorMeta::from_value(w, "foo.weight", &output_path).unwrap();
        assert_eq!(wm.dtype, "U32", "weight must be retyped to U32");
        assert_eq!(wm.shape, vec![4, 8], "weight shape must be [4, 32/4=8]");

        // foo.weight_scale_inv → foo.scales, U8, shape [4, 1]
        assert!(
            !hdr.contains_key("foo.weight_scale_inv"),
            "old scale key must be gone"
        );
        let s = hdr
            .get("foo.scales")
            .expect("foo.scales must exist in output");
        let sm = TensorMeta::from_value(s, "foo.scales", &output_path).unwrap();
        assert_eq!(sm.dtype, "U8", "scales dtype must be unchanged");
        assert_eq!(sm.shape, vec![4, 1], "scales shape must be unchanged");
    }

    #[test]
    fn happy_path_payload_bytes_unchanged() {
        let tc = TestCheckpoint::new();
        tc.convert_shard1().unwrap();
        tc.convert_shard2().unwrap();

        let in1 = tc.input().join("model-00001-of-00002.safetensors");
        let out1 = tc.output().join("model-00001-of-00002.safetensors");

        // Read full input data region.
        let mut inf = File::open(&in1).unwrap();
        let (_, _) = read_st_header(&mut inf, &in1).unwrap();
        let in_data_start = inf.stream_position().unwrap();
        let mut in_data = Vec::new();
        inf.read_to_end(&mut in_data).unwrap();

        // Read full output data region.
        let mut outf = File::open(&out1).unwrap();
        let (_, _) = read_st_header(&mut outf, &out1).unwrap();
        let mut out_data = Vec::new();
        outf.read_to_end(&mut out_data).unwrap();

        let _ = in_data_start; // suppress unused warning
        assert_eq!(
            in_data, out_data,
            "data region must be byte-for-byte identical"
        );
    }

    #[test]
    fn happy_path_passthrough_tensor_unchanged() {
        let tc = TestCheckpoint::new();
        let summary = tc.convert_shard2().expect("shard2 conversion must succeed");
        assert_eq!(summary.passthrough, 1);
        assert_eq!(summary.repacked, 0);
        assert_eq!(summary.renamed, 0);

        let output_path = tc.output().join("model-00002-of-00002.safetensors");
        let hdr = read_output_header(&output_path);
        let b = hdr.get("bar.weight").expect("bar.weight must pass through");
        let bm = TensorMeta::from_value(b, "bar.weight", &output_path).unwrap();
        assert_eq!(bm.dtype, "BF16");
        assert_eq!(bm.shape, vec![4, 4]);
    }

    // ── error cases ───────────────────────────────────────────────────────────

    #[test]
    fn orphaned_scale_is_hard_error() {
        let tmp = TempDir::new().unwrap();
        let out = TempDir::new().unwrap();
        // Only scale, no weight.
        write_shard(
            tmp.path(),
            "model.safetensors",
            &[("x.weight_scale_inv", "U8", &[4, 1], &[0u8; 4])],
        );
        // weight_map has the scale but no companion weight
        let mut wm = HashMap::new();
        wm.insert(
            "x.weight_scale_inv".to_string(),
            "model.safetensors".to_string(),
        );
        let err = convert_shard(
            &tmp.path().join("model.safetensors"),
            &out.path().join("model.safetensors"),
            &wm,
            "model.safetensors",
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("x.weight_scale_inv"),
            "error must name the orphaned tensor; got: {msg}"
        );
    }

    #[test]
    fn split_pair_is_hard_error() {
        let tmp = TempDir::new().unwrap();
        let out = TempDir::new().unwrap();
        // Shard A has the scale.
        write_shard(
            tmp.path(),
            "shard_a.safetensors",
            &[("y.weight_scale_inv", "U8", &[4, 1], &[0u8; 4])],
        );
        // weight_map says companion weight is in shard_b.
        let mut wm = HashMap::new();
        wm.insert("y.weight".to_string(), "shard_b.safetensors".to_string());
        wm.insert(
            "y.weight_scale_inv".to_string(),
            "shard_a.safetensors".to_string(),
        );
        let err = convert_shard(
            &tmp.path().join("shard_a.safetensors"),
            &out.path().join("shard_a.safetensors"),
            &wm,
            "shard_a.safetensors",
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("split pair") || msg.contains("shard_b"),
            "error must describe the split pair; got: {msg}"
        );
    }

    #[test]
    fn no_overwrite_guard() {
        let tmp = TempDir::new().unwrap();
        // Create a shard file in output.
        fs::write(tmp.path().join("model.safetensors"), b"fake").unwrap();
        let err = check_output_clean(tmp.path()).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("already contains"),
            "error must say output already contains shards; got: {msg}"
        );
    }

    #[test]
    fn index_rewrite_renames_scale_keys() {
        let tmp = TempDir::new().unwrap();
        let out = TempDir::new().unwrap();
        write_index(
            tmp.path(),
            &[
                ("a.weight", "s1.safetensors"),
                ("a.weight_scale_inv", "s1.safetensors"),
                ("b.weight", "s2.safetensors"),
            ],
        );
        rewrite_index(tmp.path(), out.path()).unwrap();

        let raw = fs::read_to_string(out.path().join("model.safetensors.index.json")).unwrap();
        let val: Value = serde_json::from_str(&raw).unwrap();
        let wm = val["weight_map"].as_object().unwrap();

        assert!(
            wm.contains_key("a.scales"),
            "a.scales must appear after rewrite"
        );
        assert!(
            !wm.contains_key("a.weight_scale_inv"),
            "a.weight_scale_inv must be gone after rewrite"
        );
        assert!(wm.contains_key("a.weight"), "a.weight must survive");
        assert!(wm.contains_key("b.weight"), "b.weight must survive");
    }

    #[test]
    fn config_rewrite_removes_quantization_config_adds_quantization() {
        let tmp = TempDir::new().unwrap();
        let out = TempDir::new().unwrap();
        write_config(tmp.path());
        rewrite_config(tmp.path(), out.path()).unwrap();

        let raw = fs::read_to_string(out.path().join("config.json")).unwrap();
        let val: Value = serde_json::from_str(&raw).unwrap();

        assert!(
            val.get("quantization_config").is_none(),
            "quantization_config must be removed"
        );
        let q = val.get("quantization").expect("quantization must be added");
        assert_eq!(q["group_size"], 32);
        assert_eq!(q["bits"], 8);
        assert_eq!(q["mode"], "mxfp8");

        // Ensure other fields preserved.
        assert_eq!(val["model_type"], "test");
        assert_eq!(val["extra_field"], "preserved");
    }

    #[test]
    fn verify_catches_flipped_byte() {
        let tc = TestCheckpoint::new();
        tc.convert_shard1().unwrap();

        let output_path = tc.output().join("model-00001-of-00002.safetensors");
        // Flip the first data byte (offset 0 in data region).
        flip_byte_in_data(&output_path, 0);

        let err = verify_shard(
            &tc.input().join("model-00001-of-00002.safetensors"),
            &output_path,
            &tc.weight_map,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("byte mismatch") || msg.contains("mismatch"),
            "verify must detect the corrupted byte; got: {msg}"
        );
    }

    #[test]
    fn verify_passes_on_clean_conversion() {
        let tc = TestCheckpoint::new();
        tc.convert_shard1().unwrap();
        tc.convert_shard2().unwrap();

        verify_shard(
            &tc.input().join("model-00001-of-00002.safetensors"),
            &tc.output().join("model-00001-of-00002.safetensors"),
            &tc.weight_map,
        )
        .expect("verify must pass on a correctly converted shard");

        verify_shard(
            &tc.input().join("model-00002-of-00002.safetensors"),
            &tc.output().join("model-00002-of-00002.safetensors"),
            &tc.weight_map,
        )
        .expect("verify must pass on the passthrough shard");
    }

    #[test]
    fn shape_mismatch_is_hard_error() {
        let tmp = TempDir::new().unwrap();
        let out = TempDir::new().unwrap();
        // Weight [4, 32], scale [4, 2] — scale last dim (2) * 32 = 64 ≠ 32 → mismatch.
        let fp8_data = vec![0u8; 128]; // [4, 32]
        let scale_data = vec![127u8; 8]; // [4, 2]
        write_shard(
            tmp.path(),
            "model.safetensors",
            &[
                ("z.weight", "F8_E4M3", &[4, 32], &fp8_data),
                ("z.weight_scale_inv", "U8", &[4, 2], &scale_data),
            ],
        );
        let mut wm = HashMap::new();
        wm.insert("z.weight".to_string(), "model.safetensors".to_string());
        wm.insert(
            "z.weight_scale_inv".to_string(),
            "model.safetensors".to_string(),
        );
        let err = convert_shard(
            &tmp.path().join("model.safetensors"),
            &out.path().join("model.safetensors"),
            &wm,
            "model.safetensors",
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("z.weight") || msg.contains("shape") || msg.contains("mismatch"),
            "error must describe shape mismatch; got: {msg}"
        );
    }
}
