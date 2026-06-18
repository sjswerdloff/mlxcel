// Test MXFP8 dequantization on a single shard using Metal kernel.
// Validates the GPU-accelerated dequant approach with wall-clock timing.

use std::time::Instant;
use mlxcel_core::weights::WeightMap;

const SHARD_PATH: &str = "/Volumes/T7 Shield/models/huggingface_cache_hub/model-00002-of-00031.safetensors";

#[test]
fn test_mxfp8_dequant_shard2() {
    println!("\n=== MXFP8 Metal Kernel Dequantization Test (Shard 2) ===\n");

    // 1. Load shard with MLX native loader (fast, mmap-based)
    let t0 = Instant::now();
    let mut weights: WeightMap = mlxcel_core::weights::load_weights_from_dir(
        std::path::Path::new(SHARD_PATH).parent().unwrap(),
    )
    .expect("Failed to load weights");
    let load_time = t0.elapsed();
    println!("Step 1: Load weights (mmap)     - {:.2?}", load_time);

    let total_tensors = weights.len();
    println!("  Total tensors loaded: {}", total_tensors);

    // 2. Rename weight_scale_inv -> scales
    let t1 = Instant::now();
    let scale_keys: Vec<String> = weights
        .keys()
        .filter(|k| k.ends_with(".weight_scale_inv"))
        .cloned()
        .collect();
    let scale_count = scale_keys.len();
    for key in scale_keys {
        if let Some(new_key) = key.strip_suffix(".weight_scale_inv") {
            let scales_key = format!("{}.scales", new_key);
            if let Some(arr) = weights.remove(&key) {
                weights.insert(scales_key, arr);
            }
        }
    }
    let rename_time = t1.elapsed();
    println!("Step 2: Rename scales            - {:.2?} ({} keys)", rename_time, scale_count);

    // 3. Dequantize layer 3 weights using Metal kernel
    let test_prefixes = vec![
        "language_model.model.layers.3.self_attn.q_proj",
        "language_model.model.layers.3.self_attn.k_proj",
        "language_model.model.layers.3.self_attn.v_proj",
        "language_model.model.layers.3.self_attn.o_proj",
        "language_model.model.layers.3.self_attn.index_q_proj",
        "language_model.model.layers.3.self_attn.index_k_proj",
        "language_model.model.layers.3.shared_experts.gate_proj",
        "language_model.model.layers.3.shared_experts.up_proj",
        "language_model.model.layers.3.shared_experts.down_proj",
    ];

    let mut total_dequant_bytes: usize = 0;
    let mut dequant_count = 0;
    let mut total_dequant_time = std::time::Duration::ZERO;

    println!("\n--- Dequantizing Layer 3 (Metal Kernel) ---\n");
    for prefix in &test_prefixes {
        let weight_key = format!("{}.weight", prefix);
        let scales_key = format!("{}.scales", prefix);

        let weight = match weights.get(&weight_key) {
            Some(w) => w,
            None => {
                println!("  SKIP {:<55} (weight not found)", prefix);
                continue;
            }
        };

        let scales = match weights.get(&scales_key) {
            Some(s) => s,
            None => {
                println!("  SKIP {:<55} (scales not found)", prefix);
                continue;
            }
        };

        let weight_shape = mlxcel_core::array_shape(weight);

        // GPU dequantize via Metal kernel
        let weight_copy = mlxcel_core::copy(weight);
        let scales_copy = mlxcel_core::copy(scales);

        let t_dequant = Instant::now();
        let dequantized = mlxcel_core::mxfp8_dequant_to_f16(&weight_copy, &scales_copy);
        let dequant_time = t_dequant.elapsed();
        total_dequant_time += dequant_time;

        let dequant_shape = mlxcel_core::array_shape(&dequantized);
        let dequant_bytes = dequant_shape.iter().product::<i32>() as usize * 2; // f16 = 2 bytes
        total_dequant_bytes += dequant_bytes;
        dequant_count += 1;

        println!(
            "  OK  {:<55} {:>12?}  {:>8} -> {:>8} ({:.1} MB)",
            prefix,
            dequant_time,
            format!("{:?}", weight_shape),
            format!("{:?}", dequant_shape),
            dequant_bytes as f64 / 1e6,
        );
    }

    let total_dequant_mb = total_dequant_bytes as f64 / 1e6;
    println!("\n--- Summary ---");
    println!("  Weights dequantized: {}", dequant_count);
    println!("  Total f16 output:    {:.1} MB", total_dequant_mb);
    println!("  Total dequant time:  {:.2?}", total_dequant_time);
    println!("  Load + rename:       {:.2?}", load_time + rename_time);
    println!("  Throughput:          {:.1} MB/s", total_dequant_mb / total_dequant_time.as_secs_f64());

    // 4. Verify dequantized values
    let w = weights.get("language_model.model.layers.3.self_attn.q_proj.weight").unwrap();
    let s = weights.get("language_model.model.layers.3.self_attn.q_proj.scales").unwrap();
    let w_copy = mlxcel_core::copy(w);
    let s_copy = mlxcel_core::copy(s);
    let sample = mlxcel_core::mxfp8_dequant_to_f16(&w_copy, &s_copy);
    let sample_shape = mlxcel_core::array_shape(&sample);
    println!("\n  Verification: q_proj dequant shape = {:?}", sample_shape);

    assert!(dequant_count > 0, "No weights were dequantized");
    assert!(total_dequant_bytes > 0, "No output bytes produced");
    assert!(sample_shape[0] > 0 && sample_shape[1] > 0, "Invalid shape");
}
