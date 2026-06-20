import os
import json
import torch
from safetensors.torch import load_file, save_file

# Paths to your model directories
MXFP8_UNPRUNED_DIR = "./Minimax-M3-MXFP8-Full"
NVFP4_REAPED_DIR = "./Minimax-M3-v0-NVFP4-REAP25"
OUTPUT_DIR = "./Minimax-M3-MXFP8-REAP25"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load the index JSON maps
with open(os.path.join(MXFP8_UNPRUNED_DIR, "model.safetensors.index.json"), "r") as f:
    mxfp8_index = json.load(f)

with open(os.path.join(NVFP4_REAPED_DIR, "model.safetensors.index.json"), "r") as f:
    nvfp4_index = json.load(f)

# Group the MXFP8 tensors by their physical file names
mxfp8_shards = {}
for tensor_name, shard_file in mxfp8_index["weight_map"].items():
    mxfp8_shards.setdefault(shard_file, []).append(tensor_name)

new_weight_map = {}

# 2. Iterate through each MXFP8 shard systematically to save RAM
for shard_file, tensor_list in mxfp8_shards.items():
    print(f"Processing shard: {shard_file}...")
    
    # Load the specific unpruned MXFP8 file
    full_shard_path = os.path.join(MXFP8_UNPRUNED_DIR, shard_file)
    full_tensors = load_file(full_shard_path, device="cpu")
    
    # Track down corresponding reference REAP tensors
    # We load them on demand since they might be scattered across different NVFP4 shards
    reap_tensors_cache = {}
    pruned_shard_tensors = {}
    
    for tensor_name in tensor_list:
        full_tensor = full_tensors[tensor_name]
        
        # Look up where this tensor lives in the NVFP4 REAP model
        if tensor_name in nvfp4_index["weight_map"]:
            nvfp4_shard_file = nvfp4_index["weight_map"][tensor_name]
            
            # Load NVFP4 shard on-demand if it isn't in our active cache
            if nvfp4_shard_file not in reap_tensors_cache:
                reap_shard_path = os.path.join(NVFP4_REAPED_DIR, nvfp4_shard_file)
                reap_tensors_cache[nvfp4_shard_file] = load_file(reap_shard_path, device="cpu")
            
            reap_tensor = reap_tensors_cache[nvfp4_shard_file][tensor_name]
            
            # Check for structural MoE mismatch (expert pruning target)
            if full_tensor.shape != reap_tensor.shape and "mlp.experts" in tensor_name:
                num_reap_experts = reap_tensor.shape[0]
                print(f"  -> Pruning {tensor_name}: {full_tensor.shape[0]} to {num_reap_experts} experts")
                
                # Slicing the MXFP8 expert block using the calculated deterministic index length
                pruned_shard_tensors[tensor_name] = full_tensor[:num_reap_experts, ...]
            else:
                # Identity pass for standard layers
                pruned_shard_tensors[tensor_name] = full_tensor
        else:
            # Fallback for structural name variations across versions
            pruned_shard_tensors[tensor_name] = full_tensor
            
        new_weight_map[tensor_name] = shard_file

    # 3. Save the modified shard directly to disk and wipe memory
    output_shard_path = os.path.join(OUTPUT_DIR, shard_file)
    save_file(pruned_shard_tensors, output_shard_path)
    
    del full_tensors
    del pruned_shard_tensors
    del reap_tensors_cache

# 4. Generate the final structural metadata files
print("Finalising configuration metadata indices...")
mxfp8_index["weight_map"] = new_weight_map
with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(mxfp8_index, f, indent=2)

# Customise config to run under standard MXFP8 frameworks
with open(os.path.join(NVFP4_REAPED_DIR, "config.json"), "r") as f:
    config = json.load(f)

config["quantization_config"] = {
    "quant_method": "mxfp8",
    "activation_dtype": "fp8_e4m3fn",
    "weight_dtype": "fp8_e4m3fn"
}

with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

print("Deterministic multi-shard conversion successfully completed!")

