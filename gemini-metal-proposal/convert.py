def map_minimax_weights(hf_state_dict):
    mlx_dict = {}
    for key, tensor in hf_state_dict.items():
        # Map HuggingFace attention keys to the custom MLX sparse attention module
        new_key = key
        if "self_attn.q_proj" in key:
            new_key = key.replace("self_attn", "sparse_attn")
        elif "self_attn.k_proj" in key:
            new_key = key.replace("self_attn", "sparse_attn")
        elif "self_attn.v_proj" in key:
            new_key = key.replace("self_attn", "sparse_attn")
        elif "self_attn.o_proj" in key:
            new_key = key.replace("self_attn", "sparse_attn")
            
        # Standardize weight formats (transpose for MLX linear layers if needed)
        if len(tensor.shape) == 2:
            mlx_dict[new_key] = mx.array(tensor.numpy().T)
        else:
            mlx_dict[new_key] = mx.array(tensor.numpy())
            
    return mlx_dict
