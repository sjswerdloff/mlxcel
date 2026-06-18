import mlx.core as mx

# Load the custom C++ operators via MLX's fast kernel compiler
# Assuming msa_op.cpp and msa_kernel.metal are in the current directory
mx.eval(mx.fast.metal_custom_op("msa_kernel.metal"))

def sparse_topk_select(block_scores: mx.array, k: int) -> mx.array:
    """
    Inputs:
        block_scores: Float tensor of shape (batch, num_heads, num_blocks)
        k: The number of top blocks to select.
    Outputs:
        indices: Int32 tensor of shape (batch, num_heads, k)
    """
    return mx.core.fast.custom_op(
        "sparse_topk", 
        [block_scores], 
        {"k": k}, 
        out_shapes=[(block_scores.shape[0], block_scores.shape[1], k)],
        out_dtypes=[mx.int32]
    )[0]

def block_sparse_attention(q: mx.array, k: mx.array, v: mx.array, indices: mx.array, scale: float) -> mx.array:
    """
    Executes FlashAttention-style SDPA but restricted to KV blocks specified in `indices`.
    """
    return mx.core.fast.custom_op(
        "block_sparse_sdpa",
        [q, k, v, indices],
        {"scale": scale},
        out_shapes=[q.shape],
        out_dtypes=[q.dtype]
    )[0]

