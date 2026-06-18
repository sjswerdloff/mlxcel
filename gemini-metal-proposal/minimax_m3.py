import mlx.core as mx
import mlx.nn as nn
from interface import sparse_topk_select, block_sparse_attention

class MiniMaxSparseAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.block_size = 128
        self.top_k = config.msa_top_k
        
        # Q, K, V Projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.kv_hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.kv_hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def __call__(self, x, mask=None, cache=None):
        B, L, D = x.shape
        
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, -1, self.head_dim)
        v = self.v_proj(x).reshape(B, L, -1, self.head_dim)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        # Index Branch: Chunk queries and keys into blocks of 128
        num_blocks = k.shape[1] // self.block_size
        
        # Calculate lightweight block relevance (e.g., mean-pooled dot product per block)
        q_pool = mx.mean(q.reshape(B, -1, self.block_size, self.num_heads, self.head_dim), axis=2)
        k_pool = mx.mean(k.reshape(B, -1, self.block_size, k.shape[2], self.head_dim), axis=2)
        
        # Shape: (Batch, Heads, Num_Blocks)
        block_metrics = mx.matmul(q_pool, k_pool.transpose(0, 1, 3, 2)) 
        
        # 1. Custom Operator: Get Top-K block indices
        active_indices = sparse_topk_select(block_metrics, k=self.top_k)

        # 2. Custom Operator: Compute attention explicitly over selected blocks
        scale = 1.0 / (self.head_dim ** 0.5)
        out = block_sparse_attention(q, k, v, active_indices, scale)

        return self.o_proj(out.reshape(B, L, -1))

