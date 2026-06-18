#include <mlx/mlx.h>
#include "msa_op.h" // Your previously written sparse_topk operator

extern "C" {
    // We pass opaque pointers across the FFI boundary
    void* msa_sparse_topk_ffi(const void* block_scores_ptr, int k) {
        // Cast the raw pointer back to an MLX array
        const auto* block_scores = static_cast<const mlx::core::array*>(block_scores_ptr);
        
        // Invoke your custom MLX C++ operator
        auto op = std::make_shared<SparseTopK>(k);
        auto result = op->eval({*block_scores})[0];
        
        // Allocate a new array on the heap to pass back to Rust
        return new mlx::core::array(result);
    }

    void* msa_block_sparse_sdpa_ffi(const void* q_ptr, const void* k_ptr, const void* v_ptr, const void* indices_ptr, float scale) {
        const auto* q = static_cast<const mlx::core::array*>(q_ptr);
        const auto* k_cache = static_cast<const mlx::core::array*>(k_ptr);
        const auto* v_cache = static_cast<const mlx::core::array*>(v_ptr);
        const auto* indices = static_cast<const mlx::core::array*>(indices_ptr);

        auto op = std::make_shared<BlockSparseAttention>(scale);
        auto result = op->eval({*q, *k_cache, *v_cache, *indices})[0];
        
        return new mlx::core::array(result);
    }
}
