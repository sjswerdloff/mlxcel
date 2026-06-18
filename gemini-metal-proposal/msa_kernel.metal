#include <metal_stdlib>
using namespace metal;

// Device-side constant definition for thread coordination
constant uint THREADS_PER_GROUP = 512;
constant uint TRANSFORMATION_LIMIT = 1024;

[[kernel]] void sparse_topk_kernel(
    device const float* block_scores       [[buffer(0)]],
    device int* out_indices                [[buffer(1)]],
    constant int& k_blocks                 [[buffer(2)]],
    constant int& num_blocks               [[buffer(3)]],
    uint3 grid_id                          [[threadgroup_position_in_grid]],
    uint tid                               [[thread_position_in_threadgroup]]
) {
    // Each threadgroup processes exactly one batch-head combination
    uint head_idx = grid_id.x;
    
    // Allocate high-speed shared SRAM tile for parallel reduction
    threadgroup float local_scores[TRANSFORMATION_LIMIT];
    threadgroup int local_indices[TRANSFORMATION_LIMIT];
    
    // Dual-load layout: 512 threads cooperatively stage 1024 elements
    uint idx1 = tid;
    uint idx2 = tid + THREADS_PER_GROUP;
    
    device const float* head_scores = block_scores + (head_idx * num_blocks);
    
    // Load first chunk with out-of-bounds protection
    if (idx1 < (uint)num_blocks) {
        local_scores[idx1] = head_scores[idx1];
        local_indices[idx1] = (int)idx1;
    } else {
        local_scores[idx1] = -INFINITY; // Pad with negative infinity to sink to the bottom
        local_indices[idx1] = -1;
    }
    
    // Load second chunk
    if (idx2 < (uint)num_blocks) {
        local_scores[idx2] = head_scores[idx2];
        local_indices[idx2] = (int)idx2;
    } else {
        local_scores[idx2] = -INFINITY;
        local_indices[idx2] = -1;
    }
    
    // Memory fence ensuring all tiles are populated before sorting begins
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Iterative Parallel Bitonic Sort
    uint log2_k = 1;
    for (uint k = 2; k <= TRANSFORMATION_LIMIT; k <<= 1, log2_k++) {
        uint log2_j = log2_k - 1;
        for (uint j = k >> 1; j > 0; j >>= 1, log2_j--) {
            
            // Map thread identity to a unique element pair using bit-insertion
            uint i = ((tid >> log2_j) << (log2_j + 1)) | (tid & (j - 1));
            uint ixj = i ^ j;
            
            // Determine structural sorting direction across the interleaved sequences
            // The final phase (k == 1024) forces global descending order
            bool desc = (k == TRANSFORMATION_LIMIT) || ((i & k) == 0);
            
            float score_i = local_scores[i];
            float score_ixj = local_scores[ixj];
            
            // Compare and swap based on execution direction
            if ((score_i < score_ixj) == desc) {
                // Swap scores
                local_scores[i] = score_ixj;
                local_scores[ixj] = score_i;
                
                // Swap corresponding tracking indices
                int temp_idx = local_indices[i];
                local_indices[i] = local_indices[ixj];
                local_indices[ixj] = temp_idx;
            }
            
            // Sync all execution paths within the threadgroup before altering step strides
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    
    // Stream the highly-correlated Top-K selection indices straight out to unified memory
    device int* out_head_indices = out_indices + (head_idx * k_blocks);
    
    if (tid < (uint)k_blocks) {
        out_head_indices[tid] = local_indices[tid];
    }
    if (tid + THREADS_PER_GROUP < (uint)k_blocks) {
        out_head_indices[tid + THREADS_PER_GROUP] = local_indices[tid + THREADS_PER_GROUP];
    }
}
