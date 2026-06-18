#include <metal_stdlib>
using namespace metal;

// W is our window stride (e.g., 128 or 256)
// The Threadgroup size must exactly equal W.
constant uint W = 128;
constant uint LIMIT = 2 * W;

[[kernel]] void sliding_topk_kernel(
    device const float* block_scores       [[buffer(0)]],
    device int* out_indices                [[buffer(1)]],
    constant int& k_blocks                 [[buffer(2)]],
    constant int& num_blocks               [[buffer(3)]],
    uint3 grid_id                          [[threadgroup_position_in_grid]],
    uint tid                               [[thread_position_in_threadgroup]]
) {
    uint head_idx = grid_id.x;
    device const float* head_scores = block_scores + (head_idx * num_blocks);
    
    // Allocate SRAM buffer for 2W elements
    threadgroup float local_scores[LIMIT];
    threadgroup int local_indices[LIMIT];
    
    // 1. Initialize the running Top-W buffer with -Infinity
    local_scores[tid] = -INFINITY;
    local_indices[tid] = -1;
    
    // Wait for all threads to finish initialization
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 2. Slide the window across the entire sequence length
    for (uint offset = 0; offset < (uint)num_blocks; offset += W) {
        
        // Stream the next W elements into the SECOND half of the buffer
        uint fetch_idx = offset + tid;
        if (fetch_idx < (uint)num_blocks) {
            local_scores[tid + W] = head_scores[fetch_idx];
            local_indices[tid + W] = (int)fetch_idx;
        } else {
            // Safe padding for out-of-bounds chunks
            local_scores[tid + W] = -INFINITY;
            local_indices[tid + W] = -1;
        }
        
        // Ensure the new chunk is fully loaded before sorting
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // 3. In-place Bitonic Sort on the 2W buffer
        // Since we have W threads, each thread executes exactly one Compare-and-Swap pair
        uint log2_k = 1;
        for (uint k_step = 2; k_step <= LIMIT; k_step <<= 1, log2_k++) {
            uint log2_j = log2_k - 1;
            for (uint j_step = k_step >> 1; j_step > 0; j_step >>= 1, log2_j--) {
                
                // Bit-magic to determine thread mapping without branch divergence
                uint i = ((tid >> log2_j) << (log2_j + 1)) | (tid & (j_step - 1));
                uint ixj = i ^ j_step;
                
                // For the final step (k_step == LIMIT), this forces a descending global sort.
                // The largest elements will mathematically float to indices 0 -> W-1.
                bool dir_desc = ((i & k_step) == 0);
                
                float score_i = local_scores[i];
                float score_ixj = local_scores[ixj];
                
                if ((score_i < score_ixj) == dir_desc) {
                    local_scores[i] = score_ixj;
                    local_scores[ixj] = score_i;
                    
                    int tmp_idx = local_indices[i];
                    local_indices[i] = local_indices[ixj];
                    local_indices[ixj] = tmp_idx;
                }
                
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        // At this point, the absolute highest values seen so far are safely retained 
        // in local_scores[0 ... W-1]. The window slides forward to overwrite the bottom half.
    }
    
    // 4. Output the finalized Top-K indices directly to Unified Memory
    device int* out_head_indices = out_indices + (head_idx * k_blocks);
    
    if (tid < (uint)k_blocks) {
        out_head_indices[tid] = local_indices[tid];
    }
}
