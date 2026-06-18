#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <iostream>

using namespace mlx::core;

class SparseTopK : public fast::MetalCustomOp {
public:
    int k_blocks;
    
    SparseTopK(int k) : k_blocks(k) {}

    std::vector<array> eval(const std::vector<array>& inputs) override {
        if (inputs.empty()) {
            throw std::runtime_error("SparseTopK requires an input block tensor.");
        }
        
        auto& block_scores = inputs[0];
        
        // Force evaluation to ensure data resides cleanly in contiguous memory layout
        auto scores_contiguous = astype(block_scores, float32);

        int batch_size = block_scores.shape(0);
        int num_heads = block_scores.shape(1);
        int num_blocks = block_scores.shape(2);

        // Allocate the structured index return array
        std::vector<int> out_shape = {batch_size, num_heads, k_blocks};
        array indices = array(out_shape, int32);

        // Compute grid scale: One threadgroup assigned exclusively per batch head
        int grid_size = batch_size * num_heads;
        int threadgroup_size = 512; // Must match THREADS_PER_GROUP parameter inside MSL code

        // Access the native MLX Metal Compute pipeline environment
        auto& compute_encoder = fast::metal::get_compute_encoder();
        fast::metal::set_kernel(compute_encoder, "sparse_topk_kernel");
        
        // Bind the active data pointers inside the Unified Memory stack
        fast::metal::set_array_arg(compute_encoder, 0, scores_contiguous);
        fast::metal::set_array_arg(compute_encoder, 1, indices);
        
        // Load primitive scalars into buffer positions
        compute_encoder.setBytes(&k_blocks, sizeof(int), 2);
        compute_encoder.setBytes(&num_blocks, sizeof(int), 3);
        
        // Dispatch execution to the M3 Ultra compute clusters
        fast::metal::dispatch(compute_encoder, grid_size, threadgroup_size);
        
        return {indices};
    }
};

// Expose internal bindings to the MLX execution graph register
void register_ops() {
    fast::register_custom_op("sparse_topk", [](const std::vector<array>& inputs, const std::unordered_map<std::string, std::variant<int, float, std::string>>& kwargs) {
        int k = std::get<int>(kwargs.at("k"));
        return std::make_shared<SparseTopK>(k)->eval(inputs);
    });
}
