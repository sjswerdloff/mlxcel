#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <stdexcept>

using namespace mlx::core;

class SlidingSparseTopK : public fast::MetalCustomOp {
public:
    int k_blocks;
    
    SlidingSparseTopK(int k) : k_blocks(k) {
        // Enforce bounds based on the fixed Metal window (W=128)
        if (k > 128) {
            throw std::runtime_error("k_blocks must be <= W (128)");
        }
    }

    std::vector<array> eval(const std::vector<array>& inputs) override {
        auto& block_scores = inputs[0];
        auto scores_contiguous = astype(block_scores, float32);

        int batch_size = block_scores.shape(0);
        int num_heads = block_scores.shape(1);
        int num_blocks = block_scores.shape(2);

        // Output matrix setup
        std::vector<int> out_shape = {batch_size, num_heads, k_blocks};
        array indices = array(out_shape, int32);

        // Compute Execution Grid: 1 Threadgroup per batch-head
        int grid_size = batch_size * num_heads;
        
        // This must strictly match the constant `W` defined in the Metal shader
        int threadgroup_size = 128; 

        auto& compute_encoder = fast::metal::get_compute_encoder();
        fast::metal::set_kernel(compute_encoder, "sliding_topk_kernel");
        
        // Buffer Bindings
        fast::metal::set_array_arg(compute_encoder, 0, scores_contiguous);
        fast::metal::set_array_arg(compute_encoder, 1, indices);
        compute_encoder.setBytes(&k_blocks, sizeof(int), 2);
        compute_encoder.setBytes(&num_blocks, sizeof(int), 3);
        
        // Launch on M3 Ultra's GPU clusters
        fast::metal::dispatch(compute_encoder, grid_size, threadgroup_size);
        
        return {indices};
    }
};

void register_ops() {
    fast::register_custom_op("sliding_sparse_topk", [](const std::vector<array>& inputs, const std::unordered_map<std::string, std::variant<int, float, std::string>>& kwargs) {
        int k = std::get<int>(kwargs.at("k"));
        return std::make_shared<SlidingSparseTopK>(k)->eval(inputs);
    });
}

