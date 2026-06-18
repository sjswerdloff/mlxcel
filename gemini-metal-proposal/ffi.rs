use std::ffi::c_void;
use mlx_rs::array::Array; // Assuming mlxcel's internal MLX Rust binding path

extern "C" {
    fn msa_sparse_topk_ffi(block_scores: *const c_void, k: i32) -> *mut c_void;
    fn msa_block_sparse_sdpa_ffi(
        q: *const c_void, 
        k: *const c_void, 
        v: *const c_void, 
        indices: *const c_void, 
        scale: f32
    ) -> *mut c_void;
}

/// Safe Rust wrapper around the unsafe C++ FFI
pub fn sparse_topk(block_scores: &Array, k: i32) -> Array {
    unsafe {
        let raw_ptr = msa_sparse_topk_ffi(block_scores.as_ptr(), k);
        Array::from_raw(raw_ptr) // Re-wrap the C++ pointer into a safe Rust abstraction
    }
}

pub fn block_sparse_sdpa(q: &Array, k: &Array, v: &Array, indices: &Array, scale: f32) -> Array {
    unsafe {
        let raw_ptr = msa_block_sparse_sdpa_ffi(q.as_ptr(), k.as_ptr(), v.as_ptr(), indices.as_ptr(), scale);
        Array::from_raw(raw_ptr)
    }
}
