// Append to the bottom of build.rs

let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
if target_os == "macos" {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let metal_src = "src/lib/mlx-cpp/sliding_msa_kernel.metal";
    let metal_ir = format!("{}/sliding_msa_kernel.air", out_dir);
    let metallib = format!("{}/sliding_msa_kernel.metallib", out_dir);

    // 1. Compile Metal Shading Language to Apple Intermediate Representation (AIR)
    std::process::Command::new("xcrun")
        .args(&["-sdk", "macosx", "metal", "-O3", "-c", metal_src, "-o", &metal_ir])
        .status()
        .expect("Failed to compile MSA Metal shader");

    // 2. Link AIR to a compiled Metallib
    std::process::Command::new("xcrun")
        .args(&["-sdk", "macosx", "metallib", &metal_ir, "-o", &metallib])
        .status()
        .expect("Failed to link MSA Metallib");

    // 3. Compile the C++ FFI bindings using the 'cc' crate
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .flag("-O3")
        .file("src/lib/mlx-cpp/msa_op.cpp")
        .file("src/lib/mlx-cpp/msa_ffi.cpp")
        // Ensure the compiler can find MLX's native C++ headers inside the mlxcel tree
        .include("src/lib/mlx-cpp") 
        .compile("msa_custom_ops");

    // Tell Cargo to re-run this script ONLY if your custom source files change
    println!("cargo:rerun-if-changed=src/lib/mlx-cpp/sliding_msa_kernel.metal");
    println!("cargo:rerun-if-changed=src/lib/mlx-cpp/msa_op.cpp");
    println!("cargo:rerun-if-changed=src/lib/mlx-cpp/msa_ffi.cpp");
}
