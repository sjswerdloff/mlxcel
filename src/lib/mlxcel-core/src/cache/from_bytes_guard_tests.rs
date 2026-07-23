//! Guard: the KV-cache restore path must never call the unvalidated FFI
//! constructors (the nocopy and f16 raw-byte variants).
//!
//! An audit found the restore path uses only the validated, copying
//! `ffi::from_bytes` (length-checked in cpp/mlx_cxx_bridge.cpp). The two
//! forbidden constructors skip length, alignment, and lifetime validation:
//! a short or misaligned buffer means an out-of-bounds read, and the nocopy
//! variant aliases caller memory with a use-after-free contract. On a
//! consciousness-restore path that is not an acceptable failure mode.
//!
//! This test re-runs the "zero call sites" census on every `cargo test` so
//! the point-in-time audit finding stays true. If it fires, either remove
//! the forbidden call (use `ffi::from_bytes`) or -- if a nocopy path is ever
//! genuinely required -- bring that change through review with this guard
//! updated deliberately, not deleted.

use std::fs;
use std::path::{Path, PathBuf};

/// Forbidden constructor names, assembled via `concat!` so this file's own
/// source never contains the literal tokens. That makes the guard immune to
/// scanning itself even if the filename-based self-skip below ever breaks
/// (defense in depth; the skip is still applied as the primary mechanism).
const FORBIDDEN: [&str; 2] = [
    concat!("from_bytes", "_nocopy"),
    concat!("from_bytes", "_f16"),
];

/// This guard file's own name (skipped while scanning).
const SELF_NAME: &str = concat!("from_bytes", "_guard_tests.rs");

/// Minimum number of .rs files we expect to scan. The cache module currently
/// has ~25 Rust sources (cache.rs + cache/ + cache/turbo/). If the walk finds
/// fewer than this, path resolution has drifted and the guard would be
/// passing vacuously -- fail loud instead of certifying nothing.
const MIN_EXPECTED_SOURCES: usize = 10;

fn collect_rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("guard: cannot read dir {}: {e}", dir.display()));
    for entry in entries {
        let entry =
            entry.unwrap_or_else(|e| panic!("guard: bad dir entry in {}: {e}", dir.display()));
        let path = entry.path();
        if path.is_dir() {
            collect_rust_sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn cache_restore_path_never_uses_unvalidated_from_bytes_constructors() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let cache_dir = manifest_dir.join("src").join("cache");
    let cache_root = manifest_dir.join("src").join("cache.rs");

    assert!(
        cache_dir.is_dir(),
        "guard: expected cache module dir at {} -- crate layout changed; \
         update this guard rather than letting it scan nothing",
        cache_dir.display()
    );
    assert!(
        cache_root.is_file(),
        "guard: expected cache module root at {} -- crate layout changed; \
         update this guard rather than letting it scan nothing",
        cache_root.display()
    );

    // Scan the cache module root (src/cache.rs) plus everything under
    // src/cache/ recursively (includes cache/turbo/). Only .rs files:
    // cache/kvarn_fixtures/ holds binary fixtures.
    let mut sources = vec![cache_root];
    collect_rust_sources(&cache_dir, &mut sources);

    assert!(
        sources.len() >= MIN_EXPECTED_SOURCES,
        "guard: only found {} Rust sources under {} -- the walk is broken \
         and this guard would pass vacuously",
        sources.len(),
        cache_dir.display()
    );

    let mut violations = Vec::new();
    for path in &sources {
        if path.file_name().is_some_and(|name| name == SELF_NAME) {
            continue; // the guard necessarily discusses the forbidden names
        }
        let text = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("guard: cannot read {}: {e}", path.display()));
        for (line_idx, line) in text.lines().enumerate() {
            for needle in FORBIDDEN {
                if line.contains(needle) {
                    violations.push(format!(
                        "  {}:{}: `{}` in: {}",
                        path.display(),
                        line_idx + 1,
                        needle,
                        line.trim()
                    ));
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "Unvalidated FFI constructor referenced in the KV-cache module.\n\
         The cache restore path must only use the validated, copying \
         `ffi::from_bytes` (length/dtype/overflow-checked in \
         cpp/mlx_cxx_bridge.cpp). The constructors found below perform no \
         length or alignment validation and can OOB-read or dangle during \
         consciousness restore.\n\
         Offending sites:\n{}",
        violations.join("\n")
    );
}
