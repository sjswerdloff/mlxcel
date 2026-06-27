// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::{
    RuntimeDevice, parse_memory_size, parse_runtime_device, resolve_runtime_device,
    should_warn_cpu_only_on_nvidia_host,
};

#[test]
fn parse_runtime_device_accepts_cpu() {
    assert_eq!(parse_runtime_device("cpu"), Some(RuntimeDevice::Cpu));
}

#[test]
fn parse_runtime_device_accepts_gpu_aliases() {
    assert_eq!(parse_runtime_device("gpu"), Some(RuntimeDevice::Gpu));
    assert_eq!(parse_runtime_device("Metal"), Some(RuntimeDevice::Gpu));
}

#[test]
fn parse_runtime_device_rejects_unknown_values() {
    assert_eq!(parse_runtime_device("tpu"), None);
}

#[test]
fn resolve_runtime_device_defaults_to_gpu() {
    assert_eq!(resolve_runtime_device(None), (RuntimeDevice::Gpu, None));
}

#[test]
fn resolve_runtime_device_preserves_invalid_override() {
    assert_eq!(
        resolve_runtime_device(Some("mps")),
        (RuntimeDevice::Gpu, Some("mps".to_string()))
    );
}

#[test]
fn parse_memory_size_gb() {
    assert_eq!(parse_memory_size("64GB"), Some(64 * 1024 * 1024 * 1024));
    assert_eq!(parse_memory_size("128gb"), Some(128 * 1024 * 1024 * 1024));
}

#[test]
fn parse_memory_size_mb() {
    assert_eq!(parse_memory_size("512MB"), Some(512 * 1024 * 1024));
}

#[test]
fn parse_memory_size_bytes() {
    assert_eq!(parse_memory_size("1073741824"), Some(1073741824));
}

#[test]
fn parse_memory_size_fractional_gb() {
    // 1.5 GB
    assert_eq!(
        parse_memory_size("1.5GB"),
        Some((1.5 * 1024.0 * 1024.0 * 1024.0) as usize)
    );
}

#[test]
fn parse_memory_size_invalid() {
    assert_eq!(parse_memory_size("abc"), None);
}

#[test]
fn warns_only_for_cpu_fallback_on_nvidia_host_without_cuda() {
    use RuntimeDevice::{Cpu, Gpu};
    // Footgun: wanted GPU, fell back to CPU, no cuda feature, NVIDIA host present.
    assert!(should_warn_cpu_only_on_nvidia_host(Gpu, Cpu, false, true));
    // cuda-capable build that fell back to CPU is a genuine no-GPU host, not the footgun.
    assert!(!should_warn_cpu_only_on_nvidia_host(Gpu, Cpu, true, true));
    // Genuine CPU-only Linux box (no NVIDIA device node): no nag.
    assert!(!should_warn_cpu_only_on_nvidia_host(Gpu, Cpu, false, false));
    // Explicit MLXCEL_DEVICE=cpu (requested == Cpu): respect the override.
    assert!(!should_warn_cpu_only_on_nvidia_host(Cpu, Cpu, false, true));
    // Already running on the GPU: nothing to warn about.
    assert!(!should_warn_cpu_only_on_nvidia_host(Gpu, Gpu, false, true));
}

// -- Metal cache cap + memory monitor resolution --
//
// Honest scope: these tests cover the env-var parsing and early-return
// (disabled) paths in `resolve_metal_cache_limit` and
// `resolve_memory_monitor_interval`. They do NOT cover the FFI call
// `mlxcel_core::memory::set_cache_limit` actually taking effect inside
// MLX — there is no public getter for the cache limit, mirroring the
// existing `resolve_memory_limit` situation. Real verification is the
// `scripts/smoke_test_metal_cache_monitor.sh` end-to-end, which boots
// the server and checks the log lines + observed cache_bytes.
//
// Env-var tests serialise via an explicit mutex because cargo runs
// tests in parallel inside one process and `std::env::set_var` is
// process-global. Without serialisation the env state observed by one
// test gets clobbered by another.

use super::{parse_metal_cache_limit_value, resolve_memory_monitor_interval};
use std::sync::Mutex;
use std::time::Duration;

// One mutex shared across env-mutating tests in this module. The MLX
// allocator API (`mlxcel_core::memory::set_cache_limit`) is also
// process-global, but we never call it from these tests — the cache-cap
// path was deliberately factored so the pure parser is testable
// without touching MLX state.
static ENV_MUTEX: Mutex<()> = Mutex::new(());

/// RAII helper: set an env var on construction, restore on Drop.
/// Restoring is what lets a panicking test not poison other tests'
/// view of the environment.
struct EnvGuard {
    name: &'static str,
    prev: Option<String>,
}

impl EnvGuard {
    fn set(name: &'static str, value: &str) -> Self {
        let prev = std::env::var(name).ok();
        // SAFETY: the ENV_MUTEX guard serialises all env mutation in
        // this test module, so no other test thread is reading or
        // writing the environment concurrently. The unsafe contract on
        // set_var is "no other thread is accessing the environment" —
        // we hold the mutex for the lifetime of this guard, so we own
        // exclusive access.
        unsafe {
            std::env::set_var(name, value);
        }
        Self { name, prev }
    }
    fn unset(name: &'static str) -> Self {
        let prev = std::env::var(name).ok();
        // SAFETY: same as `set` — exclusive access guaranteed by the
        // mutex held throughout this test.
        unsafe {
            std::env::remove_var(name);
        }
        Self { name, prev }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: still holding the ENV_MUTEX guard at the call site
        // (the EnvGuard is bound to the test's stack which is below
        // the mutex guard). Exclusive access is intact.
        unsafe {
            match &self.prev {
                Some(v) => std::env::set_var(self.name, v),
                None => std::env::remove_var(self.name),
            }
        }
    }
}

// -- parse_metal_cache_limit_value (pure, no FFI side effect) --
//
// These exercise the env-var → byte-count parsing only. The actual
// `mlxcel_core::memory::set_cache_limit` call is deliberately NOT
// invoked here because polluting MLX's process-global allocator state
// from within the test runner has been observed to break sibling model
// tests (gemma4_mtp_target etc). The disabled-path tests on
// `resolve_metal_cache_limit` below short-circuit before the FFI, so
// they remain safe.

#[test]
fn parse_metal_cache_limit_value_none_for_unset() {
    assert_eq!(parse_metal_cache_limit_value(None), None);
}

#[test]
fn parse_metal_cache_limit_value_none_for_zero() {
    assert_eq!(parse_metal_cache_limit_value(Some("0")), None);
}

#[test]
fn parse_metal_cache_limit_value_none_for_none_keyword() {
    assert_eq!(parse_metal_cache_limit_value(Some("none")), None);
    assert_eq!(parse_metal_cache_limit_value(Some("NONE")), None);
}

#[test]
fn parse_metal_cache_limit_value_none_for_empty() {
    assert_eq!(parse_metal_cache_limit_value(Some("")), None);
}

#[test]
fn parse_metal_cache_limit_value_gb_value() {
    assert_eq!(
        parse_metal_cache_limit_value(Some("4GB")),
        Some(4 * 1024 * 1024 * 1024)
    );
}

#[test]
fn parse_metal_cache_limit_value_mb_value() {
    assert_eq!(
        parse_metal_cache_limit_value(Some("512MB")),
        Some(512 * 1024 * 1024)
    );
}

#[test]
fn parse_metal_cache_limit_value_plain_bytes() {
    assert_eq!(
        parse_metal_cache_limit_value(Some("8589934592")),
        Some(8589934592)
    );
}

#[test]
fn parse_metal_cache_limit_value_garbage_returns_none() {
    assert_eq!(parse_metal_cache_limit_value(Some("not-a-size")), None);
}

#[test]
fn parse_metal_cache_limit_value_fractional_gb() {
    // Reuses parse_memory_size, which accepts fractional GB.
    assert_eq!(
        parse_metal_cache_limit_value(Some("0.5GB")),
        Some((0.5 * 1024.0 * 1024.0 * 1024.0) as usize)
    );
}

#[test]
fn resolve_memory_monitor_interval_disabled_when_unset() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::unset("MLXCEL_MEMORY_MONITOR_INTERVAL_SECS");
    assert_eq!(resolve_memory_monitor_interval(), None);
}

#[test]
fn resolve_memory_monitor_interval_disabled_when_zero() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("MLXCEL_MEMORY_MONITOR_INTERVAL_SECS", "0");
    assert_eq!(resolve_memory_monitor_interval(), None);
}

#[test]
fn resolve_memory_monitor_interval_disabled_when_none_keyword() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("MLXCEL_MEMORY_MONITOR_INTERVAL_SECS", "none");
    assert_eq!(resolve_memory_monitor_interval(), None);
}

#[test]
fn resolve_memory_monitor_interval_disabled_when_empty() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("MLXCEL_MEMORY_MONITOR_INTERVAL_SECS", "");
    assert_eq!(resolve_memory_monitor_interval(), None);
}

#[test]
fn resolve_memory_monitor_interval_returns_seconds_for_positive_int() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("MLXCEL_MEMORY_MONITOR_INTERVAL_SECS", "60");
    assert_eq!(
        resolve_memory_monitor_interval(),
        Some(Duration::from_secs(60))
    );
}

#[test]
fn resolve_memory_monitor_interval_returns_none_for_garbage() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("MLXCEL_MEMORY_MONITOR_INTERVAL_SECS", "sixty");
    assert_eq!(resolve_memory_monitor_interval(), None);
}

#[test]
fn resolve_memory_monitor_interval_returns_none_for_negative() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set("MLXCEL_MEMORY_MONITOR_INTERVAL_SECS", "-1");
    assert_eq!(resolve_memory_monitor_interval(), None);
}
