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

//! Runtime/device selection helpers shared by all inference entry points.
//!
//! CLI generation and the HTTP server both rely on the same environment-based
//! device resolution so CPU overrides and GPU wired-limit behavior stay
//! consistent regardless of how inference is entered.

use std::fmt;
use std::time::Duration;

const RUNTIME_DEVICE_ENV: &str = "MLXCEL_DEVICE";
const WIRED_LIMIT_ENV: &str = "MLXCEL_WIRED_LIMIT";
/// Issue #55: optional soft cap on the MLX allocator. When set, the
/// runtime calls `mlxcel_core::memory::set_memory_limit(...)` at startup
/// so MLX raises an exception once allocations would push the working
/// set past this value, instead of thrashing or OOM-killing the process.
/// Used by the future preflight capstone (#56). Accepts the same syntax
/// as `MLXCEL_WIRED_LIMIT`: plain bytes, `NGB`, or `NMB`. Unset means
/// "do not override MLX's default limit".
const MEMORY_LIMIT_ENV: &str = "MLXCEL_MEMORY_LIMIT";
/// Cap on MLX's free-buffer cache (the allocator's free list). Without
/// a cap, the free list grows toward Metal's 1.5x recommended working
/// set ceiling over long-running sessions, eventually starving fresh
/// allocations on a Mac that has many tens of GB of unified memory in
/// circulation. Calls `mlxcel_core::memory::set_cache_limit(...)` at
/// startup. Same syntax as `MLXCEL_MEMORY_LIMIT` (plain bytes, `NGB`,
/// `NMB`). Unset / `0` / `none` means "do not override MLX's default
/// behaviour" (the unbounded growth). Set this low (e.g. 4–8 GB) and
/// rely on the monitor below to confirm the cap is sized correctly.
const METAL_CACHE_LIMIT_ENV: &str = "MLXCEL_METAL_CACHE_LIMIT";
/// Seconds between MLX memory-snapshot tracing emissions. When set, the
/// server spawns a tokio task that periodically reads
/// `(active, cache, peak, limit)` and emits a structured INFO line so
/// operators can tune `MLXCEL_METAL_CACHE_LIMIT` empirically. `0` / unset
/// disables the monitor. Recommended initial value: `60`.
const MEMORY_MONITOR_INTERVAL_ENV: &str = "MLXCEL_MEMORY_MONITOR_INTERVAL_SECS";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeDevice {
    Cpu,
    Gpu,
}

impl RuntimeDevice {
    const fn uses_gpu(self) -> bool {
        matches!(self, Self::Gpu)
    }
}

impl fmt::Display for RuntimeDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Gpu => {
                #[cfg(feature = "cuda")]
                return write!(f, "NVIDIA GPU (CUDA)");
                #[cfg(target_os = "macos")]
                return write!(f, "Apple GPU (Metal)");
                #[cfg(not(any(feature = "cuda", target_os = "macos")))]
                write!(f, "GPU")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeSetup {
    pub device: RuntimeDevice,
    pub wired_limit_bytes: Option<usize>,
    /// Soft MLX allocator memory limit applied via `MLXCEL_MEMORY_LIMIT`
    /// (issue #55). `None` when the env var was unset or invalid and
    /// MLX's default limit is in effect.
    pub memory_limit_bytes: Option<usize>,
    /// Cap on MLX's free-buffer cache applied via `MLXCEL_METAL_CACHE_LIMIT`.
    /// `None` when the env var was unset and the free list grows
    /// unbounded (MLX default).
    pub metal_cache_limit_bytes: Option<usize>,
    /// Memory-snapshot monitor interval applied via
    /// `MLXCEL_MEMORY_MONITOR_INTERVAL_SECS`. `None` disables the
    /// monitor; otherwise the server spawns a periodic tokio task at
    /// this interval.
    pub memory_monitor_interval: Option<Duration>,
    pub invalid_device_override: Option<String>,
}

pub fn initialize_runtime() -> RuntimeSetup {
    let (requested_device, invalid_device_override) =
        resolve_runtime_device(std::env::var(RUNTIME_DEVICE_ENV).ok().as_deref());

    if requested_device == RuntimeDevice::Cpu {
        mlxcel_core::set_default_device(false);
    }

    let device = if mlxcel_core::is_gpu_available() {
        RuntimeDevice::Gpu
    } else {
        RuntimeDevice::Cpu
    };

    // Footgun guard: a default `cargo build --release` on Linux omits the `cuda`
    // feature and silently runs MLX on the CPU. If the user wanted the GPU but
    // this CPU-only binary fell back to CPU on a host that has an NVIDIA GPU,
    // say so loudly instead of crawling at a fraction of GPU speed.
    if should_warn_cpu_only_on_nvidia_host(
        requested_device,
        device,
        cfg!(feature = "cuda"),
        nvidia_host_present(),
    ) {
        warn_cpu_only_on_nvidia_host();
    }

    let wired_limit_bytes = if device.uses_gpu() {
        resolve_wired_limit()
    } else {
        None
    };

    // Issue #55: apply optional soft allocator cap regardless of device.
    // The MLX no-gpu CPU allocator also honours `set_memory_limit()`, so
    // the preflight (#56) can use this on Linux/CI just as on Apple
    // Silicon.
    let memory_limit_bytes = resolve_memory_limit();

    // Cap MLX's free-buffer cache. Unset → MLX default (unbounded
    // growth, the long-session OOM cause). The monitor lets operators
    // tune this with real data instead of guessing.
    let metal_cache_limit_bytes = resolve_metal_cache_limit();
    let memory_monitor_interval = resolve_memory_monitor_interval();

    RuntimeSetup {
        device,
        wired_limit_bytes,
        memory_limit_bytes,
        metal_cache_limit_bytes,
        memory_monitor_interval,
        invalid_device_override,
    }
}

fn resolve_runtime_device(value: Option<&str>) -> (RuntimeDevice, Option<String>) {
    match value {
        Some(raw) => match parse_runtime_device(raw) {
            Some(device) => (device, None),
            None => (RuntimeDevice::Gpu, Some(raw.to_owned())),
        },
        None => (RuntimeDevice::Gpu, None),
    }
}

/// Resolve wired memory limit from MLXCEL_WIRED_LIMIT env var.
///
/// Default: set to gpu_max_memory_size (matches Python mlx-lm's wired_limit context manager).
/// This is critical for large models (>50% of GPU memory) to avoid weight eviction.
///
/// - Not set or "max": set to gpu_max_memory_size (default, matches Python mlx-lm)
/// - "0" or "none": disable wired limit
/// - Number (bytes) or "NGB"/"NMB": explicit limit
fn resolve_wired_limit() -> Option<usize> {
    let raw = std::env::var(WIRED_LIMIT_ENV).ok();
    let limit = match raw.as_deref() {
        Some("0") | Some("none") | Some("NONE") => return None,
        None | Some("") | Some("max") | Some("MAX") => mlxcel_core::gpu_max_memory_size(),
        Some(s) => parse_memory_size(s).unwrap_or(mlxcel_core::gpu_max_memory_size()),
    };
    if limit > 0 {
        mlxcel_core::set_wired_limit(limit);
        Some(limit)
    } else {
        None
    }
}

/// Resolve the MLX allocator soft limit from MLXCEL_MEMORY_LIMIT (issue #55).
///
/// Returns the limit actually applied to MLX, or `None` when the env var
/// is unset / explicitly disabled. This is the hook the capstone preflight
/// (#56) drives when a model is too large to fit comfortably — calling
/// `mlxcel_core::memory::set_memory_limit` makes MLX raise an exception
/// during evaluation instead of thrashing the system allocator.
fn resolve_memory_limit() -> Option<usize> {
    let raw = std::env::var(MEMORY_LIMIT_ENV).ok();
    let bytes = match raw.as_deref() {
        Some("0") | Some("none") | Some("NONE") | None | Some("") => return None,
        Some(s) => parse_memory_size(s)?,
    };
    if bytes == 0 {
        return None;
    }
    mlxcel_core::memory::set_memory_limit(bytes as u64);
    Some(bytes)
}

/// Pure parser for MLXCEL_METAL_CACHE_LIMIT raw value. Returns the byte
/// count to apply, or `None` for the disabled-by-design inputs (`0`,
/// `none`/`NONE`, empty, unset).
///
/// Split out from [`resolve_metal_cache_limit`] so unit tests can verify
/// the parsing logic without invoking `set_cache_limit`. The FFI is
/// process-global and polluting it inside the test runner has broken
/// sibling model tests in this crate.
fn parse_metal_cache_limit_value(raw: Option<&str>) -> Option<usize> {
    let bytes = match raw {
        Some("0") | Some("none") | Some("NONE") | None | Some("") => return None,
        Some(s) => parse_memory_size(s)?,
    };
    if bytes == 0 {
        return None;
    }
    Some(bytes)
}

/// Resolve the MLX free-buffer cache cap from MLXCEL_METAL_CACHE_LIMIT.
///
/// Without a cap, MLX's allocator keeps freed buffers in an internal
/// free list to amortise the next allocation of the same shape. Over a
/// long-running session with varied shapes the free list grows toward
/// Metal's 1.5x recommended-working-set ceiling — eventually a fresh
/// allocation is refused with `[metal::malloc] Resource limit exceeded`
/// and the request panics.
///
/// Setting a cap returns excess freed buffers to the OS immediately.
/// The trade-off: slightly slower first allocation of an unfamiliar
/// shape (must round-trip to the OS) for predictable peak memory.
/// In practice the hot working set is small (the few activation /
/// scratch shapes that recur during prefill and decode), so a cap on
/// the order of `4–8 GB` typically has no observable perf impact.
///
/// Returns the cap actually applied, or `None` when the env var was
/// unset / explicitly disabled (`0`, `none`). The `set_cache_limit`
/// FFI is process-global; this function is the only intended caller.
fn resolve_metal_cache_limit() -> Option<usize> {
    let raw = std::env::var(METAL_CACHE_LIMIT_ENV).ok();
    let bytes = parse_metal_cache_limit_value(raw.as_deref())?;
    mlxcel_core::memory::set_cache_limit(bytes as u64);
    Some(bytes)
}

/// Resolve the memory-snapshot monitor cadence from
/// MLXCEL_MEMORY_MONITOR_INTERVAL_SECS.
///
/// `0` / unset disables the monitor entirely. Any positive integer N
/// means "spawn a tokio task that wakes every N seconds and emits a
/// structured tracing line with active / cache / peak / limit bytes."
/// Lets an operator empirically observe the MLX allocator's behaviour
/// and tune `MLXCEL_METAL_CACHE_LIMIT` with real data rather than
/// guesswork.
fn resolve_memory_monitor_interval() -> Option<Duration> {
    let raw = std::env::var(MEMORY_MONITOR_INTERVAL_ENV).ok();
    let secs = match raw.as_deref() {
        Some("0") | Some("none") | Some("NONE") | None | Some("") => return None,
        Some(s) => s.trim().parse::<u64>().ok()?,
    };
    if secs == 0 {
        return None;
    }
    Some(Duration::from_secs(secs))
}

/// Spawn the periodic MLX memory-snapshot tracing task.
///
/// Reads `(active, cache, peak, limit)` from
/// [`mlxcel_core::memory::snapshot`] at the configured interval and
/// emits a structured INFO tracing line. Designed to run for the
/// server's lifetime; cancelling the returned `JoinHandle` stops the
/// monitor. Logging is the only side effect — no shared mutable state.
///
/// The interval should be measured in tens of seconds. The snapshot
/// reads are cheap (just counter loads from the MLX allocator) but
/// emitting at sub-second cadence would clutter the log without
/// telling the operator anything new about steady-state behaviour.
///
/// Used by: server startup after [`initialize_runtime`] when
/// `RuntimeSetup::memory_monitor_interval` is `Some`.
pub fn spawn_memory_monitor(interval: Duration) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        // Skip the immediate first-tick fire — we just started; let
        // one interval elapse so the operator sees a real measurement
        // rather than the post-startup zero-state.
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        ticker.tick().await;
        loop {
            ticker.tick().await;
            let snap = mlxcel_core::memory::snapshot();
            tracing::info!(
                target: "mlxcel::memory_monitor",
                active_bytes = snap.active_bytes,
                cache_bytes = snap.cache_bytes,
                peak_bytes = snap.peak_bytes,
                limit_bytes = snap.limit_bytes,
                "MLX memory snapshot"
            );
        }
    })
}

/// Parse a memory size string: plain bytes, "NGB", or "NMB".
fn parse_memory_size(s: &str) -> Option<usize> {
    let s = s.trim().to_ascii_uppercase();
    if let Some(n) = s.strip_suffix("GB") {
        n.trim()
            .parse::<f64>()
            .ok()
            .map(|v| (v * 1024.0 * 1024.0 * 1024.0) as usize)
    } else if let Some(n) = s.strip_suffix("MB") {
        n.trim()
            .parse::<f64>()
            .ok()
            .map(|v| (v * 1024.0 * 1024.0) as usize)
    } else {
        s.parse::<usize>().ok()
    }
}

fn parse_runtime_device(value: &str) -> Option<RuntimeDevice> {
    match value.trim().to_ascii_lowercase().as_str() {
        "cpu" => Some(RuntimeDevice::Cpu),
        "gpu" | "metal" => Some(RuntimeDevice::Gpu),
        _ => None,
    }
}

/// Detect an NVIDIA GPU without linking CUDA. The kernel driver exposes these
/// paths whether or not this binary was built with the `cuda` feature, so a
/// CPU-only build can still tell it is sitting on an NVIDIA host.
fn nvidia_host_present() -> bool {
    std::path::Path::new("/dev/nvidiactl").exists()
        || std::path::Path::new("/proc/driver/nvidia/version").exists()
}

/// Whether to warn that a CPU-only build is wasting an NVIDIA GPU. True only
/// when the GPU was wanted, the runtime fell back to CPU, this binary lacks the
/// `cuda` feature (so it can never use the GPU), and an NVIDIA host is present.
/// An explicit `MLXCEL_DEVICE=cpu` (`requested == Cpu`) suppresses the warning,
/// and a `cuda`-capable build that fell back to CPU is a genuine no-GPU host,
/// not the footgun.
fn should_warn_cpu_only_on_nvidia_host(
    requested: RuntimeDevice,
    resolved: RuntimeDevice,
    cuda_build: bool,
    nvidia_host: bool,
) -> bool {
    requested == RuntimeDevice::Gpu && resolved == RuntimeDevice::Cpu && !cuda_build && nvidia_host
}

/// Loud one-time startup warning for the CPU-only-build-on-NVIDIA-host footgun.
fn warn_cpu_only_on_nvidia_host() {
    eprintln!(
        "warning: an NVIDIA GPU is present but this mlxcel binary was built \
         without CUDA support, so it is running on the CPU (orders of magnitude \
         slower).\n         \
         Rebuild with the `cuda` feature: \
         `MLX_CUDA_ARCHITECTURES=<arch> cargo build --release --features cuda` \
         (or `cargo cuda`).\n         \
         See docs/installation.md (Linux with CUDA)."
    );
}

#[cfg(test)]
#[path = "runtime_tests.rs"]
mod tests;
