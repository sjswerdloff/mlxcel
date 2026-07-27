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

//! Stream-selection wrappers for generation-time pipelining.
//!
//! Generation owners ([`crate::generate::CxxGenerator`],
//! [`crate::speculative::SpeculativeGenerator`], and the server-side
//! `BatchScheduler`) create a dedicated MLX stream up front and install
//! it as the default for the worker thread that drives the generation
//! loop. Until that stream was a plain
//! `mlx::core::new_stream(Device::gpu)` instance, which means every
//! owner had to be constructed on the same thread that ran the loop —
//! otherwise the stream would be "owned" by the wrong thread and any
//! `synchronize()` call would target a different physical stream than
//! the one that dispatched the work.
//!
//! upstream `mlx-vlm` PR #1050 (commit `728fab1`)
//! introduces `mlx::core::ThreadLocalStream`: a TLS-backed handle that
//! resolves to a per-thread `Stream` on demand. The generation owners
//! now hold a `ThreadLocalStream` and resolve it on the worker thread
//! at install time, so dispatch and synchronization always pair up on
//! the same per-thread stream regardless of which thread constructed
//! the generator.
//!
//! See `docs/bridge-overhead-microbench.md` for the per-op cost
//! microbench used to validate that this change is at worst a no-op
//! and at best a small per-step latency win on multi-worker
//! deployments where construction and execution can happen on
//! different threads.

use crate::ffi;
use crate::ffi::{MlxStream, MlxThreadLocalStream};
use crate::UniquePtr;

/// Create a thread-local generation stream bound to the GPU device.
///
/// Returns `None` on CPU-only builds (where `is_gpu_available()` is
/// false) so callers can fall back to the MLX default stream.
///
/// The returned [`MlxThreadLocalStream`] handle is independent of the
/// thread it was created on — pass it to
/// [`install_thread_local_default_stream`] from any thread that wants
/// to bind its dedicated per-thread stream as the default for
/// subsequent MLX dispatches on that thread.
///
/// Used by: CxxGenerator, SpeculativeGenerator, BatchScheduler, AudioWorker
pub fn new_thread_local_generation_stream() -> Option<UniquePtr<MlxThreadLocalStream>> {
    if ffi::is_gpu_available() {
        Some(ffi::new_thread_local_stream_gpu())
    } else {
        None
    }
}

/// Resolve the calling thread's `MlxStream` from a thread-local handle
/// and install it as the default stream for that thread.
///
/// This is the worker-thread-side counterpart of
/// [`new_thread_local_generation_stream`]: the generator owner is
/// typically constructed on a control thread, then later runs on a
/// dedicated worker thread. Calling this method **on the worker thread**
/// (e.g. at the top of the generation loop) ensures every subsequent
/// MLX op on that thread is dispatched on the same per-thread stream
/// that synchronization will target.
///
/// `None` is a safe no-op for CPU-only builds.
///
/// Used by: CxxGenerator, SpeculativeGenerator, BatchScheduler, AudioWorker
pub fn install_thread_local_default_stream(tls: Option<&UniquePtr<MlxThreadLocalStream>>) {
    if let Some(tls) = tls {
        // NO teardown finalizer is armed here. Arming one used to happen on
        // this line; it made every thread that installed a stream crash at
        // thread exit. See the "Per-thread MLX teardown" notes below and
        // `install_then_exit_thread_does_not_kill_the_process`.
        let stream = ffi::stream_from_thread_local_stream(tls);
        ffi::set_default_stream(&stream);
    }
}

/// Synchronize the calling thread's stream associated with a
/// thread-local handle.
///
/// Equivalent to resolving the handle on the calling thread and
/// calling `synchronize_stream`, but uses MLX's
/// `synchronize(ThreadLocalStream)` overload so the synchronization is
/// guaranteed to target the same per-thread stream that previously
/// dispatched work via this handle.
///
/// `None` is a safe no-op (matches the CPU-only build path of
/// [`new_thread_local_generation_stream`]).
///
/// Used by: AudioWorker (after each request); tests. Generation-loop callers
/// rely on the default stream installed by
/// [`install_thread_local_default_stream`] and on per-op `eval` for
/// synchronization.
pub fn synchronize_thread_local_stream(tls: Option<&UniquePtr<MlxThreadLocalStream>>) {
    if let Some(tls) = tls {
        ffi::synchronize_thread_local_stream(tls);
    }
}

/// RAII guard that restores the calling thread's MLX default stream on drop.
///
/// Constructed by capturing the current default stream before installing a
/// new one. When the guard is dropped the previous stream is restored,
/// leaving the thread in exactly the state it was in before the installation.
///
/// This is primarily useful in tests that call
/// [`install_thread_local_default_stream`] and must not leak the mutated
/// per-thread state into subsequent test cases that run on the same thread.
///
/// # Example
///
/// ```ignore
/// let _guard = DefaultStreamGuard::capture();
/// install_thread_local_default_stream(Some(&tls));
/// // … test body …
/// // guard restores previous default stream here
/// ```
#[must_use]
pub struct DefaultStreamGuard {
    previous: UniquePtr<MlxStream>,
}

impl DefaultStreamGuard {
    /// Capture the calling thread's current default stream.
    ///
    /// The returned guard will restore that stream when dropped.
    pub fn capture() -> Self {
        Self {
            previous: ffi::default_stream(),
        }
    }
}

impl Drop for DefaultStreamGuard {
    fn drop(&mut self) {
        ffi::set_default_stream(&self.previous);
    }
}

/// Legacy helper — create a non-thread-local GPU stream.
///
/// Kept for backward compatibility with any external user of the
/// `mlxcel_core::streams` module that has not yet migrated to the
/// thread-local API. New code should call
/// [`new_thread_local_generation_stream`] instead.
///
/// Used by: external crates only; in-tree generation owners now use
/// [`new_thread_local_generation_stream`].
#[deprecated(
    since = "26.5.9",
    note = "Use `new_thread_local_generation_stream` so dispatch and synchronization stay on the same per-thread stream."
)]
pub fn new_generation_stream() -> Option<UniquePtr<MlxStream>> {
    if ffi::is_gpu_available() {
        Some(ffi::new_gpu_stream())
    } else {
        None
    }
}

/// Legacy helper — install a previously created `MlxStream` as the
/// default stream.
///
/// Used together with the deprecated
/// [`new_generation_stream`]. Prefer
/// [`install_thread_local_default_stream`] in new code.
#[deprecated(
    since = "26.5.9",
    note = "Use `install_thread_local_default_stream` to bind the per-thread MLX stream."
)]
pub fn install_default_stream(stream: Option<&UniquePtr<MlxStream>>) {
    if let Some(stream) = stream {
        ffi::set_default_stream(stream);
    }
}

// ---------------------------------------------------------------------------
// Per-thread MLX teardown — EXPLICIT ONLY
// ---------------------------------------------------------------------------
//
// MLX maintains per-thread state (see `mlx/backend/metal/device.cpp`:
// `get_command_encoders()` is itself a `static thread_local`). Releasing
// that state on thread exit is desirable — but it MUST NOT be done from a
// destructor that runs during thread teardown.
//
// WHY (measured 2026-07-27, not reasoned):
//
// A previous revision registered a `thread_local!` RAII guard whose `Drop`
// called `ffi::clear_streams()`. Rust's `thread_local!` destructors and
// libc++'s `__cxa_thread_atexit` destructors are DIFFERENT mechanisms with
// no defined interleaving, and on this platform MLX's own C++ thread-locals
// are torn down FIRST. By the time the Rust guard ran, MLX's per-thread
// state was already destroyed, so:
//
//   * `clear_streams()`      -> hard trap (SIGTRAP), no C++ exception
//   * `synchronize_default()` -> `std::out_of_range: vector`
//
// This was deterministic, and it killed the test process at thread exit
// AFTER the test body had passed — e.g. every run of
// `speculative::tests::speculative_generate_max_tokens_one_emits_first_non_eos_token`,
// which capped a `--test-threads=1` suite run at 1069 of 1136 tests.
// Re-ordering the arming relative to MLX touches does NOT help: the
// ordering is not ours to control. Removing the guard restored the suite
// to 1134 passed / 2 ignored / 0 failed.
//
// The rule this encodes: **never call MLX from a thread-exit destructor.**
// Release per-thread state explicitly, from the thread body, while the
// thread is still alive — that is what [`finalize_thread`] is for.

/// Release the calling thread's MLX per-thread state.
///
/// Call this **from the thread body, before the thread returns** — never
/// from a `Drop` that runs during thread teardown (see the module notes
/// above: MLX's own thread-locals are already gone by then, and the call
/// traps).
///
/// Optional: MLX's per-thread containers are themselves `thread_local`
/// and self-destruct, so a worker that simply exits is not leaking. This
/// exists for long-lived processes that want the release to happen at a
/// known point rather than at thread exit.
///
/// Idempotent. Safe no-op semantics are the caller's responsibility only
/// in the sense that it must be on a live thread.
///
/// # DELIBERATELY UNWIRED — read this before adding a caller
///
/// Nothing in production calls this, and that is on purpose. So is the
/// same fact about [`shutdown`]. **There is no measured evidence that any
/// per-thread release is required**: MLX's own per-thread containers are
/// `thread_local` and self-destruct, and `main` — which has none of this
/// machinery at all — does not exhibit the flake this was built for.
///
/// This function exists to demonstrate, executably, the *correct* shape
/// (release from a live thread body), as the contrast partner to the
/// destructor version that was fatal. That is its whole job today.
///
/// **The historical warning, because it already happened once.** Commit
/// `2cda250` added `clear_streams` plumbing and said in its own message:
/// *"Neither production nor test paths currently invoke `clear_streams`
/// anywhere"* and *"the test-teardown flake is not proven fixed"*. An
/// unwired primitive with a plausible name was then auto-wired into
/// `install_thread_local_default_stream` without that verification, and
/// every thread installing a generation stream began dying at thread exit
/// — the deterministic version of the intermittent crash it was meant to
/// prevent.
///
/// So: **do not wire this because it looks like it belongs somewhere.**
/// Wire it only together with a measurement showing the release is needed
/// and a test that reddens without it. Machinery and its verification land
/// in the same change, or neither lands.
pub fn finalize_thread() {
    ffi::clear_streams();
}

/// Explicit main-thread shutdown for graceful process exit.
///
/// Call from a `SIGTERM` handler or immediately before returning from
/// `main()`. Synchronizes the default stream, releases the main
/// thread's stream registry entries, and clears MLX's global memory
/// cache — leaving the runtime in a state where the subsequent C++
/// static-destructor phase has nothing racing against it on this
/// thread.
///
/// Idempotent, but only meaningful once per process. Does NOT finalize
/// other threads' state — a worker thread that wants an explicit release
/// calls [`finalize_thread`] from its own body before returning.
///
/// Sound because it runs on a live thread. The same three calls from a
/// thread-exit destructor trap; see the "Per-thread MLX teardown" notes.
///
/// **Also DELIBERATELY UNWIRED — zero callers.** Same reasoning and same
/// historical warning as [`finalize_thread`]; read that before adding one.
/// A SIGTERM handler is the intended eventual home, but it lands with the
/// measurement, not ahead of it.
pub fn shutdown() {
    ffi::synchronize_default();
    ffi::clear_streams();
    ffi::clear_memory_cache();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Refuse to run vacuously.
    ///
    /// The teardown tests below only discriminate anything when a GPU is
    /// present: without one, `new_thread_local_generation_stream()` returns
    /// `None`, `install_thread_local_default_stream(None)` is a no-op, and
    /// the test degenerates into "run an MLX op on a thread" — green under
    /// any finalizer, correct or catastrophic.
    ///
    /// So the precondition is an assertion, not a silent branch. On a host
    /// without a GPU these tests FAIL rather than pass emptily, and the
    /// failure text says what to do about it. Turning that into a
    /// deliberate `#[ignore]` for a genuinely CPU-only target is a decision
    /// someone makes on purpose and can be seen making — which is the whole
    /// point.
    fn require_gpu_or_refuse(test: &str) {
        assert!(
            ffi::is_gpu_available(),
            "VACUOUS: `{test}` cannot fail on this host — no GPU, so \
             install_thread_local_default_stream is a no-op and the teardown \
             path under test is never exercised. This test is REFUSING to \
             report a green it did not earn. If this target is genuinely \
             CPU-only, gate the test with an explicit #[ignore] naming that \
             reason (see cold_store::tests::sharpened_kv_concurrency_probe \
             for the precedent) rather than letting it pass emptily."
        );
    }

    /// REGRESSION GUARD — a thread that installs a generation stream, uses
    /// MLX, and then EXITS must not take the process with it.
    ///
    /// This is the shape that a `thread_local!` teardown finalizer broke:
    /// the thread body completed and passed, then the guard's `Drop` called
    /// into MLX after MLX's own C++ thread-locals had been destroyed, and
    /// the process died with SIGTRAP (`clear_streams`) or
    /// `std::out_of_range` (`synchronize_default`).
    ///
    /// The mutation that must redden this test: re-introduce any MLX call
    /// in a `thread_local!` `Drop` reachable from
    /// `install_thread_local_default_stream`. Verified 2026-07-27 — with
    /// that arming restored the process dies at the `join()` below and the
    /// whole suite goes red, which is exactly the intended failure signal.
    ///
    /// A green run here is NOT decoration: the assertion is that we reach
    /// the line after `join()` at all.
    ///
    /// DECLARED BLIND SPOTS — what a green run here does NOT establish
    /// (asked for by Violet at QE; stated so absence is never inferred as
    /// coverage). This test catches ONE failure mode: a thread-exit
    /// destructor that faults. It is silent about:
    ///
    /// 1. ~~CPU-only builds — it goes VACUOUS.~~ **CLOSED** — this is now
    ///    enforced, not documented. Violet's point: a doc comment is
    ///    fail-visible to a *reviewer*, and CI is not a reviewer — it reads
    ///    exit codes, where a vacuous green is indistinguishable from a real
    ///    one. `require_gpu_or_refuse` below makes the vacuous case fail
    ///    loudly in the OUTPUT. Silent-pass is the one outcome not available
    ///    to a test whose job is catching a silent crash.
    /// 2. **Corruption without a fault.** A finalizer that releases the
    ///    wrong thread's state, or releases too early, and does not crash,
    ///    stays green here.
    /// 3. **Under-release.** This is green today *because* nothing releases
    ///    per-thread state. If a release is ever genuinely required, its
    ///    absence is invisible to this test.
    /// 4. **Anything needing two or more concurrent MLX threads** — one
    ///    worker only. The separate, still-undiagnosed parallel-suite
    ///    SIGSEGV is out of scope here.
    /// 5. **Process-exit races** rather than thread-exit races — the
    ///    original `2cda250` concern (static destructors interleaving with
    ///    `MlxArray` destructors on other unwinding threads) is NOT
    ///    exercised.
    /// 6. **Non-Metal backends** — only the path this host takes.
    #[test]
    fn install_then_exit_thread_does_not_kill_the_process() {
        use crate::dtype;

        require_gpu_or_refuse("install_then_exit_thread_does_not_kill_the_process");

        let handle = std::thread::spawn(|| {
            let tls = new_thread_local_generation_stream();
            install_thread_local_default_stream(tls.as_ref());
            // Touch MLX so this thread genuinely populates per-thread state;
            // without a real op the teardown path under test is not exercised.
            let a = ffi::ones(&[2, 2], dtype::FLOAT32);
            ffi::eval(&a);
            ffi::array_shape(&a)
        });

        let shape = handle.join().expect("worker thread must not panic");
        assert_eq!(
            shape,
            vec![2, 2],
            "worker thread must have run a real MLX op"
        );
    }

    /// Explicit finalization from a LIVE thread body is sound — the
    /// counterpart to the guard above. Same work, but the thread releases
    /// MLX state itself before returning rather than leaving it to a
    /// destructor.
    #[test]
    fn finalize_thread_from_a_live_thread_body_is_sound() {
        require_gpu_or_refuse("finalize_thread_from_a_live_thread_body_is_sound");

        let handle = std::thread::spawn(|| {
            let tls = new_thread_local_generation_stream();
            install_thread_local_default_stream(tls.as_ref());
            let a = crate::ffi::ones(&[3, 1], crate::dtype::FLOAT32);
            ffi::eval(&a);
            finalize_thread();
            true
        });
        assert!(
            handle.join().expect("worker thread must not panic"),
            "finalize_thread must return normally on a live thread"
        );
    }

    /// Smoke test: the TLS handle factory either succeeds (GPU build)
    /// or returns `None` cleanly (CPU-only build).
    #[test]
    fn new_thread_local_generation_stream_is_total() {
        let _ = new_thread_local_generation_stream();
    }

    /// `new_generation_stream` (the deprecated, non-thread-local
    /// helper) keeps working. We intentionally still exercise it so
    /// that any external consumer that imports it continues to build.
    #[test]
    #[allow(deprecated)]
    fn legacy_new_generation_stream_is_total() {
        let _ = new_generation_stream();
    }

    /// Resolving the same TLS handle twice on the same thread returns
    /// `MlxStream` wrappers that are distinct allocations (each call
    /// produces a fresh `unique_ptr`) but represent the same underlying
    /// per-thread MLX stream. We assert the allocations are independent
    /// by pointer comparison; MLX's per-thread invariant is provided by
    /// upstream and validated by upstream's own test suite.
    ///
    /// Skipped on CPU-only builds (no TLS handle to resolve).
    #[test]
    fn resolved_stream_wrappers_are_independent_allocations() {
        let Some(tls) = new_thread_local_generation_stream() else {
            // CPU-only build: nothing to verify.
            return;
        };
        let stream_a = ffi::stream_from_thread_local_stream(&tls);
        let stream_b = ffi::stream_from_thread_local_stream(&tls);

        // Two `make_unique` calls in the C++ resolver always produce
        // distinct heap allocations for the wrapper struct, so a
        // different address is the cheapest structural check that the
        // bridge is genuinely returning two separately owned wrappers
        // (rather than, say, alias-aliasing a single static handle).
        let ptr_a: *const MlxStream = stream_a.as_ref().expect("stream A non-null");
        let ptr_b: *const MlxStream = stream_b.as_ref().expect("stream B non-null");
        assert_ne!(
            ptr_a, ptr_b,
            "stream_from_thread_local_stream must produce independent wrappers per call"
        );
    }

    /// Round-trip: resolving the handle on the main thread, installing
    /// it as default, dispatching a tiny op, and synchronizing through
    /// the same handle does not panic and leaves MLX in a usable state
    /// for subsequent dispatches.
    ///
    /// A [`DefaultStreamGuard`] captures the previous default stream
    /// before installation so that the per-thread state is restored on
    /// exit, regardless of whether the test panics. This prevents state
    /// from leaking into other tests that run on the same thread.
    ///
    /// Skipped on CPU-only builds.
    #[test]
    fn install_and_synchronize_round_trip_works() {
        let Some(tls) = new_thread_local_generation_stream() else {
            return;
        };
        // Capture previous default stream; restored on drop.
        let _guard = DefaultStreamGuard::capture();
        install_thread_local_default_stream(Some(&tls));
        // A trivial op on the now-installed default stream — verifies
        // that the resolved stream is wired correctly into MLX's
        // dispatch system.
        let arr = ffi::zeros(&[1, 1], crate::dtype::FLOAT32);
        ffi::eval(&arr);
        synchronize_thread_local_stream(Some(&tls));
    }

    /// The TLS-backed install path is a no-op when handed `None` (the
    /// CPU-only build case). It must not panic, and must not leave
    /// MLX's default stream in a bad state — a follow-up trivial op
    /// must still succeed.
    #[test]
    fn install_thread_local_default_stream_is_noop_on_none() {
        install_thread_local_default_stream(None);
        synchronize_thread_local_stream(None);
        // Sanity: MLX is still usable.
        let arr = ffi::zeros(&[1, 1], crate::dtype::FLOAT32);
        ffi::eval(&arr);
    }
}
