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

//! Runtime-reloadable decode-path configuration (harness plan §H2).
//!
//! The problem this solves: comparing decode paths (v1 full-window vs K1
//! gathered) on a DEEP context today costs a server restart plus a full
//! re-prefill — at 300K that is the dominant experiment cost, and prompt-cache
//! adoption cannot bridge it (KVarN8 snapshots are refused by design). This
//! module makes the path a runtime value so ONE resident session can A/B
//! paths between generation segments with zero reloads.
//!
//! Semantics — the override can only DISABLE gathering, never force it:
//! - `auto` (default): the structural predicate at the dispatch site decides,
//!   exactly as before this module existed.
//! - `v1`: the gathered path is switched off; every decode step takes the
//!   full-window fetch. Safe on any cache mode.
//!
//! There is deliberately no `gathered` value: the structural predicate
//! (block-fetch support, decode-shaped chunk, unsaturated selector, healthy
//! idx lockstep) is load-bearing for memory safety and correctness — a forced
//! gather on an ineligible step would fetch against the wrong cache shape.
//! Future path variants (`gathered_sdpa`, `qmm_union`, `qmm_gather` — plan
//! §H1/§H3) will extend [`KvarnDecodePath`] to select AMONG implementations
//! when the predicate holds, keeping the same only-narrowing rule.
//!
//! Reload surfaces, all echoing the effective config (fail-loud — a probe
//! must never mis-attribute a measurement to the wrong path):
//! - TOML file named by `MLXCEL_DECODE_CONFIG`, re-read on SIGHUP;
//! - `GET/POST /admin/decode-config` (see `server::routes::decode_config`);
//! - every HTTP response carries `x-mlxcel-decode-config: path=..; v=N`.
//!
//! The version counter is process-local and bumped on every successful apply
//! (never read from the file — a stale file must not be able to replay an old
//! version number). Hot-path cost: [`gathered_enabled`] is one relaxed atomic
//! load per attention forward; the full snapshot is only built on the admin
//! and echo surfaces.
//!
//! A parse failure on reload leaves the running config UNCHANGED and logs the
//! error: degrading a resident 300K session over a typo'd TOML would destroy
//! exactly the state this module exists to preserve.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{OnceLock, RwLock};

use serde::{Deserialize, Serialize};

/// Which decode path the kvarn8 dispatch site is allowed to take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvarnDecodePath {
    /// The structural predicate decides (production default; today's behavior).
    Auto,
    /// Gathering disabled: every decode step takes the v1 full-window fetch.
    V1,
}

impl std::fmt::Display for KvarnDecodePath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvarnDecodePath::Auto => write!(f, "auto"),
            KvarnDecodePath::V1 => write!(f, "v1"),
        }
    }
}

impl std::str::FromStr for KvarnDecodePath {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(KvarnDecodePath::Auto),
            "v1" => Ok(KvarnDecodePath::V1),
            other => Err(format!(
                "unknown kvarn_decode_path '{other}' (expected 'auto' or 'v1')"
            )),
        }
    }
}

/// The effective config, as reported on every echo surface.
#[derive(Debug, Clone, Serialize)]
pub struct EffectiveConfig {
    pub kvarn_decode_path: KvarnDecodePath,
    /// Process-local monotonic apply counter (0 = untouched defaults).
    pub version: u64,
    /// What applied the current value: "default" | "file" | "sighup" | "api"
    /// | "api-reload".
    pub source: String,
    /// The TOML file consulted on reload, if `MLXCEL_DECODE_CONFIG` is set.
    pub config_file: Option<PathBuf>,
}

/// On-disk shape of the `MLXCEL_DECODE_CONFIG` TOML file.
#[derive(Debug, Deserialize)]
struct FileConfig {
    kvarn_decode_path: Option<String>,
}

/// Reloadable store. Kept as a struct (rather than free functions over
/// globals) so tests exercise real instances without cross-test global
/// pollution — the model test suites must keep seeing pristine defaults.
pub struct Store {
    inner: RwLock<Inner>,
    /// Hot-path mirror of `inner.path == V1`. The dispatch site reads ONLY
    /// this, one relaxed load per attention forward.
    gather_disabled: AtomicBool,
    config_file: Option<PathBuf>,
}

struct Inner {
    path: KvarnDecodePath,
    source: String,
    /// Lives INSIDE the lock, on purpose (Clement's H2 review): path and
    /// version must move together, or a snapshot racing an apply could echo
    /// (new path, old version) — two different configs sharing a version
    /// number, which is precisely the mis-attribution the echo surfaces
    /// exist to make impossible.
    version: u64,
}

impl Store {
    pub fn new(config_file: Option<PathBuf>) -> Self {
        Store {
            inner: RwLock::new(Inner {
                path: KvarnDecodePath::Auto,
                source: "default".to_string(),
                version: 0,
            }),
            gather_disabled: AtomicBool::new(false),
            config_file,
        }
    }

    /// True when the gathered kvarn8 decode path may run (i.e. not forced v1).
    pub fn gathered_enabled(&self) -> bool {
        !self.gather_disabled.load(Ordering::Relaxed)
    }

    pub fn snapshot(&self) -> EffectiveConfig {
        let inner = self.inner.read().expect("decode_config lock poisoned");
        EffectiveConfig {
            kvarn_decode_path: inner.path,
            version: inner.version,
            source: inner.source.clone(),
            config_file: self.config_file.clone(),
        }
    }

    /// Apply a new path. Bumps the version, updates the hot-path mirror, and
    /// logs the effective config (the fail-loud echo).
    ///
    /// Invariant: (path, version, source) update atomically under the write
    /// lock and are only ever read together under the read lock, so every
    /// snapshot — hence every response-header echo — is a pair some apply
    /// produced (or the (auto, 0) default). Sequenced probes get perfect
    /// attribution; a response already in flight when apply lands is
    /// ambiguous by nature, which the harness handles by switching paths
    /// between generation segments, never during one.
    pub fn apply(&self, path: KvarnDecodePath, source: &str) -> EffectiveConfig {
        let snap = {
            let mut inner = self.inner.write().expect("decode_config lock poisoned");
            inner.path = path;
            inner.source = source.to_string();
            inner.version += 1;
            EffectiveConfig {
                kvarn_decode_path: inner.path,
                version: inner.version,
                source: inner.source.clone(),
                config_file: self.config_file.clone(),
            }
        };
        // The mirror only gates dispatch behavior (never attribution), so a
        // relaxed store after the lock is fine: the echo reads the locked
        // state, and mid-apply dispatch ambiguity is the documented
        // in-flight case above.
        self.gather_disabled
            .store(path == KvarnDecodePath::V1, Ordering::Relaxed);
        tracing::info!(
            "decode_config applied: kvarn_decode_path={} version={} source={}",
            snap.kvarn_decode_path,
            snap.version,
            snap.source
        );
        snap
    }

    /// Re-read the TOML file (if configured). A missing or invalid file
    /// leaves the running config unchanged — degrade loudly, never
    /// destructively, under a resident session.
    pub fn reload_from_file(&self, source: &str) -> Result<EffectiveConfig, String> {
        let Some(path) = self.config_file.as_ref() else {
            return Err("MLXCEL_DECODE_CONFIG not set; nothing to reload".to_string());
        };
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        let parsed: FileConfig =
            toml::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
        let decode_path = match parsed.kvarn_decode_path {
            Some(s) => s.parse::<KvarnDecodePath>()?,
            // An empty file is a valid "reset to default".
            None => KvarnDecodePath::Auto,
        };
        Ok(self.apply(decode_path, source))
    }
}

static STORE: OnceLock<Store> = OnceLock::new();

fn global() -> &'static Store {
    STORE.get_or_init(|| {
        let config_file = std::env::var_os("MLXCEL_DECODE_CONFIG").map(PathBuf::from);
        Store::new(config_file)
    })
}

/// Hot-path check for the kvarn8 dispatch site (one relaxed atomic load).
pub fn gathered_enabled() -> bool {
    global().gathered_enabled()
}

pub fn snapshot() -> EffectiveConfig {
    global().snapshot()
}

pub fn apply(path: KvarnDecodePath, source: &str) -> EffectiveConfig {
    global().apply(path, source)
}

pub fn reload_from_file(source: &str) -> Result<EffectiveConfig, String> {
    global().reload_from_file(source)
}

/// Load the initial file config (if any) and install the SIGHUP watcher.
/// Idempotent; safe to call from `create_app` (only spawns when a tokio
/// runtime is present, so bare test invocations stay inert).
pub fn init_and_watch() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        let store = global();
        if store.config_file.is_some() {
            match store.reload_from_file("file") {
                Ok(_) => {}
                Err(e) => tracing::warn!("decode_config initial load failed: {e}"),
            }
        } else {
            // Still echo the effective default at boot so probes always have
            // one authoritative line to check.
            let snap = store.snapshot();
            tracing::info!(
                "decode_config defaults: kvarn_decode_path={} version={} source={}",
                snap.kvarn_decode_path,
                snap.version,
                snap.source
            );
        }
        #[cfg(unix)]
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn(async {
                    let Ok(mut hup) =
                        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup())
                    else {
                        tracing::warn!("decode_config: could not install SIGHUP handler");
                        return;
                    };
                    tracing::info!("decode_config: SIGHUP watcher installed");
                    loop {
                        hup.recv().await;
                        match global().reload_from_file("sighup") {
                            Ok(snap) => tracing::info!(
                                "decode_config SIGHUP reload: kvarn_decode_path={} version={}",
                                snap.kvarn_decode_path,
                                snap.version
                            ),
                            Err(e) => tracing::warn!("decode_config SIGHUP reload failed: {e}"),
                        }
                    }
                });
            }
            Err(_) => {
                // Installation is once-per-process: a first call from a bare
                // (non-runtime) context permanently forgoes SIGHUP reload, so
                // say so out loud rather than degrading silently (Clement's
                // H2 review). The admin endpoint remains fully functional.
                tracing::warn!(
                    "decode_config: no tokio runtime at init — SIGHUP reload unavailable \
                     for this process (admin endpoint unaffected)"
                );
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn defaults_are_auto_gathering_enabled_version_zero() {
        let s = Store::new(None);
        assert!(s.gathered_enabled());
        let snap = s.snapshot();
        assert_eq!(snap.kvarn_decode_path, KvarnDecodePath::Auto);
        assert_eq!(snap.version, 0);
        assert_eq!(snap.source, "default");
    }

    #[test]
    fn apply_v1_disables_gathering_and_bumps_version() {
        let s = Store::new(None);
        let snap = s.apply(KvarnDecodePath::V1, "api");
        assert!(!s.gathered_enabled());
        assert_eq!(snap.version, 1);
        assert_eq!(snap.source, "api");
        let snap = s.apply(KvarnDecodePath::Auto, "api");
        assert!(s.gathered_enabled());
        assert_eq!(snap.version, 2);
    }

    #[test]
    fn path_parse_accepts_auto_v1_rejects_others() {
        assert_eq!("auto".parse::<KvarnDecodePath>(), Ok(KvarnDecodePath::Auto));
        assert_eq!(" V1 ".parse::<KvarnDecodePath>(), Ok(KvarnDecodePath::V1));
        assert!("gathered".parse::<KvarnDecodePath>().is_err());
        assert!("".parse::<KvarnDecodePath>().is_err());
    }

    #[test]
    fn reload_reads_file_and_records_source() {
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        writeln!(f, "kvarn_decode_path = \"v1\"").unwrap();
        let s = Store::new(Some(f.path().to_path_buf()));
        let snap = s.reload_from_file("sighup").expect("reload");
        assert_eq!(snap.kvarn_decode_path, KvarnDecodePath::V1);
        assert_eq!(snap.source, "sighup");
        assert!(!s.gathered_enabled());
    }

    #[test]
    fn reload_with_empty_file_resets_to_auto() {
        let f = tempfile::NamedTempFile::new().expect("tempfile");
        let s = Store::new(Some(f.path().to_path_buf()));
        s.apply(KvarnDecodePath::V1, "api");
        let snap = s.reload_from_file("sighup").expect("reload");
        assert_eq!(snap.kvarn_decode_path, KvarnDecodePath::Auto);
        assert!(s.gathered_enabled());
    }

    #[test]
    fn bad_file_leaves_config_unchanged() {
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        writeln!(f, "kvarn_decode_path = \"warp_drive\"").unwrap();
        let s = Store::new(Some(f.path().to_path_buf()));
        s.apply(KvarnDecodePath::V1, "api");
        let before = s.snapshot();
        let err = s.reload_from_file("sighup").unwrap_err();
        assert!(err.contains("warp_drive"), "error names the bad value: {err}");
        let after = s.snapshot();
        assert_eq!(after.kvarn_decode_path, KvarnDecodePath::V1);
        assert_eq!(after.version, before.version, "failed reload must not bump");
        assert!(!s.gathered_enabled());
    }

    #[test]
    fn missing_file_is_an_error_not_a_panic() {
        let s = Store::new(Some(PathBuf::from("/nonexistent/decode_config.toml")));
        assert!(s.reload_from_file("sighup").is_err());
        assert!(s.gathered_enabled(), "config unchanged on missing file");
    }

    #[test]
    fn no_file_configured_reload_is_a_clean_error() {
        let s = Store::new(None);
        let err = s.reload_from_file("sighup").unwrap_err();
        assert!(err.contains("MLXCEL_DECODE_CONFIG"));
    }

    /// Clement's H2 review regression: a snapshot racing an apply must never
    /// observe a torn (path, version) pair — every echoed pair has to be one
    /// some apply produced (or the (auto, 0) default). Red on the
    /// version-outside-the-lock implementation; green with version inside.
    #[test]
    fn snapshot_never_observes_a_torn_path_version_pair() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool as StopFlag;

        let s = Arc::new(Store::new(None));
        let stop = Arc::new(StopFlag::new(false));
        const APPLIES: u64 = 500;

        let reader = {
            let s = Arc::clone(&s);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut seen = Vec::new();
                while !stop.load(Ordering::Relaxed) {
                    let snap = s.snapshot();
                    seen.push((snap.kvarn_decode_path, snap.version));
                }
                seen
            })
        };

        // Alternate paths so every version has exactly one valid partner:
        // odd versions are V1, even are Auto (version 0 = default Auto).
        for i in 1..=APPLIES {
            let expect = if i % 2 == 1 {
                KvarnDecodePath::V1
            } else {
                KvarnDecodePath::Auto
            };
            let snap = s.apply(expect, "test");
            assert_eq!((snap.kvarn_decode_path, snap.version), (expect, i));
        }
        stop.store(true, Ordering::Relaxed);
        let seen = reader.join().expect("reader thread");

        for (path, version) in seen {
            let valid = if version % 2 == 1 {
                KvarnDecodePath::V1
            } else {
                KvarnDecodePath::Auto
            };
            assert_eq!(
                path, valid,
                "torn pair echoed: version {version} paired with {path}"
            );
        }
    }
}
