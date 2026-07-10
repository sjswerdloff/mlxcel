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
//! ## Runtime keys (reloadable on a live process)
//!
//! - `kvarn_decode_path = "auto" | "v1"` — the override can only DISABLE
//!   gathering, never force it. `auto` (default): the structural predicate at
//!   the dispatch site decides. `v1`: every decode step takes the full-window
//!   fetch. There is deliberately no `gathered` value: the structural
//!   predicate (block-fetch support, decode-shaped chunk, unsaturated
//!   selector, healthy idx lockstep) is load-bearing for memory safety and
//!   correctness — a forced gather on an ineligible step would fetch against
//!   the wrong cache shape.
//! - `msa_core = "blocked" | "sdpa"` — which CORE the gathered flow uses when
//!   it runs (§H1, approach G). Select-among-implementations: it picks between
//!   two contract-equivalent cores but never widens WHERE gathering happens —
//!   the only-narrowing rule above still owns that.
//!
//! The 2026-07-10 rank matrix proved fetch and core are ORTHOGONAL axes
//! (each selected independently at 8K/300K/500K), so path variants join as
//! separate keys rather than as new [`KvarnDecodePath`] values — superseding
//! this module's original §H1/§H3 extension note.
//!
//! ## Construction keys (boot-frozen)
//!
//! The `[construction]` TOML section is read ONCE, at the first file load;
//! these choices shape what caches store or which stored representation the
//! fetch reads, so they cannot flip under a resident session:
//!
//! - `msa_fetch = "dequant" | "qmm"` — the gathered flow's fetch
//!   implementation on KVarN8 caches (C, qmm-fetch fused core).
//! - `fp16_gathered = true | false` — lets Fp16 caches report block-fetch
//!   support so the gathered flow runs on fp16 buffers (process-wide
//!   capability latch in `mlxcel_core::cache`).
//!
//! Config says INTENT, cache structure says CAN — both must hold. Every
//! construction key sits ABOVE a structural gate at its dispatch site (e.g.
//! `msa_fetch = "qmm"` still falls through to the dequant fetch+core pair
//! whenever `kvarn_qmm_state()` returns `None`); enabling a key never
//! bypasses those gates.
//!
//! A later reload whose `[construction]` section diverges from the running
//! values WARNS (restart required) and still applies the runtime keys — a
//! sweep toggling runtime keys must not fail because a frozen key sits in the
//! same file. `POST /admin/decode-config` refuses construction keys outright.
//!
//! ## Env seeding (bench compatibility)
//!
//! The legacy env instruments (`MLXCEL_MSA_CORE=sdpa`, `MLXCEL_MSA_FETCH=qmm`,
//! `MLXCEL_FP16_GATHERED=1`) seed this store's INITIAL defaults and remain
//! fully functional for bench binaries that never load a TOML — but the store
//! is the single source of truth at every dispatch site; the env vars are
//! consulted exactly once, at first touch.
//!
//! ## Reload surfaces
//!
//! All echo the effective config (fail-loud — a probe must never
//! mis-attribute a measurement to the wrong path):
//! - TOML file named by `MLXCEL_DECODE_CONFIG`, re-read on SIGHUP;
//! - `GET/POST /admin/decode-config` (see `server::routes::decode_config`);
//! - every HTTP response carries `x-mlxcel-decode-config: path=..; core=..; v=N`.
//!
//! The version counter is process-local and bumped on every successful apply
//! (never read from the file — a stale file must not be able to replay an old
//! version number). Hot-path cost: [`gathered_enabled`] and [`msa_core_sdpa`]
//! are one relaxed atomic load each per attention forward; [`msa_fetch_qmm`]
//! is one `OnceLock` read; the full snapshot is only built on the admin and
//! echo surfaces.
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

/// Which core the gathered decode flow uses when the structural predicate
/// lets it run (§H1, approach G). Select-among-implementations only — this
/// never widens where gathering happens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MsaCore {
    /// Blocked-gather core (take_along_axis remap + blocked softmax).
    Blocked,
    /// Fused-SDPA masked core (G): one fast_scaled_dot_product_attention
    /// call over the compact window with a block-membership mask.
    Sdpa,
}

impl std::fmt::Display for MsaCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MsaCore::Blocked => write!(f, "blocked"),
            MsaCore::Sdpa => write!(f, "sdpa"),
        }
    }
}

impl std::str::FromStr for MsaCore {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "blocked" => Ok(MsaCore::Blocked),
            "sdpa" => Ok(MsaCore::Sdpa),
            other => Err(format!(
                "unknown msa_core '{other}' (expected 'blocked' or 'sdpa')"
            )),
        }
    }
}

/// Which fetch implementation the gathered flow uses on KVarN8 caches.
/// Construction-frozen: the qmm core reads the stored representation
/// directly (pool view + fold scalars), so the choice is bound to how the
/// resident cache was built and cannot flip under it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MsaFetch {
    /// Dequant-after-gather fetch feeding a separate core (K1 + G-eligible).
    Dequant,
    /// C: qmm-fetch fused core — scores/attends straight off stored u8 codes
    /// via gather_qmm; no dequant chain, no compact-window materialization.
    Qmm,
}

impl std::fmt::Display for MsaFetch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MsaFetch::Dequant => write!(f, "dequant"),
            MsaFetch::Qmm => write!(f, "qmm"),
        }
    }
}

impl std::str::FromStr for MsaFetch {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "dequant" => Ok(MsaFetch::Dequant),
            "qmm" => Ok(MsaFetch::Qmm),
            other => Err(format!(
                "unknown msa_fetch '{other}' (expected 'dequant' or 'qmm')"
            )),
        }
    }
}

/// Boot-frozen construction values, as reported on the echo surfaces.
/// `fp16_gathered` is read from the process-wide capability latch in
/// `mlxcel_core::cache` (the single source of truth the fp16 dispatch
/// actually consults); `msa_fetch` from this store's construction latch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ConstructionSnapshot {
    pub msa_fetch: MsaFetch,
    pub fp16_gathered: bool,
}

/// Initial defaults, seeded once from the legacy env instruments so bench
/// binaries that never load a TOML keep their exact behavior. Injected into
/// [`Store::new`] rather than read inside it, so tests exercise seeding
/// without process-global env mutation.
#[derive(Debug, Clone, Copy)]
pub struct StoreDefaults {
    pub msa_core: MsaCore,
    pub msa_fetch: MsaFetch,
}

impl Default for StoreDefaults {
    fn default() -> Self {
        StoreDefaults {
            msa_core: MsaCore::Blocked,
            msa_fetch: MsaFetch::Dequant,
        }
    }
}

impl StoreDefaults {
    /// Env-seeded defaults. Exact legacy semantics preserved: any value other
    /// than the enabling literal (including parse garbage) means the default
    /// implementation — the env instruments were switches, not parsers.
    pub fn from_env() -> Self {
        StoreDefaults {
            msa_core: if std::env::var("MLXCEL_MSA_CORE").is_ok_and(|v| v == "sdpa") {
                MsaCore::Sdpa
            } else {
                MsaCore::Blocked
            },
            msa_fetch: if std::env::var("MLXCEL_MSA_FETCH").is_ok_and(|v| v == "qmm") {
                MsaFetch::Qmm
            } else {
                MsaFetch::Dequant
            },
        }
    }
}

/// The effective config, as reported on every echo surface.
#[derive(Debug, Clone, Serialize)]
pub struct EffectiveConfig {
    pub kvarn_decode_path: KvarnDecodePath,
    pub msa_core: MsaCore,
    /// Process-local monotonic apply counter (0 = untouched defaults).
    pub version: u64,
    /// What applied the current value: "default" | "file" | "sighup" | "api"
    /// | "api-reload".
    pub source: String,
    /// The TOML file consulted on reload, if `MLXCEL_DECODE_CONFIG` is set.
    pub config_file: Option<PathBuf>,
    /// Boot-frozen construction values (never bump `version`).
    pub construction: ConstructionSnapshot,
}

/// On-disk shape of the `MLXCEL_DECODE_CONFIG` TOML file.
#[derive(Debug, Deserialize)]
struct FileConfig {
    kvarn_decode_path: Option<String>,
    msa_core: Option<String>,
    construction: Option<FileConstruction>,
}

#[derive(Debug, Deserialize)]
struct FileConstruction {
    msa_fetch: Option<String>,
    fp16_gathered: Option<bool>,
}

/// A partial runtime update: absent fields keep their current value. One
/// apply = one version bump, however many fields it carries — the version
/// counts config STATES, not fields.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConfigUpdate {
    pub path: Option<KvarnDecodePath>,
    pub msa_core: Option<MsaCore>,
}

/// Reloadable store. Kept as a struct (rather than free functions over
/// globals) so tests exercise real instances without cross-test global
/// pollution — the model test suites must keep seeing pristine defaults.
/// (Exception: `ConstructionSnapshot::fp16_gathered` reads the process-wide
/// latch in `mlxcel_core`; instance stores report the process value, which
/// no test in this crate mutates.)
pub struct Store {
    inner: RwLock<Inner>,
    /// Hot-path mirror of `inner.path == V1`. The dispatch site reads ONLY
    /// this, one relaxed load per attention forward.
    gather_disabled: AtomicBool,
    /// Hot-path mirror of `inner.msa_core == Sdpa`. Same contract.
    sdpa_core: AtomicBool,
    /// Construction latch: set at the first file load (or lazily to
    /// `defaults` at first dispatch read, for processes that never load a
    /// TOML — the bench path). Frozen thereafter.
    msa_fetch: OnceLock<MsaFetch>,
    defaults: StoreDefaults,
    config_file: Option<PathBuf>,
}

struct Inner {
    path: KvarnDecodePath,
    msa_core: MsaCore,
    source: String,
    /// Lives INSIDE the lock, on purpose (Clement's H2 review): the runtime
    /// values and version must move together, or a snapshot racing an apply
    /// could echo (new values, old version) — two different configs sharing
    /// a version number, which is precisely the mis-attribution the echo
    /// surfaces exist to make impossible.
    version: u64,
}

impl Store {
    pub fn new(config_file: Option<PathBuf>, defaults: StoreDefaults) -> Self {
        Store {
            inner: RwLock::new(Inner {
                path: KvarnDecodePath::Auto,
                msa_core: defaults.msa_core,
                source: "default".to_string(),
                version: 0,
            }),
            gather_disabled: AtomicBool::new(false),
            sdpa_core: AtomicBool::new(defaults.msa_core == MsaCore::Sdpa),
            msa_fetch: OnceLock::new(),
            defaults,
            config_file,
        }
    }

    /// True when the gathered kvarn8 decode path may run (i.e. not forced v1).
    pub fn gathered_enabled(&self) -> bool {
        !self.gather_disabled.load(Ordering::Relaxed)
    }

    /// True when the gathered flow should use the fused-SDPA masked core (G).
    pub fn msa_core_sdpa(&self) -> bool {
        self.sdpa_core.load(Ordering::Relaxed)
    }

    /// True when the gathered flow should use the C qmm-fetch fused core.
    /// First read latches the construction default for processes that never
    /// load a TOML (bench compatibility).
    pub fn msa_fetch_qmm(&self) -> bool {
        *self.msa_fetch.get_or_init(|| self.defaults.msa_fetch) == MsaFetch::Qmm
    }

    fn construction_snapshot(&self) -> ConstructionSnapshot {
        ConstructionSnapshot {
            msa_fetch: *self.msa_fetch.get_or_init(|| self.defaults.msa_fetch),
            fp16_gathered: mlxcel_core::cache::fp16_gathered_effective(),
        }
    }

    pub fn snapshot(&self) -> EffectiveConfig {
        let (path, msa_core, version, source) = {
            let inner = self.inner.read().expect("decode_config lock poisoned");
            (
                inner.path,
                inner.msa_core,
                inner.version,
                inner.source.clone(),
            )
        };
        EffectiveConfig {
            kvarn_decode_path: path,
            msa_core,
            version,
            source,
            config_file: self.config_file.clone(),
            construction: self.construction_snapshot(),
        }
    }

    /// Apply a partial runtime update. Bumps the version once, updates the
    /// hot-path mirrors, and logs the effective config (the fail-loud echo).
    ///
    /// Invariant: (path, msa_core, version, source) update atomically under
    /// the write lock and are only ever read together under the read lock, so
    /// every snapshot — hence every response-header echo — is a tuple some
    /// apply produced (or the version-0 default). Sequenced probes get
    /// perfect attribution; a response already in flight when apply lands is
    /// ambiguous by nature, which the harness handles by switching paths
    /// between generation segments, never during one.
    pub fn apply(&self, update: ConfigUpdate, source: &str) -> EffectiveConfig {
        let (path, msa_core, snap) = {
            let mut inner = self.inner.write().expect("decode_config lock poisoned");
            if let Some(path) = update.path {
                inner.path = path;
            }
            if let Some(core) = update.msa_core {
                inner.msa_core = core;
            }
            inner.source = source.to_string();
            inner.version += 1;
            (
                inner.path,
                inner.msa_core,
                EffectiveConfig {
                    kvarn_decode_path: inner.path,
                    msa_core: inner.msa_core,
                    version: inner.version,
                    source: inner.source.clone(),
                    config_file: self.config_file.clone(),
                    construction: self.construction_snapshot(),
                },
            )
        };
        // The mirrors only gate dispatch behavior (never attribution), so
        // relaxed stores after the lock are fine: the echo reads the locked
        // state, and mid-apply dispatch ambiguity is the documented
        // in-flight case above.
        self.gather_disabled
            .store(path == KvarnDecodePath::V1, Ordering::Relaxed);
        self.sdpa_core
            .store(msa_core == MsaCore::Sdpa, Ordering::Relaxed);
        tracing::info!(
            "decode_config applied: kvarn_decode_path={} msa_core={} version={} source={}",
            snap.kvarn_decode_path,
            snap.msa_core,
            snap.version,
            snap.source
        );
        snap
    }

    /// Re-read the TOML file (if configured). A missing or invalid file
    /// leaves the running config unchanged — degrade loudly, never
    /// destructively, under a resident session. Every key is validated
    /// BEFORE anything applies: a reload is all-or-nothing.
    pub fn reload_from_file(&self, source: &str) -> Result<EffectiveConfig, String> {
        let Some(path) = self.config_file.as_ref() else {
            return Err("MLXCEL_DECODE_CONFIG not set; nothing to reload".to_string());
        };
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        let parsed: FileConfig =
            toml::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
        // Validate every key before applying any (all-or-nothing). An absent
        // runtime key is a valid "reset to this process's default".
        let decode_path = match parsed.kvarn_decode_path {
            Some(s) => s.parse::<KvarnDecodePath>()?,
            None => KvarnDecodePath::Auto,
        };
        let msa_core = match parsed.msa_core {
            Some(s) => s.parse::<MsaCore>()?,
            None => self.defaults.msa_core,
        };
        let construction_fetch =
            match parsed.construction.as_ref().and_then(|c| c.msa_fetch.as_deref()) {
                Some(s) => Some(s.parse::<MsaFetch>()?),
                None => None,
            };
        let construction_fp16 = parsed.construction.as_ref().and_then(|c| c.fp16_gathered);

        // Construction latch: first load freezes; later divergence warns and
        // is otherwise ignored — runtime keys must still apply.
        let desired_fetch = construction_fetch.unwrap_or(self.defaults.msa_fetch);
        let frozen_fetch = *self.msa_fetch.get_or_init(|| desired_fetch);
        if frozen_fetch != desired_fetch {
            tracing::warn!(
                "decode_config: construction key msa_fetch={} differs from the running \
                 value {} — construction keys are boot-frozen; restart required. \
                 Runtime keys from this reload still apply.",
                desired_fetch,
                frozen_fetch
            );
        }
        if let Some(want_fp16) = construction_fp16 {
            if let Err(current) = mlxcel_core::cache::init_fp16_gathered(want_fp16) {
                tracing::warn!(
                    "decode_config: construction key fp16_gathered={} differs from the \
                     latched process value {} — construction keys are boot-frozen; \
                     restart required. Runtime keys from this reload still apply.",
                    want_fp16,
                    current
                );
            }
        }

        Ok(self.apply(
            ConfigUpdate {
                path: Some(decode_path),
                msa_core: Some(msa_core),
            },
            source,
        ))
    }
}

static STORE: OnceLock<Store> = OnceLock::new();

fn global() -> &'static Store {
    STORE.get_or_init(|| {
        let config_file = std::env::var_os("MLXCEL_DECODE_CONFIG").map(PathBuf::from);
        Store::new(config_file, StoreDefaults::from_env())
    })
}

/// Hot-path check for the kvarn8 dispatch site (one relaxed atomic load).
pub fn gathered_enabled() -> bool {
    global().gathered_enabled()
}

/// Hot-path check for the gathered flow's core selection (one relaxed load).
pub fn msa_core_sdpa() -> bool {
    global().msa_core_sdpa()
}

/// Construction check for the gathered flow's fetch selection (one
/// `OnceLock` read; latches the env-seeded default on first touch).
pub fn msa_fetch_qmm() -> bool {
    global().msa_fetch_qmm()
}

pub fn snapshot() -> EffectiveConfig {
    global().snapshot()
}

pub fn apply(update: ConfigUpdate, source: &str) -> EffectiveConfig {
    global().apply(update, source)
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
                "decode_config defaults: kvarn_decode_path={} msa_core={} version={} source={}",
                snap.kvarn_decode_path,
                snap.msa_core,
                snap.version,
                snap.source
            );
        }
        // Boot artifact for the frozen keys, whatever set them (file or
        // env-seeded default) — the one authoritative construction line.
        let cons = store.construction_snapshot();
        tracing::info!(
            "decode_config construction (boot-frozen): msa_fetch={} fp16_gathered={}",
            cons.msa_fetch,
            cons.fp16_gathered
        );
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
                                "decode_config SIGHUP reload: kvarn_decode_path={} msa_core={} version={}",
                                snap.kvarn_decode_path,
                                snap.msa_core,
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

    fn store(config_file: Option<PathBuf>) -> Store {
        Store::new(config_file, StoreDefaults::default())
    }

    #[test]
    fn defaults_are_auto_blocked_gathering_enabled_version_zero() {
        let s = store(None);
        assert!(s.gathered_enabled());
        assert!(!s.msa_core_sdpa());
        let snap = s.snapshot();
        assert_eq!(snap.kvarn_decode_path, KvarnDecodePath::Auto);
        assert_eq!(snap.msa_core, MsaCore::Blocked);
        assert_eq!(snap.version, 0);
        assert_eq!(snap.source, "default");
        assert_eq!(snap.construction.msa_fetch, MsaFetch::Dequant);
    }

    #[test]
    fn env_seeded_defaults_shape_initial_state_without_version_bump() {
        // Injected defaults stand in for the env seed (StoreDefaults::from_env
        // is a trivial mapping; the seeding CONTRACT is what's pinned here).
        let s = Store::new(
            None,
            StoreDefaults {
                msa_core: MsaCore::Sdpa,
                msa_fetch: MsaFetch::Qmm,
            },
        );
        assert!(s.msa_core_sdpa(), "env-seeded core default drives the mirror");
        assert!(s.msa_fetch_qmm(), "env-seeded fetch default drives dispatch");
        let snap = s.snapshot();
        assert_eq!(snap.msa_core, MsaCore::Sdpa);
        assert_eq!(snap.construction.msa_fetch, MsaFetch::Qmm);
        assert_eq!(snap.version, 0, "seeding is a default, not an apply");
        assert_eq!(snap.source, "default");
    }

    #[test]
    fn apply_v1_disables_gathering_and_bumps_version() {
        let s = store(None);
        let snap = s.apply(
            ConfigUpdate {
                path: Some(KvarnDecodePath::V1),
                msa_core: None,
            },
            "api",
        );
        assert!(!s.gathered_enabled());
        assert_eq!(snap.version, 1);
        assert_eq!(snap.source, "api");
        let snap = s.apply(
            ConfigUpdate {
                path: Some(KvarnDecodePath::Auto),
                msa_core: None,
            },
            "api",
        );
        assert!(s.gathered_enabled());
        assert_eq!(snap.version, 2);
    }

    #[test]
    fn apply_msa_core_flips_mirror_and_leaves_path_untouched() {
        let s = store(None);
        s.apply(
            ConfigUpdate {
                path: Some(KvarnDecodePath::V1),
                msa_core: None,
            },
            "api",
        );
        let snap = s.apply(
            ConfigUpdate {
                path: None,
                msa_core: Some(MsaCore::Sdpa),
            },
            "api",
        );
        assert!(s.msa_core_sdpa());
        assert_eq!(
            snap.kvarn_decode_path,
            KvarnDecodePath::V1,
            "partial update must not reset the other key"
        );
        assert_eq!(snap.msa_core, MsaCore::Sdpa);
        assert_eq!(snap.version, 2);
        let snap = s.apply(
            ConfigUpdate {
                path: None,
                msa_core: Some(MsaCore::Blocked),
            },
            "api",
        );
        assert!(!s.msa_core_sdpa());
        assert_eq!(snap.version, 3);
    }

    #[test]
    fn empty_update_still_bumps_version_and_records_source() {
        // An empty ConfigUpdate is a no-op on values but a real apply (the
        // admin surface guards against sending one; the store stays simple).
        let s = store(None);
        let snap = s.apply(ConfigUpdate::default(), "api");
        assert_eq!(snap.version, 1);
        assert_eq!(snap.kvarn_decode_path, KvarnDecodePath::Auto);
        assert_eq!(snap.msa_core, MsaCore::Blocked);
    }

    #[test]
    fn path_parse_accepts_auto_v1_rejects_others() {
        assert_eq!("auto".parse::<KvarnDecodePath>(), Ok(KvarnDecodePath::Auto));
        assert_eq!(" V1 ".parse::<KvarnDecodePath>(), Ok(KvarnDecodePath::V1));
        assert!("gathered".parse::<KvarnDecodePath>().is_err());
        assert!("".parse::<KvarnDecodePath>().is_err());
    }

    #[test]
    fn msa_core_parse_accepts_blocked_sdpa_rejects_others() {
        assert_eq!("blocked".parse::<MsaCore>(), Ok(MsaCore::Blocked));
        assert_eq!(" SDPA ".parse::<MsaCore>(), Ok(MsaCore::Sdpa));
        assert!("fused".parse::<MsaCore>().is_err());
        assert!("".parse::<MsaCore>().is_err());
    }

    #[test]
    fn msa_fetch_parse_accepts_dequant_qmm_rejects_others() {
        assert_eq!("dequant".parse::<MsaFetch>(), Ok(MsaFetch::Dequant));
        assert_eq!(" Qmm ".parse::<MsaFetch>(), Ok(MsaFetch::Qmm));
        assert!("gathered".parse::<MsaFetch>().is_err());
        assert!("".parse::<MsaFetch>().is_err());
    }

    #[test]
    fn reload_reads_both_runtime_keys_and_records_source() {
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        writeln!(f, "kvarn_decode_path = \"v1\"").unwrap();
        writeln!(f, "msa_core = \"sdpa\"").unwrap();
        let s = store(Some(f.path().to_path_buf()));
        let snap = s.reload_from_file("sighup").expect("reload");
        assert_eq!(snap.kvarn_decode_path, KvarnDecodePath::V1);
        assert_eq!(snap.msa_core, MsaCore::Sdpa);
        assert_eq!(snap.source, "sighup");
        assert!(!s.gathered_enabled());
        assert!(s.msa_core_sdpa());
    }

    #[test]
    fn reload_with_empty_file_resets_to_process_defaults() {
        // "Default" means THIS process's env-seeded default, not a hardcoded
        // value — a bench that booted with MLXCEL_MSA_CORE=sdpa resets to
        // sdpa, not blocked.
        let f = tempfile::NamedTempFile::new().expect("tempfile");
        let s = Store::new(
            Some(f.path().to_path_buf()),
            StoreDefaults {
                msa_core: MsaCore::Sdpa,
                msa_fetch: MsaFetch::Dequant,
            },
        );
        s.apply(
            ConfigUpdate {
                path: Some(KvarnDecodePath::V1),
                msa_core: Some(MsaCore::Blocked),
            },
            "api",
        );
        let snap = s.reload_from_file("sighup").expect("reload");
        assert_eq!(snap.kvarn_decode_path, KvarnDecodePath::Auto);
        assert_eq!(snap.msa_core, MsaCore::Sdpa, "reset lands on the seeded default");
        assert!(s.gathered_enabled());
        assert!(s.msa_core_sdpa());
    }

    #[test]
    fn bad_file_leaves_config_unchanged() {
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        writeln!(f, "kvarn_decode_path = \"warp_drive\"").unwrap();
        let s = store(Some(f.path().to_path_buf()));
        s.apply(
            ConfigUpdate {
                path: Some(KvarnDecodePath::V1),
                msa_core: None,
            },
            "api",
        );
        let before = s.snapshot();
        let err = s.reload_from_file("sighup").unwrap_err();
        assert!(err.contains("warp_drive"), "error names the bad value: {err}");
        let after = s.snapshot();
        assert_eq!(after.kvarn_decode_path, KvarnDecodePath::V1);
        assert_eq!(after.version, before.version, "failed reload must not bump");
        assert!(!s.gathered_enabled());
    }

    #[test]
    fn bad_msa_core_rejects_the_whole_reload() {
        // All-or-nothing: a good path plus a bad core applies NEITHER.
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        writeln!(f, "kvarn_decode_path = \"v1\"").unwrap();
        writeln!(f, "msa_core = \"fused\"").unwrap();
        let s = store(Some(f.path().to_path_buf()));
        let before = s.snapshot();
        let err = s.reload_from_file("sighup").unwrap_err();
        assert!(err.contains("fused"));
        let after = s.snapshot();
        assert_eq!(after.kvarn_decode_path, KvarnDecodePath::Auto);
        assert_eq!(after.version, before.version);
        assert!(s.gathered_enabled(), "no partial apply on a bad reload");
    }

    #[test]
    fn missing_file_is_an_error_not_a_panic() {
        let s = store(Some(PathBuf::from("/nonexistent/decode_config.toml")));
        assert!(s.reload_from_file("sighup").is_err());
        assert!(s.gathered_enabled(), "config unchanged on missing file");
    }

    #[test]
    fn no_file_configured_reload_is_a_clean_error() {
        let s = store(None);
        let err = s.reload_from_file("sighup").unwrap_err();
        assert!(err.contains("MLXCEL_DECODE_CONFIG"));
    }

    #[test]
    fn construction_msa_fetch_latches_on_first_load() {
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        writeln!(f, "[construction]").unwrap();
        writeln!(f, "msa_fetch = \"qmm\"").unwrap();
        let s = store(Some(f.path().to_path_buf()));
        s.reload_from_file("file").expect("initial load");
        assert!(s.msa_fetch_qmm());
        assert_eq!(s.snapshot().construction.msa_fetch, MsaFetch::Qmm);
    }

    #[test]
    fn construction_divergence_on_reload_warns_and_keeps_frozen_value() {
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        writeln!(f, "[construction]").unwrap();
        writeln!(f, "msa_fetch = \"qmm\"").unwrap();
        let s = store(Some(f.path().to_path_buf()));
        s.reload_from_file("file").expect("initial load");
        assert!(s.msa_fetch_qmm());

        // Rewrite the file to diverge on the frozen key AND flip a runtime
        // key: the runtime key must still apply; the frozen key must not.
        let mut f2 = std::fs::File::create(f.path()).expect("rewrite");
        writeln!(f2, "kvarn_decode_path = \"v1\"").unwrap();
        writeln!(f2, "[construction]").unwrap();
        writeln!(f2, "msa_fetch = \"dequant\"").unwrap();
        drop(f2);
        let snap = s.reload_from_file("sighup").expect("divergent reload succeeds");
        assert_eq!(
            snap.kvarn_decode_path,
            KvarnDecodePath::V1,
            "runtime key applied"
        );
        assert!(s.msa_fetch_qmm(), "construction key stays frozen");
        assert_eq!(snap.construction.msa_fetch, MsaFetch::Qmm);
    }

    #[test]
    fn msa_fetch_lazy_latches_the_default_when_no_file_ever_loads() {
        // The bench path: dispatch reads before any reload — first touch
        // freezes the env-seeded default, and a LATER file load cannot
        // change it (warn-only), because caches may already exist.
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        writeln!(f, "[construction]").unwrap();
        writeln!(f, "msa_fetch = \"qmm\"").unwrap();
        let s = store(Some(f.path().to_path_buf()));
        assert!(!s.msa_fetch_qmm(), "defaults latch on first dispatch read");
        s.reload_from_file("file").expect("late load");
        assert!(!s.msa_fetch_qmm(), "late file cannot thaw a frozen key");
    }

    #[test]
    fn bad_construction_msa_fetch_rejects_the_whole_reload() {
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        writeln!(f, "kvarn_decode_path = \"v1\"").unwrap();
        writeln!(f, "[construction]").unwrap();
        writeln!(f, "msa_fetch = \"warp\"").unwrap();
        let s = store(Some(f.path().to_path_buf()));
        let err = s.reload_from_file("file").unwrap_err();
        assert!(err.contains("warp"));
        assert!(s.gathered_enabled());
        assert_eq!(s.snapshot().version, 0, "nothing applied");
    }

    /// Clement's H2 review regression, extended to the runtime triple: a
    /// snapshot racing an apply must never observe a torn (path, msa_core,
    /// version) tuple — every echoed tuple has to be one some apply produced
    /// (or the version-0 default). Red on any field updated outside the
    /// write lock; green with all runtime fields inside.
    #[test]
    fn snapshot_never_observes_a_torn_runtime_tuple() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool as StopFlag;

        let s = Arc::new(store(None));
        let stop = Arc::new(StopFlag::new(false));
        const APPLIES: u64 = 500;

        let reader = {
            let s = Arc::clone(&s);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut seen = Vec::new();
                while !stop.load(Ordering::Relaxed) {
                    let snap = s.snapshot();
                    seen.push((snap.kvarn_decode_path, snap.msa_core, snap.version));
                }
                seen
            })
        };

        // Alternate BOTH runtime keys so every version has exactly one valid
        // partner pair: odd versions are (V1, Sdpa), even are (Auto, Blocked)
        // (version 0 = default (Auto, Blocked)).
        for i in 1..=APPLIES {
            let (expect_path, expect_core) = if i % 2 == 1 {
                (KvarnDecodePath::V1, MsaCore::Sdpa)
            } else {
                (KvarnDecodePath::Auto, MsaCore::Blocked)
            };
            let snap = s.apply(
                ConfigUpdate {
                    path: Some(expect_path),
                    msa_core: Some(expect_core),
                },
                "test",
            );
            assert_eq!(
                (snap.kvarn_decode_path, snap.msa_core, snap.version),
                (expect_path, expect_core, i)
            );
        }
        stop.store(true, Ordering::Relaxed);
        let seen = reader.join().expect("reader thread");

        for (path, core, version) in seen {
            let (valid_path, valid_core) = if version % 2 == 1 {
                (KvarnDecodePath::V1, MsaCore::Sdpa)
            } else {
                (KvarnDecodePath::Auto, MsaCore::Blocked)
            };
            assert_eq!(
                (path, core),
                (valid_path, valid_core),
                "torn tuple echoed at version {version}"
            );
        }
    }
}
