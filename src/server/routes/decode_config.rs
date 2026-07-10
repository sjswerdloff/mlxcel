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

//! Admin surface for the runtime decode-path config (harness plan §H2).
//!
//! `GET /admin/decode-config` returns the effective config; `POST` applies
//! runtime keys (`{"kvarn_decode_path": "v1"}`, `{"msa_core": "sdpa"}`),
//! re-reads the TOML file (`{"reload": true}`), or both — reload first, then
//! the explicit keys win. Construction keys (`msa_fetch`, `fp16_gathered`)
//! are boot-frozen and REFUSED here with a pointed error — never silently
//! ignored, because a probe that believes it changed the fetch mode would
//! mis-attribute every measurement after. Every response body IS the
//! effective config, so a probe driving the server always gets the
//! authoritative state in-band. The endpoint sits behind the same
//! `api_key_auth` middleware as every other route (the same posture as
//! `/v1/cache/reset`, the existing state-mutating admin precedent).

use axum::Json;
use axum::http::StatusCode;
use serde::Deserialize;

use crate::decode_config::{self, ConfigUpdate, EffectiveConfig};
use crate::server::types::ErrorResponse;

#[derive(Debug, Deserialize)]
pub struct DecodeConfigUpdate {
    /// New path to apply: "auto" | "v1".
    pub kvarn_decode_path: Option<String>,
    /// New gathered-flow core to apply: "blocked" | "sdpa".
    pub msa_core: Option<String>,
    /// Re-read the `MLXCEL_DECODE_CONFIG` TOML before applying explicit keys.
    #[serde(default)]
    pub reload: bool,
    /// Construction key — named here so posting it fails loudly instead of
    /// being silently dropped as an unknown field. Always refused.
    pub msa_fetch: Option<serde_json::Value>,
    /// Construction key — same posture as `msa_fetch`. Always refused.
    pub fp16_gathered: Option<serde_json::Value>,
}

/// GET /admin/decode-config — report the effective config.
pub async fn decode_config_get() -> Json<EffectiveConfig> {
    Json(decode_config::snapshot())
}

/// POST /admin/decode-config — apply an update and report the result.
pub async fn decode_config_set(
    Json(update): Json<DecodeConfigUpdate>,
) -> Result<Json<EffectiveConfig>, (StatusCode, Json<ErrorResponse>)> {
    apply_update(update).map(Json).map_err(|msg| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(msg, "invalid_decode_config")),
        )
    })
}

/// Pure application logic, split from the axum plumbing so tests drive it
/// without an `AppState` (the `build_stats_response` pattern from
/// `routes/cache.rs`).
fn apply_update(update: DecodeConfigUpdate) -> Result<EffectiveConfig, String> {
    // Construction keys first: refuse before doing ANYTHING else, so a
    // request mixing a runtime key with a construction key applies nothing
    // (all-or-nothing, same rule as file reload).
    let refused: Vec<&str> = [
        update.msa_fetch.as_ref().map(|_| "msa_fetch"),
        update.fp16_gathered.as_ref().map(|_| "fp16_gathered"),
    ]
    .into_iter()
    .flatten()
    .collect();
    if !refused.is_empty() {
        return Err(format!(
            "{} {} construction key(s) — boot-frozen; set in the MLXCEL_DECODE_CONFIG \
             TOML [construction] section and restart the server. A live process cannot \
             change them, and nothing from this request was applied.",
            refused.join(", "),
            if refused.len() == 1 { "is a" } else { "are" },
        ));
    }
    if !update.reload && update.kvarn_decode_path.is_none() && update.msa_core.is_none() {
        return Err(
            "empty update: set 'kvarn_decode_path' (\"auto\" | \"v1\"), 'msa_core' \
             (\"blocked\" | \"sdpa\"), and/or 'reload': true"
                .to_string(),
        );
    }
    // Parse every explicit key BEFORE applying anything (all-or-nothing).
    let path = update
        .kvarn_decode_path
        .as_deref()
        .map(str::parse)
        .transpose()?;
    let msa_core = update.msa_core.as_deref().map(str::parse).transpose()?;

    let mut effective = None;
    if update.reload {
        effective = Some(decode_config::reload_from_file("api-reload")?);
    }
    if path.is_some() || msa_core.is_some() {
        effective = Some(decode_config::apply(ConfigUpdate { path, msa_core }, "api"));
    }
    Ok(effective.expect("guarded above: at least one action ran"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode_config::{KvarnDecodePath, MsaCore};

    // These tests go through the process-global store (the route surface has
    // no other store), so each asserts on the RESULT of its own apply rather
    // than on absolute global state — order- and parallelism-independent.

    fn update() -> DecodeConfigUpdate {
        DecodeConfigUpdate {
            kvarn_decode_path: None,
            msa_core: None,
            reload: false,
            msa_fetch: None,
            fp16_gathered: None,
        }
    }

    #[test]
    fn empty_update_is_rejected_and_names_every_runtime_key() {
        let err = apply_update(update()).unwrap_err();
        assert!(err.contains("empty update"));
        assert!(err.contains("kvarn_decode_path"));
        assert!(err.contains("msa_core"));
    }

    #[test]
    fn bad_path_string_is_rejected_with_the_offending_value() {
        let err = apply_update(DecodeConfigUpdate {
            kvarn_decode_path: Some("warp_drive".to_string()),
            ..update()
        })
        .unwrap_err();
        assert!(err.contains("warp_drive"));
    }

    #[test]
    fn bad_msa_core_string_is_rejected_with_the_offending_value() {
        let err = apply_update(DecodeConfigUpdate {
            msa_core: Some("fused".to_string()),
            ..update()
        })
        .unwrap_err();
        assert!(err.contains("fused"));
    }

    #[test]
    fn apply_path_reports_the_applied_value_and_api_source() {
        let snap = apply_update(DecodeConfigUpdate {
            kvarn_decode_path: Some("auto".to_string()),
            ..update()
        })
        .expect("valid update");
        assert_eq!(snap.kvarn_decode_path, KvarnDecodePath::Auto);
        assert_eq!(snap.source, "api");
    }

    #[test]
    fn apply_both_runtime_keys_lands_in_one_snapshot() {
        let snap = apply_update(DecodeConfigUpdate {
            kvarn_decode_path: Some("auto".to_string()),
            msa_core: Some("blocked".to_string()),
            ..update()
        })
        .expect("valid update");
        assert_eq!(snap.kvarn_decode_path, KvarnDecodePath::Auto);
        assert_eq!(snap.msa_core, MsaCore::Blocked);
        assert_eq!(snap.source, "api");
    }

    #[test]
    fn construction_keys_are_refused_with_a_pointed_error() {
        let err = apply_update(DecodeConfigUpdate {
            msa_fetch: Some(serde_json::json!("qmm")),
            ..update()
        })
        .unwrap_err();
        assert!(err.contains("msa_fetch"));
        assert!(err.contains("construction"));
        assert!(err.contains("restart"));
    }

    #[test]
    fn construction_key_mixed_with_runtime_key_applies_nothing() {
        // The runtime key is valid on its own; the construction key must
        // poison the WHOLE request (all-or-nothing), proven by the error
        // and by the absence of a bump in the next self-applied snapshot
        // chain (each test asserts only on its own applies).
        let err = apply_update(DecodeConfigUpdate {
            kvarn_decode_path: Some("auto".to_string()),
            fp16_gathered: Some(serde_json::json!(true)),
            ..update()
        })
        .unwrap_err();
        assert!(err.contains("fp16_gathered"));
        assert!(err.contains("nothing from this request was applied"));
    }
}
