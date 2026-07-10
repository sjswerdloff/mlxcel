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
//! `GET /admin/decode-config` returns the effective config; `POST` applies a
//! new path (`{"kvarn_decode_path": "v1"}`), re-reads the TOML file
//! (`{"reload": true}`), or both — reload first, then the explicit path wins.
//! Every response body IS the effective config, so a probe driving the server
//! always gets the authoritative state in-band. The endpoint sits behind the
//! same `api_key_auth` middleware as every other route (the same posture as
//! `/v1/cache/reset`, the existing state-mutating admin precedent).

use axum::Json;
use axum::http::StatusCode;
use serde::Deserialize;

use crate::decode_config::{self, EffectiveConfig};
use crate::server::types::ErrorResponse;

#[derive(Debug, Deserialize)]
pub struct DecodeConfigUpdate {
    /// New path to apply: "auto" | "v1".
    pub kvarn_decode_path: Option<String>,
    /// Re-read the `MLXCEL_DECODE_CONFIG` TOML before applying `kvarn_decode_path`.
    #[serde(default)]
    pub reload: bool,
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
    if !update.reload && update.kvarn_decode_path.is_none() {
        return Err(
            "empty update: set 'kvarn_decode_path' (\"auto\" | \"v1\") and/or 'reload': true"
                .to_string(),
        );
    }
    let mut effective = None;
    if update.reload {
        effective = Some(decode_config::reload_from_file("api-reload")?);
    }
    if let Some(path) = update.kvarn_decode_path {
        let parsed = path.parse()?;
        effective = Some(decode_config::apply(parsed, "api"));
    }
    Ok(effective.expect("guarded above: at least one action ran"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode_config::KvarnDecodePath;

    // These tests go through the process-global store (the route surface has
    // no other store), so each asserts on the RESULT of its own apply rather
    // than on absolute global state — order- and parallelism-independent.

    #[test]
    fn empty_update_is_rejected() {
        let err = apply_update(DecodeConfigUpdate {
            kvarn_decode_path: None,
            reload: false,
        })
        .unwrap_err();
        assert!(err.contains("empty update"));
    }

    #[test]
    fn bad_path_string_is_rejected_with_the_offending_value() {
        let err = apply_update(DecodeConfigUpdate {
            kvarn_decode_path: Some("warp_drive".to_string()),
            reload: false,
        })
        .unwrap_err();
        assert!(err.contains("warp_drive"));
    }

    #[test]
    fn apply_path_reports_the_applied_value_and_api_source() {
        let snap = apply_update(DecodeConfigUpdate {
            kvarn_decode_path: Some("auto".to_string()),
            reload: false,
        })
        .expect("valid update");
        assert_eq!(snap.kvarn_decode_path, KvarnDecodePath::Auto);
        assert_eq!(snap.source, "api");
    }
}
