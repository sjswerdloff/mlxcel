#!/usr/bin/env bash
# Flip `use_sparse_attention` in a MiniMax-M3 checkpoint's config.json,
# with a one-time backup. Used by the MSA-vs-dense fidelity protocol
# (msa_dense_fidelity_capture.sh / msa_dense_fidelity_diff.py) to force
# the dense dispatch path — mlxcel reads the flag at model load, so the
# server must be restarted after toggling.
#
# Usage: m3_toggle_sparse.sh <MODEL_DIR> on|off
#   MODEL_DIR: directory containing config.json
#   on  -> use_sparse_attention = true  (normal MSA serving)
#   off -> use_sparse_attention = false (every layer dispatches dense)
#
# The first toggle writes config.json.pre_toggle_backup next to the
# config; restore it any time with:
#   cp "$MODEL_DIR/config.json.pre_toggle_backup" "$MODEL_DIR/config.json"

set -euo pipefail

MODEL_DIR="${1:?usage: m3_toggle_sparse.sh <MODEL_DIR> on|off}"
STATE="${2:?usage: m3_toggle_sparse.sh <MODEL_DIR> on|off}"
CFG="$MODEL_DIR/config.json"

[[ -f "$CFG" ]] || { echo "no config.json in $MODEL_DIR" >&2; exit 2; }
case "$STATE" in
  on) VALUE=true ;;
  off) VALUE=false ;;
  *) echo "state must be 'on' or 'off', got '$STATE'" >&2; exit 2 ;;
esac

[[ -f "$CFG.pre_toggle_backup" ]] || cp "$CFG" "$CFG.pre_toggle_backup"

python3 - "$CFG" "$VALUE" << 'EOF'
import json, sys

path, value = sys.argv[1], sys.argv[2] == "true"
with open(path) as f:
    cfg = json.load(f)

holder = cfg.get("text_config", cfg)
sac = holder.get("sparse_attention_config")
if sac is None or "use_sparse_attention" not in sac:
    sys.exit(f"{path}: no sparse_attention_config.use_sparse_attention key")
before = sac["use_sparse_attention"]
sac["use_sparse_attention"] = value
with open(path, "w") as f:
    json.dump(cfg, f, indent=1)
    f.write("\n")
print(f"use_sparse_attention: {before} -> {value}")
EOF

echo "Toggled. Restart the server for this to take effect (flag is read at model load)."
