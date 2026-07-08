#!/usr/bin/env python3
# Export KVarN reference vectors (results/kvarn_reference_vectors.npz, produced
# by kvarn_decomposition_experiment.py from the mlx-kvarn REFERENCE
# implementation) as flat little-endian binaries + a JSON manifest, for
# consumption by Rust unit tests via include_bytes!.
#
# The Rust port's pipeline stages are pinned to these exact values so
# "verified" means verified against the reference implementation's output,
# not against a re-derivation of the same math (cycle-65 lesson).
#
# k8 note: the npz holds the k4/iters=4 pipeline snapshot. The k8 fixtures are
# regenerated here by rerunning the reference pipeline at bits=8 on the SAME
# input tiles, so both bit-widths are pinned from one source of truth.

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    "/Users/stuartswerdloff/ai/ClaudeInstanceHomeOffices/clement-7074f29f/"
    "kindled_projects/mlx-kvarn",
)
import mlx.core as mx  # noqa: E402
from mlx_kvarn.hadamard import hadamard_rotate  # noqa: E402
from mlx_kvarn.sinkhorn import variance_normalize_batched  # noqa: E402
from mlx_kvarn.quant import asymmetric_rtn_per_row  # noqa: E402

OUT = Path("src/lib/mlxcel-core/src/cache/kvarn_fixtures")
NPZ = Path("results/kvarn_reference_vectors.npz")
SINKHORN_ITERS = 4


def dump(name: str, arr: np.ndarray, manifest: dict) -> None:
    arr = np.ascontiguousarray(arr)
    path = OUT / f"{name}.bin"
    arr.tofile(path)
    manifest[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(NPZ)
    tiles = z["input_tiles"].astype(np.float32)  # [2, 128, 128]

    manifest: dict = {"sinkhorn_iters": SINKHORN_ITERS, "source": str(NPZ)}
    dump("input_tiles_f32", tiles, manifest)

    x = mx.array(tiles)
    rot = hadamard_rotate(x)
    balanced, s_col, s_row = variance_normalize_batched(rot, iterations=SINKHORN_ITERS)
    mx.eval(rot, balanced, s_col, s_row)
    dump("rotated_f32", np.array(rot, copy=False).astype(np.float32), manifest)
    dump("balanced_f32", np.array(balanced, copy=False).astype(np.float32), manifest)
    dump("s_col_f32", np.array(s_col, copy=False).astype(np.float32), manifest)
    dump("s_row_f32", np.array(s_row, copy=False).astype(np.float32), manifest)

    for bits in (8, 4):
        q, scale, zp = asymmetric_rtn_per_row(balanced, bits)
        mx.eval(q, scale, zp)
        dump(f"q{bits}_u8", np.array(q, copy=False).astype(np.uint8), manifest)
        dump(f"scale{bits}_f32", np.array(scale, copy=False).astype(np.float32), manifest)
        dump(f"zp{bits}_f32", np.array(zp, copy=False).astype(np.float32), manifest)
        # Full-pipeline dequant in the ROTATED frame (before unrotation), the
        # exact quantity the Rust dequant stage must reproduce.
        deq_rot = (
            np.array(q, copy=False).astype(np.float32)
            * np.array(scale, copy=False)
            + np.array(zp, copy=False)
        ) * np.array(s_col, copy=False) * np.array(s_row, copy=False)
        dump(f"dequant_rot{bits}_f32", deq_rot.astype(np.float32), manifest)

    with open(OUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    total = sum(p.stat().st_size for p in OUT.glob("*.bin"))
    print(f"wrote {len(manifest) - 2} arrays, {total // 1024} KiB -> {OUT}")


if __name__ == "__main__":
    main()
