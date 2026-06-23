#!/usr/bin/env python3
"""Hypothesis A verification: where in the network do positions collapse?

Reads the binary dumps produced by `MLXCEL_M3_DEBUG=1` and computes cosine
similarity / L2 distance between the three sampled positions (pos0 = start,
pos1 = middle, pos2 = end) at each captured layer.

A healthy transformer should show:
  - Embedding: positions completely different (cos_sim ~ 0)
  - Each layer: positions remain meaningfully distinct (cos_sim < 0.99)
A broken-collapse transformer shows:
  - cos_sim shooting to ~1.0 at some specific layer = the converger

Usage: uv run --with numpy --with ml_dtypes python3 scripts/verify_collapse.py
"""
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

PROBE_DIR = Path("/tmp")
HIDDEN = 6144


def load(label: str) -> np.ndarray:
    """Load a /tmp/m3_probe_{label}.bin file as float32."""
    path = PROBE_DIR / f"m3_probe_{label}.bin"
    if not path.exists():
        return None
    raw = path.read_bytes()
    if len(raw) == HIDDEN * 2:
        # BF16: HIDDEN elements
        u16 = np.frombuffer(raw, dtype=np.uint16)
        return u16.view(bfloat16).astype(np.float32)
    elif len(raw) == HIDDEN * 4:
        # F32
        return np.frombuffer(raw, dtype=np.float32)
    else:
        raise ValueError(f"unexpected size {len(raw)} for {path}")


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def main():
    captures = [
        ("embed",    "after embedding lookup"),
        ("l0_input", "  layer 0 input (= embed)"),
        ("l0_after_input_norm",     "  layer 0 after input_norm"),
        ("l0_after_attn",           "  layer 0 after self_attn"),
        ("l0_after_attn_residual",  "  layer 0 after attn residual"),
        ("l0_after_post_norm",      "  layer 0 after post_attn_norm"),
        ("l0_after_ff",             "  layer 0 after MLP"),
        ("l0_output",               "  layer 0 output"),
        ("layer0",   "after layer 0 (dense MLP) [whole]"),
        ("layer2",   "after layer 2 (last dense layer)"),
        ("layer3",   "after layer 3 (first MoE layer)"),
        ("layer30",  "after layer 30 (mid MoE)"),
        ("layer59",  "after layer 59 (final MoE)"),
        ("final_norm", "after final RMSNorm"),
    ]

    print(f"{'Layer':<26} {'cos(p0,p1)':>11} {'cos(p0,p2)':>11} {'cos(p1,p2)':>11} {'L2(p0,p2)':>11}")
    print("-" * 75)
    for name, desc in captures:
        p0 = load(f"{name}_pos0")
        p1 = load(f"{name}_pos1")
        p2 = load(f"{name}_pos2")
        if p0 is None or p1 is None or p2 is None:
            print(f"  [missing] {name}")
            continue
        s01 = cos_sim(p0, p1)
        s02 = cos_sim(p0, p2)
        s12 = cos_sim(p1, p2)
        l2  = float(np.linalg.norm(p0 - p2))
        print(f"{desc:<26} {s01:>11.4f} {s02:>11.4f} {s12:>11.4f} {l2:>11.2f}")

    print()
    print("Reading: cos_sim near 1.0 means positions are nearly identical (collapse).")
    print("Healthy:   embed << 0.5,   each layer well below 0.99")
    print("Broken:    cos_sim jumps to >0.99 at the layer that introduces collapse")


if __name__ == "__main__":
    main()
