#!/usr/bin/env python
"""convert_weights.py — the five NERVE checkpoints (Phips/NERVE @ c23588c36988, Apache-2.0) in MLX layout.

    oracle/.venv/bin/python oracle/convert_weights.py [--out Sources/NERVEMLX/Resources] [--log oracle/reports/convert.log]

writes `<out>/<checkpoint>-mlx.safetensors` for each of the five pinned checkpoints.

Key contract (S0): the keys ARE the upstream state-dict keys — `stem.weight`, `blocks.{0..23}.conv{1,2}.weight`,
`upsampler.0.weight` — 50 tensors, every conv `bias=False`. The only change is the layout of each 4-D conv weight,
PyTorch (O, I, kH, kW) → MLX (O, kH, kW, I); dtype stays fp32 (N3 decides whether a half lane ships, and a half
lane casts these at load). The source sha256 is verified against the pinned Hub LFS oid before anything is
written, and recorded — with the output's — in the file's safetensors metadata and in the conversion log.
Every array is materialised through numpy (no lazy tensors).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nerve_oracle as oracle  # noqa: E402  (pins + sha256 only; no torch needed here)


def expected_keys(n_blocks: int = 24) -> list[str]:
    keys = ["stem.weight", "upsampler.0.weight"]
    for i in range(n_blocks):
        keys += [f"blocks.{i}.conv1.weight", f"blocks.{i}.conv2.weight"]
    return sorted(keys)


def main() -> None:
    root = os.path.dirname(HERE)
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(root, "Sources", "NERVEMLX", "Resources"))
    p.add_argument("--log", default=os.path.join(HERE, "reports", "convert.log"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.dirname(a.log), exist_ok=True)

    lines = [f"# convert_weights.py — {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}",
             f"# source {oracle.REPO} @ {oracle.REVISION} (models/*.safetensors)", ""]
    for name, pinned in oracle.CHECKPOINTS.items():
        src = oracle.checkpoint_path(name)
        got = oracle.sha256(src)
        if got != pinned:
            sys.exit(f"{name}: source sha256 {got} != pinned {pinned}")
        with safe_open(src, framework="numpy") as fh:
            src_meta = fh.metadata() or {}
        sd = load_file(src)
        if sorted(sd) != expected_keys():
            sys.exit(f"{name}: key contract broken: {sorted(set(sd) ^ set(expected_keys()))}")
        out, n_conv = {}, 0
        for k, v in sd.items():
            assert v.dtype == np.float32 and v.ndim == 4, (k, v.dtype, v.shape)
            out[k] = np.ascontiguousarray(v.transpose(0, 2, 3, 1))  # (O,I,kH,kW) -> (O,kH,kW,I)
            n_conv += 1
        assert n_conv == 50, n_conv
        scale = int(round((sd["upsampler.0.weight"].shape[0] // 3) ** 0.5))
        meta = {
            "format": "mlx",
            "layout": "OHWI",
            "model": "NERVE",
            "scale": str(scale),
            "source_repo": oracle.REPO,
            "source_revision": oracle.REVISION,
            "source_file": f"models/{name}.safetensors",
            "source_sha256": pinned,
            "license": "Apache-2.0",
            "author": "Philip Hofmann (Phips)",
        }
        dst = os.path.join(a.out, f"{name}-mlx.safetensors")
        save_file(out, dst, metadata=meta)
        n_params = sum(int(np.prod(v.shape)) for v in out.values())
        line = (f"{name}: x{scale} · {len(out)} tensors · {n_params:,} params · src sha256 {pinned} "
                f"(src metadata {json.dumps(src_meta, sort_keys=True)}) → {os.path.relpath(dst, root)} "
                f"sha256 {oracle.sha256(dst)} ({os.path.getsize(dst):,} B)")
        print(line)
        lines.append(line)
    with open(a.log, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"log → {os.path.relpath(a.log, root)}")


if __name__ == "__main__":
    main()
