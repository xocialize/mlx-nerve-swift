#!/usr/bin/env python
"""dump_goldens.py — parity fixtures from THE oracle (upstream nerve_arch.py, `nerve_oracle.py`) and the second
oracle (the author's ONNX exports under ONNX Runtime), CPU fp32 on both. Every array is saved NHWC float32 (the
Swift port's layout), C-contiguous.

    oracle/.venv/bin/python oracle/dump_goldens.py s1                # S1 taps, 5 checkpoints × {64², 37×53}
    oracle/.venv/bin/python oracle/dump_goldens.py e2e <ckpt|all>    # N2: the 27 bench cells, torch + ONNX

s1  → oracle/goldens/s1/<ckpt>_<H>x<W>.safetensors — `input`, the eight taps upstream's forward computes
      (stem · block0 · block23 · body [= stem + blocks(stem), the global residual] · head [conv before the shuffle] ·
      shuffle · bicubic [F.interpolate's own result, recorded, not re-derived] · output) and `onnx` (ORT's output).
      Also the committed compact test golden Tests/NERVEMLXTests/Resources/goldens_s1_37x53.safetensors: the shared
      37×53 input, all eight taps for 4x_NERVE_OTF_fidelity and 2x_NERVE_OTF_gan (both shuffle factors), and the
      torch + ONNX outputs of all five checkpoints.
e2e → oracle/goldens/e2e/<ckpt>/<cell>.safetensors — `input` (the 512² bench low, PIL-decoded /255), `torch`,
      `onnx`; metadata carries the torch↔ONNX PSNR (dB, peak 1, unclamped).

Inputs: S1 uses two PROCEDURAL, licence-free images (seeded; 8-bit levels like a decoded PNG): smooth colour
gradients, hard-edged blocks, thin lines and fine texture — so every tap and the bicubic border see edges, and the
committed golden carries no third-party pixels. 64² and 37×53 (odd, non-square: the bicubic border). e2e uses a
directory of `*__low.png` bench inputs (`--bench <dir>` or $NERVE_BENCH_DIR); its goldens stay local.
"""
from __future__ import annotations

import glob
import math
import os
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.numpy import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import nerve_oracle as oracle  # noqa: E402

BENCH = os.environ.get("NERVE_BENCH_DIR", "")
S1_DIR = os.path.join(HERE, "goldens", "s1")
E2E_DIR = os.path.join(HERE, "goldens", "e2e")
TEST_GOLDEN = os.path.join(ROOT, "Tests", "NERVEMLXTests", "Resources", "goldens_s1_37x53.safetensors")

torch.set_num_threads(os.cpu_count() or 8)


def nhwc(t: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(t.detach().permute(0, 2, 3, 1).contiguous().float().numpy())


def load_png(path: str) -> np.ndarray:
    """[H, W, 3] float32 in [0, 1] — PIL decode, the same path the bench's torch arms used."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def to_nchw(hwc: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(hwc.transpose(2, 0, 1)))[None]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return math.inf if mse == 0 else 10.0 * math.log10(1.0 / mse)


class _FProxy:
    """Stands in for `nerve_arch`'s module-global `F` during one forward: forwards every attribute to the real
    torch.nn.functional, and records the tensor `F.interpolate` returns (the bicubic base upstream adds)."""

    def __init__(self, real):
        self._real = real
        self.last_interpolate = None

    def __getattr__(self, name):
        return getattr(self._real, name)

    def interpolate(self, *args, **kwargs):
        y = self._real.interpolate(*args, **kwargs)
        self.last_interpolate = y
        return y


def taps(model, x: torch.Tensor) -> dict[str, np.ndarray]:
    arch = oracle.nerve_arch()
    rec: dict[str, torch.Tensor] = {}
    hooks = [
        model.stem.register_forward_hook(lambda m, i, o: rec.__setitem__("stem", o)),
        model.blocks[0].register_forward_hook(lambda m, i, o: rec.__setitem__("block0", o)),
        model.blocks[len(model.blocks) - 1].register_forward_hook(lambda m, i, o: rec.__setitem__("block23", o)),
        model.upsampler[0].register_forward_pre_hook(lambda m, i: rec.__setitem__("body", i[0])),
        model.upsampler[0].register_forward_hook(lambda m, i, o: rec.__setitem__("head", o)),
        model.upsampler[1].register_forward_hook(lambda m, i, o: rec.__setitem__("shuffle", o)),
    ]
    proxy = _FProxy(arch.F)
    arch.F = proxy
    try:
        with torch.inference_mode():
            y = model(x)
    finally:
        arch.F = proxy._real
        for h in hooks:
            h.remove()
    rec["bicubic"] = proxy.last_interpolate
    rec["output"] = y
    # The hooks and the recorded interpolate are upstream's own intermediates; sanity-check they compose.
    assert torch.equal(rec["shuffle"] + rec["bicubic"], y)
    return {k: nhwc(v) for k, v in rec.items()}


def onnx_run(sess, x_nchw: np.ndarray) -> np.ndarray:
    name = sess.get_inputs()[0].name
    y = sess.run(None, {name: np.ascontiguousarray(x_nchw, dtype=np.float32)})[0]
    return np.ascontiguousarray(y.transpose(0, 2, 3, 1).astype(np.float32))


def procedural(h: int, w: int, seed: int) -> np.ndarray:
    """A licence-free natural-ish test image: gradients + hard edges + thin lines + texture, on 8-bit levels.
    (A fixture must exercise edges: the first 37×53 fixture — a crop of a flat colour field — passed every tap
    and read 128 dB under the alignCorners:true probe, since bicubic of a constant is the constant under any
    sampling grid. The probe caught the fixture, not the port.)"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.stack([0.5 + 0.4 * np.sin(xx / w * 2.3 * np.pi + 0.7),
                    0.5 + 0.4 * np.cos(yy / h * 1.7 * np.pi + 0.3),
                    (xx + yy) / (w + h)], axis=-1)
    for _ in range(7):
        y0, x0 = int(rng.integers(0, h - 3)), int(rng.integers(0, w - 3))
        y1, x1 = y0 + int(rng.integers(3, max(4, h // 2))), x0 + int(rng.integers(3, max(4, w // 2)))
        img[y0:y1, x0:x1] = rng.uniform(0, 1, 3)
    for x in rng.integers(0, w, 2):
        img[:, int(x)] = 1.0
    for y in rng.integers(0, h, 2):
        img[int(y), :] = 0.0
    img += rng.normal(0, 0.03, img.shape)
    return (np.round(np.clip(img, 0, 1) * 255) / 255).astype(np.float32)


def s1_inputs() -> dict[str, np.ndarray]:
    return {"64x64": procedural(64, 64, 20260928), "37x53": procedural(37, 53, 20260929)}


def dump_s1() -> None:
    os.makedirs(S1_DIR, exist_ok=True)
    inputs = s1_inputs()
    for key, img in inputs.items():
        print(f"input {key}: shape {img.shape}  mean {img.mean():.4f}  std {img.std():.4f}  "
              f"[{img.min():.3f}, {img.max():.3f}]")
    compact: dict[str, np.ndarray] = {"input": np.ascontiguousarray(inputs["37x53"][None])}
    for name in oracle.CHECKPOINTS:
        model, scale, _ = oracle.load(name)
        sess = oracle.onnx_session(name)
        for key, img in inputs.items():
            x = to_nchw(img)
            t = taps(model, x)
            ort = onnx_run(sess, x.numpy())
            arrays = {"input": np.ascontiguousarray(img[None]), **t, "onnx": ort}
            path = os.path.join(S1_DIR, f"{name}_{key}.safetensors")
            save_file(arrays, path, metadata={"checkpoint": name, "scale": str(scale), "size": key,
                                              "torch_vs_onnx_psnr": f"{psnr(t['output'], ort):.2f}"})
            print(f"  {name} {key}: x{scale} out {t['output'].shape}  torch↔ONNX {psnr(t['output'], ort):.1f} dB  "
                  f"max|Δ| {np.abs(t['output'] - ort).max():.2e}  |stem|max {np.abs(t['stem']).max():.2f}  "
                  f"|body|max {np.abs(t['body']).max():.2f}  |head|max {np.abs(t['head']).max():.3f}")
            if key == "37x53":
                compact[f"{name}.output"] = t["output"]
                compact[f"{name}.onnx"] = ort
                if name in ("4x_NERVE_OTF_fidelity", "2x_NERVE_OTF_gan"):
                    for tap, v in t.items():
                        if tap != "output":
                            compact[f"{name}.{tap}"] = v
    os.makedirs(os.path.dirname(TEST_GOLDEN), exist_ok=True)
    save_file(compact, TEST_GOLDEN, metadata={"source_revision": oracle.REVISION,
                                             "producer": "oracle/dump_goldens.py s1 (upstream nerve_arch.py, CPU fp32)"})
    print(f"compact test golden → {os.path.relpath(TEST_GOLDEN, ROOT)} ({os.path.getsize(TEST_GOLDEN):,} B, "
          f"{len(compact)} arrays)")


def bench_cells() -> list[str]:
    if not BENCH or not os.path.isdir(BENCH):
        sys.exit("e2e needs the bench inputs: --bench <dir of *__low.png> or NERVE_BENCH_DIR")
    return sorted(glob.glob(os.path.join(BENCH, "*__low.png")))


def dump_e2e(name: str) -> None:
    model, scale, _ = oracle.load(name)
    sess = oracle.onnx_session(name)
    out_dir = os.path.join(E2E_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    worst = math.inf
    for path in bench_cells():
        cell = os.path.basename(path)[: -len("__low.png")]
        img = load_png(path)
        x = to_nchw(img)
        with torch.inference_mode():
            y = nhwc(model(x))
        ort = onnx_run(sess, x.numpy())
        p = psnr(y, ort)
        worst = min(worst, p)
        save_file({"input": np.ascontiguousarray(img[None]), "torch": y, "onnx": ort},
                  os.path.join(out_dir, f"{cell}.safetensors"),
                  metadata={"checkpoint": name, "scale": str(scale), "cell": cell, "torch_vs_onnx_psnr": f"{p:.2f}"})
        print(f"  {name} {cell}: {img.shape[1]}x{img.shape[0]} → {y.shape[2]}x{y.shape[1]}  torch↔ONNX {p:.1f} dB")
    print(f"{name}: {len(bench_cells())} cells, worst torch↔ONNX {worst:.1f} dB → {os.path.relpath(out_dir, ROOT)}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("s1", "e2e"):
        sys.exit(__doc__)
    if "--bench" in sys.argv:
        i = sys.argv.index("--bench")
        BENCH = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    if sys.argv[1] == "s1":
        dump_s1()
    else:
        which = sys.argv[2] if len(sys.argv) > 2 else "all"
        for n in (oracle.CHECKPOINTS if which == "all" else [which]):
            dump_e2e(n)
