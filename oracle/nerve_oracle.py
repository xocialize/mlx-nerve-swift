"""nerve_oracle.py — THE oracle: upstream `nerve_arch.py` (Phips/NERVE @ c23588c36988, Apache-2.0) executed
verbatim, with only its three imports satisfied from outside:

  spandrel.util.store_hyperparameters   the REAL function — spandrel==0.4.2 installed with
                                        `uv pip install --target oracle/pydeps --no-deps spandrel==0.4.2`;
                                        `spandrel/__init__.py` eagerly imports every architecture (einops,
                                        torchvision…), so the package object is registered bare and only
                                        `spandrel.util` (stdlib-only) is executed
  traiNNer.utils.icnr.icnr_reinit       the pinned repo's own `icnr.py` (init-only; overwritten by the strict load)
  traiNNer.utils.registry               ARCH_REGISTRY / SPANDREL_REGISTRY whose `register()` is the identity

Nothing in the model maths is re-implemented here. `load(name)` builds `nerve_arch.nerve(scale=s)` with the scale
derived from the head weight (`upsampler.0.weight` = 3·s² output channels) and loads the EMA state dict
`strict=True`. `Tools/nerve-eval/scripts/nerve_ref.py` (a 60-line re-implementation used on the bench) is a
cross-check, never the oracle.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(HERE, "upstream")
PYDEPS = os.path.join(HERE, "pydeps")
WEIGHTS = os.path.join(HERE, "weights", "upstream")
ONNX_DIR = os.path.join(HERE, "onnx")

REPO = "Phips/NERVE"
REVISION = "c23588c36988fc2058ea3d7db80646d3ecb12295"
# The five checkpoints of the pinned revision and their Hub LFS sha256 (verified on download, 2026-09-28).
CHECKPOINTS = {
    "4x_NERVE_OTF_fidelity": "e05f589bde0ba0bb18b55e881b0160b52986df71ce472a6fa28c1e81179adae7",
    "4x_NERVE_release": "07bee958c168b561ec8d31ffe319ba6aa8ad3b67726633994e3d65687708fb98",
    "2x_NERVE_release": "241dc6a6cb2e3ef8bd66493c36bb6a900bf4ca97491d0d1462d42ecb4d989693",
    "4x_NERVE_OTF_gan": "b1c0c041293ec0d327ee3b59253cd27177190c403760fe1f0e8f88084e6a2faf",
    "2x_NERVE_OTF_gan": "478459e1428997ba79aa1ac4b5825f34494e8d76387d220594b0e191bef7023c",
}
# The author's dynamic fp32 opset-20 ONNX export of each checkpoint (the second, independent oracle).
ONNX = {
    "4x_NERVE_OTF_fidelity": "4x_NERVE_OTF_fidelity_1x3xHxW_fp32_op20.onnx",
    "4x_NERVE_release": "4x_NERVE_1x3xHxW_fp32_op20.onnx",
    "2x_NERVE_release": "2x_NERVE_1x3xHxW_fp32_op20.onnx",
    "4x_NERVE_OTF_gan": "4x_NERVE_OTF_gan_1x3xHxW_fp32_op20.onnx",
    "2x_NERVE_OTF_gan": "2x_NERVE_OTF_gan_1x3xHxW_fp32_op20.onnx",
}
ONNX_SHA256 = {
    "4x_NERVE_OTF_fidelity": "510308a2f49a1abf1ef1f9a57fd76e6c946f993231f3dc2b3d67d6c95be4c4db",
    "4x_NERVE_release": "eeecdc430db756f36096b110ee275d860feab6b5c036a7c94e55c82d2d01b616",
    "2x_NERVE_release": "a499bc54cb9d442a94f63fab201de61c72b32bf88bab6dffffbd5abbb43cb5f7",
    "4x_NERVE_OTF_gan": "cb2513d504cdec4694d82c8a226609f75b684963d5f44b1a482987fa987fcdb2",
    "2x_NERVE_OTF_gan": "85a6f5d8afc1b8cd1d2513ab6f87dafc083c921e9d5a7fb948ccaa98f3c7cb38",
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _install_import_shims() -> None:
    if "traiNNer.utils.registry" in sys.modules:
        return
    # spandrel: register the package bare so `spandrel.util` resolves from the --target install without
    # executing spandrel/__init__.py (which imports all ~40 architectures and their third-party deps).
    spandrel = types.ModuleType("spandrel")
    spandrel.__path__ = [os.path.join(PYDEPS, "spandrel")]
    sys.modules["spandrel"] = spandrel

    # traiNNer: package skeleton + the pinned repo's icnr.py as traiNNer.utils.icnr.
    for name in ("traiNNer", "traiNNer.utils"):
        mod = types.ModuleType(name)
        mod.__path__ = []
        sys.modules[name] = mod
    spec = importlib.util.spec_from_file_location("traiNNer.utils.icnr", os.path.join(UPSTREAM, "icnr.py"))
    icnr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(icnr)
    sys.modules["traiNNer.utils.icnr"] = icnr

    class _Registry:
        def register(self, *args, **kwargs):  # identity decorator
            return lambda obj: obj

    registry = types.ModuleType("traiNNer.utils.registry")
    registry.ARCH_REGISTRY = _Registry()
    registry.SPANDREL_REGISTRY = _Registry()
    sys.modules["traiNNer.utils.registry"] = registry


def nerve_arch():
    """The upstream module, executed verbatim."""
    _install_import_shims()
    if "nerve_arch" not in sys.modules:
        spec = importlib.util.spec_from_file_location("nerve_arch", os.path.join(UPSTREAM, "nerve_arch.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["nerve_arch"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["nerve_arch"]


def checkpoint_path(name: str) -> str:
    return os.path.join(WEIGHTS, f"{name}.safetensors")


def onnx_path(name: str) -> str:
    return os.path.join(ONNX_DIR, ONNX[name])


def load(name: str, verify_sha: bool = True):
    """Build upstream `nerve(scale=s)` for checkpoint `name` and load it strictly. Returns (model, scale, sd)."""
    import torch
    from safetensors.torch import load_file

    path = checkpoint_path(name)
    if verify_sha:
        got = sha256(path)
        if got != CHECKPOINTS[name]:
            raise RuntimeError(f"{name}: sha256 {got} != pinned {CHECKPOINTS[name]}")
    sd = load_file(path)
    # Derive the scale from the head weight, never from the file name (plan §2.3: the 2× and 4× checkpoints
    # differ ONLY in upsampler.0.weight's output channels, 12 vs 48).
    scale = int(math.isqrt(sd["upsampler.0.weight"].shape[0] // 3))
    arch = nerve_arch()
    model = arch.nerve(scale=scale)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()
    torch.set_grad_enabled(False)
    return model, scale, sd


def onnx_session(name: str, verify_sha: bool = True):
    """ONNX Runtime CPU session over the author's export (fp32, no graph optimisations that change maths)."""
    import onnxruntime as ort

    path = onnx_path(name)
    if verify_sha:
        got = sha256(path)
        if got != ONNX_SHA256[name]:
            raise RuntimeError(f"{name} onnx: sha256 {got} != pinned {ONNX_SHA256[name]}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 8
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


if __name__ == "__main__":
    for name in CHECKPOINTS:
        m, s, sd = load(name)
        n = sum(p.numel() for p in m.parameters())
        print(f"{name}: x{s}  {len(sd)} tensors  {n:,} params  hyperparameters={m.hyperparameters}  "
              f"dtypes={sorted({str(v.dtype) for v in sd.values()})}")
