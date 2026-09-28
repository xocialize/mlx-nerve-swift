#!/bin/bash
# fetch_upstream.sh — rebuild everything the oracle needs from pinned upstream sources. Nothing third-party is
# vendored in this repo except the converted weights; this script re-fetches the rest byte-for-byte and checks
# every sha256 against the Hub LFS oid recorded on 2026-09-28.
#
#   Phips/NERVE  c23588c36988fc2058ea3d7db80646d3ecb12295   (Apache-2.0: nerve_arch.py, icnr.py, the five
#                                                          checkpoints, the author's five fp32 opset-20 ONNX exports)
#   spandrel     0.4.2 (MIT) — `--target oracle/pydeps --no-deps`; only spandrel.util.store_hyperparameters runs
#
# Then: oracle/.venv/bin/python oracle/nerve_oracle.py        (strict-loads all five through upstream code)
#       oracle/.venv/bin/python oracle/convert_weights.py     (→ Sources/NERVEMLX/Resources/*-mlx.safetensors)
#       oracle/.venv/bin/python oracle/dump_goldens.py s1|e2e (→ oracle/goldens/, Tests/…/goldens_s1_37x53)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REV=c23588c36988fc2058ea3d7db80646d3ecb12295
BASE="https://huggingface.co/Phips/NERVE/resolve/$REV"
mkdir -p "$HERE/upstream/configs" "$HERE/upstream/docs" "$HERE/upstream/scripts" "$HERE/weights/upstream" "$HERE/onnx"

for f in nerve_arch.py icnr.py LICENSE README.md docs/ABLATIONS.md scripts/export_dynamic.py \
         configs/4x_NERVE_OTF_fidelity.yml configs/4x_NERVE_release.yml configs/2x_NERVE_release.yml \
         configs/4x_NERVE_OTF_gan.yml configs/4x_NERVE_onnx.yml; do
  curl -fsSL "$BASE/$f" -o "$HERE/upstream/$f"
done

check() {  # <file> <sha256>
  local got; got=$(shasum -a 256 "$1" | cut -d' ' -f1)
  [ "$got" = "$2" ] || { echo "sha256 MISMATCH $1: $got != $2" >&2; exit 1; }
  echo "ok  $2  $(basename "$1")"
}

while read -r name sha; do
  curl -fsSL "$BASE/models/$name.safetensors" -o "$HERE/weights/upstream/$name.safetensors"
  check "$HERE/weights/upstream/$name.safetensors" "$sha"
done <<'EOF'
4x_NERVE_OTF_fidelity e05f589bde0ba0bb18b55e881b0160b52986df71ce472a6fa28c1e81179adae7
4x_NERVE_release 07bee958c168b561ec8d31ffe319ba6aa8ad3b67726633994e3d65687708fb98
2x_NERVE_release 241dc6a6cb2e3ef8bd66493c36bb6a900bf4ca97491d0d1462d42ecb4d989693
4x_NERVE_OTF_gan b1c0c041293ec0d327ee3b59253cd27177190c403760fe1f0e8f88084e6a2faf
2x_NERVE_OTF_gan 478459e1428997ba79aa1ac4b5825f34494e8d76387d220594b0e191bef7023c
EOF

while read -r file sha; do
  curl -fsSL "$BASE/onnx/$file" -o "$HERE/onnx/$file"
  check "$HERE/onnx/$file" "$sha"
done <<'EOF'
4x_NERVE_OTF_fidelity_1x3xHxW_fp32_op20.onnx 510308a2f49a1abf1ef1f9a57fd76e6c946f993231f3dc2b3d67d6c95be4c4db
4x_NERVE_1x3xHxW_fp32_op20.onnx eeecdc430db756f36096b110ee275d860feab6b5c036a7c94e55c82d2d01b616
2x_NERVE_1x3xHxW_fp32_op20.onnx a499bc54cb9d442a94f63fab201de61c72b32bf88bab6dffffbd5abbb43cb5f7
4x_NERVE_OTF_gan_1x3xHxW_fp32_op20.onnx cb2513d504cdec4694d82c8a226609f75b684963d5f44b1a482987fa987fcdb2
2x_NERVE_OTF_gan_1x3xHxW_fp32_op20.onnx 85a6f5d8afc1b8cd1d2513ab6f87dafc083c921e9d5a7fb948ccaa98f3c7cb38
EOF

if [ ! -x "$HERE/.venv/bin/python" ]; then
  uv venv --python 3.12 "$HERE/.venv"
  uv pip install --python "$HERE/.venv/bin/python" torch numpy safetensors onnxruntime onnx pillow einops
fi
uv pip install --python "$HERE/.venv/bin/python" --target "$HERE/pydeps" --no-deps spandrel==0.4.2
echo "oracle ready: $HERE"
