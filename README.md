# mlx-nerve-swift

**NERVE ×2 / ×4 super-resolution for MLXEngine — the provenance-clean fast `imageUpscale` tier.** A Swift/MLX port
of [**NERVE**](https://huggingface.co/Phips/NERVE) ("Norm-free Efficient Restoration for Various Edge devices") by
**Philip Hofmann (Phips)**: a 1.8 M-parameter, pure-convolution network whose released weights are Apache-2.0 and
were trained **only** on a CC0 corpus. Numerically the port matches the author's PyTorch code and ONNX exports to
**≥ 137 dB** at 512² → 2048² on every released checkpoint. It runs at **≈ 21 ms per output megapixel** on an M5 Max
(fp16), about 1.4× Real-ESRGAN's compact net and 0.16× RealPLKSR.

| | |
|---|---|
| capability | `imageUpscale`, native **×4** and native **×2** (scale 1/3 via the ×4 result post-downsampled); `appliedScale` reports what ran |
| architecture | 5×5 stem → 24 × (conv3×3 → ReLU → conv3×3 + x) → global residual → conv3×3 (64 → 3·s²) → pixel shuffle → + bicubic(x); no bias, no norm, no attention. 1,801,920 params (×4) / 1,781,184 (×2) |
| weights | the five released checkpoints, **vendored** (7.1–7.2 MB fp32 each, MLX layout) — no download, ever |
| lanes | **fp16** conv body (default) · fp32 (parity lane; `BudgetAware` drops it to fp16 when loaded into less than its footprint). The bicubic base and the sum are fp32 on both |
| licences | weights **Apache-2.0** (Phips) · port code **Apache-2.0** (derived from `nerve_arch.py`) — C7/C8 both permissive |
| engine posture | contract 1.48: bundled-weights MAT (`needsDownload == false`), CAN (entry + per-tile checkpoints, `RunProgress(.upsample)` per tile), C14 inference mode, split footprint measured, `QuantConfigured` + `BudgetAware` |
| siblings | `mlx-realesrgan-swift` (the tier NERVE replaces as Forge's fast `.upscale`; this package reuses its tile driver) · `mlx-realplksr-swift` (the fidelity tier) |

## Variants

| `NERVEVariant` | ×4 checkpoint | ×2 request | use |
|---|---|---|---|
| **`.fidelity`** (default) | `4x_NERVE_OTF_fidelity` | native `2x_NERVE_OTF_gan` | real-world input: JPEG, noise, blur. Best full-reference arm on the damaged Forge bench cells |
| `.clean` | `4x_NERVE_release` | native `2x_NERVE_release` | clean sources (bicubic-downscale prior; passes JPEG artefacts through) |
| `.sharp` | `4x_NERVE_OTF_gan` | native `2x_NERVE_OTF_gan` | the trainer's GAN showcase — more texture |

No ×2 fidelity checkpoint exists upstream. On a ×2 bench, `.fidelity`'s ×2 requests go to the native ×2 GAN
checkpoint rather than ×4 + downsample: it won 17/27 cells on SSIMULACRA2 (+0.94 mean), ran faster (HD → 4K 1.20 s
vs 1.40 s) and never builds the 8K intermediate. The one counter-signal, recorded: its fidelity guard reads lower on
photo (it invents a little more).

```swift
let engine = MLXServeEngine()
let id = try await engine.register(NERVEUpscalePackage.registration, configuration: NERVEConfiguration())
let out = try await engine.run(ImageUpscaleRequest(image: image, scale: 2), package: id) as! ImageUpscaleResponse
```

## Products

- **`NERVEMLX`** — engine-agnostic core: `NERVE` (the network, NHWC, isomorphic to upstream) and `NERVE_Playback`
  (a `PlaybackTier` over RealESRGANMLX's shared `MLXTileProcessor`).
- **`MLXNERVE`** — the MLXEngine `ModelPackage`: `NERVEUpscalePackage` + `NERVEConfiguration`.
- **`nerve-smoke`** — the CLI gate lane (S0/S1/N2 parity, the N3/N4 studies), the real-engine drive with the memory
  report, and the live cancel probe. `Tools/nerve-bench` (a separate package, so the library never depends on its
  siblings) holds the interleaved timing harness and the ×2 bench.

## Attribution and provenance

- **Model and architecture:** NERVE by **Philip Hofmann (Phips)**, part of his BODY suite —
  [huggingface.co/Phips/NERVE](https://huggingface.co/Phips/NERVE), pinned at revision
  `c23588c36988fc2058ea3d7db80646d3ecb12295`. Code and all five checkpoints are **Apache-2.0** (the repository's
  `LICENSE`, the model card, and the README's License section). The LICENSE here is byte-identical to upstream's.
- **Training:** traiNNer-redux (Apache-2.0).
- **Training data — the only corpus named:**
  [`Phips/lucid-cc0-v2-hc-512`](https://huggingface.co/datasets/Phips/lucid-cc0-v2-hc-512) (CC0-1.0, 100,866
  lossless 512² tiles), filtered from `Phips/lucid-cc0-v2` (CC0-1.0), tiled from
  [`nyuuzyou/pxhere`](https://huggingface.co/datasets/nyuuzyou/pxhere) (CC0-1.0), a re-host of
  [pxhere.com](https://pxhere.com/en/license), whose platform-wide licence is CC0. The chain asserts nothing
  beyond what upstream asserts. The residual risk is a misattributed upload on the stock platform, not a licence
  term. Xocialize accepted platform-declared, uploader-asserted CC0 as shippable provenance for this tier on
  2026-09-28 (bridge decision **AB-D-0106**).
- **What changed from upstream:** each conv weight is re-laid out OIHW → OHWI (`oracle/convert_weights.py`; the
  source sha256 is in every file's safetensors metadata). The bicubic base is computed as an exact sub-pixel
  convolution instead of an interpolation call (same numbers, see below). Nothing is retrained or fine-tuned.

## Parity (PORTING-SPEC.md has the full record)

The oracle is **upstream `nerve_arch.py` executed verbatim** (its three imports satisfied from outside:
spandrel 0.4.2's real `store_hyperparameters`, the pinned repo's `icnr.py`, identity registries). The second oracle
is the author's own fp32 ONNX exports under ONNX Runtime. The two agree with each other at 141–149 dB.

- **S0** key contract: 50/50 tensors × 5 checkpoints, 0 missing / 0 unused.
- **S1** eight sub-op taps × 5 checkpoints × {64², 37×53} (procedural, licence-free fixtures), chained and
  isolated: worst relative error 4.0e-6 (isolated) / 3.6e-6 (chained through 50 convs); the pixel shuffle is exact.
  Probes that must fail do fail: the `(r, r, C)` shuffle order and OWHI (spatially transposed) kernels at
  ≥ 19,000× their tolerance; torch's bicubic with `alignCorners: true` at 25–29 dB. The bicubic error is uniform
  across the border band.
- **N2** end to end, 512² → 2048² on 27 bench cells × all five checkpoints: worst **138.3 dB vs torch, 138.5 dB vs
  ONNX** (GPU, compiled). The same holds on the CPU stream, and the numbers are identical with `MLX_ENABLE_TF32` on or off:
  NERVE's 64-channel convs sit outside every TF32 path, so its fp32 lane is genuinely fp32.
- **N3** fp16 body vs fp32, 27 cells × 5 checkpoints: worst 70.5 dB (`.sharp` ×4), 77.6 dB (`.fidelity`); 8-bit
  output within 2 levels on `.fidelity`.

## Performance and memory (M5 Max, 2026-09-28)

Forward only, whole-frame, interleaved with RealPLKSR as the calibration anchor, bracketed by an idle GPU:

| input → ×4 | NERVE fp16 | NERVE fp32 | Real-ESRGAN | RealPLKSR (anchor) |
|---|---:|---:|---:|---:|
| 512² | 16.8 ms/Mpx | 19.0 | 12.3 | 105.1 |
| 1024² | 21.3 | 24.1 | 15.7 | 132.9 |
| 1080×1920 | 21.0 | 24.3 | 15.0 | 129.8 |

Through the real `MLXServeEngine` (raw BGRA in/out), fp16: 256² 0.023 s · 512² 0.086 s · 1024² 0.354 s ·
1080×1920 → 4320×7680 **0.75 s** · 2160×3840 → 8640×15360 5.5 s (tiled); ×2 HD → 4K **0.59 s**. The declared split
footprint (phys_footprint basis, measured at those five sizes) is 256 MB resident + 5.9 GB activation (fp16) /
8.2 GB (fp32). The manifest comment has the table.

## Two design points worth knowing

- **Bicubic as a sub-pixel convolution.** Torch's `F.interpolate(mode="bicubic", align_corners=False)` at an
  integer scale is a fixed 5×5 phase kernel on the edge-padded input followed by a pixel shuffle, so it is added
  to the head before the one shared shuffle. `MLXNN.Upsample(.cubic)` matches torch as well (it is kept as
  `NERVE.bicubicReference` and gated beside it). But it materialises 16 output-resolution gathers — 185 ms and
  6.4 GB at 1080p → 4K, against 9 ms and 1 GB here.
- **Exact tiling.** NERVE's receptive field is 51 LR px, so plain overlapped tiles blend outputs computed from
  different context: 62.9 dB against whole-frame, with the error ×15,578 concentrated in the seam bands. Tiles
  here get a 56 px halo of real neighbouring pixels, cropped to the frame before the forward, so the model's own zero
  padding lands exactly on the image edge. Tiled output then equals whole-frame output bit-for-bit, or within one
  level on a handful of samples. Whole-frame (≤ `wholeFrameMaxPixels`, default 1920×1080) vs tiled (512 px tiles)
  is purely a memory/speed choice.

## Gates

```bash
swift build -c release --build-system swiftbuild --target NERVESmoke     # → .build/out/Products/Release/nerve-smoke
swift test --build-system swiftbuild                                      # core + package suites (CPU stream)
oracle/fetch_upstream.sh                                                  # the oracle: pinned upstream + sha256 checks
nerve-smoke keys oracle/weights/upstream                                  # S0
nerve-smoke gate s1 oracle/goldens/s1 [--gpu]                             # S1 + probes
nerve-smoke gate e2e oracle/goldens/e2e --gpu --compiled                  # N2
nerve-smoke engine in.png out.png [--variant …] [--scale 2] [--fp32]      # real engine + memory report
nerve-smoke cancel in.png --resize 2160x3840 --after 1                    # live CAN probe
```

The suites cover C0–C14 (manifest, licences, requirements, descriptor, codec seams), MAT-1..5 per variant ×
precision, CAN-1..3, the C14 INF gate on the loaded graph (and its inversion), S1/N2 twins on a committed 37×53
golden, scale routing, per-tile `RunProgress`, a deterministic mid-run cancel, and exact-halo tiled == whole-frame.

## Licence

Apache-2.0 — see `LICENSE` and `NOTICE`. The weights are Philip Hofmann's, redistributed under their Apache-2.0
terms.
