# mlx-nerve-swift — porting spec (NERVE → Swift/MLX)

Reference: **Phips/NERVE** (Philip Hofmann) at HF revision **`c23588c36988fc2058ea3d7db80646d3ecb12295`** —
`nerve_arch.py` (Apache-2.0, 86 lines) and the five released checkpoints (Apache-2.0, trained only on the CC0 corpus
`Phips/lucid-cc0-v2-hc-512` ← `nyuuzyou/pxhere` ← pxhere.com; provenance approved as [[AB-D-0106]]). Plan and gate
vocabulary: `mlxengine-forge/Docs/NERVE-HEART-PORT-PLAN.md` §1–§2 (phases N0–N7), evaluation and the reasons for the
port: `mlxengine-forge/Docs/NERVE-EVAL.md`. Bridge task **AB-T-0186** (Steps 2–3). Path B, PyTorch → Swift directly
with per-sub-op goldens from the PyTorch oracle; no Python-MLX rung (the whole net is conv / ReLU / add / pixel
shuffle / bicubic, every op with a fleet donor).

## What is ported (isomorphic to upstream)

| Swift (`NERVEMLX`) | upstream `nerve_arch.py` | notes |
|---|---|---|
| `NerveBlock` | `NerveBlock` | `x + conv2(relu(conv1(x)))`, 3×3, 64 ch, `bias=False` |
| `NERVE` | `NERVE` | 5×5 stem → 24 blocks → global residual → `upsampler.0` (3×3, 64 → 3·s²) → pixel shuffle → + bicubic |
| `pixelShuffleNHWC` | `nn.PixelShuffle` | channel `c·s·s + i·s + j` → (c, i, j) — torch's order, NHWC |
| `NERVE.bicubic` / `BicubicPhaseKernel` | `F.interpolate(…, "bicubic", align_corners=False)` | Keys a = −0.75, half-pixel, clamped borders, no antialias — as the exact sub-pixel conv (edge-pad 2 → 5×5 phase kernel → shuffle), summed into the head before its shuffle; `MLXNN.Upsample(.cubic)` kept as `bicubicReference` (see Findings) |
| — | `icnr_reinit` | initialisation only; nothing to port |
| `NERVE_Playback` | — | `PlaybackTier` over RealESRGANMLX's shared `MLXTileProcessor` (whole-frame ≤ ceiling, else tiles, optional halo) |

Weights: `oracle/convert_weights.py` writes each checkpoint as `Sources/NERVEMLX/Resources/<name>-mlx.safetensors` —
the SAME 50 keys, each conv weight OIHW → OHWI, fp32, with the source's sha256 (verified against the Hub LFS oid
first) in the safetensors metadata and in `oracle/reports/convert.log`. The loader derives scale, width and depth
from the file (the ×2 and ×4 checkpoints differ only in `upsampler.0.weight`'s 12 vs 48 output channels) and refuses
a partial set, a disagreeing scale, or an OIHW file.

## The oracle

`oracle/nerve_oracle.py` executes **upstream `nerve_arch.py` verbatim**; only its three imports are satisfied from
outside: `spandrel.util.store_hyperparameters` is the real function (spandrel 0.4.2 via
`uv pip install --target oracle/pydeps --no-deps`; the package object is registered bare so its eager all-architectures
`__init__` never runs), `traiNNer.utils.icnr` is the pinned repo's own `icnr.py`, and the registry decorators are the
identity. Second, independent oracle: the author's five fp32 opset-20 ONNX exports under ONNX Runtime 1.30 (CPU).
`oracle/fetch_upstream.sh` re-fetches all of it and checks every sha256; `oracle/dump_goldens.py` writes the fixtures
(CPU fp32, NHWC, C-contiguous). `Tools/nerve-eval/scripts/nerve_ref.py` (the bench's re-implementation) was not used.

Torch ↔ ONNX agree at **141.5–149.4 dB** on every fixture and bench cell (`oracle/reports/dump-e2e.log`), so either
oracle reads the same answer.

## Phase gates (stamp AFTER the run — a row is `pending` until a tool result says otherwise)

| phase | gate | status |
|---|---|---|
| N0 / S0 key contract | `nerve-smoke keys oracle/weights/upstream`: generated keys == headers, all five checkpoints, 0 missing / 0 unused, F32, OHWI, same keys as upstream with the OIHW transposition of every shape | **PASSED 2026-09-28** — 50/50 tensors ×5, 1,801,920 (×4) / 1,781,184 (×2) params, module tree == contract at ×2 and ×4 (`oracle/reports/s0-keys.log`) |
| N1 / S1 sub-ops | `nerve-smoke gate s1 oracle/goldens/s1` (CPU; `--gpu` too): 8 taps × 5 checkpoints × {64², 37×53} on PROCEDURAL licence-free inputs (gradients, hard edges, thin lines, texture; 8-bit levels), each CHAINED (the model's forward) and ISOLATED (the sub-op on the oracle's own input); rel tolerances in `Sources/Smoke/Gates.swift`; + probes | **PASSED 2026-09-28** — worst isolated: stem 3.5e-7 · block0 5.6e-7 · 24-block chain 4.0e-6 · head 1.2e-6 · shuffle **0** · bicubic 3.2e-7 (sub-pixel form; the MLXNN Upsample reference ≤ 4.0e-7) · output 0; worst chained: head 3.6e-6; vs ONNX 135.5–141.4 dB. GPU lane (TF32 off AND on — identical): chained head 3.3e-6 (`s1-cpu.log`, `s1-gpu-notf32.log`, `s1-gpu-tf32.log`). First pass (on bench crops, before the fixtures were made licence-free): worst chained head 9.6e-6 — the evidence for the chained-tail tolerance |
| N1 probes | (r,r,C) shuffle must fail; OWHI kernels (the shape-identical spatial transpose) must fail; bicubic error uniform across the 2·s border band; `alignCorners: true` must fail | **PASSED 2026-09-28** — (r,r,C) 42,890–80,429× its tolerance; OWHI block0 19,607–29,073× the chained-tail tolerance; border/interior max ×0.80–1.00; `alignCorners: true` 25.2–28.7 dB |
| N2 e2e | `nerve-smoke gate e2e oracle/goldens/e2e --gpu`: Swift vs torch ≥ 90 dB AND vs ONNX ≥ 90 dB at 512², all 27 bench cells × all five checkpoints | **PASSED 2026-09-28** — first forward (MLXNN cubic Upsample): worst 137.4 dB vs torch, 137.3 dB vs ONNX; final forward (sub-pixel bicubic, below): worst **138.3 dB vs torch, 138.5 dB vs ONNX** (4x_NERVE_OTF_gan), per checkpoint 138.3–141.8 dB, compiled (`n2-e2e-gpu-notf32.log`). TF32 on == off (`n2-e2e-gpu-tf32-compiled.log`); CPU lane (fidelity) 140.7 dB vs torch, 140.9 dB vs ONNX (`n2-e2e-cpu-fidelity.log`) |
| N3 dtype | fp16 vs fp32 end to end on the 27 cells + wall time; ship fp16 only if ≥ ~50 dB and measurably faster | **PASSED 2026-09-28 → fp16 is the default lane** — fp16 conv body (bicubic base + sum fp32) vs fp32, all 5 checkpoints × 27 cells: worst **70.5 dB** (gan ×4), 77.6 dB (fidelity ×4); 8-bit output ≤ 2 levels (fidelity), ≤ 7 (gan ×4), ≤ 3.7 % of samples changed (`n3-fidelity.log`). bf16 58.1–65.7 dB (rejected); fp16 incl. the base 68.8–71.9 dB (why the base stays fp32). Speed, idle bracket: fp16 1.13–1.16× fp32 (16.8 vs 19.0 / 21.3 vs 24.1 / 21.0 vs 24.3 ms per output Mpx at 512² / 1024² / 1080×1920), peak MLX 0.89 vs 1.66 / 1.38 vs 2.44 / 2.69 vs 4.78 GB (`perf-receipt.log`). fp32 stays the parity lane, `BudgetAware` → fp16 under pressure |
| N4 tiling | tiled vs whole-frame (≥ 45 dB, no seam concentration) → `wholeFrameMaxPixels`, overlap, halo | **PASSED 2026-09-28** — plain tiles (the template's 256/32): worst 62.9 dB (27 lows) / 65.3 dB (10 HD stills), max 10–11 levels, error ×4,450–72,000 concentrated in the seam bands — ≥ 45 dB but seam-concentrated. Frame-cropped HALO tiles (H56): 27 lows T256/O0 **bit-identical**, O16 104.8 dB; HD T512/O0 **123.1 dB** (max 1 level, only where the clamped last tile overlaps), O16 111.3 dB (`n4-tile-512.log`, `n4-tile-hd.log`). Defaults: tiles **512 / overlap 0 / halo 56** (halo overhead 1.49× per interior tile, 256: 2.07×), `wholeFrameMaxPixels` 1920×1080 (whole vs tiled is now a pure memory/speed choice) |
| N5 ×2 route | ×2 bench (1024² references, 512² lows, 3 regimes, the 9 ×4-bench stills): native `2x_NERVE_OTF_gan` vs `.fidelity` ×4 + downsample, FR + wall time | **DECIDED 2026-09-28 → native `2x_NERVE_OTF_gan`** — SSIMULACRA2: gan2x wins 17/27 cells, mean +0.94 (bicubic +2.41, jpeg +1.60 — 7/9, lanczos −1.19); package wall time 512² 79 vs 110 ms, HD → 4K **1.20 vs 1.40 s**, and no 8K intermediate. Recorded counter-signal: DC-worst (fidelity guard) photo 0.843 vs 0.897 on jpeg — the GAN checkpoint invents more. Both NERVE routes beat Real-ESRGAN ×4 + down by 10–25 FR everywhere; on clean photo `2x_NERVE_release` (`.clean`) leads (54.0) (`n5-x2-score.csv`, `n5-x2-tables.md`, `n5-x2-wall.log`) |
| N6 package | `NERVEUpscalePackage` C0–C14 + MAT-1..5 (bundled, `needsDownload == false`) + CAN-1..3; split footprint at five sizes; real-engine drive with the timing bracket | **PASSED 2026-09-28** — suites: core 15 XCTest + package 13 XCTest + 16 Swift Testing, all green (MAT per variant × precision, `needsDownload == false` via the real engine, CAN-1..3, INF gate on the loaded graph + inversion, live CPU-lane routes / per-tile `RunProgress` / deterministic mid-run cancel / exact-halo tiled == whole-frame). Real engine (`n6-engine.log`), fp16 ×4: 256² 0.023 s · 512² 0.086 s · 1024² 0.354 s · 1080×1920 **0.750 s (22.6 ms/Mpx)** · 2160×3840 tiled 5.47 s; ×2 HD → 4K 0.586 s; activation (phys peak − floor) 0.44 / 1.04 / 2.12 / 4.17 / 4.70 GB, floor 158–221 MB → declared fp16 256 MB + 5.9 GB, fp32 256 MB + 8.2 GB. Live CAN on the GPU through `MLXServeEngine`: CancellationError unwrapped, time-to-throw 69–135 ms ≈ one tile (138 ms) of a 5.5 s run (`can-live-gpu.log`) |
| N7 publish | `xocialize/mlx-nerve-swift` public, v0.1.0, README model card, topics, registry row | pending |

## Findings

- **A flat fixture passes everything.** The first 37×53 S1 crop (a flat region of a bench graphic) was a flat colour
  field: all eight taps passed and the `alignCorners: true` probe read **128 dB** — bicubic of a constant is the
  constant under any sampling grid. The probe caught the fixture, not the port. The next crop was chosen by gradient
  energy (probe 26.7–30.2 dB); the committed fixtures are now PROCEDURAL (seeded gradients, hard edges, thin lines,
  texture on 8-bit levels) — licence-free, reproducible by anyone, and harder (|body| up to 17 vs ≤ 4): probe
  25.2–28.7 dB. The bench cells stay internal (N2–N5 name them photo-a…f / graphic-a…d).
- **Relative error on a small residual.** On that flat crop the head's output was a ±0.007–0.03 residual, so its
  chained rel error read 1.0–1.4e-5 while the absolute error was 2.8e-7 — rounding inherited from the 64-channel body.
  Isolating each sub-op on the oracle's own input separates the two: the head conv alone is 1.2e-6. The chained tail
  (block23 → shuffle) is gated at 2e-5, held against the (r,r,C) and OWHI probes (≥ 34,000× that tolerance).
- **TF32 is irrelevant to NERVE on M5.** Its 64-channel 3×3 convs sit outside the Winograd window (C + O = 128 < 256)
  and the S1/N2 numbers are bit-for-bit the same with `MLX_ENABLE_TF32` on or off — so, unlike VOSR2 ([[AB-L-0175]]),
  the fp32 lane is genuinely fp32 in a host that never sets the flag.

- **MLXNN's cubic `Upsample` is the memory peak.** It matches torch, but it materialises 16 output-resolution
  gathers and their weighted sum: at 1080×1920 → 4320×7680, 185 ms and 6.4 GB for the bicubic alone, most of the
  forward's 7.2 GB peak (`perf-subpixel.log`, taken under ~20 % foreign GPU load — the RATIO is the evidence). At an integer scale torch's bicubic IS a sub-pixel convolution
  (edge-pad 2 → a fixed 5×5 phase kernel → pixel shuffle), so the port adds the phases to the head before the one
  shared shuffle: 9 ms and 1 GB for the base; fp32 peak 7.2 → 4.7 GB, fp16 6.8 → 2.7 GB; parity improved (bicubic
  rel 3.4e-7 vs the Upsample's 4.5e-7; N2 137.4 → 138.3 dB). The Upsample stays as `NERVE.bicubicReference`, gated.
- **Receptive field 51 px ⇒ halo tiles, cropped to the frame.** The driver's halo replicates past the image
  edge, which is not upstream's per-layer zero padding; running only the in-frame part of each window and
  re-embedding the output makes a tile's kept output equal the whole-frame output (N4). Plain overlap cannot:
  the error sits exactly in the blend bands.
- **Shared-machine timing.** Two perf passes read every arm ~2× slow, uniformly (RealPLKSR anchor 203–302 ms/Mpx vs
  the receipt's 111): the AGX counter was idle in the brackets, but another session was running a 7-thread ONNX
  Runtime probe (20 GB) and a CPU-stream gate — unified-memory bandwidth, invisible to the GPU counter. With the
  machine genuinely quiet the anchor read 105.1 ms/Mpx (spread 1.1 ms) and the receipts above were taken; the
  ratios (NERVE ≈ 0.16–0.22× RealPLKSR, 1.4–1.8× Real-ESRGAN) held in every pass, contended or not.

## Tooling

`nerve-smoke` (Sources/Smoke; `swift build -c release --build-system swiftbuild --target NERVESmoke`, binary at
`.build/out/Products/Release/nerve-smoke`, AB-L-0171):

```
nerve-smoke keys [<upstreamModelsDir>]                                    # S0
nerve-smoke gate s1  <goldensDir> [--gpu] [--only <ckpt>]                 # S1 + probes
nerve-smoke gate e2e <goldensDir> [--gpu] [--only <ckpt>] [--fp16] [--compiled] [--min-db N]   # N2
nerve-smoke study n3 <e2eGoldensDir>                                      # N3 fidelity (fp16 / bf16 / fp16-all)
nerve-smoke study tile <dir> [--only <substring>] [--configs T/O/H,…]      # N4 tiled vs whole-frame, seam bands
nerve-smoke run <in.png> <out.png> [--ckpt …] [--fp16] [--whole-frame N] [--tile N] [--overlap N] [--halo N]
nerve-smoke engine <in.png> <out.png> [--variant …] [--scale N] [--fp32] [--resize WxH] [--png] [--repeat N]   # N6
nerve-smoke cancel <in.png> [--resize WxH] [--after S]                    # live CAN probe (GPU, real engine)
```

`Tools/nerve-bench` (separate package; `swift build -c release --build-system swiftbuild --target NERVEBench`):
`perf` (interleaved forward-only arms incl. RealPLKSR / Real-ESRGAN, AGX bracket), `prep2x` (the ×2 bench — the
vosrgate pixel path at 1024 → 512), `x2` (the package's ×2 routes: outputs for `vosrgate score` + wall time).
`oracle/n5_tables.py` prints the N5 tables from the score CSV.

Gate modes set `MLX_ENABLE_TF32=0` before the first MLX op unless `--tf32`. `swift test --build-system swiftbuild`
runs the core suite (XCTest, CPU stream: structure, C14, loader refusals, S1 + N2 twins on the committed 37×53
golden, both probes).
