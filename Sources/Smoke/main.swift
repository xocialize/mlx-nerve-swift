import CoreVideo
import Foundation
import MLX
import NERVEMLX

// nerve-smoke — the CLI lane of mlx-nerve-swift (gates that need Metal, studies, and the real-engine drive).
//
//   nerve-smoke keys [<upstreamModelsDir>]                         S0 key contract (header-only, no MLX eval)
//   nerve-smoke gate s1  <goldensDir> [--gpu] [--only <ckpt>]      S1 per-sub-op parity + probes (CPU default)
//   nerve-smoke gate e2e <goldensDir> [--gpu] [--only <ckpt>] [--fp16] [--min-db N] [--compiled]
//                                                                 N2 vs torch + vs the author's ONNX
//   nerve-smoke run <in.png> <out.png> [--ckpt fidelity4x|release4x|release2x|gan4x|gan2x] [--fp16]
//                   [--whole-frame N] [--tile N] [--overlap N] [--halo N]
//   nerve-smoke study n3|tile|x2|perf …                           (Studies.swift)
//   nerve-smoke engine <in.png> <out.png> [--variant fidelity|clean|sharp] [--scale N] [--fp16] …
//   nerve-smoke cancel <in.png> [--after S]                       live mid-run cancel probe (CAN, GPU)
//
// Gate modes pin `MLX_ENABLE_TF32=0` before the first MLX op (fp32 GPU lanes are TF32-class on M5 otherwise,
// AB-L-0175); pass `--tf32` to measure the default. Other modes leave the host default unless `--no-tf32`.

var args = Array(CommandLine.arguments.dropFirst())

func flag(_ name: String) -> Bool {
    if let i = args.firstIndex(of: name) { args.remove(at: i); return true }
    return false
}

func option(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    let v = args[i + 1]
    args.removeSubrange(i ... i + 1)
    return v
}

func intOption(_ name: String) -> Int? { option(name).flatMap(Int.init) }

let mode = args.first ?? ""
let isGate = mode == "gate" || mode == "keys"
let tf32On = flag("--tf32")
let tf32Off = flag("--no-tf32")
if (isGate && !tf32On) || tf32Off {
    setenv("MLX_ENABLE_TF32", "0", 1)
}
note("MLX_ENABLE_TF32=\(ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] ?? "(unset → mlx default, on)")")

func checkpoint(_ s: String?) throws -> NERVE_Playback.Checkpoint {
    guard let s else { return .fidelity4x }
    guard let c = NERVE_Playback.Checkpoint(rawValue: s) else { throw SmokeError("unknown checkpoint \(s)") }
    return c
}

func runOne() throws {
    let fp16 = flag("--fp16")
    let ck = try checkpoint(option("--ckpt"))
    let whole = intOption("--whole-frame") ?? NERVE_Playback.defaultWholeFrameMaxPixels
    let tile = intOption("--tile") ?? NERVE_Playback.defaultInputTileSize
    let overlap = intOption("--overlap") ?? NERVE_Playback.defaultTileOverlap
    let halo = intOption("--halo") ?? NERVE_Playback.defaultTileHalo
    guard args.count >= 3 else { throw SmokeError("run <in.png> <out.png> [options]") }
    let cg = try loadCGImage(args[1])
    let tier = try NERVE_Playback(checkpoint: ck, precision: fp16 ? .fp16 : .fp32, wholeFrameMaxPixels: whole,
                                  inputTileSize: tile, tileOverlap: overlap, tileHalo: halo)
    let pb = try pixelBuffer(from: cg)
    let (warm, tWarm) = try timed { try tier.upscale(pb, progress: nil) }
    _ = warm
    let (out, t) = try timed { try tier.upscale(pb, progress: nil) }
    try writePNG(try cgImage(from: out), args[2])
    let (rgb, w, h) = rgbFloats(out)
    let mean = rgb.reduce(0.0) { $0 + Double($1) } / Double(rgb.count)
    let lo = rgb.min() ?? 0, hi = rgb.max() ?? 0
    let mpx = Double(w * h) / 1e6
    print("OK \(tier.name) \(cg.width)x\(cg.height) → \(w)x\(h) | first \(fmtF(tWarm, 3)) s · steady \(fmtF(t, 3)) s "
          + "(\(fmtF(t * 1000 / mpx, 1)) ms/Mpx out) | \(tier.runsWholeFrame(width: cg.width, height: cg.height) ? "whole-frame" : "tiled \(tier.tileCount(width: cg.width, height: cg.height)) tiles") "
          + "| mean \(fmtF(mean, 3)) [\(fmtF(Double(lo), 3))…\(fmtF(Double(hi), 3))] | peak MLX \(MLX.GPU.snapshot().peakMemory / 1_048_576) MB")
    if hi - lo < 0.02 { note("WARN: near-uniform output — possible silent failure"); exit(3) }
}

do {
    switch mode {
    case "keys":
        let ok = try gateS0(upstreamDir: args.count > 1 ? args[1] : nil)
        note(ok ? "✅ S0 PASSED" : "❌ S0 FAILED")
        exit(ok ? 0 : 1)
    case "gate":
        let gpu = flag("--gpu")
        let only = option("--only")
        let fp16 = flag("--fp16")
        let compiled = flag("--compiled")
        let minDB = option("--min-db").flatMap(Double.init) ?? 90
        guard args.count >= 3 else { throw SmokeError("gate s1|e2e <goldensDir> …") }
        let ok: Bool
        switch args[1] {
        case "s1": ok = try gateS1(goldensDir: args[2], gpu: gpu, only: only)
        case "e2e": ok = try gateE2E(goldensDir: args[2], gpu: gpu, only: only, precision: fp16 ? .fp16 : .fp32,
                                     threshold: minDB, compiled: compiled)
        default: throw SmokeError("unknown gate \(args[1])")
        }
        exit(ok ? 0 : 1)
    case "run":
        try runOne()
        exit(0)
    case "study":
        try runStudy()
        exit(0)
    case "engine":
        try await runEngine()
        exit(0)
    case "cancel":
        try await runCancelProbe()
        exit(0)
    default:
        note("usage: nerve-smoke keys|gate|run|study|engine|cancel … (see Sources/Smoke/main.swift)")
        exit(2)
    }
} catch {
    note("FAILED: \(error)")
    exit(1)
}
