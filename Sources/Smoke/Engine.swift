import CoreGraphics
import CoreVideo
import ImageIO
import Darwin
import Foundation
import MLX
import MLXNERVE
import MLXServeCore
import MLXToolKit
import NERVEMLX

// nerve-smoke engine <in.png> <out.png> [--variant fidelity|clean|sharp] [--scale N] [--fp32] [--resize WxH]
//                    [--whole-frame N] [--tile N] [--overlap N] [--halo N] [--png] [--repeat N]
//   NERVEUpscalePackage through the REAL MLXServeEngine (register → license gate → C10 → run; the engine constructs
//   and loads the package, C13), raw BGRA in/out by default (no codec in the timed path; --png to include it).
//   Reports first-run and steady wall time, output luminance (a silent failure reads near-uniform), and the memory
//   report the manifest's split footprint is declared from: MLX active floor post-load, MLX peak during the run,
//   and the process phys_footprint (the basis the governor compares against in-app). One size per process.
//
// nerve-smoke cancel <in.png> [--resize WxH] [--after S] [--variant …]
//   The live CAN probe on the GPU: a deliberately tiled run, `Task.cancel()` after S seconds, asserting BOTH the
//   type (CancellationError, unwrapped, through the engine) and the latency (≈ one tile, not the whole run).

func vmInfo() -> task_vm_info_data_t? {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? info : nil
}

/// Current `phys_footprint` — the number the governor's R-MEM-1 trigger reads.
func physFootprint() -> UInt64 { vmInfo().map { UInt64($0.phys_footprint) } ?? 0 }

/// The kernel's lifetime high-water mark of `phys_footprint` for this process (one size per process = that run's peak).
func physFootprintPeak() -> UInt64 { vmInfo().map { UInt64($0.ledger_phys_footprint_peak) } ?? 0 }

func mb(_ b: Int) -> String { String(format: "%.0f MB", Double(b) / 1_048_576) }
func mb(_ b: UInt64) -> String { mb(Int(b)) }
func mb(_ b: Double) -> String { mb(Int(b)) }

func rawImage(_ cg: CGImage) throws -> Image {
    let pb = try pixelBuffer(from: cg)
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb), bpr = CVPixelBufferGetBytesPerRow(pb)
    var data = Data(count: w * h * 4)
    data.withUnsafeMutableBytes { dst in
        let src = CVPixelBufferGetBaseAddress(pb)!
        for y in 0 ..< h { memcpy(dst.baseAddress! + y * w * 4, src + y * bpr, w * 4) }
    }
    return Image.rawBGRA8(data: data, width: w, height: h)
}

func resized(_ cg: CGImage, _ spec: String?) throws -> CGImage {
    guard let spec else { return cg }
    let p = spec.split(separator: "x").compactMap { Int($0) }
    guard p.count == 2, let ctx = CGContext(data: nil, width: p[0], height: p[1], bitsPerComponent: 8, bytesPerRow: 0,
                                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else { throw SmokeError("bad --resize \(spec)") }
    ctx.interpolationQuality = .high
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: p[0], height: p[1]))
    return ctx.makeImage()!
}

func engineConfiguration() throws -> NERVEConfiguration {
    let variant = try option("--variant").map { v -> NERVEVariant in
        guard let x = NERVEVariant(rawValue: v) else { throw SmokeError("unknown variant \(v)") }
        return x
    } ?? .fidelity
    return NERVEConfiguration(variant: variant, precision: flag("--fp32") ? .fp32 : .fp16,
                              wholeFrameMaxPixels: intOption("--whole-frame"), inputTileSize: intOption("--tile"),
                              tileOverlap: intOption("--overlap"), tileHalo: intOption("--halo"))
}

func runEngine() async throws {
    let config = try engineConfiguration()
    let scale = intOption("--scale")
    let resize = option("--resize")
    let png = flag("--png")
    let repeats = intOption("--repeat") ?? 2
    guard args.count >= 3 else { throw SmokeError("engine <in.png> <out.png> [options]") }
    let cg = try resized(try loadCGImage(args[1]), resize)
    let image = png ? Image(format: .png, data: try encodePNG(cg), width: cg.width, height: cg.height) : try rawImage(cg)
    let footprintBefore = physFootprint()

    let engine = MLXServeEngine()
    await engine.useModelStore(ModelStore(root: nil))
    let id = try await engine.register(NERVEUpscalePackage.registration, configuration: config)
    let needs = await engine.needsDownload(.imageUpscale)

    // Resident floor (memory harness step 1): engine constructs + loads the package on a tiny warm-up run of the
    // same route (weights, compiled graph, Metal/MLX init), then the MLX cache is released.
    let tiny = try rawImage(try resized(cg, "24x16"))
    _ = try await engine.run(ImageUpscaleRequest(image: tiny, scale: scale), package: id)
    MLX.Memory.clearCache()
    let mlxFloor = MLX.Memory.activeMemory
    let physFloor = physFootprint()

    var times: [Double] = []
    var response: ImageUpscaleResponse?
    var peak = 0
    for i in 0 ..< max(1, repeats) {
        MLX.Memory.peakMemory = 0
        let (r, t) = try await timedAsync { try await engine.run(ImageUpscaleRequest(image: image, scale: scale), package: id) }
        times.append(t)
        guard let up = r as? ImageUpscaleResponse else { throw SmokeError("bad response") }
        response = up
        peak = max(peak, MLX.Memory.peakMemory)
        note("   run \(i + 1): \(fmtF(t, 3)) s")
    }
    let physPeak = physFootprintPeak()   // read BEFORE any of the smoke's own post-processing allocates
    guard let up = response else { throw SmokeError("no response") }
    // Output check + write.
    let outCG: CGImage
    if up.image.format == .rawBGRA8 {
        let pb = try pixelBuffer(rgb: [], width: 0, height: 0, raw: up.image)
        outCG = try cgImage(from: pb)
    } else {
        outCG = try loadCGImage(data: up.image.data)
    }
    try writePNG(outCG, args[2])
    let (rgb, w, h) = rgbFloats(try pixelBuffer(from: outCG))
    let mean = rgb.reduce(0.0) { $0 + Double($1) } / Double(max(rgb.count, 1))
    let lo = rgb.min() ?? 0, hi = rgb.max() ?? 0

    let outMpx = Double(w * h) / 1e6
    let steady = times.count > 1 ? median(Array(times.dropFirst())) : times[0]
    print("OK \(config.variant.rawValue) \(config.effectivePrecision.rawValue) ×\(up.appliedScale) \(cg.width)x\(cg.height) → \(w)x\(h) "
          + "| needsDownload \(needs) | first \(fmtF(times[0], 3)) s · steady \(fmtF(steady, 3)) s "
          + "(\(fmtF(steady * 1000 / outMpx, 1)) ms/Mpx out, \(png ? "png" : "raw BGRA") I/O) "
          + "| mean \(fmtF(mean, 3)) [\(fmtF(Double(lo), 3))…\(fmtF(Double(hi), 3))]")
    let activation = physPeak > physFloor ? physPeak - physFloor : 0
    print("MEM \(cg.width)x\(cg.height) ×\(up.appliedScale) \(config.effectivePrecision.rawValue) | process start \(mb(footprintBefore)) · "
          + "resident floor (post-load, cache cleared) phys \(mb(physFloor)) / MLX active \(mb(mlxFloor)) · "
          + "run peak phys \(mb(physPeak)) / MLX \(mb(peak)) · activation (phys peak − floor) \(mb(activation)) "
          + "→ declare ×1.2+256 MB = \(mb(Double(activation) * 1.2 + 268_435_456))")
    if hi - lo < 0.02 { note("WARN: near-uniform output — possible silent failure"); exit(3) }
}

func runCancelProbe() async throws {
    let config = try engineConfiguration()
    let resize = option("--resize") ?? "3840x2160"
    let after = option("--after").flatMap(Double.init) ?? 0.5
    guard args.count >= 2 else { throw SmokeError("cancel <in.png> [--resize WxH] [--after S]") }
    let cg = try resized(try loadCGImage(args[1]), resize)
    let image = try rawImage(cg)
    var cfg = config
    if cfg.wholeFrameMaxPixels == nil { cfg.wholeFrameMaxPixels = 0 }   // force the tiled path
    let engine = MLXServeEngine()
    await engine.useModelStore(ModelStore(root: nil))
    let id = try await engine.register(NERVEUpscalePackage.registration, configuration: cfg)
    // Uncancelled reference run (also warms: load + compile).
    let (_, full) = try await timedAsync { try await engine.run(ImageUpscaleRequest(image: image), package: id) }
    let tiles = try NERVE_Playback(checkpoint: cfg.variant.checkpoint4x, wholeFrameMaxPixels: 0,
                                   inputTileSize: cfg.inputTileSize ?? NERVE_Playback.defaultInputTileSize,
                                   tileOverlap: cfg.tileOverlap ?? NERVE_Playback.defaultTileOverlap)
        .tileCount(width: cg.width, height: cg.height)
    note("reference run: \(fmtF(full, 3)) s for \(tiles) tiles (≈ \(fmtF(full * 1000 / Double(tiles), 1)) ms/tile)")
    let t0 = Date()
    let task = Task { try await engine.run(ImageUpscaleRequest(image: image), package: id) }
    try await Task.sleep(nanoseconds: UInt64(after * 1e9))
    let tCancel = Date()
    task.cancel()
    let outcome = await task.result
    let thrownAt = Date()
    let latency = thrownAt.timeIntervalSince(tCancel)
    switch outcome {
    case .success:
        print("CANCEL ❌ run completed despite cancel at \(fmtF(after, 2)) s (total \(fmtF(thrownAt.timeIntervalSince(t0), 3)) s)")
        exit(1)
    case .failure(let e as CancellationError):
        _ = e
        print("CANCEL ✅ CancellationError (unwrapped, through MLXServeEngine) · cancel at \(fmtF(after, 2)) s · "
              + "time-to-throw \(fmtF(latency * 1000, 0)) ms (uncancelled run \(fmtF(full, 3)) s, \(tiles) tiles, "
              + "≈ \(fmtF(full * 1000 / Double(tiles), 0)) ms/tile)")
    case .failure(let e):
        print("CANCEL ❌ laundered: \(type(of: e)) \(e)")
        exit(1)
    }
}

func timedAsync<R>(_ body: () async throws -> R) async rethrows -> (R, Double) {
    let t0 = DispatchTime.now().uptimeNanoseconds
    let r = try await body()
    return (r, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
}

func loadCGImage(data: Data) throws -> CGImage {
    guard let src = CGImageSourceCreateWithData(data as CFData, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw SmokeError("undecodable output") }
    return cg
}

/// A 32BGRA pixel buffer over a raw BGRA8 `Image`'s bytes.
func pixelBuffer(rgb _: [Float], width _: Int, height _: Int, raw: Image) throws -> CVPixelBuffer {
    guard let w = raw.width, let h = raw.height else { throw SmokeError("raw image without dimensions") }
    var pb: CVPixelBuffer?
    guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA,
                              [kCVPixelBufferIOSurfacePropertiesKey as String: [:]] as CFDictionary, &pb) == kCVReturnSuccess,
          let b = pb else { throw SmokeError("pixel buffer") }
    CVPixelBufferLockBaseAddress(b, [])
    defer { CVPixelBufferUnlockBaseAddress(b, []) }
    let bpr = CVPixelBufferGetBytesPerRow(b), stride = raw.bytesPerRow ?? w * 4
    raw.data.withUnsafeBytes { src in
        for y in 0 ..< h { memcpy(CVPixelBufferGetBaseAddress(b)! + y * bpr, src.baseAddress! + y * stride, w * 4) }
    }
    return b
}
