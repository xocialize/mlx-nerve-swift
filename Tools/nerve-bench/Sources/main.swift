import CoreGraphics
import CoreImage
import CoreVideo
import Foundation
import ImageIO
import Metal
import MetalPerformanceShaders
import MLX
import MLXNERVE
import MLXNN
import MLXToolKit
import NERVEMLX
import RealESRGANMLX
import RealPLKSRMLX
import UniformTypeIdentifiers

// nerve-bench — wall time for mlx-nerve-swift's decisions (N3 fp16, N5 ×2 route, N6 cost), beside the tiers NERVE
// replaces (Real-ESRGAN `general`) and sits under (RealPLKSR — also the calibration anchor, 111 ms per output Mpx
// on an idle GPU, AB-R-0365). Shared-machine rules: every run is bracketed by the AGX utilisation counter (~10 s
// before and after), arms are interleaved round-robin with a rotating start, and a number taken without the bracket
// is not a receipt.
//
//   nerve-bench perf [--sizes 512x512,1024x1024,1080x1920] [--rounds 7] [--arms nerve32,nerve16,plk,esrgan]
//                    [--bracket 10] [--no-tf32]                     forward-only, whole-frame, compiled
//   nerve-bench prep2x <stillsDir> <dumpDir>                        the ×2 bench: 1024² centre-crop references,
//                                                                   512² lows in the bicubic/lanczos/jpeg regimes
//                                                                   (vosrgate's pixel path, side 1024 → 512)
//   nerve-bench x2 <dumpDir> [--rounds 3] [--hd <png>] [--bracket 10]   N5: the package's ×2 routes, outputs
//                                                                   written for `vosrgate score`, wall time

var args = Array(CommandLine.arguments.dropFirst())
func flag(_ n: String) -> Bool { if let i = args.firstIndex(of: n) { args.remove(at: i); return true }; return false }
func option(_ n: String) -> String? {
    guard let i = args.firstIndex(of: n), i + 1 < args.count else { return nil }
    let v = args[i + 1]; args.removeSubrange(i ... i + 1); return v
}
func note(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }
func f1(_ v: Double) -> String { String(format: "%.1f", v) }
func f2(_ v: Double) -> String { String(format: "%.2f", v) }
struct BenchError: Error, CustomStringConvertible { let description: String }

if flag("--no-tf32") { setenv("MLX_ENABLE_TF32", "0", 1) }

// MARK: - timing hygiene

func gpuUtilization(seconds: Int) -> [Int] {
    var out: [Int] = []
    for i in 0 ..< seconds {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/sbin/ioreg")
        p.arguments = ["-r", "-d", "1", "-c", "AGXAccelerator"]
        let pipe = Pipe(); p.standardOutput = pipe
        try? p.run()
        let s = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        p.waitUntilExit()
        if let r = s.range(of: "\"Device Utilization %\"=") { out.append(Int(s[r.upperBound...].prefix { $0.isNumber }) ?? -1) }
        if i < seconds - 1 { Thread.sleep(forTimeInterval: 1) }
    }
    return out
}

func bracket(_ label: String, _ seconds: Int) -> [Int] {
    guard seconds > 0 else { return [] }
    let u = gpuUtilization(seconds: seconds)
    let busy = u.filter { $0 > 5 }.count
    note("[bracket \(label)] AGX Device Utilization % over \(u.count) s: \(u) — \(busy == 0 ? "idle" : "\(busy) busy sample(s)")")
    return u
}

func median(_ v: [Double]) -> Double {
    let s = v.sorted(); guard !s.isEmpty else { return .nan }
    return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
}

func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

// MARK: - perf: forward-only arms

struct Arm {
    let name: String
    let scale: Int
    let forward: (MLXArray) -> MLXArray
}

func makeArm(_ name: String) throws -> Arm {
    func nerve(_ ck: NERVE_Playback.Checkpoint, _ dtype: DType?) throws -> Arm {
        let m = try NERVE.load(from: ck.bundledWeightsURL!, expectedScale: ck.scale, dtype: dtype)
        return Arm(name: name, scale: ck.scale, forward: compile { x in m(x) })
    }
    switch name {
    case "nerve32": return try nerve(.fidelity4x, nil)
    case "nerve16": return try nerve(.fidelity4x, .float16)
    case "nerve2x32": return try nerve(.gan2x, nil)
    case "nerve2x16": return try nerve(.gan2x, .float16)
    case "bicref":   // the bicubic base alone, MLXNN Upsample (16 output-resolution gathers)
        return Arm(name: name, scale: 4, forward: compile { x in NERVE.bicubicReference(x, scale: 4) })
    case "bicsub":   // the bicubic base alone, the sub-pixel conv the port uses
        return Arm(name: name, scale: 4, forward: compile { x in NERVE.bicubic(x, scale: 4) })
    case "plk":
        let m = RealPLKSR()
        try m.loadWeights(from: RealPLKSR_Playback.Variant.webphoto.bundledWeightsURL!)
        return Arm(name: name, scale: 4, forward: compile { x in m(x) })
    case "esrgan":
        let m = SRVGGNetCompact.general()
        try m.loadWeights(from: SRVGGNetCompact_Playback.Variant.general.bundledWeightsURL!)
        return Arm(name: name, scale: 4, forward: compile { x in m(x) })
    default: throw BenchError(description: "unknown arm \(name)")
    }
}

func perf() throws {
    let sizes = (option("--sizes") ?? "512x512,1024x1024,1080x1920").split(separator: ",").map { s -> (Int, Int) in
        let p = s.split(separator: "x").map { Int($0)! }; return (p[0], p[1])
    }
    let rounds = Int(option("--rounds") ?? "7")!
    let armNames = (option("--arms") ?? "nerve32,nerve16,plk,esrgan").split(separator: ",").map(String.init)
    let br = Int(option("--bracket") ?? "10")!
    // `--cold` frees MLX's buffer cache before every timed forward (each run then pays its own multi-GB
    // allocation); the default keeps the cache warm — the steady state of a host upscaling image after image.
    let cold = flag("--cold")
    note("MLX_ENABLE_TF32=\(ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] ?? "(unset → on)") · device \(MLX.GPU.deviceInfo().architecture)")
    let arms = try armNames.map(makeArm)
    note("cache: \(cold ? "cold (clearCache before every forward)" : "warm")")
    _ = bracket("before", br)
    var table: [String] = []
    for (w, h) in sizes {
        MLXRandom.seed(20260928)
        let x = MLXRandom.uniform(low: 0, high: 1, [1, h, w, 3])
        eval(x)
        for a in arms { for _ in 0 ..< 2 { eval(a.forward(x)) } }   // compile + warm, every arm at this size
        var times: [String: [Double]] = [:], peaks: [String: Int] = [:]
        for r in 0 ..< rounds {
            for k in 0 ..< arms.count {
                let a = arms[(k + r) % arms.count]
                if cold { MLX.Memory.clearCache() }
                MLX.Memory.peakMemory = 0
                let t0 = now()
                let y = a.forward(x)
                eval(y)
                times[a.name, default: []].append(now() - t0)
                peaks[a.name] = max(peaks[a.name] ?? 0, MLX.Memory.peakMemory)
            }
        }
        var line = "\(w)x\(h):"
        let plkPerMpx = times["plk"].map { median($0) * 1000 / (Double(w * h * 16) / 1e6) }
        let esrPerMpx = times["esrgan"].map { median($0) * 1000 / (Double(w * h * 16) / 1e6) }
        for a in arms {
            let ms = median(times[a.name]!) * 1000
            let outMpx = Double(w * h * a.scale * a.scale) / 1e6
            let perMpx = ms / outMpx
            var cell = " \(a.name) \(f1(ms)) ms (\(f1(perMpx)) ms/Mpx out, spread \(f1((times[a.name]!.max()! - times[a.name]!.min()!) * 1000)) ms, peak \(peaks[a.name]! / 1_048_576) MB"
            if let e = esrPerMpx, a.name != "esrgan" { cell += ", ×\(f2(perMpx / e)) esrgan" }
            if let p = plkPerMpx, a.name != "plk" { cell += ", ×\(f2(perMpx / p)) plk" }
            line += cell + ")"
        }
        if let p = plkPerMpx { line += " | anchor: plk \(f1(p)) ms/Mpx vs 111 (AB-R-0365)" }
        note(line)
        table.append(line)
    }
    _ = bracket("after", br)
    print(table.joined(separator: "\n"))
}

// MARK: - ×2 bench prep (vosrgate's pixel path — centreCrop / resampleCG / lanczosMPS / jpegRoundTrip — at 1024 → 512)

func loadCG(_ url: URL) -> CGImage? {
    guard let s = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(s, 0, nil)
}

func writePNG(_ image: CGImage, to url: URL) {
    guard let d = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return }
    CGImageDestinationAddImage(d, image, nil); CGImageDestinationFinalize(d)
}

func centreCrop(_ image: CGImage, side: Int) -> CGImage? {
    guard image.width >= side, image.height >= side else { return nil }
    return image.cropping(to: CGRect(x: (image.width - side) / 2, y: (image.height - side) / 2, width: side, height: side))
}

func resampleCG(_ image: CGImage, side: Int, quality: CGInterpolationQuality) -> CGImage? {
    guard let space = CGColorSpace(name: CGColorSpace.sRGB),
          let ctx = CGContext(data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: 0, space: space,
                              bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
    ctx.interpolationQuality = quality
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: side, height: side))
    return ctx.makeImage()
}

func lanczosMPS(_ image: CGImage, targetSide: Int) -> CGImage? {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { return nil }
    let w = image.width, h = image.height
    var bytes = [UInt8](repeating: 0, count: w * h * 4)
    guard let space = CGColorSpace(name: CGColorSpace.sRGB),
          let ctx = CGContext(data: &bytes, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: space,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    let sd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
    sd.usage = [.shaderRead, .shaderWrite]
    guard let src = device.makeTexture(descriptor: sd) else { return nil }
    src.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0, withBytes: bytes, bytesPerRow: w * 4)
    let dd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: targetSide, height: targetSide, mipmapped: false)
    dd.usage = [.shaderRead, .shaderWrite]
    guard let dst = device.makeTexture(descriptor: dd), let cb = queue.makeCommandBuffer() else { return nil }
    MPSImageLanczosScale(device: device).encode(commandBuffer: cb, sourceTexture: src, destinationTexture: dst)
    cb.commit(); cb.waitUntilCompleted()
    var out = [UInt8](repeating: 0, count: targetSide * targetSide * 4)
    dst.getBytes(&out, bytesPerRow: targetSide * 4, from: MTLRegionMake2D(0, 0, targetSide, targetSide), mipmapLevel: 0)
    guard let provider = CGDataProvider(data: Data(out) as CFData) else { return nil }
    return CGImage(width: targetSide, height: targetSide, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: targetSide * 4,
                   space: space, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                   provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
}

func jpegRoundTrip(_ image: CGImage, quality: Double) -> CGImage? {
    let data = NSMutableData()
    guard let d = CGImageDestinationCreateWithData(data, "public.jpeg" as CFString, 1, nil) else { return nil }
    CGImageDestinationAddImage(d, image, [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
    guard CGImageDestinationFinalize(d), let s = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(s, 0, nil)
}

func downsample(_ image: CGImage, to side: Int, regime: String) -> CGImage? {
    switch regime {
    case "lanczos": return lanczosMPS(image, targetSide: side)
    case "jpeg": return resampleCG(image, side: side, quality: .high).flatMap { jpegRoundTrip($0, quality: 0.3) }
    default: return resampleCG(image, side: side, quality: .high)
    }
}

func prep2x(stills: URL, dump: URL) throws {
    try FileManager.default.createDirectory(at: dump, withIntermediateDirectories: true)
    let urls = try FileManager.default.contentsOfDirectory(at: stills, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension.lowercased() == "png" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    for url in urls {
        let name = url.deletingPathExtension().lastPathComponent
        guard let master = loadCG(url), let ref = centreCrop(master, side: 1024) else { note("  \(name): SKIP"); continue }
        writePNG(ref, to: dump.appendingPathComponent("\(name)__reference.png"))
        for regime in ["bicubic", "lanczos", "jpeg"] {
            guard let low = downsample(ref, to: 512, regime: regime) else { note("  \(name)/\(regime): SKIP"); continue }
            writePNG(low, to: dump.appendingPathComponent("\(name)__\(regime)__low.png"))
            if let a = lanczosMPS(low, targetSide: 1024) { writePNG(a, to: dump.appendingPathComponent("\(name)__\(regime)__lanczos-mps.png")) }
            if let b = resampleCG(low, side: 1024, quality: .high) { writePNG(b, to: dump.appendingPathComponent("\(name)__\(regime)__bicubic-cg.png")) }
        }
        note("  \(name): \(master.width)x\(master.height) → 1024² reference + 3 lows")
    }
}

// MARK: - x2: the package's ×2 routes

func pngImage(_ cg: CGImage) throws -> Image {
    let data = NSMutableData()
    guard let d = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else { throw BenchError(description: "png") }
    CGImageDestinationAddImage(d, cg, nil); CGImageDestinationFinalize(d)
    return Image(format: .png, data: data as Data, width: cg.width, height: cg.height)
}

func rawImage(_ cg: CGImage) throws -> Image {
    let w = cg.width, h = cg.height
    var bytes = Data(count: w * h * 4)
    try bytes.withUnsafeMutableBytes { raw in
        guard let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        else { throw BenchError(description: "ctx") }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    }
    return Image.rawBGRA8(data: bytes, width: w, height: h)
}

@InferenceActor
func x2(dump: URL, rounds: Int, hd: URL?, br: Int) async throws {
    // Arms: every NERVE ×2 route the package can take + the incumbent fast tier's ×2 behaviour for context.
    let arms: [(String, NERVEVariant)] = [("NERVE-gan2x", .sharp), ("NERVE-fid4x-down", .fidelity), ("NERVE-rel2x", .clean)]
    var pkgs: [(String, NERVEUpscalePackage)] = []
    for (n, v) in arms {
        let p = NERVEUpscalePackage(configuration: NERVEConfiguration(variant: v))   // the shipping (fp16) lane
        try await p.load()
        pkgs.append((n, p))
    }
    // Context: the incumbent fast tier's ×2 behaviour (Real-ESRGAN `general` ×4 + the same CG .high downsample).
    let esrgan = try SRVGGNetCompact_Playback(variant: .general)
    let lows = try FileManager.default.contentsOfDirectory(at: dump, includingPropertiesForKeys: nil)
        .filter { $0.lastPathComponent.hasSuffix("__low.png") }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    // 1. Outputs for the FR scorer (one pass per arm; routes resolved by the package itself).
    for low in lows {
        let cg = loadCG(low)!
        let prefix = low.lastPathComponent.replacingOccurrences(of: "__low.png", with: "")
        for (n, p) in pkgs {
            let r = try await p.run(ImageUpscaleRequest(image: try pngImage(cg), scale: 2)) as! ImageUpscaleResponse
            precondition(r.appliedScale == 2 && r.image.width == 2 * cg.width, "\(n): appliedScale \(r.appliedScale)")
            try r.image.data.write(to: dump.appendingPathComponent("\(prefix)__\(n).png"))
        }
        let e4 = try await esrgan.upscale(try pixelBuffer(cg))
        if let down = resampleCG(try cgImage(e4), side: 2 * cg.width, quality: .high) {
            writePNG(down, to: dump.appendingPathComponent("\(prefix)__RealESRGAN-4x-down.png"))
        }
    }
    note("x2: \(lows.count) lows × \(pkgs.count) arms written to \(dump.path)")
    // 2. Wall time, interleaved, raw BGRA in/out (no PNG codec in the loop): the 512² lows, then HD → 4K.
    var inputs: [(String, Image)] = []
    if let first = lows.first, let cg = loadCG(first) { inputs.append(("512x512", try rawImage(cg))) }
    if let hd, let cg = loadCG(hd) { inputs.append(("\(cg.width)x\(cg.height)", try rawImage(cg))) }
    _ = bracket("before", br)
    for (label, img) in inputs {
        for (_, p) in pkgs { _ = try await p.run(ImageUpscaleRequest(image: img, scale: 2)) }   // warm
        var times: [String: [Double]] = [:]
        for r in 0 ..< rounds {
            for k in 0 ..< pkgs.count {
                let (n, p) = pkgs[(k + r) % pkgs.count]
                MLX.Memory.clearCache()
                let t0 = now()
                _ = try await p.run(ImageUpscaleRequest(image: img, scale: 2))
                times[n, default: []].append(now() - t0)
            }
        }
        note("x2 wall \(label) → ×2 (package run, rawBGRA8): " + pkgs.map { n, _ in
            "\(n) \(f1(median(times[n]!) * 1000)) ms" }.joined(separator: " · "))
    }
    _ = bracket("after", br)
}

func pixelBuffer(_ cg: CGImage) throws -> CVPixelBuffer {
    var pb: CVPixelBuffer?
    let attrs: [String: Any] = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                                kCVPixelBufferIOSurfacePropertiesKey as String: [:]]
    guard CVPixelBufferCreate(nil, cg.width, cg.height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
          let b = pb else { throw BenchError(description: "pb") }
    CVPixelBufferLockBaseAddress(b, []); defer { CVPixelBufferUnlockBaseAddress(b, []) }
    guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(b), width: cg.width, height: cg.height, bitsPerComponent: 8,
                              bytesPerRow: CVPixelBufferGetBytesPerRow(b), space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
    else { throw BenchError(description: "ctx") }
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: cg.width, height: cg.height))
    return b
}

func cgImage(_ pb: CVPixelBuffer) throws -> CGImage {
    CVPixelBufferLockBaseAddress(pb, .readOnly); defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: CVPixelBufferGetWidth(pb),
                              height: CVPixelBufferGetHeight(pb), bitsPerComponent: 8,
                              bytesPerRow: CVPixelBufferGetBytesPerRow(pb), space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue),
          let cg = ctx.makeImage() else { throw BenchError(description: "cg") }
    return cg
}

// MARK: - main

do {
    switch args.first ?? "" {
    case "perf": try perf()
    case "prep2x":
        guard args.count >= 3 else { throw BenchError(description: "prep2x <stillsDir> <dumpDir>") }
        try prep2x(stills: URL(fileURLWithPath: args[1]), dump: URL(fileURLWithPath: args[2]))
    case "x2":
        let rounds = Int(option("--rounds") ?? "3")!
        let hd = option("--hd").map { URL(fileURLWithPath: $0) }
        let br = Int(option("--bracket") ?? "10")!
        guard args.count >= 2 else { throw BenchError(description: "x2 <dumpDir>") }
        try await x2(dump: URL(fileURLWithPath: args[1]), rounds: rounds, hd: hd, br: br)
    default:
        note("usage: nerve-bench perf|prep2x|x2 … (see Tools/nerve-bench/Sources/main.swift)")
        exit(2)
    }
    exit(0)
} catch {
    note("FAILED: \(error)")
    exit(1)
}
