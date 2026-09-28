import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers

// MARK: - errors + logging

struct SmokeError: Error, CustomStringConvertible {
    let description: String
    init(_ d: String) { description = d }
}

/// Progress / diagnostics go to stderr (unbuffered even when stdout is redirected to a file).
func note(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }

func fmtE(_ v: Double) -> String { String(format: "%.2e", v) }
func fmtF(_ v: Double, _ d: Int = 2) -> String { String(format: "%.\(d)f", v) }

// MARK: - safetensors header (pure Foundation — no MLX, S0)

struct TensorInfo: Equatable {
    let dtype: String
    let shape: [Int]
}

/// Parse a safetensors header: 8-byte little-endian length + JSON. Returns tensors and `__metadata__`.
func safetensorsHeader(_ url: URL) throws -> (tensors: [String: TensorInfo], metadata: [String: String]) {
    let fh = try FileHandle(forReadingFrom: url)
    defer { try? fh.close() }
    guard let lenData = try fh.read(upToCount: 8), lenData.count == 8 else { throw SmokeError("\(url.path): short header") }
    let len = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
    guard len > 0, len < 100_000_000, let json = try fh.read(upToCount: Int(len)) else {
        throw SmokeError("\(url.path): bad header length \(len)")
    }
    guard let obj = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
        throw SmokeError("\(url.path): header is not a JSON object")
    }
    var tensors: [String: TensorInfo] = [:]
    var metadata: [String: String] = [:]
    for (k, v) in obj {
        if k == "__metadata__" {
            metadata = (v as? [String: String]) ?? [:]
            continue
        }
        guard let d = v as? [String: Any], let dt = d["dtype"] as? String, let sh = d["shape"] as? [Int] else {
            throw SmokeError("\(url.path): malformed entry \(k)")
        }
        tensors[k] = TensorInfo(dtype: dt, shape: sh)
    }
    return (tensors, metadata)
}

// MARK: - metrics (float64 host-side so the metric adds no rounding of its own)

struct Diff {
    let maxAbs: Double
    let refMax: Double
    let cos: Double
    let psnr: Double   // peak 1, unclamped
    var rel: Double { refMax > 0 ? maxAbs / refMax : maxAbs }
}

func hostFloats(_ a: MLXArray) -> [Float] {
    let f = a.dtype == .float32 ? a : a.asType(.float32)
    eval(f)
    return f.asArray(Float.self)
}

func diff(_ y: MLXArray, _ ref: MLXArray) -> Diff {
    precondition(y.shape == ref.shape, "shape \(y.shape) vs \(ref.shape)")
    return diff(hostFloats(y), hostFloats(ref))
}

func diff(_ a: [Float], _ b: [Float]) -> Diff {
    precondition(a.count == b.count)
    var maxAbs = 0.0, refMax = 0.0, dot = 0.0, na = 0.0, nb = 0.0, se = 0.0
    for i in 0 ..< a.count {
        let x = Double(a[i]), r = Double(b[i])
        let d = x - r
        maxAbs = max(maxAbs, abs(d))
        refMax = max(refMax, abs(r))
        dot += x * r; na += x * x; nb += r * r; se += d * d
    }
    let mse = se / Double(max(a.count, 1))
    let cos = (na > 0 && nb > 0) ? dot / (na.squareRoot() * nb.squareRoot()) : (na == nb ? 1 : 0)
    return Diff(maxAbs: maxAbs, refMax: refMax, cos: cos, psnr: mse == 0 ? .infinity : 10 * log10(1 / mse))
}

/// Max |Δ| split into a border band (outermost `band` px of an NHWC image) and the interior.
func borderSplit(_ y: MLXArray, _ ref: MLXArray, band: Int) -> (border: Double, interior: Double) {
    let (h, w, c) = (ref.dim(1), ref.dim(2), ref.dim(3))
    let a = hostFloats(y), b = hostFloats(ref)
    var border = 0.0, interior = 0.0
    for yy in 0 ..< h {
        for xx in 0 ..< w {
            let isBorder = yy < band || xx < band || yy >= h - band || xx >= w - band
            for ch in 0 ..< c {
                let i = (yy * w + xx) * c + ch
                let d = abs(Double(a[i]) - Double(b[i]))
                if isBorder { border = max(border, d) } else { interior = max(interior, d) }
            }
        }
    }
    return (border, interior)
}

// MARK: - images

func loadCGImage(_ path: String) throws -> CGImage {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw SmokeError("cannot read \(path)") }
    return cg
}

func encodePNG(_ cg: CGImage) throws -> Data {
    let data = NSMutableData()
    guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil)
    else { throw SmokeError("png destination") }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw SmokeError("png finalize") }
    return data as Data
}

func writePNG(_ cg: CGImage, _ path: String) throws {
    try encodePNG(cg).write(to: URL(fileURLWithPath: path))
}

/// A CGImage drawn into a fresh 32BGRA pixel buffer (sRGB) — the same seam the package's decoder uses.
func pixelBuffer(from cg: CGImage) throws -> CVPixelBuffer {
    let w = cg.width, h = cg.height
    var pb: CVPixelBuffer?
    let attrs: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:],
    ]
    guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
          let buffer = pb else { throw SmokeError("pixel buffer \(w)x\(h)") }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer), width: w, height: h, bitsPerComponent: 8,
                              bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                | CGBitmapInfo.byteOrder32Little.rawValue) else { throw SmokeError("cg context") }
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    return buffer
}

/// A 32BGRA pixel buffer filled from NHWC RGB floats in [0, 1] (rounded, clamped).
func pixelBuffer(rgb: [Float], width w: Int, height h: Int) throws -> CVPixelBuffer {
    var pb: CVPixelBuffer?
    let attrs: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:],
    ]
    guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
          let buffer = pb else { throw SmokeError("pixel buffer \(w)x\(h)") }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    let bpr = CVPixelBufferGetBytesPerRow(buffer)
    func q(_ v: Float) -> UInt8 { UInt8((max(0, min(1, v)) * 255).rounded()) }
    for y in 0 ..< h {
        for x in 0 ..< w {
            let s = (y * w + x) * 3, d = y * bpr + x * 4
            base[d] = q(rgb[s + 2]); base[d + 1] = q(rgb[s + 1]); base[d + 2] = q(rgb[s]); base[d + 3] = 255
        }
    }
    return buffer
}

/// 32BGRA pixel buffer → NHWC RGB float [0, 1] (the tile driver's own lift: byte / 255).
func rgbFloats(_ pb: CVPixelBuffer) -> (rgb: [Float], width: Int, height: Int) {
    let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
    let bpr = CVPixelBufferGetBytesPerRow(pb)
    var out = [Float](repeating: 0, count: w * h * 3)
    for y in 0 ..< h {
        for x in 0 ..< w {
            let s = y * bpr + x * 4, d = (y * w + x) * 3
            out[d] = Float(base[s + 2]) / 255; out[d + 1] = Float(base[s + 1]) / 255; out[d + 2] = Float(base[s]) / 255
        }
    }
    return (out, w, h)
}

func cgImage(from pb: CVPixelBuffer) throws -> CGImage {
    let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: w, height: h, bitsPerComponent: 8,
                              bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                | CGBitmapInfo.byteOrder32Little.rawValue),
          let cg = ctx.makeImage() else { throw SmokeError("cg from pixel buffer") }
    return cg
}

// MARK: - timing hygiene (shared GPU — memory `shared-machine-timing`)

/// The AGX "Device Utilization %" counter, sampled `seconds` times at 1 Hz. Returns the samples.
func gpuUtilization(seconds: Int) -> [Int] {
    var samples: [Int] = []
    for i in 0 ..< seconds {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/sbin/ioreg")
        p.arguments = ["-r", "-d", "1", "-c", "AGXAccelerator"]
        let pipe = Pipe()
        p.standardOutput = pipe
        try? p.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        let s = String(decoding: data, as: UTF8.self)
        if let r = s.range(of: "\"Device Utilization %\"=") {
            let digits = s[r.upperBound...].prefix { $0.isNumber }
            if let v = Int(digits) { samples.append(v) }
        }
        if i < seconds - 1 { Thread.sleep(forTimeInterval: 1) }
    }
    return samples
}

func median(_ v: [Double]) -> Double {
    guard !v.isEmpty else { return .nan }
    let s = v.sorted()
    return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
}

/// Wall-clock seconds of `body`.
func timed<R>(_ body: () throws -> R) rethrows -> (R, Double) {
    let t0 = DispatchTime.now().uptimeNanoseconds
    let r = try body()
    return (r, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
}
