import CoreGraphics
import CoreVideo
import Foundation
import MLX
import NERVEMLX

// nerve-smoke study n3   <e2eGoldensDir> [--only <ckpt>]           N3 fidelity: fp16 / bf16 vs fp32 and vs torch
// nerve-smoke study tile <dir-of-pngs> [--only <prefix>] [--limit N] [--ckpt …] [--configs T/O/H,…]
//                                                                  N4: tiled vs whole-frame, 8-bit, seam bands
// (Wall time lives in Tools/nerve-bench — interleaved arms, the RealPLKSR anchor and the GPU-idle bracket.)

func runStudy() throws {
    guard args.count >= 3 else { throw SmokeError("study n3|tile <dir> …") }
    let only = option("--only")
    switch args[1] {
    case "n3": try studyN3(goldensDir: args[2], only: only)
    case "tile":
        try studyTile(dir: args[2], only: only, limit: intOption("--limit"), checkpoint: try checkpoint(option("--ckpt")),
                      configs: option("--configs"))
    default: throw SmokeError("unknown study \(args[1])")
    }
}

// MARK: - 8-bit helpers

/// The product's quantisation: clamp to [0, 1], ×255, round.
func quantize8(_ y: MLXArray) -> [UInt8] {
    let q = MLX.round(MLX.clip(y.asType(.float32), min: 0, max: 1) * 255).asType(.uint8)
    eval(q)
    return q.asArray(UInt8.self)
}

struct Diff8 {
    let psnr: Double      // peak 255
    let maxLevels: Int
    let fracDiff: Double  // share of samples that differ at all
}

func diff8(_ a: [UInt8], _ b: [UInt8]) -> Diff8 {
    var se = 0.0, mx = 0, n = 0
    for i in 0 ..< a.count {
        let d = Int(a[i]) - Int(b[i])
        if d != 0 { n += 1; se += Double(d * d); mx = max(mx, abs(d)) }
    }
    let mse = se / Double(a.count)
    return Diff8(psnr: mse == 0 ? .infinity : 10 * log10(255 * 255 / mse), maxLevels: mx,
                 fracDiff: Double(n) / Double(a.count))
}

// MARK: - N3 — fp16 (and bf16, informational) against fp32

func studyN3(goldensDir: String, only: String?) throws {
    let root = URL(fileURLWithPath: goldensDir)
    let dirs = try FileManager.default.contentsOfDirectory(atPath: goldensDir)
        .filter { d in NERVE_Playback.Checkpoint.allCases.contains { $0.upstreamName == d } && (only.map { d.hasPrefix($0) } ?? true) }
        .sorted()
    for dir in dirs {
        let ck = NERVE_Playback.Checkpoint.allCases.first { $0.upstreamName == dir }!
        let url = ck.bundledWeightsURL!
        let m32 = try NERVE.load(from: url, expectedScale: ck.scale)
        let m16 = try NERVE.load(from: url, expectedScale: ck.scale, dtype: .float16)
        let mbf = try NERVE.load(from: url, expectedScale: ck.scale, dtype: .bfloat16)
        let f32 = compile { x in m32(x) }, f16 = compile { x in m16(x) }, fbf = compile { x in mbf(x) }
        // The whole graph at half precision, bicubic base included — the design the fp16 lane does NOT use.
        let f16all = compile { x in m16(x.asType(.float16)).asType(.float32) }
        var rows: [String: [Double]] = [:]
        var worst8: [String: Diff8] = [:]
        let cells = try FileManager.default.contentsOfDirectory(atPath: root.appendingPathComponent(dir).path)
            .filter { $0.hasSuffix(".safetensors") }.sorted()
        for c in cells {
            let g = try MLX.loadArrays(url: root.appendingPathComponent(dir).appendingPathComponent(c))
            let x = g["input"]!, torch = g["torch"]!
            let y32 = f32(x), y16 = f16(x), ybf = fbf(x), y16a = f16all(x)
            eval(y32, y16, ybf, y16a)
            let q32 = quantize8(y32), qt = quantize8(torch)
            for (name, y) in [("fp16", y16), ("bf16", ybf), ("fp16-all", y16a), ("fp32", y32)] {
                rows["\(name) vs fp32", default: []].append(diff(y, y32).psnr)
                rows["\(name) vs torch", default: []].append(diff(y, torch).psnr)
                let d8 = diff8(quantize8(y), name == "fp32" ? qt : q32)
                rows["\(name) 8-bit vs \(name == "fp32" ? "torch" : "fp32")", default: []].append(d8.psnr)
                let key = name
                if let w = worst8[key] {
                    worst8[key] = Diff8(psnr: min(w.psnr, d8.psnr), maxLevels: max(w.maxLevels, d8.maxLevels),
                                        fracDiff: max(w.fracDiff, d8.fracDiff))
                } else { worst8[key] = d8 }
            }
            note("   \(dir)/\(c.dropLast(12)): fp16 \(fmtF(diff(y16, y32).psnr, 1)) dB · bf16 \(fmtF(diff(ybf, y32).psnr, 1)) dB · "
                 + "fp16-all \(fmtF(diff(y16a, y32).psnr, 1)) dB (vs fp32)")
        }
        note("N3 \(dir) (\(cells.count) cells, GPU, float PSNR peak 1 — worst / median):")
        for k in rows.keys.sorted() {
            let v = rows[k]!
            note("   \(k): worst \(fmtF(v.min() ?? .nan, 1)) dB · median \(fmtF(median(v), 1)) dB")
        }
        for k in worst8.keys.sorted() {
            let w = worst8[k]!
            note("   \(k) 8-bit output: worst max |Δ| \(w.maxLevels) level(s), worst share of samples changed "
                 + "\(fmtF(100 * w.fracDiff, 3))%")
        }
    }
}

// MARK: - N4 — tiled against whole-frame

struct TileConfig: CustomStringConvertible {
    let tile: Int, overlap: Int, halo: Int
    var description: String { "T\(tile)/O\(overlap)/H\(halo)" }
}

/// Output pixels (per axis) that more than one tile covers — the feathered seam bands.
func seamMask(extent: Int, tile: Int, overlap: Int, scale: Int) -> [Bool] {
    let step = max(tile - overlap, 1)
    var origins: [Int] = []
    for o in stride(from: 0, to: extent, by: step) {
        let c = min(o, max(0, extent - tile))
        if origins.last != c { origins.append(c) }
    }
    var cover = [Int](repeating: 0, count: extent)
    for o in origins { for i in o ..< min(o + tile, extent) { cover[i] += 1 } }
    var out = [Bool](repeating: false, count: extent * scale)
    for i in 0 ..< extent where cover[i] > 1 { for j in 0 ..< scale { out[i * scale + j] = true } }
    return out
}

func studyTile(dir: String, only: String?, limit: Int?, checkpoint ck: NERVE_Playback.Checkpoint, configs: String?) throws {
    let grid: [TileConfig] = (configs ?? "256/32/0,256/64/0,512/32/0,128/32/0,256/16/56,256/0/56,512/16/56,256/16/64,384/16/56")
        .split(separator: ",").map { c in
            let p = c.split(separator: "/").map { Int($0)! }
            return TileConfig(tile: p[0], overlap: p[1], halo: p[2])
        }
    var files = try FileManager.default.contentsOfDirectory(atPath: dir)
        .filter { f in f.hasSuffix(".png") && (only.map { f.contains($0) } ?? true) }.sorted()
    if let limit { files = Array(files.prefix(limit)) }
    let whole = try NERVE_Playback(checkpoint: ck, wholeFrameMaxPixels: Int.max)
    let tiers = try grid.map { try NERVE_Playback(checkpoint: ck, wholeFrameMaxPixels: 0, inputTileSize: $0.tile,
                                                  tileOverlap: $0.overlap, tileHalo: $0.halo) }
    var worst: [String: (psnr: Double, maxL: Int, frac: Double, band: Double)] = [:]
    for f in files {
        let cg = try loadCGImage((dir as NSString).appendingPathComponent(f))
        let pb = try pixelBuffer(from: cg)
        let (w, h) = (cg.width, cg.height)
        let ref = rgbFloats(try whole.upscale(pb, progress: nil)).rgb.map { UInt8(($0 * 255).rounded()) }
        var cells: [String] = []
        for (cfg, tier) in zip(grid, tiers) {
            let out = rgbFloats(try tier.upscale(pb, progress: nil)).rgb.map { UInt8(($0 * 255).rounded()) }
            let d = diff8(out, ref)
            // Seam concentration: squared error per sample inside the blended bands vs outside them.
            let s = ck.scale
            let mx = seamMask(extent: w, tile: cfg.tile, overlap: cfg.overlap, scale: s)
            let my = seamMask(extent: h, tile: cfg.tile, overlap: cfg.overlap, scale: s)
            var seIn = 0.0, nIn = 0, seOut = 0.0, nOut = 0
            for yy in 0 ..< h * s {
                for xx in 0 ..< w * s {
                    let inBand = mx[xx] || my[yy]
                    for c in 0 ..< 3 {
                        let i = (yy * w * s + xx) * 3 + c
                        let e = Double(Int(out[i]) - Int(ref[i]))
                        if inBand { seIn += e * e; nIn += 1 } else { seOut += e * e; nOut += 1 }
                    }
                }
            }
            let mseIn = nIn > 0 ? seIn / Double(nIn) : 0, mseOut = nOut > 0 ? seOut / Double(nOut) : 0
            let band = mseOut > 0 ? mseIn / mseOut : (mseIn > 0 ? .infinity : 1)
            let key = cfg.description
            let prev = worst[key] ?? (.infinity, 0, 0, 0)
            worst[key] = (min(prev.psnr, d.psnr), max(prev.maxL, d.maxLevels), max(prev.frac, d.fracDiff),
                          max(prev.band, band.isFinite ? band : 1e9))
            cells.append("\(key) \(d.psnr.isFinite ? fmtF(d.psnr, 1) : "∞") dB (max \(d.maxLevels), band×\(band.isFinite ? fmtF(band, 1) : "∞"))")
        }
        note("   \(f) \(w)x\(h): " + cells.joined(separator: " · "))
    }
    note("N4 \(ck.upstreamName) over \(files.count) images — tiled vs whole-frame, 8-bit (worst per geometry):")
    for cfg in grid {
        let r = worst[cfg.description]!
        note("   \(cfg): worst \(r.psnr.isFinite ? fmtF(r.psnr, 1) : "∞") dB · max |Δ| \(r.maxL) · worst share changed "
             + "\(fmtF(100 * r.frac, 3))% · worst seam-band MSE ratio ×\(fmtF(r.band, 2))")
    }
}
