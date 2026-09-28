//
//  NERVETests.swift — core tests: the key contract and exact parameter counts (weight-free), inference mode
//  (C14), the vendored checkpoints' full-coverage load and the loader's refusals, and S1 PARITY against the
//  committed golden produced by THE oracle (upstream nerve_arch.py executed on the pinned checkpoints,
//  oracle/dump_goldens.py s1): every sub-op tap for 4x_NERVE_OTF_fidelity and 2x_NERVE_OTF_gan (both shuffle
//  factors), and the end-to-end output of all five checkpoints vs torch AND vs the author's ONNX export.
//
//  XCTest on the CPU stream: fp32 on CPU is the numerically strict setting, and `swift test` cannot rely on the
//  Metal bundle being staged for GPU work (GPU lanes live in `nerve-smoke gate … --gpu`). Build with
//  `--build-system swiftbuild` so the metallib MLX initialises on first op exists.
//

import Foundation
import MLX
import MLXNN
import XCTest
@testable import NERVEMLX

private func withCPU<R>(_ body: () throws -> R) rethrows -> R {
    try Device.withDefaultDevice(Device(.cpu), body)
}

private func parameterCount(_ m: Module) -> Int {
    m.parameters().flattened().reduce(0) { $0 + $1.1.size }
}

/// Relative max error (max|Δ| / max|ref|) and PSNR (peak 1, unclamped), float64 on the host.
private func compare(_ y: MLXArray, _ ref: MLXArray) -> (rel: Double, psnr: Double, maxAbs: Double) {
    XCTAssertEqual(y.shape, ref.shape)
    let a = y.asType(.float32).asArray(Float.self), b = ref.asType(.float32).asArray(Float.self)
    var maxAbs = 0.0, refMax = 0.0, se = 0.0
    for i in 0 ..< min(a.count, b.count) {
        let d = Double(a[i]) - Double(b[i])
        maxAbs = max(maxAbs, abs(d)); refMax = max(refMax, abs(Double(b[i]))); se += d * d
    }
    let mse = se / Double(max(a.count, 1))
    return (refMax > 0 ? maxAbs / refMax : maxAbs, mse == 0 ? .infinity : 10 * log10(1 / mse), maxAbs)
}

final class NERVEStructureTests: XCTestCase {

    func testModuleTreeIsTheCheckpointKeyContract() {
        // Weight-free, no eval: the generated key set IS the upstream state-dict key set (S0's structural twin).
        for s in [2, 4] {
            let keys = Set(NERVE(upscale: s).parameters().flattened().map(\.0))
            XCTAssertEqual(keys, NERVE.expectedKeys(), "×\(s)")
            XCTAssertEqual(keys.count, 50)
        }
    }

    func testExactParameterCounts() {
        withCPU {
            XCTAssertEqual(parameterCount(NERVE(upscale: 4)), NERVE.parameterCount4x)   // 1,801,920
            XCTAssertEqual(parameterCount(NERVE(upscale: 2)), NERVE.parameterCount2x)   // 1,781,184
        }
    }

    func testForwardShapes() {
        withCPU {
            for s in [2, 4] {
                let y = NERVE(upscale: s)(MLXArray.zeros([1, 21, 34, 3]))
                eval(y)
                XCTAssertEqual(y.shape, [1, 21 * s, 34 * s, 3])
            }
        }
    }

    /// C14: born in inference mode at the construction choke point (nothing in NERVE is train-dependent — no
    /// norm, no dropout — but the graph must report it; the package-level INF gate checks the LOADED graph).
    func testBornInInferenceMode() {
        let m = NERVE()
        XCTAssertFalse(m.training)
        XCTAssertTrue(m.blocks.allSatisfy { !$0.training && !$0.conv1.training && !$0.conv2.training })
        XCTAssertFalse(m.stem.training)
        XCTAssertFalse(m.upsampler[0].training)
    }

    /// The pixel-shuffle helper matches torch's channel ordering on a hand-built case: channel c·r·r + i·r + j of
    /// input pixel (y, x) lands at output (y·r + i, x·r + j), channel c.
    func testPixelShuffleOrderingByConstruction() {
        withCPU {
            let r = 2, c = 3, h = 2, w = 2
            let x = MLXArray((0 ..< (h * w * c * r * r)).map { Float($0) }, [1, h, w, c * r * r])
            let y = pixelShuffleNHWC(x, r)
            eval(y)
            XCTAssertEqual(y.shape, [1, h * r, w * r, c])
            let xv = x.asArray(Float.self), yv = y.asArray(Float.self)
            for yy in 0 ..< h { for xx in 0 ..< w { for ch in 0 ..< c { for i in 0 ..< r { for j in 0 ..< r {
                let src = ((yy * w + xx) * (c * r * r)) + ch * r * r + i * r + j
                let dst = (((yy * r + i) * (w * r)) + (xx * r + j)) * c + ch
                XCTAssertEqual(yv[dst], xv[src])
            } } } } }
        }
    }
}

final class NERVECheckpointTests: XCTestCase {

    func testEveryVendoredCheckpointLoadsWithFullCoverageAndItsOwnScale() throws {
        try withCPU {
            for ck in NERVE_Playback.Checkpoint.allCases {
                let url = try XCTUnwrap(ck.bundledWeightsURL, "\(ck) missing from the bundle")
                let m = try NERVE.load(from: url, expectedScale: ck.scale)
                XCTAssertEqual(m.upscale, ck.scale)
                XCTAssertEqual(parameterCount(m), ck.scale == 4 ? NERVE.parameterCount4x : NERVE.parameterCount2x)
                XCTAssertFalse(m.training)
                // Something non-trivial landed (not the random init's range, not zeros).
                XCTAssertGreaterThan(MLX.abs(m.stem.weight).sum().item(Float.self), 0)
            }
        }
    }

    /// The scale is read from the head weight; a caller expecting another scale is refused (plan §2.3).
    func testScaleDisagreementIsRefused() throws {
        let url = try XCTUnwrap(NERVE_Playback.Checkpoint.gan2x.bundledWeightsURL)
        XCTAssertThrowsError(try withCPU { try NERVE.load(from: url, expectedScale: 4) }) { error in
            guard case NERVEError.scaleMismatch(let e, let c) = error else { return XCTFail("\(error)") }
            XCTAssertEqual(e, 4); XCTAssertEqual(c, 2)
        }
    }

    func testPartialCheckpointIsRefused() throws {
        let url = try XCTUnwrap(NERVE_Playback.Checkpoint.fidelity4x.bundledWeightsURL)
        try withCPU {
            var sd = try MLX.loadArrays(url: url)
            sd.removeValue(forKey: "blocks.23.conv2.weight")
            XCTAssertThrowsError(try NERVE.load(sd)) { error in
                guard case NERVEError.parameterMismatch(let missing, _) = error else { return XCTFail("\(error)") }
                XCTAssertEqual(missing, ["blocks.23.conv2.weight"])
            }
        }
    }

    /// An upstream (OIHW) state dict must never load as if it were MLX layout.
    func testUpstreamLayoutIsRefused() throws {
        let url = try XCTUnwrap(NERVE_Playback.Checkpoint.fidelity4x.bundledWeightsURL)
        try withCPU {
            let oihw = try MLX.loadArrays(url: url).mapValues { $0.transposed(0, 3, 1, 2) }
            XCTAssertThrowsError(try NERVE.load(oihw))
        }
    }

    func testPlaybackTierDefaultsAndScales() throws {
        for ck in NERVE_Playback.Checkpoint.allCases {
            let tier = try NERVE_Playback(checkpoint: ck)
            XCTAssertEqual(tier.scaleFactor, ck.scale)
            XCTAssertEqual(tier.inputTileSize, NERVE_Playback.defaultInputTileSize)
            XCTAssertEqual(tier.tileOverlap, NERVE_Playback.defaultTileOverlap)
            XCTAssertEqual(tier.tileHalo, NERVE_Playback.defaultTileHalo)
            XCTAssertEqual(tier.wholeFrameMaxPixels, NERVE_Playback.defaultWholeFrameMaxPixels)
            XCTAssertNil(tier.loadedModel, "weights load lazily")
        }
        XCTAssertEqual(try NERVE_Playback(checkpoint: .fidelity4x).name, "nerve-fidelity-x4")
        XCTAssertEqual(try NERVE_Playback(checkpoint: .gan2x, precision: .fp16).name, "nerve-sharp-x2-fp16")
    }

    /// The tile-count helper walks the same deduplicated clamped grid as the shared driver.
    func testTileCountMatchesTheDriverGrid() throws {
        let tier = try NERVE_Playback(checkpoint: .fidelity4x, wholeFrameMaxPixels: 0, inputTileSize: 256, tileOverlap: 32)
        XCTAssertEqual(tier.tileCount(width: 256, height: 256), 1)
        XCTAssertEqual(tier.tileCount(width: 100, height: 60), 1)         // smaller than one tile
        XCTAssertEqual(tier.tileCount(width: 480, height: 256), 2)        // origins 0, 224
        XCTAssertEqual(tier.tileCount(width: 1920, height: 1080), 9 * 5)  // 0,224,…,1568,1664 × 0,224,…,824
        let whole = try NERVE_Playback(checkpoint: .fidelity4x)
        XCTAssertEqual(whole.tileCount(width: 1920, height: 1080), 1)     // whole-frame path
    }
}

final class NERVEParityTests: XCTestCase {

    private func golden() throws -> [String: MLXArray] {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "goldens_s1_37x53", withExtension: "safetensors"))
        return try MLX.loadArrays(url: url)
    }

    private static let taps = ["stem", "block0", "block23", "body", "head", "shuffle", "bicubic", "output"]
    // Same tables as the CLI gate (Sources/Smoke/Gates.swift) — see PORTING-SPEC.md S1 for the evidence.
    private static let chained: [String: Double] = ["stem": 2e-6, "block0": 1e-5, "block23": 2e-5, "body": 2e-5,
                                                    "head": 2e-5, "shuffle": 2e-5, "bicubic": 2e-6, "output": 1e-5]
    private static let isolated: [String: Double] = ["stem": 2e-6, "block0": 1e-5, "block23": 1e-5, "body": 1e-5,
                                                     "head": 1e-5, "shuffle": 0, "bicubic": 2e-6, "output": 2e-6]

    /// S1 at 37×53 (odd, non-square — the bicubic border) for both shuffle factors: every sub-op, chained through
    /// the model's own forward AND isolated on the oracle's input for it.
    func testSubOpParityBothScales() throws {
        let g = try golden()
        let x = try XCTUnwrap(g["input"])
        try withCPU {
            for ck in [NERVE_Playback.Checkpoint.fidelity4x, .gan2x] {
                let m = try NERVE.load(from: ck.bundledWeightsURL!, expectedScale: ck.scale)
                var ref: [String: MLXArray] = [:]
                for t in Self.taps { ref[t] = g["\(ck.upstreamName).\(t)"] }
                let chained = m.taps(x), iso = m.isolatedTaps(input: x, golden: ref)
                for t in Self.taps {
                    let r = try XCTUnwrap(ref[t], "\(ck) \(t)")
                    let c = compare(try XCTUnwrap(chained[t]), r), i = compare(try XCTUnwrap(iso[t]), r)
                    XCTAssertLessThanOrEqual(c.rel, Self.chained[t]!, "\(ck) chained \(t)")
                    XCTAssertLessThanOrEqual(i.rel, Self.isolated[t]!, "\(ck) isolated \(t)")
                }
            }
        }
    }

    /// N2's twin at fixture size: all five checkpoints end to end vs torch and vs the author's ONNX (≥ 90 dB).
    func testEndToEndAllCheckpointsVsTorchAndONNX() throws {
        let g = try golden()
        let x = try XCTUnwrap(g["input"])
        try withCPU {
            for ck in NERVE_Playback.Checkpoint.allCases {
                let y = try NERVE.load(from: ck.bundledWeightsURL!, expectedScale: ck.scale)(x)
                let vsTorch = compare(y, try XCTUnwrap(g["\(ck.upstreamName).output"]))
                let vsONNX = compare(y, try XCTUnwrap(g["\(ck.upstreamName).onnx"]))
                XCTAssertGreaterThanOrEqual(vsTorch.psnr, 90, "\(ck) vs torch")
                XCTAssertGreaterThanOrEqual(vsONNX.psnr, 90, "\(ck) vs ONNX")
            }
        }
    }

    /// Probe: the (r, r, C) reading of the shuffle's channel axis is shape-identical and must fail loudly.
    func testPixelShuffleWrongOrderFailsLoudly() throws {
        let g = try golden()
        try withCPU {
            for name in ["4x_NERVE_OTF_fidelity", "2x_NERVE_OTF_gan"] {
                let head = try XCTUnwrap(g["\(name).head"]), ref = try XCTUnwrap(g["\(name).shuffle"])
                let r = name.hasPrefix("4x") ? 4 : 2
                let (b, h, w, crr) = (head.dim(0), head.dim(1), head.dim(2), head.dim(3))
                let wrong = head.reshaped([b, h, w, r, r, crr / (r * r)]).transposed(0, 1, 3, 2, 4, 5)
                    .reshaped([b, h * r, w * r, crr / (r * r)])
                XCTAssertGreaterThan(compare(wrong, ref).rel, 0.1, "\(name): wrong order not caught")
                XCTAssertEqual(compare(pixelShuffleNHWC(head, r), ref).maxAbs, 0, "\(name): shuffle not exact")
            }
        }
    }

    /// Probe: torch's bicubic (a = −0.75, half-pixel, clamped borders) — error uniform across the border band,
    /// and the shape-safe `alignCorners: true` misreading far off.
    func testBicubicIsTorchsAndBorderUniform() throws {
        let g = try golden()
        let x = try XCTUnwrap(g["input"])
        try withCPU {
            for (name, s) in [("4x_NERVE_OTF_fidelity", 4), ("2x_NERVE_OTF_gan", 2)] {
                let ref = try XCTUnwrap(g["\(name).bicubic"])
                let y = NERVE.bicubic(x, scale: s)
                XCTAssertLessThanOrEqual(compare(y, ref).rel, 2e-6, "\(name)")
                // The MLXNN Upsample form the sub-pixel conv replaced matches torch too.
                XCTAssertLessThanOrEqual(compare(NERVE.bicubicReference(x, scale: s), ref).rel, 2e-6, "\(name) reference")
                // Border band (2·s px) vs interior: rounding is uniform, a border bug concentrates.
                let a = y.asArray(Float.self), b = ref.asArray(Float.self)
                let (h, w) = (ref.dim(1), ref.dim(2)), band = 2 * s
                var border = 0.0, interior = 0.0
                for yy in 0 ..< h { for xx in 0 ..< w { for c in 0 ..< 3 {
                    let i = (yy * w + xx) * 3 + c, d = abs(Double(a[i]) - Double(b[i]))
                    if yy < band || xx < band || yy >= h - band || xx >= w - band { border = max(border, d) }
                    else { interior = max(interior, d) }
                } } }
                XCTAssertLessThanOrEqual(border, max(4 * interior, 1e-6), "\(name): border-concentrated")
                let ac = Upsample(scaleFactor: .float(Float(s)), mode: .cubic(alignCorners: true))(x)
                XCTAssertLessThan(compare(ac, ref).psnr, 40, "\(name): alignCorners:true not caught")
            }
        }
    }
}
