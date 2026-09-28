//
//  LiveCPUTests.swift — the package driven for real on the CPU stream (tiny inputs, so `swift test` needs no GPU):
//  scale routing and `appliedScale`, the package output == the core forward, exact-halo tiling == whole-frame, per-
//  tile RunProgress, a MID-RUN cancel surfacing within one tile, and the C14 INF gate on the LOADED graph.
//
//  The default device is pinned with the async `Device.withDefaultDevice` (a TaskLocal — it follows the run onto
//  the package's InferenceActor, since that is the same task).
//

import XCTest
import CoreGraphics
import Foundation
import MLX
import MLXNN
import MLXToolKit
import MLXServeConformance
import MLXServeConformanceNN
import NERVEMLX
@testable import MLXNERVE

/// C14 seam — lives in the TEST target so the shipping target takes no dependency on the conformance library.
extension NERVEUpscalePackage: InferenceModeInspectable {
    public func inferenceModeFlags() async -> [InferenceModeConformance.ModuleTrainingFlag] {
        InferenceModeConformance.flags(of: loadedModels)
    }
}

private final class ReportLog: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [RunPhaseReport] = []
    func append(_ r: RunPhaseReport) { lock.lock(); items.append(r); lock.unlock() }
    var all: [RunPhaseReport] { lock.lock(); defer { lock.unlock() }; return items }
}

final class LiveCPUTests: XCTestCase {

    /// A small image with edges, gradients and colour (so outputs are not trivially flat), as raw BGRA8.
    static func testImage(width w: Int = 40, height h: Int = 28) -> Image {
        var bytes = Data(count: w * h * 4)
        bytes.withUnsafeMutableBytes { raw in
            let p = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
            for y in 0 ..< h {
                for x in 0 ..< w {
                    let i = (y * w + x) * 4
                    let edge: UInt8 = (x / 7 + y / 5) % 2 == 0 ? 220 : 30
                    p[i] = UInt8((x * 255) / max(w - 1, 1))          // B: horizontal ramp
                    p[i + 1] = edge                                    // G: checker edges
                    p[i + 2] = UInt8((y * 255) / max(h - 1, 1))      // R: vertical ramp
                    p[i + 3] = 255
                }
            }
        }
        return Image.rawBGRA8(data: bytes, width: w, height: h)
    }

    private func onCPU<R>(_ body: () async throws -> R) async rethrows -> R {
        try await Device.withDefaultDevice(Device(.cpu)) { try await body() }
    }

    private func run(_ cfg: NERVEConfiguration, _ image: Image, scale: Int? = nil,
                     sink: RunProgress.Sink? = nil) async throws -> ImageUpscaleResponse {
        try await onCPU {
            let pkg = NERVEUpscalePackage(configuration: cfg)
            try await pkg.load()
            let req = ImageUpscaleRequest(image: image, scale: scale)
            let r = try await RunProgress.$sink.withValue(sink) { try await pkg.run(req) }
            return try XCTUnwrap(r as? ImageUpscaleResponse)
        }
    }

    func testRoutesDeliverTheAppliedScale() async throws {
        let img = Self.testImage()
        let cases: [(NERVEVariant, Int?, Int)] = [(.fidelity, nil, 4), (.fidelity, 2, 2), (.fidelity, 3, 3),
                                                  (.sharp, 2, 2), (.clean, 2, 2), (.sharp, 4, 4), (.clean, 8, 4)]
        for (variant, scale, applied) in cases {
            let r = try await run(NERVEConfiguration(variant: variant), img, scale: scale)
            XCTAssertEqual(r.appliedScale, applied, "\(variant) scale \(String(describing: scale))")
            XCTAssertEqual(r.image.width, 40 * applied)
            XCTAssertEqual(r.image.height, 28 * applied)
            XCTAssertEqual(r.image.format, .rawBGRA8, "raw in ⇒ raw out")
        }
    }

    /// The package's whole-frame output is the core forward on the same /255 input, quantised the same way.
    func testPackageOutputIsTheCoreForward() async throws {
        let img = Self.testImage()
        for precision in NERVEPrecision.allCases {
            let r = try await run(NERVEConfiguration(variant: .fidelity, precision: precision), img)
            let direct: [UInt8] = try await onCPU {
                var rgb = [Float](repeating: 0, count: 40 * 28 * 3)
                img.data.withUnsafeBytes { raw in
                    let p = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
                    for i in 0 ..< 40 * 28 {
                        rgb[i * 3] = Float(p[i * 4 + 2]) / 255; rgb[i * 3 + 1] = Float(p[i * 4 + 1]) / 255
                        rgb[i * 3 + 2] = Float(p[i * 4]) / 255
                    }
                }
                let m = try NERVE.load(from: NERVE_Playback.Checkpoint.fidelity4x.bundledWeightsURL!, expectedScale: 4,
                                       dtype: precision == .fp16 ? .float16 : nil)
                let y = MLX.round(MLX.clip(m(MLXArray(rgb, [1, 28, 40, 3])), min: 0, max: 1) * 255).asType(.uint8)
                eval(y)
                return y.asArray(UInt8.self)
            }
            var maxL = 0, changed = 0
            r.image.data.withUnsafeBytes { raw in
                let p = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
                for i in 0 ..< 160 * 112 {
                    for (c, off) in [(0, 2), (1, 1), (2, 0)] {
                        let d = abs(Int(p[i * 4 + off]) - Int(direct[i * 3 + c]))
                        maxL = max(maxL, d); if d > 0 { changed += 1 }
                    }
                }
            }
            // The package runs the COMPILED forward (fused elementwise ops): at most rounding-boundary flips.
            XCTAssertLessThanOrEqual(maxL, 1, "\(precision)")
            XCTAssertLessThanOrEqual(Double(changed) / Double(160 * 112 * 3), 0.001, "\(precision)")
        }
    }

    /// N4's design, pinned: tiles with a halo ≥ the 51 px receptive field, cropped to the frame before the forward,
    /// reproduce the whole-frame output — seams and image borders included.
    func testExactHaloTilingEqualsWholeFrame() async throws {
        // 160 px wide so interior tiles' 136 px windows (32 + 2·52) do NOT cover the frame — the halo, not the
        // frame, bounds their context; 24 px tall (< one tile) exercises the frame-crop on the other axis.
        let img = Self.testImage(width: 160, height: 24)
        let whole = try await run(NERVEConfiguration(variant: .fidelity, precision: .fp32, wholeFrameMaxPixels: Int.max), img)
        let tiled = try await run(NERVEConfiguration(variant: .fidelity, precision: .fp32, wholeFrameMaxPixels: 0,
                                                     inputTileSize: 32, tileOverlap: 8, tileHalo: 52), img)
        XCTAssertEqual(whole.image.data.count, tiled.image.data.count)
        var maxL = 0, changed = 0
        for (a, b) in zip(whole.image.data, tiled.image.data) {
            let d = abs(Int(a) - Int(b)); maxL = max(maxL, d); if d > 0 { changed += 1 }
        }
        XCTAssertLessThanOrEqual(maxL, 1)
        XCTAssertLessThanOrEqual(Double(changed) / Double(whole.image.data.count), 0.001)
        // …and plain tiles (no halo) DON'T: the probe that keeps the halo honest.
        let plain = try await run(NERVEConfiguration(variant: .fidelity, precision: .fp32, wholeFrameMaxPixels: 0,
                                                     inputTileSize: 32, tileOverlap: 8, tileHalo: 0), img)
        let plainMax = zip(whole.image.data, plain.image.data).map { abs(Int($0) - Int($1)) }.max() ?? 0
        XCTAssertGreaterThan(plainMax, 1, "plain 32 px tiles should visibly disagree with whole-frame")
    }

    func testRunProgressReportsEveryTile() async throws {
        let img = Self.testImage(width: 72, height: 44)
        let cfg = NERVEConfiguration(variant: .sharp, wholeFrameMaxPixels: 0, inputTileSize: 32, tileOverlap: 8, tileHalo: 0)
        let log = ReportLog()
        _ = try await run(cfg, img, scale: 2) { log.append($0) }
        let tier = try NERVE_Playback(checkpoint: .gan2x, wholeFrameMaxPixels: 0, inputTileSize: 32, tileOverlap: 8)
        let total = tier.tileCount(width: 72, height: 44)
        let up = log.all.filter { $0.phase == .upsample }
        XCTAssertGreaterThan(total, 1)
        XCTAssertEqual(up.map(\.step), Array(0 ... total))
        XCTAssertTrue(up.allSatisfy { $0.totalSteps == total })
    }

    /// The live half of CAN: a cancel landing after the first tile surfaces as an UNWRAPPED CancellationError
    /// before the second tile completes — deterministic, because the progress sink runs synchronously on the
    /// run's own task and cancels it there.
    func testMidRunCancelSurfacesWithinOneTile() async throws {
        let img = Self.testImage(width: 72, height: 44)
        let cfg = NERVEConfiguration(variant: .fidelity, wholeFrameMaxPixels: 0, inputTileSize: 32, tileOverlap: 8, tileHalo: 0)
        let log = ReportLog()
        do {
            _ = try await run(cfg, img) { report in
                log.append(report)
                if report.phase == .upsample, report.step == 1 { withUnsafeCurrentTask { $0?.cancel() } }
            }
            XCTFail("a cancelled run must not complete")
        } catch is CancellationError {
            let steps = log.all.filter { $0.phase == .upsample }.compactMap(\.step)
            XCTAssertEqual(steps, [0, 1], "stopped at the next tile boundary")
        } catch {
            XCTFail("cancellation laundered into \(type(of: error)): \(error)")
        }
    }

    /// C14 INF gate on the LOADED graph (every checkpoint the variant can load), and proof the gate can fail.
    /// NERVE's construction choke point is `NERVE.init` itself — every load path (`NERVE.load(from:)`,
    /// `NERVE.load(_:)`, `NERVE_Playback`, this package) funnels through it — so the inversion is shown by
    /// flipping the loaded graph back to training mode.
    func testInferenceModeGateOnTheLoadedGraph() async throws {
        try await onCPU {
            for variant in NERVEVariant.allCases {
                let pkg = NERVEUpscalePackage(configuration: NERVEConfiguration(variant: variant))
                try await pkg.load()
                let before = await pkg.loadedRoles
                XCTAssertTrue(before.isEmpty, "weights load lazily")
                try await pkg.loadWeightsNow()
                let roles = await pkg.loadedRoles
                XCTAssertEqual(roles, Set(variant.checkpoints.map { "x\($0.scale)" }))
                let report = await InferenceModeConformance.check(pkg, posture: .moduleGraph)
                XCTAssertTrue(report.passed, "\(variant): \(report.summary)")
                await pkg.setTrainingForTesting(true)
                let bad = await InferenceModeConformance.check(pkg, posture: .moduleGraph)
                XCTAssertFalse(bad.passed, "\(variant): a training-mode graph must fail INF-1")
                await pkg.setTrainingForTesting(false)
            }
        }
    }
}
