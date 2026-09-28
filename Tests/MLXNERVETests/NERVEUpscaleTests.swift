import Testing
import Foundation
import CoreGraphics
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
import MLXToolKit
import NERVEMLX
@testable import MLXNERVE

/// Offline conformance checks (C0–C13 surface) — no Metal evaluation. Live runs are in `LiveCPUTests` (CPU
/// stream) and `nerve-smoke engine` (the real MLXServeEngine on the GPU).
struct NERVEUpscaleTests {

    @Test func manifestIsImageUpscaleAndPermissiveOnBothLayers() {
        let m = NERVEUpscalePackage.manifest
        #expect(m.capabilities == [.imageUpscale])
        #expect(m.license.weightLicense == .apache2)      // C7 — Phips/NERVE's weights, Apache-2.0
        #expect(m.license.portCodeLicense == .apache2)    // C8 — derived from nerve_arch.py (Apache-2.0)
        #expect(LicensePolicy.permissiveOnly.evaluate(m.license) == .admitted)
    }

    @Test func contractVersionIsDeclared() {
        #expect(NERVEUpscalePackage.manifest.contractVersion == ContractVersion.current)   // C0
    }

    @Test func provenancePointsAtOurNamespace() {
        // Fleet policy (2026-08-03): every weight source names a namespace WE control — the weights ship
        // vendored in this repo; the upstream pin lives in the README and each file's metadata.
        #expect(NERVEUpscalePackage.manifest.provenance.sourceRepo.hasPrefix("xocialize/"))
    }

    @Test func manifestRequirements() {
        let r = NERVEUpscalePackage.manifest.requirements
        #expect(r.requiredBackends.contains(.metalGPU))
        #expect(r.os.minMacOS == SemanticVersion(major: 26, minor: 0, patch: 0))
        #expect(r.chipFloor == nil)
    }

    /// Efficiency adoption (engine 1.14): both lanes declare the split (floor + activation peak), and the fp16
    /// lane is the lighter one.
    @Test func splitFootprintDeclaredPerLane() {
        let fps = NERVEUpscalePackage.manifest.requirements.footprints
        let fp32 = fps.first { $0.quant == .fp32 }, fp16 = fps.first { $0.quant == .fp16 }
        #expect(fp32 != nil && fp16 != nil)
        #expect((fp32?.peakActivationBytes ?? 0) > 0 && (fp16?.peakActivationBytes ?? 0) > 0)
        #expect((fp16?.peakActivationBytes ?? .max) < (fp32?.peakActivationBytes ?? 0))
    }

    /// `QuantConfigured` so the governor charges the selected lane's footprint, not largest-that-fits.
    @Test func quantConfiguredPerPrecision() {
        let c16: any PackageConfiguration = NERVEConfiguration()
        let c32: any PackageConfiguration = NERVEConfiguration(precision: .fp32)
        #expect((c16 as? QuantConfigured)?.quant == .fp16)   // N3: fp16 is the default lane
        #expect((c32 as? QuantConfigured)?.quant == .fp32)
    }

    /// `BudgetAware`: an fp32 registration loaded into less than its fp32 footprint runs fp16; fp16 never moves.
    @Test func budgetAwareDropsTheParityLaneUnderPressure() {
        var c = NERVEConfiguration(precision: .fp32)
        #expect(c.effectivePrecision == .fp32)                        // no figure stamped → as configured
        c.availableBudgetBytes = NERVEUpscalePackage.requiredBytes(.fp32)
        #expect(c.effectivePrecision == .fp32)                        // exactly enough
        c.availableBudgetBytes = NERVEUpscalePackage.requiredBytes(.fp32) - 1
        #expect(c.effectivePrecision == .fp16)                        // under → the lighter lane
        var h = NERVEConfiguration(precision: .fp16)
        h.availableBudgetBytes = 1
        #expect(h.effectivePrecision == .fp16)
        let asAny: any PackageConfiguration = c
        #expect(asAny is BudgetAware)
    }

    @Test func surfaceIsTheCanonicalUpscaleDescriptor() {
        let s = NERVEUpscalePackage.manifest.surfaces.first
        #expect(NERVEUpscalePackage.manifest.surfaces.count == 1)
        #expect(s?.name == "nerve-upscale")
        #expect(s?.capability == .imageUpscale)
        #expect(s?.parameters.first?.kind == .image)
        #expect(s?.parameters.contains { $0.name == "scale" && !$0.required } == true)
        #expect((s?.summary.count ?? 0) > 40)   // C11: written for the model audience
    }

    @Test func registrationConstructs() throws {
        let reg = NERVEUpscalePackage.registration
        #expect(reg.manifest.capabilities == [.imageUpscale])
        let pkg = try reg.makePackage(NERVEConfiguration())
        #expect(pkg is NERVEUpscalePackage)
    }

    @Test func variantsMapToTheUpstreamCheckpoints() {
        #expect(NERVEConfiguration().variant == .fidelity)
        #expect(NERVEVariant.fidelity.checkpoint4x.upstreamName == "4x_NERVE_OTF_fidelity")
        #expect(NERVEVariant.clean.checkpoint4x.upstreamName == "4x_NERVE_release")
        #expect(NERVEVariant.clean.checkpoint2x?.upstreamName == "2x_NERVE_release")
        #expect(NERVEVariant.sharp.checkpoint4x.upstreamName == "4x_NERVE_OTF_gan")
        #expect(NERVEVariant.sharp.checkpoint2x?.upstreamName == "2x_NERVE_OTF_gan")
        #expect(NERVEVariant.fidelity.checkpoint2x == NERVEConfiguration.fidelity2xRoute)
        for v in NERVEVariant.allCases {
            #expect(v.checkpoint4x.scale == 4)
            #expect(v.checkpoint2x.map { $0.scale == 2 } ?? true)
            #expect(Set(v.checkpoints).count == v.checkpoints.count)
        }
    }

    /// Scale routing: native ×4 for nil / ≥ 4; native ×2 where the variant has a ×2 checkpoint; otherwise the ×4
    /// checkpoint post-downsampled — and `appliedScale` always the scale actually delivered.
    @Test func scaleRouting() {
        for v in NERVEVariant.allCases {
            #expect(NERVEUpscalePackage.route(variant: v, requestedScale: nil) == (v.checkpoint4x, 4))
            #expect(NERVEUpscalePackage.route(variant: v, requestedScale: 4) == (v.checkpoint4x, 4))
            #expect(NERVEUpscalePackage.route(variant: v, requestedScale: 8) == (v.checkpoint4x, 4))
            #expect(NERVEUpscalePackage.route(variant: v, requestedScale: 0) == (v.checkpoint4x, 4))
            #expect(NERVEUpscalePackage.route(variant: v, requestedScale: 3) == (v.checkpoint4x, 3))
            #expect(NERVEUpscalePackage.route(variant: v, requestedScale: 1) == (v.checkpoint4x, 1))
            let two = NERVEUpscalePackage.route(variant: v, requestedScale: 2)
            #expect(two.applied == 2)
            #expect(two.checkpoint == (v.checkpoint2x ?? v.checkpoint4x))
        }
    }

    @Test func configurationCodableRoundTrips() throws {
        var c = NERVEConfiguration(variant: .sharp, precision: .fp32, wholeFrameMaxPixels: 123,
                                   inputTileSize: 64, tileOverlap: 8, tileHalo: 52)
        c.availableBudgetBytes = 42
        let back = try JSONDecoder().decode(NERVEConfiguration.self, from: JSONEncoder().encode(c))
        #expect(back.variant == .sharp)
        #expect(back.precision == .fp32)
        // Host-specific knobs are never persisted.
        #expect(back.wholeFrameMaxPixels == nil && back.inputTileSize == nil && back.tileOverlap == nil)
        #expect(back.tileHalo == nil && back.availableBudgetBytes == nil && back.modelsRootDirectory == nil)
        // Older payloads without `precision` decode to the default lane.
        let legacy = try JSONDecoder().decode(NERVEConfiguration.self, from: Data(#"{"variant":"clean"}"#.utf8))
        #expect(legacy.variant == .clean && legacy.precision == NERVEConfiguration.defaultPrecision)
    }

    @Test func pngRoundTripsThroughPixelBuffer() throws {
        let png = try #require(Self.makePNG(width: 32, height: 32))
        let pb = try NERVEUpscalePackage.decodeToPixelBuffer(Image(format: .png, data: png, width: 32, height: 32))
        #expect(CVPixelBufferGetWidth(pb) == 32)
        let back = try #require(NERVEUpscalePackage.encodePNG(pb))
        #expect(back.prefix(4) == Data([0x89, 0x50, 0x4E, 0x47]))
    }

    @Test func rawBGRA8RoundTripsBitIdentical() throws {
        let w = 8, h = 4
        let bytes = Data((0..<(w * h * 4)).map { UInt8($0 % 256) })
        let pb = try NERVEUpscalePackage.decodeToPixelBuffer(Image.rawBGRA8(data: bytes, width: w, height: h))
        #expect(CVPixelBufferGetWidth(pb) == w && CVPixelBufferGetHeight(pb) == h)
        let back = try #require(NERVEUpscalePackage.encodeRawBGRA8(pb))
        #expect(back.format == .rawBGRA8)
        #expect(back.width == w && back.height == h && back.bytesPerRow == nil)
        #expect(back.data == bytes)
    }

    @Test func rawBGRA8MissingDimensionsThrows() {
        #expect(throws: NERVEPackageError.self) {
            _ = try NERVEUpscalePackage.decodeToPixelBuffer(Image(format: .rawBGRA8, data: Data(count: 16)))
        }
    }

    /// The sub-native `scale` path yields the requested dimensions as a valid 32BGRA buffer.
    @Test func resizePixelBufferProducesRequestedDimensions() throws {
        let png = try #require(Self.makePNG(width: 64, height: 64))
        let pb = try NERVEUpscalePackage.decodeToPixelBuffer(Image(format: .png, data: png, width: 64, height: 64))
        let scaled = try NERVEUpscalePackage.resizePixelBuffer(pb, toWidth: 32, height: 32)
        #expect(CVPixelBufferGetWidth(scaled) == 32 && CVPixelBufferGetHeight(scaled) == 32)
        #expect(CVPixelBufferGetPixelFormatType(scaled) == kCVPixelFormatType_32BGRA)
    }

    static func makePNG(width: Int, height: Int) -> Data? {
        guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.setFillColor(CGColor(gray: 0.6, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        guard let cg = ctx.makeImage() else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }
}
