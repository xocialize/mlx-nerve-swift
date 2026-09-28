import Foundation
import CoreGraphics
import CoreImage
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXNN
import MLXToolKit
import NERVEMLX

/// Errors at the NERVE package boundary.
public enum NERVEPackageError: Error, Equatable {
    case imageDecodeFailed(String)
    case imageEncodeFailed
}

/// An MLXEngine `imageUpscale` package over **NERVE** (Phips/NERVE, Philip Hofmann; Apache-2.0 code and weights,
/// trained only on the CC0 LUCID corpus) — the provenance-clean **fast tier**: a 1.8 M-parameter pure-conv
/// network (5×5 stem → 24 conv-ReLU-conv residual blocks → PixelShuffle head → + PyTorch bicubic) at the cost
/// class of Real-ESRGAN's compact net, which it replaces as Forge's `.upscale` backer.
///
/// A thin conformance wrapper over the standalone `NERVEMLX` core; all model logic (the network, tiling via the
/// shared driver, NHWC) lives there. The five checkpoints are vendored in the core bundle — `load()` involves
/// **no download**.
///
/// Scale routing (the response's `appliedScale` always reports what actually ran):
/// - `scale` `nil` or `≥ 4` → the variant's ×4 checkpoint at native ×4.
/// - `scale == 2` → a native ×2 checkpoint: `2x_NERVE_release` for `.clean`, `2x_NERVE_OTF_gan` for `.sharp` AND
///   `.fidelity` (no ×2 fidelity checkpoint exists upstream; the ×2 bench chose the GAN one over ×4 + downsample —
///   PORTING-SPEC.md N5). HD → 4K therefore never builds the 8K intermediate.
/// - `scale == 1` or `3` → the ×4 result post-downsampled to `input × scale`.
@InferenceActor
public final class NERVEUpscalePackage: ModelPackage {
    public typealias Configuration = NERVEConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: Phips/NERVE's checkpoints are Apache-2.0 (repo LICENSE + card + README §License), trained only
            // on Phips/lucid-cc0-v2-hc-512 (CC0-1.0 ← nyuuzyou/pxhere ← pxhere.com, platform-declared CC0 —
            // shippable under AB-D-0106). C8: the port is derived from nerve_arch.py (Apache-2.0) → Apache-2.0.
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            // Fleet durability policy: provenance names a namespace we control — the weights are vendored in the
            // xocialize repo itself (no network source at runtime); the upstream pin is in the README + metadata.
            provenance: Provenance(sourceRepo: "xocialize/mlx-nerve-swift", revision: "v0.1.0", tier: 1),
            requirements: RequirementsManifest(
                // Split footprint (engine 1.14), measured at N6 — the table is on `footprints` below.
                footprints: NERVEUpscalePackage.footprints,
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                ImageUpscaleContract.descriptor(
                    name: "nerve-upscale",
                    summary: "NERVE fast super-resolution (x2/x4 native) for photos and graphics: restores JPEG and "
                        + "real-world degradations at Real-ESRGAN-class cost; 1.8M-param pure-conv, CC0-trained; tile-based."
                )
            ]
        )
    }

    private let configuration: Configuration
    private var tiers: [NERVE_Playback.Checkpoint: NERVE_Playback] = [:]

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard tiers.isEmpty else { return }
        // Weights are vendored in the core bundle; construction validates their presence and the core lazy-loads
        // tensors on the first upscale that needs them. `nil` keeps each core default.
        var built: [NERVE_Playback.Checkpoint: NERVE_Playback] = [:]
        for ck in configuration.variant.checkpoints {
            built[ck] = try NERVE_Playback(
                checkpoint: ck, precision: configuration.effectivePrecision.core,
                wholeFrameMaxPixels: configuration.wholeFrameMaxPixels ?? NERVE_Playback.defaultWholeFrameMaxPixels,
                inputTileSize: configuration.inputTileSize ?? NERVE_Playback.defaultInputTileSize,
                tileOverlap: configuration.tileOverlap ?? NERVE_Playback.defaultTileOverlap,
                tileHalo: configuration.tileHalo ?? NERVE_Playback.defaultTileHalo)
        }
        tiers = built
    }

    public func unload() async {
        tiers = [:]
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS (not just drop refs)
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before notLoaded validation. Mid-run cadence:
        // the shared tile driver checkpoints once per tile, rethrown unchanged.
        try Task.checkCancellation()
        guard !tiers.isEmpty else { throw PackageError.notLoaded }
        guard request.capability == .imageUpscale,
              let req = request as? ImageUpscaleRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        let inPB = try Self.decodeToPixelBuffer(req.image)
        let inW = CVPixelBufferGetWidth(inPB), inH = CVPixelBufferGetHeight(inPB)

        let route = Self.route(variant: configuration.variant, requestedScale: req.scale)
        guard let tier = tiers[route.checkpoint] else { throw PackageError.notLoaded }

        RunProgress.report(.upsample, step: 0, totalSteps: tier.tileCount(width: inW, height: inH))
        let nativePB = try tier.upscale(inPB) { done, total in
            RunProgress.report(.upsample, step: done, totalSteps: total)
        }

        let outPB: CVPixelBuffer
        if route.applied != route.checkpoint.scale {
            try Task.checkCancellation()
            RunProgress.report(.postprocess)
            outPB = try Self.resizePixelBuffer(nativePB, toWidth: inW * route.applied, height: inH * route.applied)
        } else {
            outPB = nativePB
        }

        let w = CVPixelBufferGetWidth(outPB), h = CVPixelBufferGetHeight(outPB)
        // Output mirrors the input format: rawBGRA8 in ⇒ rawBGRA8 out (no re-encode); else .png.
        let outImage: Image
        if req.image.format == .rawBGRA8 {
            guard let raw = Self.encodeRawBGRA8(outPB) else { throw NERVEPackageError.imageEncodeFailed }
            outImage = raw
        } else {
            guard let png = Self.encodePNG(outPB) else { throw NERVEPackageError.imageEncodeFailed }
            outImage = Image(format: .png, data: png, width: w, height: h)
        }
        return ImageUpscaleResponse(image: outImage, appliedScale: route.applied)
    }

    // MARK: - Scale routing

    /// Which checkpoint serves a request, and the scale the response reports.
    nonisolated static func route(variant: NERVEVariant, requestedScale: Int?)
        -> (checkpoint: NERVE_Playback.Checkpoint, applied: Int) {
        guard let s = requestedScale, s > 0, s < 4 else { return (variant.checkpoint4x, 4) }
        if s == 2, let native2x = variant.checkpoint2x { return (native2x, 2) }
        return (variant.checkpoint4x, s)
    }

    /// The loaded networks by role (`x4`, `x2`) — the C14 inference-mode seam (nil until first use).
    var loadedModels: [String: Module?] {
        var out: [String: Module?] = [:]
        for (ck, tier) in tiers { out["x\(ck.scale)"] = tier.loadedModel }
        return out
    }

    /// Roles whose network is loaded (Sendable view of `loadedModels`).
    var loadedRoles: Set<String> {
        Set(loadedModels.compactMap { $0.value == nil ? nil : $0.key })
    }

    /// Eagerly load every tier's weights (the INF gate needs the LOADED graph; a run loads lazily).
    func loadWeightsNow() throws {
        for tier in tiers.values { try tier.prepare() }
    }

    /// Test seam for the INF gate's inversion: flip every loaded graph's training flag, on the actor.
    func setTrainingForTesting(_ training: Bool) {
        for case let m? in loadedModels.values { m.train(training) }
    }

    // MARK: - Image codec (identical to the Real-ESRGAN / RealPLKSR siblings' — the canonical Image ↔ BGRA seam)

    /// Decode a canonical `Image` (.png/.jpeg/.rawBGRA8) to a BGRA `CVPixelBuffer`.
    nonisolated static func decodeToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        if image.format == .rawBGRA8 { return try rawBGRA8ToPixelBuffer(image) }
        guard let source = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw NERVEPackageError.imageDecodeFailed("unreadable \(image.format.rawValue) data")
        }
        let w = cg.width, h = cg.height
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw NERVEPackageError.imageDecodeFailed("pixel buffer allocation (\(w)x\(h))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let ctx = CGContext(
                data: base, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue) else {
            throw NERVEPackageError.imageDecodeFailed("CGContext for BGRA draw")
        }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    /// Encode a BGRA `CVPixelBuffer` as PNG bytes.
    nonisolated static func encodePNG(_ pb: CVPixelBuffer) -> Data? {
        let ci = CIImage(cvPixelBuffer: pb)
        let ctx = CIContext(options: [.cacheIntermediates: false])
        guard let cg = ctx.createCGImage(ci, from: ci.extent) else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }

    /// Wrap raw interleaved BGRA8 bytes straight into a 32BGRA `CVPixelBuffer` — no decode.
    nonisolated static func rawBGRA8ToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        guard let w = image.width, let h = image.height, w > 0, h > 0 else {
            throw NERVEPackageError.imageDecodeFailed("rawBGRA8 requires width/height")
        }
        let srcStride = image.bytesPerRow ?? (w * 4)
        guard srcStride >= w * 4, image.data.count >= srcStride * h else {
            throw NERVEPackageError.imageDecodeFailed(
                "rawBGRA8 data too small (\(image.data.count) < \(srcStride * h))")
        }
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw NERVEPackageError.imageDecodeFailed("pixel buffer allocation (\(w)x\(h))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw NERVEPackageError.imageDecodeFailed("pixel buffer base address")
        }
        let dstStride = CVPixelBufferGetBytesPerRow(buffer)
        let rowBytes = min(srcStride, dstStride)
        image.data.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            guard let srcBase = src.baseAddress else { return }
            for row in 0..<h {
                memcpy(base.advanced(by: row * dstStride), srcBase.advanced(by: row * srcStride), rowBytes)
            }
        }
        return buffer
    }

    /// High-quality downsample of a 32BGRA `CVPixelBuffer` to `w`×`h` (a new 32BGRA buffer).
    nonisolated static func resizePixelBuffer(_ src: CVPixelBuffer, toWidth w: Int, height h: Int) throws -> CVPixelBuffer {
        guard w > 0, h > 0 else { throw NERVEPackageError.imageEncodeFailed }
        let ci = CIImage(cvPixelBuffer: src)
        let ctx = CIContext(options: [.cacheIntermediates: false])
        guard let cg = ctx.createCGImage(ci, from: ci.extent) else { throw NERVEPackageError.imageEncodeFailed }
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw NERVEPackageError.imageEncodeFailed
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let outCtx = CGContext(
                data: base, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue) else {
            throw NERVEPackageError.imageEncodeFailed
        }
        outCtx.interpolationQuality = .high
        outCtx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    /// Emit a 32BGRA `CVPixelBuffer` as tightly-packed raw BGRA8 `Image` bytes.
    nonisolated static func encodeRawBGRA8(_ pb: CVPixelBuffer) -> Image? {
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard w > 0, h > 0 else { return nil }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let srcStride = CVPixelBufferGetBytesPerRow(pb)
        let dstStride = w * 4
        var out = Data(count: dstStride * h)
        out.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
            guard let dstBase = dst.baseAddress else { return }
            for row in 0..<h {
                memcpy(dstBase.advanced(by: row * dstStride), base.advanced(by: row * srcStride), dstStride)
            }
        }
        return Image.rawBGRA8(data: out, width: w, height: h)
    }
}

extension NERVEUpscalePackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(NERVEUpscalePackage.self)
    }

    /// Resident + peak activation the manifest declares for a lane (the `BudgetAware` threshold).
    public nonisolated static func requiredBytes(_ precision: NERVEPrecision) -> UInt64 {
        guard let fp = footprints.first(where: { $0.quant == precision.quant }) else { return .max }
        return fp.residentBytes + fp.peakActivationBytes
    }

    /// Split footprint (engine 1.14), MEASURED (N6, 2026-09-28, M5 Max, release `nerve-smoke engine`: the real
    /// MLXServeEngine, raw BGRA in/out, `.fidelity`, one size per process). Basis = the process `phys_footprint`
    /// the governor's R-MEM-1 trigger reads: resident floor = post-load with the MLX cache cleared (after a 24×16
    /// warm-up run of the same route); activation = the kernel's lifetime `ledger_phys_footprint_peak` − that floor.
    ///
    ///   input → ×4          path                 steady    ms/Mpx out   floor phys   activation   MLX peak
    ///   256²                whole-frame  fp16    0.023 s   22.1         158 MB       441 MB       431 MB
    ///   512²                whole-frame  fp16    0.086 s   20.4         160 MB       1039 MB      846 MB
    ///   1024²               whole-frame  fp16    0.354 s   21.1         165 MB       2122 MB      1312 MB
    ///   1080×1920           whole-frame  fp16    0.750 s   22.6         173 MB       4167 MB      2591 MB
    ///   2160×3840           tiled 512/0/56 fp16  5.472 s   41.2         221 MB       4701 MB      898 MB
    ///   1080×1920 (×2)      whole-frame  fp16    0.586 s   70.6         165 MB       2909 MB      2401 MB
    ///   1080×1920           whole-frame  fp32    0.757 s   22.8         154 MB       6629 MB      4682 MB
    ///   2160×3840           tiled 512/0/56 fp32  6.676 s   50.3         209 MB       5279 MB      1736 MB
    ///
    /// The floor is MLX-active weights (4 MB fp16 / 7 MB fp32 — one checkpoint; both native scales ≤ 15 MB) plus the
    /// process's Metal/MLX runtime. Tiled, the MLX working set is bounded; what grows with the input past the
    /// whole-frame ceiling is the output (a 4K input at ×4 is a 133 Mpx, 531 MB BGRA frame, held twice). DECLARED over
    /// the envelope measured — inputs up to 3840×2160 at ×4 — with the harness headroom (×1.2 + 256 MB):
    ///   fp16: resident 256 MB · activation 4701 MB × 1.2 + 256 MB ≈ 5.9 GB
    ///   fp32: resident 256 MB · activation 6629 MB × 1.2 + 256 MB ≈ 8.2 GB
    /// (CLI process basis; the in-app re-baseline rides the image-fleet batch, AB-T-0019.)
    nonisolated static var footprints: [QuantFootprint] {
        [
            QuantFootprint(quant: .fp16, residentBytes: 256_000_000, peakActivationBytes: 5_900_000_000),
            QuantFootprint(quant: .fp32, residentBytes: 256_000_000, peakActivationBytes: 8_200_000_000),
        ]
    }
}
