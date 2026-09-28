// NERVE_Playback.swift
//
// A `PlaybackTier` over NERVE: model + the SHARED tile driver (RealESRGANMLX.MLXTileProcessor), the pairing the
// shipped Real-ESRGAN and RealPLKSR tiers use — so the three backends are drop-in siblings behind one call and
// share the tile compositor's fidelity tests.
//
// Weights: Resources/<checkpoint>-mlx.safetensors — the five checkpoints of Phips/NERVE @ c23588c36988 in MLX
// (OHWI) layout, fp32, sha256 of every source recorded in the file metadata and oracle/reports/convert.log.
// Vendored, loaded lazily on the first upscale; the scale is read from the checkpoint's head and must agree with
// the checkpoint case (a disagreement throws, never runs).
//
// Tile geometry (N4, see PORTING-SPEC.md): the receptive field radius is 51 LR px (5×5 stem = 2, 48 3×3 convs = 48,
// 3×3 head = 1), so a plain overlap blends two outputs computed from different context. The tiled path therefore
// hands each tile a HALO of real neighbouring pixels (`MLXTileProcessor.process(_:halo:forward:)`), crops the
// window to the frame before the forward (so the model's own zero padding lands exactly on the image edge, as in
// upstream's whole-frame run) and re-embeds the result for the driver's centre crop. With halo ≥ 51 every kept
// output pixel sees its whole receptive field: tiled == whole-frame up to fp rounding. Whole-frame vs tiled is then
// a pure memory/speed choice, made per machine by `wholeFrameMaxPixels`.

import CoreVideo
import Foundation
import MLX
import MLXNN
import RealESRGANMLX

public final class NERVE_Playback: PlaybackTier, @unchecked Sendable {

    /// One of the five released checkpoints (vendored in this module's bundle).
    public enum Checkpoint: String, Sendable, CaseIterable {
        /// `4x_NERVE_OTF_fidelity` — OTF (Real-ESRGAN-style) degradations, fidelity losses. The default.
        case fidelity4x
        /// `4x_NERVE_release` — clean bicubic-downscale pretrain.
        case release4x
        /// `2x_NERVE_release` — clean bicubic-downscale pretrain, ×2.
        case release2x
        /// `4x_NERVE_OTF_gan` — OTF degradations, GAN + perceptual losses (the trainer's showcase).
        case gan4x
        /// `2x_NERVE_OTF_gan` — OTF degradations, GAN + perceptual losses, ×2.
        case gan2x

        /// The upstream checkpoint stem (`models/<stem>.safetensors` at the pinned revision).
        public var upstreamName: String {
            switch self {
            case .fidelity4x: return "4x_NERVE_OTF_fidelity"
            case .release4x: return "4x_NERVE_release"
            case .release2x: return "2x_NERVE_release"
            case .gan4x: return "4x_NERVE_OTF_gan"
            case .gan2x: return "2x_NERVE_OTF_gan"
            }
        }

        /// Resource stem (no extension) of the vendored MLX-layout safetensors file.
        public var safetensorsName: String { "\(upstreamName)-mlx" }

        /// The native scale of the checkpoint (its head weight is checked against this at load).
        public var scale: Int {
            switch self {
            case .fidelity4x, .release4x, .gan4x: return 4
            case .release2x, .gan2x: return 2
            }
        }

        /// The vendored checkpoint's URL inside the core's resource bundle (nil = stripped bundle).
        public var bundledWeightsURL: URL? {
            Bundle.module.url(forResource: safetensorsName, withExtension: "safetensors")
        }

        var tierName: String {
            switch self {
            case .fidelity4x: return "nerve-fidelity-x4"
            case .release4x: return "nerve-clean-x4"
            case .release2x: return "nerve-clean-x2"
            case .gan4x: return "nerve-sharp-x4"
            case .gan2x: return "nerve-sharp-x2"
            }
        }
    }

    /// Compute precision of the conv body. The bicubic base and the output always stay fp32.
    public enum Precision: String, Sendable, CaseIterable {
        /// fp32 weights and activations — the parity lane (TF32-class on M5 unless the host process sets
        /// `MLX_ENABLE_TF32=0`, AB-L-0175).
        case fp32
        /// fp16 weights and activations in the conv body; input cast once, output cast back to fp32 before the
        /// fp32 bicubic base is added.
        case fp16

        var dtype: DType { self == .fp16 ? .float16 : .float32 }
    }

    // MARK: - Defaults (measured — PORTING-SPEC.md N4 / N6)

    /// N4: frame-cropped halo tiles reproduce the whole-frame output (27 bench lows: bit-identical at overlap 0;
    /// 10 HD stills: bit-identical or ≤ 1 level on a few samples where the clamped last tile overlaps its
    /// neighbour), where the template's plain 256/32 tiles read 62.9 dB with the error ×15,578 concentrated in the
    /// seam bands. 512 px tiles keep the halo's extra work at (624/512)² = 1.49× per interior tile (256: 2.07×).
    public static let defaultInputTileSize = 512
    /// No feathering is needed between tiles that already agree; 0 also spends no pixels on overlap.
    public static let defaultTileOverlap = 0
    /// ≥ the 51 px receptive-field radius (5×5 stem 2 + 48 3×3 convs + 3×3 head 1), with margin.
    public static let defaultTileHalo = 56
    public static let defaultWholeFrameMaxPixels = 1920 * 1080

    // MARK: - PlaybackTier surface

    public let name: String
    public let scaleFactor: Int
    public let inputTileSize: Int
    public let tileOverlap: Int
    public var inputResolution: (width: Int, height: Int) { (inputTileSize, inputTileSize) }
    public var outputResolution: (width: Int, height: Int) {
        (inputTileSize * scaleFactor, inputTileSize * scaleFactor)
    }
    public let checkpoint: Checkpoint
    public let precision: Precision

    /// Input-pixel ceiling for the single-pass (no tiles, no seams) path; `0` forces tiling.
    public let wholeFrameMaxPixels: Int
    /// Real-context margin handed to each tile on the tiled path (input px); `0` = the plain overlap driver.
    public let tileHalo: Int

    // MARK: - Internals

    private let tileProcessor: MLXTileProcessor
    private let weightsURL: URL
    private let loadLock = NSLock()
    private var model: NERVE?
    private var compiledForward: (@Sendable (MLXArray) -> MLXArray)?

    // MARK: - Init

    /// Construction validates the vendored checkpoint is PRESENT (a stripped bundle fails here, not at the first
    /// upscale); the tensors load lazily on the first `upscale`.
    public init(checkpoint: Checkpoint = .fidelity4x, precision: Precision = .fp32,
                wholeFrameMaxPixels: Int = NERVE_Playback.defaultWholeFrameMaxPixels,
                inputTileSize: Int = NERVE_Playback.defaultInputTileSize,
                tileOverlap: Int = NERVE_Playback.defaultTileOverlap,
                tileHalo: Int = NERVE_Playback.defaultTileHalo) throws {
        precondition(inputTileSize > 0 && tileOverlap >= 0 && tileOverlap < inputTileSize && tileHalo >= 0)
        self.checkpoint = checkpoint
        self.precision = precision
        self.name = precision == .fp32 ? checkpoint.tierName : "\(checkpoint.tierName)-\(precision.rawValue)"
        self.scaleFactor = checkpoint.scale
        self.wholeFrameMaxPixels = wholeFrameMaxPixels
        self.inputTileSize = inputTileSize
        self.tileOverlap = tileOverlap
        self.tileHalo = tileHalo

        guard let url = checkpoint.bundledWeightsURL else {
            throw PlaybackTierError.weightsNotFound(checkpoint.safetensorsName)
        }
        self.weightsURL = url
        self.tileProcessor = MLXTileProcessor(tileSize: inputTileSize, overlap: tileOverlap, scale: checkpoint.scale)
    }

    // MARK: - PlaybackTier impl

    public func upscale(_ buffer: CVPixelBuffer) async throws -> CVPixelBuffer {
        try upscale(buffer, progress: nil)
    }

    /// Whether a `width × height` input runs whole-frame (one forward) rather than tiled.
    public func runsWholeFrame(width: Int, height: Int) -> Bool {
        wholeFrameMaxPixels > 0 && width * height <= wholeFrameMaxPixels
    }

    /// Number of tiles the tiled path runs for a `width × height` input — the same deduplicated clamped grid
    /// `MLXTileProcessor.process` walks (1 on the whole-frame path).
    public func tileCount(width: Int, height: Int) -> Int {
        if runsWholeFrame(width: width, height: height) { return 1 }
        func axis(_ extent: Int) -> Int {
            let step = max(inputTileSize - tileOverlap, 1)
            var origins: [Int] = []
            for o in stride(from: 0, to: extent, by: step) {
                let c = min(o, max(0, extent - inputTileSize))
                if origins.last != c { origins.append(c) }
            }
            return origins.count
        }
        return axis(width) * axis(height)
    }

    /// Upscale with an optional per-unit progress callback `(done, total)` — called once per tile on the tiled
    /// path (after the tile's forward) and once on the whole-frame path, synchronously on the caller's task, so
    /// a task-local progress sink (MLXToolKit's `RunProgress`) bound around the call sees every report.
    public func upscale(_ buffer: CVPixelBuffer, progress: ((Int, Int) -> Void)?) throws -> CVPixelBuffer {
        let run = try ensureReady()
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        do {
            if runsWholeFrame(width: w, height: h) {
                let out = try tileProcessor.processWholeFrame(buffer) { x in
                    let y = run(x)
                    MLX.eval(y)
                    return y
                }
                progress?(1, 1)
                return out
            }
            let total = tileCount(width: w, height: h)
            var done = 0
            let halo = tileHalo, s = scaleFactor
            return try tileProcessor.process(buffer, halo: halo) { window, region in
                let y: MLXArray
                if halo > 0 {
                    // The driver edge-REPLICATES the part of a halo window that leaves the frame; upstream
                    // zero-pads at every conv layer at the image border, which no input padding reproduces.
                    // So run only the in-frame part of the window (the model's own zero padding then falls
                    // exactly on the image edge) and re-embed its output at the same offset — the driver
                    // crops the tile's centre, which never reaches the re-embedding margin. Interior sides
                    // keep `halo` px of real context: with halo ≥ the 51 px receptive field radius, a tile's
                    // kept output equals the whole-frame output up to fp rounding.
                    let side = window.dim(1)
                    let wx = region.x - halo, wy = region.y - halo
                    let x0 = max(0, -wx), y0 = max(0, -wy)
                    let x1 = min(side, w - wx), y1 = min(side, h - wy)
                    let inFrame = run(window[0..., y0 ..< y1, x0 ..< x1, 0...])
                    y = padded(inFrame, widths: [0, .init((y0 * s, (side - y1) * s)),
                                                 .init((x0 * s, (side - x1) * s)), 0])
                } else {
                    y = run(window)
                }
                MLX.eval(y)
                done += 1
                progress?(done, total)
                return y
            }
        } catch let err as PlaybackTierError {
            throw err
        } catch is CancellationError {
            // CAN-2: never launder a cancellation — the per-tile checkpoint's CancellationError must reach the
            // engine unchanged.
            throw CancellationError()
        } catch {
            throw PlaybackTierError.inferenceError(String(describing: error))
        }
    }

    // MARK: - Weights + compile

    /// Load weights (once) and build the compiled forward (once) — the same strategy as the Real-ESRGAN and
    /// RealPLKSR tiers so an A/B between them runs through identical machinery.
    private func ensureReady() throws -> @Sendable (MLXArray) -> MLXArray {
        loadLock.lock()
        defer { loadLock.unlock() }
        if let f = compiledForward { return f }
        let m: NERVE
        do {
            m = try NERVE.load(from: weightsURL, expectedScale: checkpoint.scale,
                               dtype: precision == .fp32 ? nil : precision.dtype)
        } catch {
            throw PlaybackTierError.modelLoadFailed(String(describing: error))
        }
        model = m
        let f = compile { x in m(x) }
        compiledForward = f
        return f
    }

    /// Load the weights and build the compiled forward now instead of on the first upscale.
    public func prepare() throws {
        _ = try ensureReady()
    }

    /// The loaded network (nil until the first upscale) — the C14 inference-mode seam.
    public var loadedModel: NERVE? {
        loadLock.lock()
        defer { loadLock.unlock() }
        return model
    }
}
