import Foundation
import MLXToolKit
import NERVEMLX

/// Which NERVE checkpoint family serves the upscale (all bundled in the core — no download).
///
/// | variant | ×4 checkpoint | ×2 request | role |
/// |---|---|---|---|
/// | `.fidelity` (default) | `4x_NERVE_OTF_fidelity` | native `2x_NERVE_OTF_gan` (no ×2 fidelity checkpoint exists; N5) | best FR on damaged input |
/// | `.clean` | `4x_NERVE_release` | native `2x_NERVE_release` | clean sources (bicubic-downscale prior) |
/// | `.sharp` | `4x_NERVE_OTF_gan` | native `2x_NERVE_OTF_gan` | RealPLKSR-class texture (GAN) |
public enum NERVEVariant: String, Codable, Sendable, CaseIterable {
    /// `4x_NERVE_OTF_fidelity` — OTF degradations, fidelity losses. The best full-reference arm on the damaged
    /// Forge bench cells (NERVE-EVAL §5.2). Default.
    case fidelity
    /// `4x_NERVE_release` / `2x_NERVE_release` — clean bicubic prior; passes JPEG artefacts through.
    case clean
    /// `4x_NERVE_OTF_gan` / `2x_NERVE_OTF_gan` — OTF degradations, GAN + perceptual losses.
    case sharp

    /// The native-×4 checkpoint.
    public var checkpoint4x: NERVE_Playback.Checkpoint {
        switch self {
        case .fidelity: return .fidelity4x
        case .clean: return .release4x
        case .sharp: return .gan4x
        }
    }

    /// The checkpoint that serves a ×2 request natively; `nil` = the ×4 checkpoint + a downsample.
    ///
    /// `.fidelity` has no native ×2 checkpoint upstream. The ×2 bench (PORTING-SPEC.md N5) measured the
    /// alternative — borrowing `2x_NERVE_OTF_gan` — against `.fidelity` ×4 + downsample, and the native ×2 won
    /// (`NERVEConfiguration.fidelity2xRoute`).
    public var checkpoint2x: NERVE_Playback.Checkpoint? {
        switch self {
        case .fidelity: return NERVEConfiguration.fidelity2xRoute
        case .clean: return .release2x
        case .sharp: return .gan2x
        }
    }

    /// Every checkpoint this variant can load (the bundled-sources declaration).
    public var checkpoints: [NERVE_Playback.Checkpoint] {
        [checkpoint4x] + (checkpoint2x.map { [$0] } ?? [])
    }
}

/// Compute precision of the conv body (the bicubic base and the output stay fp32 on both lanes).
///
/// N3 (PORTING-SPEC.md): the fp16 body reads 70.5–79.2 dB worst-case against fp32 over the 27 bench cells × all
/// five checkpoints (8-bit output ≤ 2 levels on `.fidelity`), runs 1.1–1.3× faster and roughly halves the peak —
/// so it is the default; fp32 is the parity lane (bit-for-bit the same with TF32 on or off — NERVE's 64-channel
/// convs sit outside every TF32 path).
public enum NERVEPrecision: String, Codable, Sendable, CaseIterable {
    /// fp32 — the parity lane.
    case fp32
    /// fp16 conv body, fp32 bicubic base and sum — the default.
    case fp16

    var core: NERVE_Playback.Precision { self == .fp16 ? .fp16 : .fp32 }
    var quant: Quant { self == .fp16 ? .fp16 : .fp32 }
}

/// Init-time configuration for `NERVEUpscalePackage` (C9). The checkpoints are **vendored in the core package
/// bundle** — 36 MB ships with the code, so a fresh machine never downloads anything and `load()` never touches
/// the network.
///
/// Materialization posture (engine ≥ 0.24.0, contract 1.17): `ModelStorable` so the engine can stamp its store
/// root (MAT-1), and `BundledWeightSourcing` declaring the selected variant's checkpoints as bundled sources —
/// the MAT gate verifies them PRESENT on a fresh machine and `needsDownload` reads `false` on a fresh install.
/// `WeightSourcing` stays undeclared by design (no fresh-machine network source; declaring one would either lie
/// about the missing set or force a download of weights already in the binary). `WeightPrewarming` pages the
/// checkpoints in before `load()`.
public struct NERVEConfiguration: PackageConfiguration, ModelStorable {
    /// `.fidelity`'s ×2 route (N5, PORTING-SPEC.md): the native `2x_NERVE_OTF_gan` checkpoint. On the ×2 bench
    /// (1024² references, 512² lows, 27 cells) it beat `.fidelity` ×4 + downsample on mean SSIMULACRA2 (+0.94; 17/27
    /// cells; jpeg regime +1.60, 7/9) and on wall time (HD → 4K 1.20 s vs 1.40 s), and it never builds the 8K
    /// intermediate. Its fidelity guard reads lower on photo (DC-worst 0.843 vs 0.897, jpeg) — the GAN checkpoint
    /// invents a little more — recorded, not decisive under the plan's FR + wall-time rule.
    static let fidelity2xRoute: NERVE_Playback.Checkpoint? = .gan2x

    /// The package default precision (N3).
    public static let defaultPrecision: NERVEPrecision = .fp16

    public var variant: NERVEVariant
    public var precision: NERVEPrecision
    /// Engine-chosen models root. Stamped by the engine from its `ModelStore`; unused (weights are bundled) but
    /// keeps the config store-addressable (MAT-1). Environment-specific → excluded from `Codable`.
    public var modelsRootDirectory: URL?

    /// Whole-frame fast-path ceiling in INPUT pixels, forwarded to the core; `nil` keeps the core default.
    /// Host-specific (the host gates it per machine) → excluded from `Codable`.
    public var wholeFrameMaxPixels: Int?
    /// Tile geometry for the tiled path; `nil` keeps the core defaults. Excluded from `Codable`.
    public var inputTileSize: Int?
    public var tileOverlap: Int?
    public var tileHalo: Int?
    /// `BudgetAware`: stamped by the engine at load time with the headroom the package is loading into.
    public var availableBudgetBytes: UInt64?

    public init(variant: NERVEVariant = .fidelity, precision: NERVEPrecision = NERVEConfiguration.defaultPrecision,
                modelsRootDirectory: URL? = nil, wholeFrameMaxPixels: Int? = nil,
                inputTileSize: Int? = nil, tileOverlap: Int? = nil, tileHalo: Int? = nil) {
        self.variant = variant
        self.precision = precision
        self.modelsRootDirectory = modelsRootDirectory
        self.wholeFrameMaxPixels = wholeFrameMaxPixels
        self.inputTileSize = inputTileSize
        self.tileOverlap = tileOverlap
        self.tileHalo = tileHalo
    }

    private enum CodingKeys: String, CodingKey {
        case variant, precision
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        variant = try c.decodeIfPresent(NERVEVariant.self, forKey: .variant) ?? .fidelity
        precision = try c.decodeIfPresent(NERVEPrecision.self, forKey: .precision) ?? Self.defaultPrecision
    }
}

/// The bundled-weights declaration (contract 1.17): the selected variant's vendored checkpoints — one role per
/// scale — verified PRESENT by the MAT gate and read by the engine's `needsDownload` as "nothing to fetch".
extension NERVEConfiguration: BundledWeightSourcing {
    public var bundledWeightSources: [BundledWeightSource] {
        variant.checkpoints.map { ck in
            BundledWeightSource(role: "checkpoint-x\(ck.scale)", url: ck.bundledWeightsURL)
        }
    }
}

/// Cold-start page-in: the prewarmer pages the variant's 7 MB checkpoints before first load.
extension NERVEConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        variant.checkpoints.compactMap(\.bundledWeightsURL)
    }
}

/// `QuantConfigured` (engine 1.14): the governor charges the `QuantFootprint` of the selected lane.
extension NERVEConfiguration: QuantConfigured {
    public var quant: Quant { precision.quant }
}

/// `BudgetAware` (the dtype lever): an fp32 (parity-lane) registration that is loaded into less headroom than its
/// declared fp32 footprint runs the fp16 body instead — the lighter lane at 70+ dB, under a reservation the
/// governor already sized for fp32. The fp16 default never changes.
extension NERVEConfiguration: BudgetAware {
    /// The precision `load()` actually uses.
    public var effectivePrecision: NERVEPrecision {
        guard precision == .fp32, let budget = availableBudgetBytes else { return precision }
        return budget < NERVEUpscalePackage.requiredBytes(.fp32) ? .fp16 : .fp32
    }
}
