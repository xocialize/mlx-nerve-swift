// NERVE — MLX-Swift port of Phips/NERVE `nerve_arch.py` (Apache-2.0), pinned at HF revision
// c23588c36988fc2058ea3d7db80646d3ecb12295.
//
// Isomorphic to upstream: same class names (`NerveBlock`, `NERVE`), same decomposition, same forward order, same
// parameter paths (`stem`, `blocks.N.conv{1,2}`, `upsampler.0`) — the checkpoint keys ARE the upstream state-dict
// keys; only each conv weight's layout changes (OIHW → OHWI, `oracle/convert_weights.py`). PyTorch → MLX op
// substitutions only:
//   • NCHW → NHWC throughout (MLX convs are channels-last).
//   • `nn.PixelShuffle(s)` → `pixelShuffleNHWC` (MLXNN has no channel-last pixel shuffle); channel `c·s·s + i·s + j`
//     → (c, i, j), exactly torch's ordering. The `(s, s, C)` reading is shape-identical and wrong — the S1 gate
//     runs it as a committed probe that must fail loudly.
//   • `F.interpolate(x, scale_factor=s, mode="bicubic", align_corners=False)` — Keys a = −0.75, half-pixel centres,
//     clamped borders, no antialias; torch's kernel, not Pillow's — → the same resize as a fixed SUB-PIXEL CONV
//     (edge-pad 2, 5×5 phase kernel, pixel shuffle; `BicubicPhaseKernel`), added to the head BEFORE the one shared
//     shuffle. The one deviation in form (not in numbers): `MLXNN.Upsample(.cubic)` matches torch as well but
//     materialises 16 output-resolution gathers (~6 GB at 1080p → 4K); it stays as `bicubicReference`, gated beside
//     it at S1. `alignCorners: true` is shape-safe and wrong; the S1 gate probes it too.
//   • ICNR (`icnr_reinit`) is initialisation only — nothing to port.
//
//   stem  Conv5×5 (3 → 64, no bias)
//   body  24 × NerveBlock: x + Conv3×3(ReLU(Conv3×3(x)))           (64 ch, no bias, no norm, no gate)
//         global residual: stem + body(stem)
//   head  Conv3×3 (64 → 3·s², no bias) → PixelShuffle(s)
//   out   head + bicubic(x, ×s)                                    (here: shuffle(head + bicubic phases))
//
// 1,801,920 params at ×4 (`upsampler.0.weight` [48, 3, 3, 64] in MLX layout), 1,781,184 at ×2 ([12, 3, 3, 64]).
// The scale is DERIVED FROM THE HEAD WEIGHT when loading (`NERVE.load(from:)`) — never from a flag that could
// disagree with it (NERVE-HEART-PORT-PLAN §2.3).

import Foundation
import MLX
import MLXNN

/// Errors from the NERVE core.
public enum NERVEError: Error, Sendable, CustomStringConvertible {
    case weightsNotFound(String)
    case loadFailed(String)
    /// The checkpoint's key set does not cover the module tree exactly — a silently partial load is the failure
    /// mode that produces plausible-but-wrong output, so it is loud.
    case parameterMismatch(missing: [String], extra: [String])
    /// A checkpoint whose head weight implies a different scale than the one the caller asked for.
    case scaleMismatch(expected: Int, checkpoint: Int)
    /// A checkpoint that is not in the MLX (OHWI) layout — e.g. the upstream OIHW file vendored by mistake.
    case wrongLayout(String)

    public var description: String {
        switch self {
        case .weightsNotFound(let p): return "NERVE weights not found: \(p)"
        case .loadFailed(let d): return "NERVE weight load failed: \(d)"
        case .parameterMismatch(let m, let e): return "NERVE parameter mismatch — missing \(m) extra \(e)"
        case .scaleMismatch(let e, let c): return "NERVE scale mismatch — expected ×\(e), checkpoint head is ×\(c)"
        case .wrongLayout(let d): return "NERVE checkpoint layout: \(d)"
        }
    }
}

// MARK: - ops

/// NHWC pixel shuffle matching `torch.nn.PixelShuffle` on NCHW.
///
/// Torch views (B, C·r·r, H, W) as (B, C, r₁, r₂, H, W) — channel `c·r·r + i·r + j` maps to (c, i, j) — then
/// permutes to (B, C, H, r₁, W, r₂). Channel-last equivalent: (B,H,W,C·r·r) → (B,H,W,C,r₁,r₂) → (B,H,r₁,W,r₂,C).
public func pixelShuffleNHWC(_ x: MLXArray, _ r: Int) -> MLXArray {
    let (b, h, w, crr) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let c = crr / (r * r)
    return x.reshaped([b, h, w, c, r, r])
        .transposed(0, 1, 4, 2, 5, 3)      // B, H, r₁, W, r₂, C
        .reshaped([b, h * r, w * r, c])
}

// MARK: - bicubic as a sub-pixel convolution

/// Torch's `F.interpolate(x, scale_factor=s, mode="bicubic", align_corners=False)` (Keys a = −0.75, half-pixel
/// centres, tap indices clamped to the image, no antialias) at an INTEGER scale `s` is a fixed sub-pixel
/// convolution. Output pixel `s·i + p` samples source position `i + δₚ`, `δₚ = (p + ½)/s − ½ ∈ (−½, ½)`: the four
/// taps `i−2 … i+1` when `δₚ < 0` (t = δₚ + 1) or `i−1 … i+2` when `δₚ ≥ 0` (t = δₚ), with 1-D weights that depend
/// on the phase alone; torch's clamped indices are exactly replicate padding by 2. So the whole resize is: edge-pad
/// by 2 → a VALID 5×5 conv taking colour `c` to its `s²` phase channels `c·s² + p·s + q` (the pixel-shuffle order)
/// → pixel shuffle.
///
/// Why not `MLXNN.Upsample(.cubic)`, which matches torch too: it materialises 16 full-OUTPUT-resolution gathers and
/// their weighted sum — at a 1080×1920 → 4320×7680 frame, ≈ 6 GB of the forward's 7.2 GB peak (N6). The phase conv
/// costs 3,600 MAC per input pixel (0.2 % of the network's) and writes one output-sized tensor. Parity against
/// torch is gated at S1 beside the Upsample reference (`NERVE.bicubicReference`).
final class BicubicPhaseKernel: @unchecked Sendable {   // a box — Module reflection must never see the constant
    let scale: Int
    let weight: MLXArray   // [3·s², 5, 5, 3], OHWI, block-diagonal over colour

    init(scale s: Int) {
        let w = Self.weights1D(scale: s)
        var k = [Float](repeating: 0, count: 3 * s * s * 5 * 5 * 3)
        for c in 0 ..< 3 {
            for p in 0 ..< s {
                for q in 0 ..< s {
                    let o = c * s * s + p * s + q
                    for a in 0 ..< 5 {
                        for b in 0 ..< 5 {
                            // Outer product in Double, one rounding to Float.
                            k[((o * 5 + a) * 5 + b) * 3 + c] = Float(w[p][a] * w[q][b])
                        }
                    }
                }
            }
        }
        scale = s
        weight = MLXArray(k, [3 * s * s, 5, 5, 3])
    }

    /// 1-D weights per phase over the 5-tap window `i−2 … i+2` (one tap always 0), from torch's own coefficient
    /// functions (`cubic_convolution1/2`, A = −0.75), in Double.
    static func weights1D(scale s: Int) -> [[Double]] {
        let A = -0.75
        func cc1(_ x: Double) -> Double { ((A + 2) * x - (A + 3)) * x * x + 1 }
        func cc2(_ x: Double) -> Double { ((A * x - 5 * A) * x + 8 * A) * x - 4 * A }
        return (0 ..< s).map { p in
            let delta = (Double(p) + 0.5) / Double(s) - 0.5
            let t = delta < 0 ? delta + 1 : delta
            let four = [cc2(t + 1), cc1(t), cc1(1 - t), cc2(2 - t)]
            return delta < 0 ? four + [0] : [0] + four
        }
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var cache: [Int: BicubicPhaseKernel] = [:]

    static func forScale(_ s: Int) -> BicubicPhaseKernel {
        lock.lock()
        defer { lock.unlock() }
        if let k = cache[s] { return k }
        let k = BicubicPhaseKernel(scale: s)
        cache[s] = k
        return k
    }
}

// MARK: - modules

/// Conv-ReLU-Conv residual block. No normalization, no shortcuts.
final class NerveBlock: Module, UnaryLayer {
    let conv1: Conv2d
    let conv2: Conv2d

    init(_ dim: Int) {
        conv1 = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 3, padding: 1, bias: false)
        conv2 = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 3, padding: 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let out = conv2(relu(conv1(x)))
        return x + out
    }
}

/// 5x5 stem -> n_blocks residual blocks -> PixelShuffle head -> + bicubic(x).
///
/// Input/output are NHWC RGB in [0, 1] (float32, or float16 on the half lane); output is `upscale`× the input.
public final class NERVE: Module {
    let stem: Conv2d
    let blocks: [NerveBlock]
    /// Upstream `nn.Sequential(Conv2d, PixelShuffle)`: index 0 carries the only weight, the shuffle (index 1) has
    /// none — so a one-element array reproduces the checkpoint path `upsampler.0.weight` exactly.
    let upsampler: [Conv2d]
    /// The fixed bicubic phase kernel for this scale (a boxed constant — not a parameter, not in the key set).
    let bicubicKernel: BicubicPhaseKernel
    public let upscale: Int
    public let dim: Int
    public let nBlocks: Int

    /// Exact parameter counts of the released checkpoints (dim 64, 24 blocks).
    public static let parameterCount4x = 1_801_920
    public static let parameterCount2x = 1_781_184

    public init(inCh: Int = 3, outCh: Int = 3, dim: Int = 64, nBlocks: Int = 24, upscale: Int = 4) {
        self.upscale = upscale
        self.dim = dim
        self.nBlocks = nBlocks
        stem = Conv2d(inputChannels: inCh, outputChannels: dim, kernelSize: 5, padding: 2, bias: false)
        blocks = (0 ..< nBlocks).map { _ in NerveBlock(dim) }
        upsampler = [Conv2d(inputChannels: dim, outputChannels: outCh * upscale * upscale,
                            kernelSize: 3, padding: 1, bias: false)]
        bicubicKernel = BicubicPhaseKernel.forScale(upscale)
        super.init()
        // C14: born in inference mode at the construction choke point. Nothing in NERVE is train/eval-dependent
        // (no norm, no dropout), but the loaded graph must REPORT `training == false`.
        train(false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let feat = stem(bodyInput(x))
        var body = feat
        for b in blocks { body = b(body) }
        let head = upsampler[0](feat + body)
        // upstream: pixel_shuffle(head) + F.interpolate(x, bicubic). The bicubic is itself a sub-pixel conv whose
        // channels are in the same (c, p, q) order and the shuffle is a pure permutation, so ONE shuffle of the sum
        // is the same numbers as shuffling each and adding (the add sees identical operands) — one output-sized
        // tensor instead of three.
        return pixelShuffleNHWC(atInputPrecision(head, x) + Self.bicubicPhases(x, bicubicKernel), upscale)
    }

    /// Half lane (weights cast at load): the conv body runs at the weights' dtype while the bicubic base and the
    /// sum stay at the input's precision — the base carries most of the output's energy, and 8-bit input levels
    /// (k/255) are not exact in fp16. On the fp32 lane both helpers are the identity, i.e. upstream exactly.
    @inline(__always) func bodyInput(_ x: MLXArray) -> MLXArray {
        let dt = stem.weight.dtype
        return x.dtype == dt ? x : x.asType(dt)
    }

    @inline(__always) func atInputPrecision(_ y: MLXArray, _ x: MLXArray) -> MLXArray {
        y.dtype == x.dtype ? y : y.asType(x.dtype)
    }

    /// The bicubic base in phase-channel form, `[B, H, W, 3·s²]` (edge-pad 2 → VALID 5×5 conv).
    static func bicubicPhases(_ x: MLXArray, _ k: BicubicPhaseKernel) -> MLXArray {
        let w = k.weight.dtype == x.dtype ? k.weight : k.weight.asType(x.dtype)
        return conv2d(padded(x, widths: [0, 2, 2, 0], mode: .edge), w)
    }

    /// `F.interpolate(x, scale_factor=s, mode="bicubic", align_corners=False)` on NHWC — as the sub-pixel conv.
    public static func bicubic(_ x: MLXArray, scale: Int) -> MLXArray {
        pixelShuffleNHWC(bicubicPhases(x, .forScale(scale)), scale)
    }

    /// The same resize through `MLXNN.Upsample(.cubic(alignCorners: false))` — the reference the sub-pixel form is
    /// gated beside (it matches torch too, at 16 output-resolution gathers of memory).
    public static func bicubicReference(_ x: MLXArray, scale: Int) -> MLXArray {
        Upsample(scaleFactor: .float(Float(scale)), mode: .cubic(alignCorners: false))(x)
    }
}

// MARK: - taps (S1 parity)

extension NERVE {
    /// The eight intermediates the S1 gate compares, computed by the same code path as `callAsFunction`.
    public func taps(_ x: MLXArray) -> [String: MLXArray] {
        var t: [String: MLXArray] = [:]
        let feat = stem(bodyInput(x))
        t["stem"] = feat
        var body = feat
        for (i, b) in blocks.enumerated() {
            body = b(body)
            if i == 0 { t["block0"] = body }
        }
        t["block23"] = body
        let global = feat + body
        t["body"] = global
        let head = upsampler[0](global)
        t["head"] = head
        let shuffled = pixelShuffleNHWC(head, upscale)
        t["shuffle"] = shuffled
        t["bicubic"] = Self.bicubic(x, scale: upscale)
        t["output"] = pixelShuffleNHWC(atInputPrecision(head, x) + Self.bicubicPhases(x, bicubicKernel), upscale)
        return t
    }

    /// Each sub-op applied to the ORACLE'S input of that sub-op (`golden` holds upstream's taps) — isolates a
    /// sub-op's own rounding from the error it inherits through the chain. `block23` is the 24-block body run from
    /// the golden stem (the chain is the sub-op there); `output` is the final sum of the golden shuffle and base.
    public func isolatedTaps(input x: MLXArray, golden g: [String: MLXArray]) -> [String: MLXArray] {
        var t: [String: MLXArray] = [:]
        t["stem"] = stem(bodyInput(x))
        if let s = g["stem"] {
            t["block0"] = blocks[0](s)
            var body = s
            for b in blocks { body = b(body) }
            t["block23"] = body
            t["body"] = s + body
        }
        if let body = g["body"] { t["head"] = upsampler[0](body) }
        if let head = g["head"] { t["shuffle"] = pixelShuffleNHWC(head, upscale) }
        t["bicubic"] = Self.bicubic(x, scale: upscale)
        if let sh = g["shuffle"], let base = g["bicubic"] { t["output"] = sh + base }
        return t
    }
}

// MARK: - weight loading

extension NERVE {
    /// The checkpoint key contract for a given depth (S0): `stem.weight`, `blocks.{0..n-1}.conv{1,2}.weight`,
    /// `upsampler.0.weight` — every conv `bias=False`, so 2 + 2n tensors (50 for the released depth).
    public static func expectedKeys(nBlocks: Int = 24) -> Set<String> {
        var keys: Set<String> = ["stem.weight", "upsampler.0.weight"]
        for i in 0 ..< nBlocks {
            keys.insert("blocks.\(i).conv1.weight")
            keys.insert("blocks.\(i).conv2.weight")
        }
        return keys
    }

    /// Build a NERVE whose scale, width and depth are read from the checkpoint itself, and load it with full
    /// coverage. `expectedScale`, when given, must agree with the head weight (else `scaleMismatch`).
    ///
    /// `dtype` casts every weight at load (the half-precision lane); `nil` keeps the file's fp32.
    public static func load(from url: URL, expectedScale: Int? = nil, dtype: DType? = nil) throws -> NERVE {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw NERVEError.weightsNotFound(url.path)
        }
        let arrays: [String: MLXArray]
        let metadata: [String: String]
        do {
            (arrays, metadata) = try MLX.loadArraysAndMetadata(url: url)
        } catch {
            throw NERVEError.loadFailed(String(describing: error))
        }
        if let layout = metadata["layout"], layout != "OHWI" {
            throw NERVEError.wrongLayout("metadata layout \(layout), expected OHWI (run oracle/convert_weights.py)")
        }
        return try load(arrays, expectedScale: expectedScale, dtype: dtype)
    }

    /// As `load(from:)`, from already-loaded MLX-layout arrays.
    public static func load(_ arrays: [String: MLXArray], expectedScale: Int? = nil,
                            dtype: DType? = nil) throws -> NERVE {
        guard let head = arrays["upsampler.0.weight"], head.ndim == 4,
              let stemW = arrays["stem.weight"], stemW.ndim == 4 else {
            throw NERVEError.loadFailed("checkpoint missing stem.weight / upsampler.0.weight")
        }
        // MLX layout (O, kH, kW, I): the head's O is 3·s², its I the width; the stem's kernel is 5×5.
        guard head.dim(1) == 3, head.dim(2) == 3, stemW.dim(1) == 5, stemW.dim(2) == 5 else {
            throw NERVEError.wrongLayout("stem \(stemW.shape) / head \(head.shape) are not OHWI")
        }
        let outCh = 3
        let s2 = head.dim(0) / outCh
        let scale = Int(Double(s2).squareRoot().rounded())
        guard scale * scale * outCh == head.dim(0) else {
            throw NERVEError.loadFailed("head has \(head.dim(0)) output channels — not 3·s²")
        }
        if let expectedScale, expectedScale != scale {
            throw NERVEError.scaleMismatch(expected: expectedScale, checkpoint: scale)
        }
        let nBlocks = arrays.keys.filter { $0.hasPrefix("blocks.") && $0.hasSuffix(".conv1.weight") }.count
        let model = NERVE(dim: stemW.dim(0), nBlocks: nBlocks, upscale: scale)
        try model.loadWeights(arrays, dtype: dtype)
        return model
    }

    /// Load MLX-layout arrays whose keys must equal this module's parameter paths exactly (0 missing / 0 unused).
    public func loadWeights(_ arrays: [String: MLXArray], dtype: DType? = nil) throws {
        let expected = Set(parameters().flattened().map { $0.0 })
        let provided = Set(arrays.keys)
        if expected != provided {
            throw NERVEError.parameterMismatch(
                missing: Array(expected.subtracting(provided).sorted().prefix(5)),
                extra: Array(provided.subtracting(expected).sorted().prefix(5)))
        }
        let flat = arrays.map { (key, value) in (key, dtype.map { value.asType($0) } ?? value) }
        do {
            try update(parameters: ModuleParameters.unflattened(flat), verify: .all)
        } catch {
            throw NERVEError.loadFailed(String(describing: error))
        }
        // Materialise now (CPU-stream load + any cast), so the first forward's command buffer never carries it.
        eval(self)
    }
}
