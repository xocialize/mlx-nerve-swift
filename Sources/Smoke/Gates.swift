import Foundation
import MLX
import MLXNN
import NERVEMLX

// MARK: - S0 — the key contract

/// Generated keys (and shapes) of a weight-free NERVE at the checkpoint's scale == the safetensors header of every
/// vendored checkpoint: 0 missing / 0 unused, all F32, all OHWI. With `upstreamDir`, also checks each upstream
/// file carries the SAME keys with the OIHW shapes of the vendored ones and the pinned sha256 in our metadata.
func gateS0(upstreamDir: String?) throws -> Bool {
    var ok = true
    for ck in NERVE_Playback.Checkpoint.allCases {
        guard let url = ck.bundledWeightsURL else {
            note("S0 \(ck.upstreamName): ❌ not in the bundle"); ok = false; continue
        }
        let (tensors, meta) = try safetensorsHeader(url)
        let model = NERVE(upscale: ck.scale)
        let generated = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
        let missing = Set(generated.keys).subtracting(tensors.keys).sorted()
        let unused = Set(tensors.keys).subtracting(generated.keys).sorted()
        let badShape = generated.filter { k, s in tensors[k].map { $0.shape != s } ?? false }.keys.sorted()
        let dtypes = Set(tensors.values.map(\.dtype))
        let params = tensors.values.reduce(0) { $0 + $1.shape.reduce(1, *) }
        let expectParams = ck.scale == 4 ? NERVE.parameterCount4x : NERVE.parameterCount2x
        var line = "S0 \(ck.upstreamName) ×\(ck.scale): \(tensors.count) tensors · generated \(generated.count) · "
            + "missing \(missing.count) · unused \(unused.count) · shape≠ \(badShape.count) · dtypes \(dtypes.sorted()) · "
            + "\(params) params · layout \(meta["layout"] ?? "?")"
        var pass = missing.isEmpty && unused.isEmpty && badShape.isEmpty && dtypes == ["F32"]
            && tensors.count == 50 && params == expectParams && meta["layout"] == "OHWI"
            && Set(tensors.keys) == NERVE.expectedKeys()
        if let upstreamDir {
            let up = URL(fileURLWithPath: upstreamDir).appendingPathComponent("\(ck.upstreamName).safetensors")
            let (ut, _) = try safetensorsHeader(up)
            let sameKeys = Set(ut.keys) == Set(tensors.keys)
            let oihw = tensors.allSatisfy { k, v in
                guard let u = ut[k], v.shape.count == 4 else { return false }
                return u.shape == [v.shape[0], v.shape[3], v.shape[1], v.shape[2]] && u.dtype == v.dtype
            }
            line += " · upstream keys \(sameKeys ? "==" : "≠") · OIHW↔OHWI \(oihw ? "✓" : "✗") · src sha \(meta["source_sha256"]?.prefix(12) ?? "?")"
            pass = pass && sameKeys && oihw
        }
        note(line + (pass ? "  ✅" : "  ❌"))
        if !missing.isEmpty || !unused.isEmpty { note("   missing \(missing.prefix(5)) unused \(unused.prefix(5))") }
        ok = ok && pass
    }
    // The structural twin: the module tree's key set == the declared contract (weight-free, no eval).
    for s in [2, 4] {
        let keys = Set(NERVE(upscale: s).parameters().flattened().map(\.0))
        let pass = keys == NERVE.expectedKeys()
        note("S0 module tree ×\(s): \(keys.count) keys == contract \(pass ? "✅" : "❌")")
        ok = ok && pass
    }
    return ok
}

// MARK: - S1 — per-sub-op parity + probes

/// Tolerances: RELATIVE max error (max|Δ| / max|ref|), two tables (swift-port-parity "gate-threshold discipline").
///
/// ISOLATED — each sub-op fed the oracle's own input for it (`NERVE.isolatedTaps`): 2e-6 for the ≤64-channel
/// primitives (the 75-term stem, the 16-tap bicubic, the final add), 1e-5 for the 576-term 3×3 convs and the
/// 24-block chain, and the shuffle — a pure permutation — must be EXACT.
///
/// CHAINED — the model's own forward: the tail of the chain (block23 → body → head → shuffle) accumulates the
/// rounding of 50 convs and measured up to 9.6e-6 (head, 4x fidelity 64²) on the first run, i.e. ON a 1e-5 line.
/// A gate on the noise floor is a coin flip, so the chained tail is 2e-5 — and every loosened tolerance is held
/// against the failure it must still catch: the (r,r,C) shuffle probe (~1.4 rel) and the OWHI kernel probe (the
/// shape-identical spatial transpose of every 3×3 kernel) both print their margin against it.
let s1Chained: [String: Double] = [
    "stem": 2e-6, "block0": 1e-5, "block23": 2e-5, "body": 2e-5,
    "head": 2e-5, "shuffle": 2e-5, "bicubic": 2e-6, "output": 1e-5,
]
let s1Isolated: [String: Double] = [
    "stem": 2e-6, "block0": 1e-5, "block23": 1e-5, "body": 1e-5,
    "head": 1e-5, "shuffle": 0, "bicubic": 2e-6, "output": 2e-6,
]
let s1Order = ["stem", "block0", "block23", "body", "head", "shuffle", "bicubic", "output"]

/// The `(r, r, C)` misreading of the channel axis — shape-identical to the correct shuffle, and wrong.
func pixelShuffleWrongOrder(_ x: MLXArray, _ r: Int) -> MLXArray {
    let (b, h, w, crr) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let c = crr / (r * r)
    return x.reshaped([b, h, w, r, r, c]).transposed(0, 1, 3, 2, 4, 5).reshaped([b, h * r, w * r, c])
}

func gateS1(goldensDir: String, gpu: Bool, only: String?) throws -> Bool {
    let files = try FileManager.default.contentsOfDirectory(atPath: goldensDir)
        .filter { f in f.hasSuffix(".safetensors") && (only.map { f.hasPrefix($0) } ?? true) }.sorted()
    guard !files.isEmpty else { throw SmokeError("no S1 goldens in \(goldensDir)") }
    var ok = true
    var worst: [String: Double] = [:], worstIso: [String: Double] = [:]
    for f in files {
        let (g, meta) = try MLX.loadArraysAndMetadata(url: URL(fileURLWithPath: goldensDir).appendingPathComponent(f))
        guard let name = meta["checkpoint"], let ck = NERVE_Playback.Checkpoint.allCases.first(where: { $0.upstreamName == name }),
              let x = g["input"] else { throw SmokeError("\(f): bad golden") }
        let run = { () throws -> ([String: MLXArray], [String: MLXArray], MLXArray) in
            let model = try NERVE.load(from: ck.bundledWeightsURL!, expectedScale: ck.scale)
            let t = model.taps(x)
            let iso = model.isolatedTaps(input: x, golden: g)
            // Probe 3 — the OWHI misreading of every conv kernel (spatial transpose; shape-identical at 3×3).
            var owhi = try MLX.loadArrays(url: ck.bundledWeightsURL!)
            for (k, v) in owhi where k.hasPrefix("blocks.") { owhi[k] = v.transposed(0, 2, 1, 3) }
            let wrongBlock0 = try NERVE.load(owhi, expectedScale: ck.scale).isolatedTaps(input: x, golden: g)["block0"]!
            eval(Array(t.values), Array(iso.values), wrongBlock0)
            return (t, iso, wrongBlock0)
        }
        let (t, iso, owhiBlock0) = try gpu ? run() : Device.withDefaultDevice(Device(.cpu)) { try run() }
        var cells: [String] = [], isoCells: [String] = []
        for tap in s1Order {
            guard let ref = g[tap], let y = t[tap], let yi = iso[tap] else { throw SmokeError("\(f): missing tap \(tap)") }
            let d = diff(y, ref), di = diff(yi, ref)
            let pass = d.rel <= s1Chained[tap]! && y.shape == ref.shape
            let passIso = di.rel <= s1Isolated[tap]! && yi.shape == ref.shape
            worst[tap] = max(worst[tap] ?? 0, d.rel)
            worstIso[tap] = max(worstIso[tap] ?? 0, di.rel)
            ok = ok && pass && passIso
            cells.append("\(tap) \(fmtE(d.rel))\(pass ? "" : "❌")")
            isoCells.append("\(tap) \(fmtE(di.rel))\(passIso ? "" : "❌")")
        }
        // Second oracle: the author's ONNX export, end to end.
        let dOnnx = diff(t["output"]!, g["onnx"]!)
        let onnxPass = dOnnx.psnr >= 90
        ok = ok && onnxPass
        note("S1 \(f) [\(gpu ? "gpu" : "cpu")] chained: " + cells.joined(separator: " · ")
             + " · vs ONNX \(fmtF(dOnnx.psnr, 1)) dB\(onnxPass ? "" : "❌")")
        note("   isolated: " + isoCells.joined(separator: " · "))

        // Probe 1 — pixel-shuffle ordering: the (r, r, C) reading must fail loudly.
        let wrong = pixelShuffleWrongOrder(g["head"]!, ck.scale)
        let dWrong = diff(wrong, g["shuffle"]!)
        let dRight = diff(pixelShuffleNHWC(g["head"]!, ck.scale), g["shuffle"]!)
        let probe1 = dWrong.rel > 100 * s1Chained["shuffle"]!
        ok = ok && probe1 && dRight.maxAbs == 0
        // Probe 3 (reported beside 1) — OWHI kernels on block 0.
        let dOWHI = diff(owhiBlock0, g["block0"]!)
        let probe3 = dOWHI.rel > 100 * s1Chained["block23"]!
        ok = ok && probe3
        note("   probe OWHI kernels: block0 rel \(fmtE(dOWHI.rel)) = \(fmtF(dOWHI.rel / s1Chained["block23"]!, 0))× the chained tail tol "
             + "\(probe3 ? "✅" : "❌")")
        // Probe 2 — bicubic: error uniform (not border-concentrated), and alignCorners: true caught.
        let band = 2 * ck.scale
        let split = borderSplit(t["bicubic"]!, g["bicubic"]!, band: band)
        let ac = Upsample(scaleFactor: .float(Float(ck.scale)), mode: .cubic(alignCorners: true))(x)
        let dAC = diff(ac, g["bicubic"]!)
        // The MLXNN Upsample reference (what the sub-pixel form replaced) against the same golden.
        let dRef = diff(NERVE.bicubicReference(x, scale: ck.scale), g["bicubic"]!)
        let uniform = split.border <= max(4 * split.interior, 1e-6)
        let probe2 = uniform && dAC.psnr < 40 && dRef.rel <= s1Chained["bicubic"]!
        ok = ok && probe2
        note("   probes: shuffle (r,r,C) rel \(fmtE(dWrong.rel)) = \(fmtF(dWrong.rel / s1Chained["shuffle"]!, 0))× tol, "
             + "correct-order on the golden head max|Δ| \(dRight.maxAbs) \(probe1 && dRight.maxAbs == 0 ? "✅" : "❌") · "
             + "bicubic border(\(band)px) max \(fmtE(split.border)) vs interior \(fmtE(split.interior)) "
             + "(×\(fmtF(split.interior > 0 ? split.border / split.interior : 0, 2))), alignCorners:true "
             + "\(fmtF(dAC.psnr, 1)) dB · MLXNN Upsample reference rel \(fmtE(dRef.rel)) \(probe2 ? "✅" : "❌")")
    }
    note("S1 worst rel per tap, chained: " + s1Order.map { "\($0) \(fmtE(worst[$0] ?? .nan))" }.joined(separator: " · "))
    note("S1 worst rel per tap, isolated: " + s1Order.map { "\($0) \(fmtE(worstIso[$0] ?? .nan))" }.joined(separator: " · "))
    note(ok ? "✅ S1 PASSED (\(files.count) goldens)" : "❌ S1 FAILED")
    return ok
}

// MARK: - N2 — end to end vs torch and vs the author's ONNX

func gateE2E(goldensDir: String, gpu: Bool, only: String?, precision: NERVE_Playback.Precision,
             threshold: Double, compiled: Bool) throws -> Bool {
    let root = URL(fileURLWithPath: goldensDir)
    let ckDirs = try FileManager.default.contentsOfDirectory(atPath: goldensDir)
        .filter { d in NERVE_Playback.Checkpoint.allCases.contains { $0.upstreamName == d } && (only.map { d.hasPrefix($0) } ?? true) }
        .sorted()
    guard !ckDirs.isEmpty else { throw SmokeError("no e2e goldens in \(goldensDir)") }
    var ok = true
    for dir in ckDirs {
        let ck = NERVE_Playback.Checkpoint.allCases.first { $0.upstreamName == dir }!
        let cells = try FileManager.default.contentsOfDirectory(atPath: root.appendingPathComponent(dir).path)
            .filter { $0.hasSuffix(".safetensors") }.sorted()
        let body = { () throws -> (Double, Double, Double, Double, Int) in
            let model = try NERVE.load(from: ck.bundledWeightsURL!, expectedScale: ck.scale,
                                       dtype: precision == .fp32 ? nil : .float16)
            let fwd: (MLXArray) -> MLXArray = compiled ? compile { x in model(x) } : { x in model(x) }
            var worstTorch = Double.infinity, worstOnnx = Double.infinity, worstRel = 0.0, oracleFloor = Double.infinity
            for c in cells {
                let (g, meta) = try MLX.loadArraysAndMetadata(url: root.appendingPathComponent(dir).appendingPathComponent(c))
                let y = fwd(g["input"]!)
                eval(y)
                let dt = diff(y, g["torch"]!), dx = diff(y, g["onnx"]!)
                worstTorch = min(worstTorch, dt.psnr); worstOnnx = min(worstOnnx, dx.psnr)
                worstRel = max(worstRel, dt.rel)
                oracleFloor = min(oracleFloor, Double(meta["torch_vs_onnx_psnr"] ?? "") ?? .infinity)
                note("   \(dir)/\(c.dropLast(12)): vs torch \(fmtF(dt.psnr, 1)) dB (max|Δ| \(fmtE(dt.maxAbs))) · vs ONNX \(fmtF(dx.psnr, 1)) dB")
            }
            return (worstTorch, worstOnnx, worstRel, oracleFloor, cells.count)
        }
        let (wt, wo, wr, floor, n) = try gpu ? body() : Device.withDefaultDevice(Device(.cpu)) { try body() }
        let pass = wt >= threshold && wo >= threshold
        ok = ok && pass
        note("N2 \(dir) [\(gpu ? "gpu" : "cpu") \(precision.rawValue)\(compiled ? " compiled" : "")] \(n) cells: "
             + "worst vs torch \(fmtF(wt, 1)) dB · worst vs ONNX \(fmtF(wo, 1)) dB · worst rel \(fmtE(wr)) · "
             + "(torch↔ONNX floor \(fmtF(floor, 1)) dB) \(pass ? "✅" : "❌ < \(threshold) dB")")
    }
    note(ok ? "✅ N2 PASSED" : "❌ N2 FAILED")
    return ok
}
