//
//  MaterializationTests.swift — the offline MAT gate (engine ≥ 0.24.0, contract 1.17 bundled vocabulary) for a
//  BUNDLED-WEIGHTS package, per variant × precision: MAT-1 store-stampable, MAT-2 bundled sources declared (one
//  role per native scale), MAT-3 hygiene, MAT-4 every vendored checkpoint verified PRESENT on a fresh
//  configuration — and the end-to-end symptom through the REAL engine: a fresh registration never reads as
//  needing a download.
//

import XCTest
import Foundation
import MLXToolKit
import MLXServeCore
import MLXServeConformance
import NERVEMLX
@testable import MLXNERVE

final class MaterializationTests: XCTestCase {

    func testFullMATGatePassesPerVariantAndPrecision() {
        for variant in NERVEVariant.allCases {
            for precision in NERVEPrecision.allCases {
                let report = MaterializationConformance.check(
                    freshConfiguration: NERVEConfiguration(variant: variant, precision: precision))
                XCTAssertTrue(report.passed, "\(variant)/\(precision):\n\(report.summary)")
            }
        }
    }

    /// One role per native scale, never a network source.
    func testBundledSourcesAreTheVariantsCheckpoints() {
        for variant in NERVEVariant.allCases {
            let cfg = NERVEConfiguration(variant: variant)
            let roles = cfg.bundledWeightSources.map(\.role)
            XCTAssertEqual(roles, variant.checkpoints.map { "checkpoint-x\($0.scale)" })
            XCTAssertFalse((cfg as Any) is WeightSourcing, "bundled-only: no network source may be declared")
        }
    }

    /// The checkpoints are actually IN the built bundle — a stripped resource must fail here, not at the first
    /// upscale — and they are the real 7 MB files, not stubs.
    func testEveryCheckpointIsBundled() {
        for ck in NERVE_Playback.Checkpoint.allCases {
            let url = ck.bundledWeightsURL
            XCTAssertNotNil(url, "\(ck) missing from bundle")
            if let url {
                let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
                XCTAssertGreaterThan(size, 7_000_000, "\(ck): the 7.1–7.2 MB checkpoint, not a stub")
            }
        }
    }

    func testFreshConfigurationPrewarmsFromBundle() {
        for variant in NERVEVariant.allCases {
            let paths = NERVEConfiguration(variant: variant).prewarmPaths
            XCTAssertEqual(paths.count, variant.checkpoints.count)
            XCTAssertTrue(paths.allSatisfy { FileManager.default.fileExists(atPath: $0.path) }, "\(variant)")
        }
    }

    /// Register is offline — no weights load until `prepare`/`run`.
    func testEngineNeedsDownloadIsFalseOnFreshRegistration() async throws {
        for variant in NERVEVariant.allCases {
            let engine = MLXServeEngine()
            _ = try await engine.register(NERVEUpscalePackage.registration,
                                          configuration: NERVEConfiguration(variant: variant))
            let needs = await engine.needsDownload(.imageUpscale)
            XCTAssertFalse(needs, "\(variant): bundled weights present, nothing to download")
        }
    }
}
