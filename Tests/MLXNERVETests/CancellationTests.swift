//
//  CancellationTests.swift — the offline CAN gate (engine ≥ 0.27.0): CAN-1/CAN-2 pre-cancelled run() propagation
//  and classification, and CAN-3 the checkpoint-cadence declaration of record. The MID-RUN half (a cancel landing
//  between tiles surfaces within one tile, unwrapped) is proven live in LiveCPUTests and `nerve-smoke cancel`.
//

import XCTest
import Foundation
import MLXToolKit
import MLXServeConformance
@testable import MLXNERVE

final class CancellationTests: XCTestCase {

    func testCANGatePreCancelledRunEveryVariant() async {
        // Construction is cheap (C13) and the entry checkpoint throws before validation, decode, or the (bundled)
        // weights are touched — offline-safe, and it holds before load() as well as after.
        for variant in NERVEVariant.allCases {
            let package = NERVEUpscalePackage(configuration: NERVEConfiguration(variant: variant))
            let report = await CancellationConformance.checkRun(
                package: package,
                request: ImageUpscaleRequest(image: Image(format: .png, data: Data())))
            XCTAssertTrue(report.passed, "\(variant): \(report.summary)")
        }
    }

    func testCANCadenceDeclaration() {
        let report = CancellationConformance.checkCadence(
            manifest: NERVEUpscalePackage.manifest,
            posture: .cadence([
                // The shared tile driver checks Task.checkCancellation once per tile (MLXTileProcessor.process,
                // top of the tile loop) and the package reports RunProgress(.upsample, step: tile, totalSteps:)
                // after each one — "chunk" = one tile. The ≤ wholeFrameMaxPixels whole-frame path is one MLX eval
                // (≈ 1 s at 1080p → 4K); there the entry checkpoint and the pre-resize checkpoint are the seams.
                .init(phase: .upsample, unit: .chunk, reportsRunProgress: true),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
