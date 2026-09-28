// swift-tools-version: 6.2
import PackageDescription

// mlx-nerve-swift — NERVE ×2/×4 super-resolution for MLXEngine: Phips/NERVE (Philip Hofmann; Apache-2.0 code +
// weights, trained only on the CC0 LUCID corpus ← nyuuzyou/pxhere ← pxhere.com), the provenance-clean fast
// `imageUpscale` tier. ONE repo, TWO products, the mlx-realesrgan-swift shape:
//   • NERVEMLX — engine-agnostic Swift/MLX core: the network (isomorphic to upstream `nerve_arch.py` @ c23588c36988)
//     + a `PlaybackTier` over the SHARED tile driver from RealESRGANMLX
//   • MLXNERVE — the MLXEngine `imageUpscale` ModelPackage over that core
// Vendored weights (the five released checkpoints, 7.1–7.2 MB fp32 each, MLX layout) — no download.
// Materialization posture: BUNDLED-WEIGHTS (ModelStorable + BundledWeightSourcing; the MAT gate verifies the
// selected variant's checkpoints PRESENT on a fresh machine; needsDownload reads false). WeightSourcing
// deliberately NOT declared. Cancellation posture (CAN gate): entry checkpoint first act of run(); the shared tile
// driver checkpoints once per tile, and RunProgress(.upsample) is reported per tile. Licences: weights Apache-2.0
// (Phips), port code Apache-2.0 (derived from nerve_arch.py). See README.md.
let package = Package(
    name: "mlx-nerve-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "NERVEMLX", targets: ["NERVEMLX"]),
        .library(name: "MLXNERVE", targets: ["MLXNERVE"]),
        .executable(name: "nerve-smoke", targets: ["NERVESmoke"]),  // gates + drive the package + measure
    ],
    dependencies: [
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.56.0"),
        .package(url: "https://github.com/xocialize/mlx-realesrgan-swift", from: "0.7.0"),  // MLXTileProcessor + PlaybackTier
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
    ],
    targets: [
        // Engine-agnostic core — NO MLXToolKit dep. Reuses the shipped tile driver rather than forking it.
        .target(
            name: "NERVEMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "RealESRGANMLX", package: "mlx-realesrgan-swift"),
            ],
            // Per-file .copy so the bundle layout is flat (forge ADR-0011).
            resources: [
                .copy("Resources/4x_NERVE_OTF_fidelity-mlx.safetensors"),
                .copy("Resources/4x_NERVE_release-mlx.safetensors"),
                .copy("Resources/2x_NERVE_release-mlx.safetensors"),
                .copy("Resources/4x_NERVE_OTF_gan-mlx.safetensors"),
                .copy("Resources/2x_NERVE_OTF_gan-mlx.safetensors"),
            ]
        ),
        // MLXEngine `imageUpscale` wrapper over the local core.
        .target(
            name: "MLXNERVE",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                "NERVEMLX",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            // The core's playback tier isn't Sendable-audited; the engine serializes lifecycle on
            // InferenceActor, so v5 mode keeps region-isolation a warning (same posture as the siblings).
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "NERVEMLXTests",
            dependencies: [
                "NERVEMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            resources: [
                .copy("Resources/goldens_s1_37x53.safetensors"),
            ]
        ),
        .testTarget(
            name: "MLXNERVETests",
            dependencies: [
                "MLXNERVE",
                "NERVEMLX",
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformanceNN", package: "mlx-engine-swift"),
            ]
        ),
        // The CLI lane: S0/S1/N2 gates (CPU or GPU stream), the N3–N5 studies, and NERVEUpscalePackage driven
        // through the REAL MLXServeEngine (register → run) with the split-footprint memory report.
        .executableTarget(
            name: "NERVESmoke",
            dependencies: [
                "MLXNERVE",
                "NERVEMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "RealESRGANMLX", package: "mlx-realesrgan-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ],
            path: "Sources/Smoke",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
