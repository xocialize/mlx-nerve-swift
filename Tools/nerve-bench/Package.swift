// swift-tools-version: 6.2
import PackageDescription

// nerve-bench — the wall-time harness for mlx-nerve-swift's N3 / N5 / N6 decisions, kept OUT of the published
// package so its dependency graph never grows a sibling-tier dependency: it times NERVE beside the tiers it
// replaces / sits under (Real-ESRGAN `general`, RealPLKSR — the latter also the calibration anchor, 111 ms per
// output Mpx on an idle GPU, AB-R-0365), interleaving arms and bracketing every run with the AGX utilisation
// counter (memory `shared-machine-timing`).
//
//   swift build -c release --build-system swiftbuild --target NERVEBench   → .build/out/Products/Release/nerve-bench
let package = Package(
    name: "nerve-bench",
    platforms: [.macOS(.v26)],
    products: [
        .executable(name: "nerve-bench", targets: ["NERVEBench"]),
    ],
    dependencies: [
        .package(path: "../.."),  // mlx-nerve-swift
        .package(url: "https://github.com/xocialize/mlx-realplksr-swift", from: "0.1.0"),
        .package(url: "https://github.com/xocialize/mlx-realesrgan-swift", from: "0.7.0"),
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.56.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
    ],
    targets: [
        .executableTarget(
            name: "NERVEBench",
            dependencies: [
                .product(name: "NERVEMLX", package: "mlx-nerve-swift"),
                .product(name: "MLXNERVE", package: "mlx-nerve-swift"),
                .product(name: "RealPLKSRMLX", package: "mlx-realplksr-swift"),
                .product(name: "RealESRGANMLX", package: "mlx-realesrgan-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
