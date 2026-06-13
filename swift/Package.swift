// swift-tools-version: 6.3

import PackageDescription

let package = Package(
    name: "VibeChess",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "VibeChessCore",
            targets: ["VibeChessCore"]
        ),
        .executable(
            name: "VibeChessMacApp",
            targets: ["VibeChessMacApp"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/swiftlang/swift-testing.git", from: "0.12.0")
    ],
    targets: [
        .target(
            name: "VibeChessCore"
        ),
        .executableTarget(
            name: "VibeChessMacApp"
        ),
        .testTarget(
            name: "VibeChessCoreTests",
            dependencies: [
                "VibeChessCore",
                .product(name: "Testing", package: "swift-testing"),
            ]
        ),
        .testTarget(
            name: "VibeChessMacAppTests",
            dependencies: [
                "VibeChessMacApp",
                .product(name: "Testing", package: "swift-testing"),
            ]
        ),
    ]
)
