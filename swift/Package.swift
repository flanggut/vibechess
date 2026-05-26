// swift-tools-version: 6.3

import PackageDescription

let package = Package(
    name: "TinyChess",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "TinyChessCore",
            targets: ["TinyChessCore"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/swiftlang/swift-testing.git", from: "0.12.0")
    ],
    targets: [
        .target(
            name: "TinyChessCore"
        ),
        .testTarget(
            name: "TinyChessCoreTests",
            dependencies: [
                "TinyChessCore",
                .product(name: "Testing", package: "swift-testing"),
            ]
        ),
    ]
)
