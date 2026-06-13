import Testing
@testable import VibeChessCore

@Test func exposesBootstrapMetadata() {
    #expect(VibeChessCore.version == "0.1.0")
    #expect(VibeChessCore.squareIndexingConvention == "0..63 with a1 == 0 and h8 == 63")
    #expect(VibeChessCore.implementationStage == "bootstrap")
}
