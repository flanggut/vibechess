import Testing
@testable import TinyChessCore

@Test func exposesBootstrapMetadata() {
    #expect(TinyChessCore.version == "0.1.0")
    #expect(TinyChessCore.squareIndexingConvention == "0..63 with a1 == 0 and h8 == 63")
    #expect(TinyChessCore.implementationStage == "bootstrap")
}
