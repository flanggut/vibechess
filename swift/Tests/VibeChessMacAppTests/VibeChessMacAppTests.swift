import Testing
@testable import VibeChessMacApp

@MainActor
@Test func appRootViewCanBeConstructed() {
    _ = VibeChessMacRootView(appState: AppState(backend: RootViewTestBackend()))
}

@Test func rootLayoutFitsBoardInsideDetailAreaAcrossWindowSizes() {
    #expect(
        RootLayoutMetrics.boardSide(in: .init(width: 520, height: 520)) == 472
    )
    #expect(
        RootLayoutMetrics.boardSide(in: .init(width: 720, height: 680)) == 632
    )
    #expect(
        RootLayoutMetrics.boardSide(in: .init(width: 1_100, height: 760)) == 712
    )
    #expect(
        RootLayoutMetrics.boardSide(in: .init(width: 2_000, height: 1_200))
            == RootLayoutMetrics.maximumBoardSide
    )
}

private actor RootViewTestBackend: BackendSession {
    func send(_ request: BackendRequest) async throws -> BackendResponse {
        BackendResponse(
            id: request.id,
            ok: true,
            state: nil,
            error: nil,
            version: nil,
            protocolVersion: nil,
            capabilities: nil,
            appliedMove: nil,
            search: nil,
            ai: nil
        )
    }
}
