import Testing
@testable import VibeChessMacApp

@MainActor
@Test func appRootViewCanBeConstructed() {
    _ = VibeChessMacRootView(appState: AppState(backend: RootViewTestBackend()))
}

@Test func rootLayoutSwitchesBetweenCompactAndWideModesAtBreakpoint() {
    #expect(
        RootLayoutMetrics.mode(for: RootLayoutMetrics.wideBreakpoint - 1) == .compact
    )
    #expect(
        RootLayoutMetrics.mode(for: RootLayoutMetrics.wideBreakpoint) == .wide
    )
}

@Test func rootLayoutKeepsBoardVisibleAndBoundedAcrossWindowSizes() {
    #expect(
        RootLayoutMetrics.boardSide(
            in: .init(width: 520, height: 520),
            mode: .compact
        ) == 480
    )
    #expect(
        RootLayoutMetrics.boardSide(
            in: .init(width: 720, height: 680),
            mode: .compact
        ) == 640
    )
    #expect(
        RootLayoutMetrics.boardSide(
            in: .init(width: 1_100, height: 760),
            mode: .wide
        ) == 720
    )
    #expect(
        RootLayoutMetrics.boardSide(
            in: .init(width: 2_000, height: 1_200),
            mode: .wide
        ) == RootLayoutMetrics.maximumBoardSide
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
