import Testing
@testable import TinyChessMacApp

@MainActor
@Test func appRootViewCanBeConstructed() {
    _ = TinyChessMacRootView(appState: AppState(backend: RootViewTestBackend()))
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
