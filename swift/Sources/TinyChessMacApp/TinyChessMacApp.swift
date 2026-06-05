import AppKit
import SwiftUI

@main
struct TinyChessMacApp: App {
    @NSApplicationDelegateAdaptor(TinyChessApplicationDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup("tinychess") {
            TinyChessMacRootView()
        }
    }
}

struct TinyChessMacRootView: View {
    @StateObject private var appState: AppState

    @MainActor
    init(appState: AppState? = nil) {
        _appState = StateObject(wrappedValue: appState ?? Self.makeDefaultAppState())
    }

    var body: some View {
        HStack(alignment: .top, spacing: 20) {
            BoardView(appState: appState, squareSize: 58)

            VStack(alignment: .leading, spacing: 14) {
                ControlsView(appState: appState)
                MoveListView(appState: appState)
            }
        }
        .padding(24)
        .frame(minWidth: 980, minHeight: 560)
        .task {
            guard appState.backendState == nil else {
                return
            }
            await appState.newGame()
        }
    }

    @MainActor
    private static func makeDefaultAppState() -> AppState {
        do {
            return try AppState()
        } catch {
            return AppState(backend: UnavailableBackend(errorDescription: String(describing: error)))
        }
    }
}

final class TinyChessApplicationDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
}

private actor UnavailableBackend: BackendSession {
    private let errorDescription: String

    init(errorDescription: String) {
        self.errorDescription = errorDescription
    }

    func send(_ request: BackendRequest) async throws -> BackendResponse {
        BackendResponse(
            id: request.id,
            ok: false,
            state: nil,
            error: BackendError(
                code: "backend_unavailable",
                message: "Could not start tinychess gui-server: \(errorDescription)"
            ),
            version: nil,
            protocolVersion: nil,
            capabilities: nil,
            appliedMove: nil,
            search: nil,
            ai: nil
        )
    }
}
