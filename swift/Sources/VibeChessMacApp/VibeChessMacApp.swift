import AppKit
import SwiftUI

@main
struct VibeChessMacApp: App {
    @NSApplicationDelegateAdaptor(VibeChessApplicationDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup("vibechess") {
            VibeChessMacRootView()
        }
        .defaultSize(width: 1_100, height: 760)
        .windowResizability(.contentMinSize)
    }
}

struct VibeChessMacRootView: View {
    @StateObject private var appState: AppState

    @MainActor
    init(appState: AppState? = nil) {
        _appState = StateObject(wrappedValue: appState ?? Self.makeDefaultAppState())
    }

    var body: some View {
        NavigationSplitView {
            ControlsView(appState: appState)
                .navigationSplitViewColumnWidth(
                    min: RootLayoutMetrics.minimumSidebarWidth,
                    ideal: RootLayoutMetrics.idealSidebarWidth,
                    max: RootLayoutMetrics.maximumSidebarWidth
                )
        } detail: {
            GeometryReader { geometry in
                let boardSide = RootLayoutMetrics.boardSide(in: geometry.size)

                ZStack {
                    Color(nsColor: .windowBackgroundColor)

                    BoardView(appState: appState, squareSize: boardSide / 8)
                        .overlay {
                            Rectangle()
                                .stroke(Color(nsColor: .separatorColor), lineWidth: 0.5)
                        }
                        .shadow(color: .black.opacity(0.16), radius: 10, y: 4)
                }
            }
            .navigationTitle("Chessboard")
        }
        .navigationSplitViewStyle(.balanced)
        .frame(
            minWidth: RootLayoutMetrics.minimumWindowWidth,
            minHeight: RootLayoutMetrics.minimumWindowHeight
        )
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

/// Native split-view sizing values kept independent from SwiftUI rendering for focused tests.
enum RootLayoutMetrics {
    static let minimumWindowWidth: CGFloat = 720
    static let minimumWindowHeight: CGFloat = 520
    static let minimumSidebarWidth: CGFloat = 270
    static let idealSidebarWidth: CGFloat = 310
    static let maximumSidebarWidth: CGFloat = 380
    static let boardPadding: CGFloat = 24
    static let maximumBoardSide: CGFloat = 800

    static func boardSide(in detailSize: CGSize) -> CGFloat {
        let availableWidth = max(0, detailSize.width - boardPadding * 2)
        let availableHeight = max(0, detailSize.height - boardPadding * 2)
        return min(availableWidth, availableHeight, maximumBoardSide).rounded(.down)
    }
}

final class VibeChessApplicationDelegate: NSObject, NSApplicationDelegate {
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
                message: "Could not start vibechess gui-server: \(errorDescription)"
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
