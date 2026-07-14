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
        GeometryReader { geometry in
            let mode = RootLayoutMetrics.mode(for: geometry.size.width)
            let boardSide = RootLayoutMetrics.boardSide(in: geometry.size, mode: mode)

            switch mode {
            case .wide:
                HStack(alignment: .top, spacing: RootLayoutMetrics.contentSpacing) {
                    BoardView(appState: appState, squareSize: boardSide / 8)

                    ScrollView {
                        sidePanel
                    }
                    .frame(width: RootLayoutMetrics.sidePanelWidth, height: boardSide)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
                .padding(RootLayoutMetrics.outerPadding)

            case .compact:
                ScrollView {
                    VStack(spacing: RootLayoutMetrics.contentSpacing) {
                        BoardView(appState: appState, squareSize: boardSide / 8)

                        sidePanel
                            .frame(maxWidth: RootLayoutMetrics.compactPanelMaxWidth)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(RootLayoutMetrics.outerPadding)
                }
            }
        }
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

    private var sidePanel: some View {
        VStack(alignment: .leading, spacing: 14) {
            ControlsView(appState: appState)
            MoveListView(appState: appState)
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
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

enum RootLayoutMode: Equatable {
    case compact
    case wide
}

/// Window-size-driven layout values kept independent from SwiftUI rendering for focused tests.
enum RootLayoutMetrics {
    static let minimumWindowWidth: CGFloat = 520
    static let minimumWindowHeight: CGFloat = 520
    static let wideBreakpoint: CGFloat = 940
    static let outerPadding: CGFloat = 20
    static let contentSpacing: CGFloat = 20
    static let sidePanelWidth: CGFloat = 320
    static let compactPanelMaxWidth: CGFloat = 560
    static let maximumBoardSide: CGFloat = 720

    static func mode(for containerWidth: CGFloat) -> RootLayoutMode {
        containerWidth >= wideBreakpoint ? .wide : .compact
    }

    static func boardSide(in containerSize: CGSize, mode: RootLayoutMode) -> CGFloat {
        let availableWidth = max(0, containerSize.width - outerPadding * 2)
        let widthLimit: CGFloat

        switch mode {
        case .wide:
            widthLimit = max(0, availableWidth - contentSpacing - sidePanelWidth)
        case .compact:
            widthLimit = availableWidth
        }

        let availableHeight = max(0, containerSize.height - outerPadding * 2)
        return min(widthLimit, availableHeight, maximumBoardSide).rounded(.down)
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
