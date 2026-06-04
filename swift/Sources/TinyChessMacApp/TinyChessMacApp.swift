import SwiftUI

@main
struct TinyChessMacApp: App {
    var body: some Scene {
        WindowGroup("tinychess") {
            TinyChessMacRootView()
        }
    }
}

struct TinyChessMacRootView: View {
    var body: some View {
        VStack(spacing: 12) {
            Text("tinychess")
                .font(.largeTitle.monospaced())
            Text("Native macOS GUI skeleton")
                .font(.headline)
            Text("Backend integration will use `tinychess gui-server`.")
                .foregroundStyle(.secondary)
        }
        .padding(32)
        .frame(minWidth: 480, minHeight: 320)
    }
}
