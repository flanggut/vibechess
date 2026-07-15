import SwiftUI

/// Controls and status sidebar for the native vibechess GUI.
struct ControlsView: View {
    @ObservedObject var appState: AppState

    var body: some View {
        Form {
            if let errorMessage = appState.errorMessage {
                Section {
                    ErrorBanner(message: errorMessage) {
                        appState.clearError()
                    }
                }
            }

            statusSection
            gameSection
            computerSection

            Section("Moves") {
                MoveListView(appState: appState)
            }
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .controlSize(.small)
        .navigationTitle("Game")
        .accessibilityLabel("Game controls")
    }

    private var statusSection: some View {
        Section("Status") {
            Text(AppStatusPresenter.statusText(
                state: appState.backendState,
                isThinking: appState.isThinking,
                humanColor: appState.humanColor
            ))
            .font(.headline)
            .fixedSize(horizontal: false, vertical: true)

            LabeledContent(
                "Human",
                value: AppStatusPresenter.colorName(appState.humanColor)
            )
            LabeledContent(
                "Board",
                value: "\(AppStatusPresenter.colorName(appState.boardOrientation)) at bottom"
            )

            if appState.isThinking {
                HStack(spacing: 8) {
                    ProgressView()
                        .controlSize(.small)
                    Text("Thinking…")
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var gameSection: some View {
        Section("Game") {
            Button {
                Task { await appState.newGame() }
            } label: {
                Label("New Game", systemImage: "arrow.clockwise")
            }
            .disabled(appState.isThinking)
            .keyboardShortcut("r", modifiers: [.command])

            Button {
                Task { await appState.undo() }
            } label: {
                Label("Undo Full Move", systemImage: "arrow.uturn.backward")
            }
            .disabled(!appState.canUndo)
            .keyboardShortcut("z", modifiers: [.command])

            VStack(alignment: .leading, spacing: 6) {
                Text("Play as")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Picker("Play as", selection: humanColorBinding) {
                    Text("White").tag(BackendColor.white)
                    Text("Black").tag(BackendColor.black)
                }
                .labelsHidden()
                .pickerStyle(.segmented)
            }
            .disabled(appState.isThinking)

            Button {
                appState.flipBoard()
            } label: {
                Label("Flip Board", systemImage: "arrow.triangle.2.circlepath")
            }
            .disabled(appState.isThinking)
        }
    }

    private var computerSection: some View {
        Section("Computer") {
            Picker("Player", selection: aiKindBinding) {
                Text("Random").tag(BackendPlayerKind.random)
                Text("MCTS").tag(BackendPlayerKind.mcts)
                Text("Neural").tag(BackendPlayerKind.neural)
            }
            .pickerStyle(.menu)

            Stepper(value: simulationsBinding, in: 1...10_000, step: 1) {
                LabeledContent(
                    "Simulations",
                    value: simulationsBinding.wrappedValue.formatted()
                )
            }

            LabeledContent("Time limit") {
                TextField("None", text: timeLimitBinding)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 96)
            }

            LabeledContent("Node budget") {
                TextField("None", text: nodeBudgetBinding)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 96)
            }

            if aiKindBinding.wrappedValue == .neural {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Checkpoint")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    TextField("Checkpoint path", text: checkpointPathBinding)
                }
            }
        }
        .disabled(appState.isThinking)
    }

    private var humanColorBinding: Binding<BackendColor> {
        Binding(
            get: { appState.humanColor },
            set: { appState.updateHumanColor($0) }
        )
    }

    private var aiKindBinding: Binding<BackendPlayerKind> {
        Binding(
            get: { appState.aiConfig.kind ?? .neural },
            set: { kind in updateAIConfig { $0.kind = kind } }
        )
    }

    private var simulationsBinding: Binding<Int> {
        Binding(
            get: { max(1, appState.aiConfig.simulations ?? 200) },
            set: { simulations in updateAIConfig { $0.simulations = max(1, simulations) } }
        )
    }

    private var timeLimitBinding: Binding<String> {
        Binding(
            get: { AppStatusPresenter.optionalNumberText(appState.aiConfig.timeLimitSeconds) },
            set: { text in updateAIConfig { $0.timeLimitSeconds = AppStatusPresenter.parseOptionalDouble(text) } }
        )
    }

    private var nodeBudgetBinding: Binding<String> {
        Binding(
            get: { AppStatusPresenter.optionalNumberText(appState.aiConfig.nodeBudget) },
            set: { text in updateAIConfig { $0.nodeBudget = AppStatusPresenter.parseOptionalInt(text) } }
        )
    }

    private var checkpointPathBinding: Binding<String> {
        Binding(
            get: { appState.aiConfig.checkpointPath ?? "" },
            set: { text in
                updateAIConfig {
                    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                    $0.checkpointPath = trimmed.isEmpty ? nil : trimmed
                }
            }
        )
    }

    private func updateAIConfig(_ update: (inout BackendAIConfig) -> Void) {
        var config = appState.aiConfig
        update(&config)
        appState.updateAIConfig(config)
    }
}


private struct ErrorBanner: View {
    var message: String
    var onDismiss: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                Text(message)
                    .font(.callout)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer()
            }
            Button("Dismiss", action: onDismiss)
                .controlSize(.small)
        }
        .padding(10)
        .background(Color.red.opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.red.opacity(0.35), lineWidth: 1)
        )
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Error: \(message)")
    }
}

/// Pure presentation helpers for status text and control parsing.
enum AppStatusPresenter {
    static func statusText(
        state: BackendState?,
        isThinking: Bool,
        humanColor: BackendColor
    ) -> String {
        if isThinking {
            return "AI thinking…"
        }
        guard let state else {
            return "Start a game to connect to the backend."
        }
        if let outcome = state.outcome {
            return outcomeText(outcome)
        }
        let side = colorName(state.sideToMove)
        let turnOwner = state.sideToMove == humanColor ? "human" : "AI"
        return "\(side) to move (\(turnOwner))"
    }

    static func outcomeText(_ outcome: BackendOutcome) -> String {
        switch outcome.reason {
        case .checkmate:
            if let winner = outcome.winner {
                return "Checkmate — \(colorName(winner)) wins"
            }
            return "Checkmate"
        case .stalemate:
            return "Draw by stalemate"
        case .fiftyMove:
            return "Draw by fifty-move rule"
        case .repetition:
            return "Draw by repetition"
        case .insufficientMaterial:
            return "Draw by insufficient material"
        case .maxPlies:
            return "Draw by maximum plies"
        }
    }

    static func colorName(_ color: BackendColor) -> String {
        switch color {
        case .white: "White"
        case .black: "Black"
        }
    }

    static func optionalNumberText<T: CustomStringConvertible>(_ value: T?) -> String {
        guard let value else {
            return ""
        }
        return value.description
    }

    static func parseOptionalInt(_ text: String) -> Int? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return nil
        }
        guard let value = Int(trimmed), value > 0 else {
            return nil
        }
        return value
    }

    static func parseOptionalDouble(_ text: String) -> Double? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return nil
        }
        guard let value = Double(trimmed), value >= 0, value.isFinite else {
            return nil
        }
        return value
    }
}
