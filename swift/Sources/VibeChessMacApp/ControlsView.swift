import SwiftUI

/// Controls and status sidebar for the native vibechess GUI.
struct ControlsView: View {
    @ObservedObject var appState: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if let errorMessage = appState.errorMessage {
                ErrorBanner(message: errorMessage) {
                    appState.clearError()
                }
            }

            StatusPanel(appState: appState)

            GroupBox("Game") {
                VStack(alignment: .leading, spacing: 10) {
                    Button("Start / Reset Game") {
                        Task { await appState.newGame() }
                    }
                    .disabled(appState.isThinking)
                    .keyboardShortcut("r", modifiers: [.command])

                    Button("Undo Last Full Move") {
                        Task { await appState.undo() }
                    }
                    .disabled(!appState.canUndo)
                    .keyboardShortcut("z", modifiers: [.command])

                    Picker("Human", selection: humanColorBinding) {
                        Text("White").tag(BackendColor.white)
                        Text("Black").tag(BackendColor.black)
                    }
                    .pickerStyle(.segmented)
                    .disabled(appState.isThinking)

                    Button("Flip Board") {
                        appState.flipBoard()
                    }
                    .disabled(appState.isThinking)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            GroupBox("AI") {
                VStack(alignment: .leading, spacing: 10) {
                    Picker("Player", selection: aiKindBinding) {
                        Text("Random").tag(BackendPlayerKind.random)
                        Text("MCTS").tag(BackendPlayerKind.mcts)
                        Text("Neural").tag(BackendPlayerKind.neural)
                    }
                    .pickerStyle(.segmented)
                    .disabled(appState.isThinking)

                    Stepper(value: simulationsBinding, in: 1...10_000, step: 1) {
                        Text("Simulations: \(simulationsBinding.wrappedValue)")
                    }
                    .disabled(appState.isThinking)

                    TextField("Time limit seconds (optional)", text: timeLimitBinding)
                        .textFieldStyle(.roundedBorder)
                        .disabled(appState.isThinking)

                    TextField("Node budget (optional)", text: nodeBudgetBinding)
                        .textFieldStyle(.roundedBorder)
                        .disabled(appState.isThinking)

                    if aiKindBinding.wrappedValue == .neural {
                        TextField("Checkpoint path", text: checkpointPathBinding)
                            .textFieldStyle(.roundedBorder)
                            .disabled(appState.isThinking)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .frame(width: 280, alignment: .topLeading)
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

private struct StatusPanel: View {
    @ObservedObject var appState: AppState

    var body: some View {
        GroupBox("Status") {
            VStack(alignment: .leading, spacing: 8) {
                Text(AppStatusPresenter.statusText(
                    state: appState.backendState,
                    isThinking: appState.isThinking,
                    humanColor: appState.humanColor
                ))
                .font(.headline)

                Text("Human: \(AppStatusPresenter.colorName(appState.humanColor))")
                    .foregroundStyle(.secondary)

                Text("Board: \(AppStatusPresenter.colorName(appState.boardOrientation)) at bottom")
                    .foregroundStyle(.secondary)

                if appState.isThinking {
                    ProgressView("Thinking…")
                        .controlSize(.small)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
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
