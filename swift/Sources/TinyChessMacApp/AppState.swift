import Combine
import Foundation

/// Mockable backend seam used by the GUI app state.
protocol BackendSession: Sendable {
    func send(_ request: BackendRequest) async throws -> BackendResponse
}

extension BackendClient: BackendSession {}

/// Main-actor view model for the native macOS GUI.
@MainActor
final class AppState: ObservableObject {
    @Published private(set) var backendState: BackendState?
    @Published private(set) var selectedSquare: String?
    @Published private(set) var isThinking = false
    @Published private(set) var errorMessage: String?
    @Published private(set) var lastAppliedMove: String?

    @Published private(set) var humanColor: BackendColor
    @Published private(set) var aiConfig: BackendAIConfig
    @Published private(set) var boardOrientation: BackendColor

    private let backend: any BackendSession
    private var nextID = 1

    init(
        backend: any BackendSession,
        initialState: BackendState? = nil,
        humanColor: BackendColor = .white,
        aiConfig: BackendAIConfig = BackendAIConfig(kind: .random, simulations: 25),
        boardOrientation: BackendColor = .white
    ) {
        self.backend = backend
        self.backendState = initialState
        self.humanColor = humanColor
        self.aiConfig = aiConfig
        self.boardOrientation = boardOrientation
    }

    convenience init(
        command: BackendProcessCommand = .developmentDefault,
        initialState: BackendState? = nil
    ) throws {
        self.init(backend: try BackendClient(command: command), initialState: initialState)
    }

    var piecesBySquare: [String: BackendPiece] {
        Dictionary(uniqueKeysWithValues: (backendState?.squares ?? []).map { ($0.square, $0) })
    }

    var legalDestinationsByFrom: [String: [String]] {
        backendState?.legalDestinationsByFrom ?? [:]
    }

    var moveHistory: [String] {
        backendState?.moves ?? []
    }

    var legalDestinationsForSelectedSquare: [String] {
        guard let selectedSquare else {
            return []
        }
        return legalDestinationsByFrom[selectedSquare] ?? []
    }

    var canUndo: Bool {
        !moveHistory.isEmpty && !isThinking
    }

    func updateHumanColor(_ color: BackendColor) {
        guard !isThinking else {
            return
        }
        humanColor = color
        boardOrientation = color
    }

    func updateAIConfig(_ config: BackendAIConfig) {
        guard !isThinking else {
            return
        }
        aiConfig = config
    }

    func flipBoard() {
        guard !isThinking else {
            return
        }
        boardOrientation = boardOrientation == .white ? .black : .white
    }

    func clearError() {
        errorMessage = nil
    }

    func selectSquare(_ square: String) {
        guard !isThinking else {
            return
        }
        if selectedSquare == square {
            selectedSquare = nil
            return
        }
        selectedSquare = legalDestinationsByFrom[square] == nil ? nil : square
    }

    func newGame(
        humanColor requestedHumanColor: BackendColor? = nil,
        aiConfig requestedAIConfig: BackendAIConfig? = nil
    ) async {
        guard beginBackendOperation() else {
            return
        }
        defer { finishBackendOperation() }

        let nextHumanColor = requestedHumanColor ?? humanColor
        let nextAIConfig = requestedAIConfig ?? aiConfig
        let request = BackendRequest(
            id: nextRequestID(),
            cmd: .newGame,
            humanColor: nextHumanColor,
            ai: nextAIConfig
        )
        guard await sendAndApply(request) != nil else {
            return
        }
        humanColor = nextHumanColor
        aiConfig = nextAIConfig
        boardOrientation = nextHumanColor
        selectedSquare = nil
        lastAppliedMove = nil
    }

    func makeSelectedMove(to destinationSquare: String) async {
        guard let selectedSquare else {
            return
        }
        await makeMove(from: selectedSquare, to: destinationSquare)
    }

    func makeMove(from sourceSquare: String, to destinationSquare: String) async {
        await makeMove("\(sourceSquare)\(destinationSquare)")
    }

    func makeMove(_ move: String) async {
        guard beginBackendOperation() else {
            return
        }
        defer { finishBackendOperation() }

        let request = BackendRequest(id: nextRequestID(), cmd: .makeMove, move: move)
        guard await sendAndApply(request) != nil else {
            return
        }
        selectedSquare = nil
    }

    func requestAIMove(aiConfig requestedAIConfig: BackendAIConfig? = nil) async {
        guard beginBackendOperation() else {
            return
        }
        defer { finishBackendOperation() }

        let config = requestedAIConfig ?? aiConfig
        let request = BackendRequest(id: nextRequestID(), cmd: .aiMove, ai: config)
        guard await sendAndApply(request) != nil else {
            return
        }
        aiConfig = config
        selectedSquare = nil
    }

    func undo(plies: Int = 2) async {
        guard beginBackendOperation() else {
            return
        }
        defer { finishBackendOperation() }

        let request = BackendRequest(id: nextRequestID(), cmd: .undo, plies: plies)
        guard await sendAndApply(request) != nil else {
            return
        }
        selectedSquare = nil
        lastAppliedMove = nil
    }

    private func beginBackendOperation() -> Bool {
        guard !isThinking else {
            return false
        }
        isThinking = true
        errorMessage = nil
        return true
    }

    private func finishBackendOperation() {
        isThinking = false
    }

    private func nextRequestID() -> BackendMessageID {
        defer { nextID += 1 }
        return .int(nextID)
    }

    private func sendAndApply(_ request: BackendRequest) async -> BackendResponse? {
        do {
            let response = try await backend.send(request)
            if response.ok {
                apply(response)
                return response
            }
            applyBackendRejection(response)
        } catch BackendClientError.backendRejected(let response) {
            applyBackendRejection(response)
        } catch {
            errorMessage = String(describing: error)
        }
        return nil
    }

    private func apply(_ response: BackendResponse) {
        if let state = response.state {
            backendState = state
        }
        if let appliedMove = response.appliedMove {
            lastAppliedMove = appliedMove
        }
        if let ai = response.ai {
            aiConfig = ai
        }
        errorMessage = nil
    }

    private func applyBackendRejection(_ response: BackendResponse) {
        if let state = response.state {
            backendState = state
        }
        if let error = response.error {
            errorMessage = "\(error.code): \(error.message)"
        } else {
            errorMessage = "backend rejected request"
        }
    }
}
