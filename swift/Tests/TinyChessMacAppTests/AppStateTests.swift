import Foundation
import Testing
@testable import TinyChessMacApp

@MainActor
@Test func appStateExposesInitialBoardAndSelectionState() async throws {
    let appState = AppState(backend: MockBackend(), initialState: startingState())

    #expect(appState.backendState?.fen == startingFEN)
    #expect(appState.piecesBySquare["e2"]?.piece == "P")
    #expect(Set(appState.legalDestinationsByFrom["e2"] ?? []) == ["e3", "e4"])
    #expect(appState.moveHistory.isEmpty)
    #expect(appState.selectedSquare == nil)
    #expect(appState.humanColor == .white)
    #expect(appState.boardOrientation == .white)
    #expect(appState.isThinking == false)
    #expect(appState.errorMessage == nil)
}

@MainActor
@Test func appStateSelectsOnlySquaresWithLegalDestinationsAndFlipsBoard() async throws {
    let appState = AppState(backend: MockBackend(), initialState: startingState())

    appState.selectSquare("a1")
    #expect(appState.selectedSquare == nil)

    appState.selectSquare("e2")
    #expect(appState.selectedSquare == "e2")
    #expect(Set(appState.legalDestinationsForSelectedSquare) == ["e3", "e4"])

    appState.selectSquare("e2")
    #expect(appState.selectedSquare == nil)

    appState.flipBoard()
    #expect(appState.boardOrientation == .black)

    appState.updateHumanColor(.black)
    #expect(appState.humanColor == .black)
    #expect(appState.boardOrientation == .black)
}

@MainActor
@Test func appStateNewGameSendsConfigAndAppliesReturnedState() async throws {
    let backend = MockBackend(
        responses: [
            BackendResponse.success(id: .int(1), state: startingState(sideToMove: .black))
        ]
    )
    let appState = AppState(backend: backend, initialState: playedState())
    let aiConfig = BackendAIConfig(kind: .mcts, simulations: 3, nodeBudget: 5)

    await appState.newGame(humanColor: .black, aiConfig: aiConfig)

    let requests = await backend.requests
    #expect(requests.count == 1)
    #expect(requests[0].cmd == .newGame)
    #expect(requests[0].humanColor == .black)
    #expect(requests[0].ai == aiConfig)
    #expect(appState.humanColor == .black)
    #expect(appState.aiConfig == aiConfig)
    #expect(appState.boardOrientation == .black)
    #expect(appState.backendState?.sideToMove == .black)
    #expect(appState.selectedSquare == nil)
    #expect(appState.lastAppliedMove == nil)
    #expect(appState.isThinking == false)
}

@MainActor
@Test func appStateMakeSelectedMoveAppliesResponseAndClearsSelection() async throws {
    let backend = MockBackend(
        responses: [
            BackendResponse.success(
                id: .int(1),
                state: playedState(sideToMove: .white),
                appliedMove: "e2e4"
            )
        ]
    )
    let appState = AppState(backend: backend, initialState: startingState())
    appState.selectSquare("e2")

    await appState.makeSelectedMove(to: "e4")

    let requests = await backend.requests
    #expect(requests.count == 1)
    #expect(requests[0].cmd == .makeMove)
    #expect(requests[0].move == "e2e4")
    #expect(appState.backendState?.moves == ["e2e4"])
    #expect(appState.lastAppliedMove == "e2e4")
    #expect(appState.selectedSquare == nil)
}

@MainActor
@Test func appStateWhiteHumanMoveAutomaticallyRequestsAIMove() async throws {
    let backend = MockBackend(
        responses: [
            BackendResponse.success(
                id: .int(1),
                state: playedState(),
                appliedMove: "e2e4"
            ),
            BackendResponse.success(
                id: .int(2),
                state: aiReplyState(),
                appliedMove: "e7e5"
            ),
        ]
    )
    let aiConfig = BackendAIConfig(kind: .random, seed: 7)
    let appState = AppState(backend: backend, initialState: startingState(), aiConfig: aiConfig)

    await appState.makeMove("e2e4")

    let requests = await backend.requests
    #expect(requests.map(\.cmd) == [.makeMove, .aiMove])
    #expect(requests[0].move == "e2e4")
    #expect(requests[1].ai == aiConfig)
    #expect(appState.backendState?.moves == ["e2e4", "e7e5"])
    #expect(appState.backendState?.sideToMove == .white)
    #expect(appState.lastAppliedMove == "e7e5")
    #expect(appState.selectedSquare == nil)
    #expect(appState.isThinking == false)
}

@MainActor
@Test func appStateBlackHumanNewGameAutomaticallyRequestsOpeningAIMove() async throws {
    let backend = MockBackend(
        responses: [
            BackendResponse.success(id: .int(1), state: startingState(sideToMove: .white)),
            BackendResponse.success(
                id: .int(2),
                state: playedState(moves: ["g1f3"], lastMove: "g1f3", sideToMove: .black),
                appliedMove: "g1f3"
            ),
        ]
    )
    let aiConfig = BackendAIConfig(kind: .mcts, simulations: 3, nodeBudget: 5)
    let appState = AppState(backend: backend, initialState: playedState())

    await appState.newGame(humanColor: .black, aiConfig: aiConfig)

    let requests = await backend.requests
    #expect(requests.map(\.cmd) == [.newGame, .aiMove])
    #expect(requests[0].humanColor == .black)
    #expect(requests[0].ai == aiConfig)
    #expect(requests[1].ai == aiConfig)
    #expect(appState.humanColor == .black)
    #expect(appState.boardOrientation == .black)
    #expect(appState.aiConfig == aiConfig)
    #expect(appState.backendState?.sideToMove == .black)
    #expect(appState.backendState?.moves == ["g1f3"])
    #expect(appState.lastAppliedMove == "g1f3")
    #expect(appState.isThinking == false)
}

@MainActor
@Test func appStateDoesNotRequestAIAfterTerminalHumanMove() async throws {
    let backend = MockBackend(
        responses: [
            BackendResponse.success(
                id: .int(1),
                state: terminalState(moves: ["e2e4"], lastMove: "e2e4"),
                appliedMove: "e2e4"
            ),
        ]
    )
    let appState = AppState(backend: backend, initialState: startingState())

    await appState.makeMove("e2e4")

    let requests = await backend.requests
    #expect(requests.map(\.cmd) == [.makeMove])
    #expect(appState.backendState?.outcome?.reason == .checkmate)
    #expect(appState.lastAppliedMove == "e2e4")
    #expect(appState.isThinking == false)
}

@MainActor
@Test func appStateBlocksDuplicateInputDuringAutomaticAIMove() async throws {
    let backend = BlockingSequenceBackend(
        responses: [
            BackendResponse.success(
                id: .int(1),
                state: playedState(),
                appliedMove: "e2e4"
            ),
            BackendResponse.success(
                id: .int(2),
                state: aiReplyState(),
                appliedMove: "e7e5"
            ),
        ],
        blockAtRequestCount: 2
    )
    let appState = AppState(backend: backend, initialState: startingState())

    let moveTask = Task { await appState.makeMove("e2e4") }
    await backend.waitUntilRequestCount(2)
    #expect(appState.isThinking)

    appState.selectSquare("e2")
    await appState.makeMove("g1f3")
    appState.flipBoard()
    appState.updateHumanColor(.black)

    #expect(appState.selectedSquare == nil)
    #expect(appState.boardOrientation == .white)
    #expect(appState.humanColor == .white)
    #expect(await backend.requestCount == 2)

    await backend.release()
    await moveTask.value

    #expect(appState.isThinking == false)
    #expect(appState.backendState?.moves == ["e2e4", "e7e5"])
    #expect(appState.lastAppliedMove == "e7e5")
    #expect(await backend.requestCount == 2)
}

@MainActor
@Test func appStateRequestAIMoveAndUndoUseBackendSeams() async throws {
    let backend = MockBackend(
        responses: [
            BackendResponse.success(
                id: .int(1),
                state: playedState(moves: ["g1f3"], lastMove: "g1f3"),
                appliedMove: "g1f3"
            ),
            BackendResponse.success(id: .int(2), state: startingState()),
        ]
    )
    let appState = AppState(backend: backend, initialState: startingState())
    let aiConfig = BackendAIConfig(kind: .random, seed: 7)

    await appState.requestAIMove(aiConfig: aiConfig)
    await appState.undo(plies: 2)

    let requests = await backend.requests
    #expect(requests.map(\.cmd) == [.aiMove, .undo])
    #expect(requests[0].ai == aiConfig)
    #expect(requests[1].plies == 2)
    #expect(appState.aiConfig == aiConfig)
    #expect(appState.backendState?.moves == [])
    #expect(appState.lastAppliedMove == nil)
}

@MainActor
@Test func appStateAppliesBackendErrorStateAndMessage() async throws {
    let response = BackendResponse.error(
        id: .int(1),
        code: "illegal_move",
        message: "illegal move",
        state: startingState()
    )
    let backend = MockBackend(responses: [response])
    let appState = AppState(backend: backend, initialState: playedState())

    await appState.makeMove("e2e5")

    #expect(appState.errorMessage == "illegal_move: illegal move")
    #expect(appState.backendState?.moves == [])
    #expect(appState.isThinking == false)
}

@MainActor
@Test func appStateBlocksInputWhileThinking() async throws {
    let backend = BlockingBackend(
        response: BackendResponse.success(
            id: .int(1),
            state: playedState(moves: ["g1f3"], lastMove: "g1f3"),
            appliedMove: "g1f3"
        )
    )
    let appState = AppState(backend: backend, initialState: startingState())

    let aiTask = Task { await appState.requestAIMove() }
    await backend.waitUntilRequestReceived()
    #expect(appState.isThinking)

    appState.selectSquare("e2")
    await appState.makeMove("e2e4")
    appState.flipBoard()
    appState.updateHumanColor(.black)

    #expect(appState.selectedSquare == nil)
    #expect(appState.boardOrientation == .white)
    #expect(appState.humanColor == .white)
    #expect(await backend.requestCount == 1)

    await backend.release()
    await aiTask.value

    #expect(appState.isThinking == false)
    #expect(appState.lastAppliedMove == "g1f3")
    #expect(await backend.requestCount == 1)
}

private let startingFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

private actor MockBackend: BackendSession {
    private(set) var requests: [BackendRequest] = []
    private var responses: [BackendResponse]

    init(responses: [BackendResponse] = []) {
        self.responses = responses
    }

    func send(_ request: BackendRequest) async throws -> BackendResponse {
        requests.append(request)
        guard !responses.isEmpty else {
            return BackendResponse.success(id: request.id, state: startingState())
        }
        var response = responses.removeFirst()
        response.id = request.id
        return response
    }
}

private actor BlockingBackend: BackendSession {
    private(set) var requests: [BackendRequest] = []
    private var response: BackendResponse
    private var releaseContinuation: CheckedContinuation<Void, Never>?
    private var requestContinuation: CheckedContinuation<Void, Never>?

    init(response: BackendResponse) {
        self.response = response
    }

    var requestCount: Int {
        requests.count
    }

    func send(_ request: BackendRequest) async throws -> BackendResponse {
        requests.append(request)
        requestContinuation?.resume()
        requestContinuation = nil
        await withCheckedContinuation { continuation in
            releaseContinuation = continuation
        }
        response.id = request.id
        return response
    }

    func waitUntilRequestReceived() async {
        if !requests.isEmpty {
            return
        }
        await withCheckedContinuation { continuation in
            requestContinuation = continuation
        }
    }

    func release() {
        releaseContinuation?.resume()
        releaseContinuation = nil
    }
}

private actor BlockingSequenceBackend: BackendSession {
    private(set) var requests: [BackendRequest] = []
    private var responses: [BackendResponse]
    private let blockAtRequestCount: Int
    private var releaseContinuation: CheckedContinuation<Void, Never>?
    private var requestCountContinuations: [Int: [CheckedContinuation<Void, Never>]] = [:]

    init(responses: [BackendResponse], blockAtRequestCount: Int) {
        self.responses = responses
        self.blockAtRequestCount = blockAtRequestCount
    }

    var requestCount: Int {
        requests.count
    }

    func send(_ request: BackendRequest) async throws -> BackendResponse {
        requests.append(request)
        resumeRequestCountWaiters()
        if requests.count == blockAtRequestCount {
            await withCheckedContinuation { continuation in
                releaseContinuation = continuation
            }
        }
        guard !responses.isEmpty else {
            return BackendResponse.success(id: request.id, state: startingState())
        }
        var response = responses.removeFirst()
        response.id = request.id
        return response
    }

    func waitUntilRequestCount(_ count: Int) async {
        if requests.count >= count {
            return
        }
        await withCheckedContinuation { continuation in
            requestCountContinuations[count, default: []].append(continuation)
        }
    }

    func release() {
        releaseContinuation?.resume()
        releaseContinuation = nil
    }

    private func resumeRequestCountWaiters() {
        for count in requestCountContinuations.keys where requests.count >= count {
            let continuations = requestCountContinuations.removeValue(forKey: count) ?? []
            for continuation in continuations {
                continuation.resume()
            }
        }
    }
}

private extension BackendResponse {
    static func success(
        id: BackendMessageID,
        state: BackendState? = nil,
        appliedMove: String? = nil,
        search: BackendSearchMetadata? = nil,
        ai: BackendAIConfig? = nil
    ) -> BackendResponse {
        BackendResponse(
            id: id,
            ok: true,
            state: state,
            error: nil,
            version: nil,
            protocolVersion: nil,
            capabilities: nil,
            appliedMove: appliedMove,
            search: search,
            ai: ai
        )
    }

    static func error(
        id: BackendMessageID,
        code: String,
        message: String,
        state: BackendState? = nil
    ) -> BackendResponse {
        BackendResponse(
            id: id,
            ok: false,
            state: state,
            error: BackendError(code: code, message: message),
            version: nil,
            protocolVersion: nil,
            capabilities: nil,
            appliedMove: nil,
            search: nil,
            ai: nil
        )
    }
}

private func startingState(sideToMove: BackendColor = .white) -> BackendState {
    BackendState(
        fen: startingFEN,
        sideToMove: sideToMove,
        squares: [
            BackendPiece(square: "e2", index: 12, piece: "P", color: .white, kind: .pawn),
            BackendPiece(square: "e7", index: 52, piece: "p", color: .black, kind: .pawn),
            BackendPiece(square: "g1", index: 6, piece: "N", color: .white, kind: .knight),
        ],
        legalMoves: ["e2e3", "e2e4", "g1f3", "g1h3"],
        legalDestinationsByFrom: ["e2": ["e3", "e4"], "g1": ["f3", "h3"]],
        moves: [],
        lastMove: nil,
        halfmoveClock: 0,
        fullmoveNumber: 1,
        outcome: nil
    )
}

private func playedState(
    moves: [String] = ["e2e4"],
    lastMove: String? = "e2e4",
    sideToMove: BackendColor = .black
) -> BackendState {
    BackendState(
        fen: "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        sideToMove: sideToMove,
        squares: [
            BackendPiece(square: "e4", index: 28, piece: "P", color: .white, kind: .pawn),
            BackendPiece(square: "e7", index: 52, piece: "p", color: .black, kind: .pawn),
        ],
        legalMoves: ["e7e5", "e7e6"],
        legalDestinationsByFrom: ["e7": ["e5", "e6"]],
        moves: moves,
        lastMove: lastMove,
        halfmoveClock: 0,
        fullmoveNumber: 1,
        outcome: nil
    )
}

private func aiReplyState() -> BackendState {
    BackendState(
        fen: "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        sideToMove: .white,
        squares: [
            BackendPiece(square: "e4", index: 28, piece: "P", color: .white, kind: .pawn),
            BackendPiece(square: "e5", index: 36, piece: "p", color: .black, kind: .pawn),
        ],
        legalMoves: ["g1f3"],
        legalDestinationsByFrom: ["g1": ["f3"]],
        moves: ["e2e4", "e7e5"],
        lastMove: "e7e5",
        halfmoveClock: 0,
        fullmoveNumber: 2,
        outcome: nil
    )
}

private func terminalState(moves: [String], lastMove: String) -> BackendState {
    BackendState(
        fen: "7k/8/8/8/8/8/8/6RK b - - 0 1",
        sideToMove: .black,
        squares: [
            BackendPiece(square: "g1", index: 6, piece: "R", color: .white, kind: .rook),
            BackendPiece(square: "h1", index: 7, piece: "K", color: .white, kind: .king),
            BackendPiece(square: "h8", index: 63, piece: "k", color: .black, kind: .king),
        ],
        legalMoves: [],
        legalDestinationsByFrom: [:],
        moves: moves,
        lastMove: lastMove,
        halfmoveClock: 0,
        fullmoveNumber: 1,
        outcome: BackendOutcome(reason: .checkmate, winner: .white, isDraw: false)
    )
}
