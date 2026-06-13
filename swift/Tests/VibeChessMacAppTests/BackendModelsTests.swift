import Foundation
import Testing
@testable import VibeChessMacApp

@Test func decodesHelloStateResponse() throws {
    let data = Data(
        """
        {
          "id": 1,
          "ok": true,
          "version": "0.1.0",
          "protocol": "vibechess-gui-v1",
          "capabilities": {
            "players": ["random", "mcts", "neural"],
            "supportsUndo": true,
            "promotion": "auto_queen"
          },
          "state": {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "sideToMove": "white",
            "squares": [
              { "square": "a1", "index": 0, "piece": "R", "color": "white", "kind": "rook" },
              { "square": "b8", "index": 57, "piece": "n", "color": "black", "kind": "knight" }
            ],
            "legalMoves": ["e2e3", "e2e4", "g1f3"],
            "legalDestinationsByFrom": { "e2": ["e3", "e4"], "g1": ["f3", "h3"] },
            "moves": [],
            "lastMove": null,
            "halfmoveClock": 0,
            "fullmoveNumber": 1,
            "outcome": null
          }
        }
        """.utf8
    )

    let response = try JSONDecoder().decode(BackendResponse.self, from: data)

    #expect(response.id == .int(1))
    #expect(response.ok)
    #expect(response.protocolVersion == "vibechess-gui-v1")
    #expect(response.capabilities?.players == [.random, .mcts, .neural])
    #expect(response.capabilities?.supportsUndo == true)
    #expect(response.capabilities?.promotion == "auto_queen")
    #expect(response.state?.sideToMove == .white)
    #expect(response.state?.squares.count == 2)
    #expect(response.state?.squares[0].piece == "R")
    #expect(response.state?.squares[1].kind == .knight)
    #expect(response.state?.legalDestinationsByFrom["e2"] == ["e3", "e4"])
    #expect(response.state?.lastMove == nil)
    #expect(response.state?.outcome == nil)
}

@Test func decodesErrorResponseWithResyncState() throws {
    let data = Data(
        """
        {
          "id": "move-1",
          "ok": false,
          "error": {
            "code": "illegal_move",
            "message": "illegal move 'e2e5' for the current position"
          },
          "state": {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "sideToMove": "white",
            "squares": [],
            "legalMoves": ["e2e4"],
            "legalDestinationsByFrom": { "e2": ["e4"] },
            "moves": [],
            "lastMove": null,
            "halfmoveClock": 0,
            "fullmoveNumber": 1,
            "outcome": null
          }
        }
        """.utf8
    )

    let response = try JSONDecoder().decode(BackendResponse.self, from: data)

    #expect(response.id == .string("move-1"))
    #expect(response.ok == false)
    #expect(response.error?.code == "illegal_move")
    #expect(response.error?.message.contains("e2e5") == true)
    #expect(response.state?.fen.hasPrefix("rnbqkbnr") == true)
}

@Test func decodesAiMoveResponseWithSearchMetadataAndOutcome() throws {
    let data = Data(
        """
        {
          "id": 5,
          "ok": true,
          "appliedMove": "g1f3",
          "search": {
            "kind": "mcts",
            "simulations": 3,
            "nodes": 4,
            "elapsedSeconds": 0.0125,
            "visitCounts": { "g1f3": 2, "e2e4": 1 }
          },
          "state": {
            "fen": "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
            "sideToMove": "black",
            "squares": [
              { "square": "f3", "index": 21, "piece": "N", "color": "white", "kind": "knight" }
            ],
            "legalMoves": ["e7e5", "g8f6"],
            "legalDestinationsByFrom": { "e7": ["e5", "e6"], "g8": ["f6", "h6"] },
            "moves": ["g1f3"],
            "lastMove": "g1f3",
            "halfmoveClock": 1,
            "fullmoveNumber": 1,
            "outcome": {
              "reason": "stalemate",
              "winner": null,
              "isDraw": true
            }
          }
        }
        """.utf8
    )

    let response = try JSONDecoder().decode(BackendResponse.self, from: data)

    #expect(response.ok)
    #expect(response.appliedMove == "g1f3")
    #expect(response.search?.kind == .mcts)
    #expect(response.search?.simulations == 3)
    #expect(response.search?.nodes == 4)
    #expect(response.search?.visitCounts?["g1f3"] == 2)
    #expect(response.state?.sideToMove == .black)
    #expect(response.state?.lastMove == "g1f3")
    #expect(response.state?.outcome?.reason == .stalemate)
    #expect(response.state?.outcome?.winner == nil)
    #expect(response.state?.outcome?.isDraw == true)
}

@Test func encodesNewGameRequestWithCamelCaseAiConfig() throws {
    let request = BackendRequest(
        id: .int(7),
        cmd: .newGame,
        humanColor: .black,
        ai: BackendAIConfig(
            kind: .mcts,
            simulations: 25,
            timeLimitSeconds: 0.5,
            nodeBudget: 100,
            maxRolloutPlies: 0,
            puctExploration: 1.5,
            temperature: 0.0,
            seed: 11
        ),
        seed: 11
    )

    let data = try JSONEncoder().encode(request)
    let object = try #require(
        JSONSerialization.jsonObject(with: data) as? [String: Any]
    )
    let ai = try #require(object["ai"] as? [String: Any])

    #expect(object["id"] as? Int == 7)
    #expect(object["cmd"] as? String == "newGame")
    #expect(object["humanColor"] as? String == "black")
    #expect(object["seed"] as? Int == 11)
    #expect(ai["kind"] as? String == "mcts")
    #expect(ai["simulations"] as? Int == 25)
    #expect(ai["timeLimitSeconds"] as? Double == 0.5)
    #expect(ai["nodeBudget"] as? Int == 100)
    #expect(ai["maxRolloutPlies"] as? Int == 0)
    #expect(ai["puctExploration"] as? Double == 1.5)
    #expect(ai["temperature"] as? Double == 0.0)
    #expect(ai["checkpointPath"] == nil)
}

@Test func encodesMakeMoveAiMoveAndUndoRequests() throws {
    let makeMove = BackendRequest(id: .string("human-1"), cmd: .makeMove, move: "e2e4")
    let aiMove = BackendRequest(
        id: .int(8),
        cmd: .aiMove,
        ai: BackendAIConfig(kind: .random, seed: 3)
    )
    let undo = BackendRequest(id: .int(9), cmd: .undo, plies: 2)

    let encoder = JSONEncoder()
    let makeMoveObject = try #require(
        JSONSerialization.jsonObject(with: try encoder.encode(makeMove)) as? [String: Any]
    )
    let aiMoveObject = try #require(
        JSONSerialization.jsonObject(with: try encoder.encode(aiMove)) as? [String: Any]
    )
    let undoObject = try #require(
        JSONSerialization.jsonObject(with: try encoder.encode(undo)) as? [String: Any]
    )
    let ai = try #require(aiMoveObject["ai"] as? [String: Any])

    #expect(makeMoveObject["id"] as? String == "human-1")
    #expect(makeMoveObject["cmd"] as? String == "makeMove")
    #expect(makeMoveObject["move"] as? String == "e2e4")
    #expect(makeMoveObject["ai"] == nil)
    #expect(aiMoveObject["cmd"] as? String == "aiMove")
    #expect(ai["kind"] as? String == "random")
    #expect(ai["seed"] as? Int == 3)
    #expect(undoObject["cmd"] as? String == "undo")
    #expect(undoObject["plies"] as? Int == 2)
}
