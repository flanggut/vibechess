import Foundation

/// Scalar request/response identifier used by the vibechess GUI JSON-lines protocol.
enum BackendMessageID: Codable, Equatable, Sendable {
    case int(Int)
    case double(Double)
    case string(String)
    case bool(Bool)
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else if let value = try? container.decode(Double.self) {
            self = .double(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else {
            throw DecodingError.typeMismatch(
                BackendMessageID.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "GUI protocol id must be a scalar JSON value"
                )
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case let .int(value):
            try container.encode(value)
        case let .double(value):
            try container.encode(value)
        case let .string(value):
            try container.encode(value)
        case let .bool(value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        }
    }
}

enum BackendCommand: String, Codable, Equatable, Sendable {
    case hello
    case newGame
    case state
    case makeMove
    case aiMove
    case undo
    case setAiConfig
    case quit
}

enum BackendColor: String, Codable, Equatable, Sendable {
    case white
    case black
}

enum BackendPlayerKind: String, Codable, Equatable, Sendable {
    case random
    case mcts
    case neural
}

enum BackendPieceKind: String, Codable, Equatable, Sendable {
    case pawn
    case knight
    case bishop
    case rook
    case queen
    case king
}

enum BackendOutcomeReason: String, Codable, Equatable, Sendable {
    case checkmate
    case stalemate
    case fiftyMove = "fifty_move"
    case repetition
    case insufficientMaterial = "insufficient_material"
    case maxPlies = "max_plies"
}

/// Request envelope for the GUI backend. Optional command-specific fields are omitted when nil.
struct BackendRequest: Codable, Equatable, Sendable {
    var id: BackendMessageID
    var cmd: BackendCommand
    var humanColor: BackendColor?
    var ai: BackendAIConfig?
    var seed: Int?
    var move: String?
    var plies: Int?

    init(
        id: BackendMessageID,
        cmd: BackendCommand,
        humanColor: BackendColor? = nil,
        ai: BackendAIConfig? = nil,
        seed: Int? = nil,
        move: String? = nil,
        plies: Int? = nil
    ) {
        self.id = id
        self.cmd = cmd
        self.humanColor = humanColor
        self.ai = ai
        self.seed = seed
        self.move = move
        self.plies = plies
    }
}

/// Response envelope for success and error messages from the GUI backend.
struct BackendResponse: Codable, Equatable, Sendable {
    var id: BackendMessageID?
    var ok: Bool
    var state: BackendState?
    var error: BackendError?
    var version: String?
    var protocolVersion: String?
    var capabilities: BackendCapabilities?
    var appliedMove: String?
    var search: BackendSearchMetadata?
    var ai: BackendAIConfig?

    enum CodingKeys: String, CodingKey {
        case id
        case ok
        case state
        case error
        case version
        case protocolVersion = "protocol"
        case capabilities
        case appliedMove
        case search
        case ai
    }
}

struct BackendError: Codable, Equatable, Sendable {
    var code: String
    var message: String
}

struct BackendCapabilities: Codable, Equatable, Sendable {
    var players: [BackendPlayerKind]
    var supportsUndo: Bool
    var promotion: String
}

struct BackendState: Codable, Equatable, Sendable {
    var fen: String
    var sideToMove: BackendColor
    var squares: [BackendPiece]
    var legalMoves: [String]
    var legalDestinationsByFrom: [String: [String]]
    var moves: [String]
    var lastMove: String?
    var halfmoveClock: Int
    var fullmoveNumber: Int
    var outcome: BackendOutcome?
}

struct BackendPiece: Codable, Equatable, Sendable {
    var square: String
    var index: Int
    var piece: String
    var color: BackendColor
    var kind: BackendPieceKind
}

struct BackendOutcome: Codable, Equatable, Sendable {
    var reason: BackendOutcomeReason
    var winner: BackendColor?
    var isDraw: Bool
}

/// AI configuration payload accepted by `newGame`, `setAiConfig`, and `aiMove` requests.
struct BackendAIConfig: Codable, Equatable, Sendable {
    var kind: BackendPlayerKind?
    var simulations: Int?
    var timeLimitSeconds: Double?
    var nodeBudget: Int?
    var maxRolloutPlies: Int?
    var checkpointPath: String?
    var puctExploration: Double?
    var temperature: Double?
    var seed: Int?

    init(
        kind: BackendPlayerKind? = nil,
        simulations: Int? = nil,
        timeLimitSeconds: Double? = nil,
        nodeBudget: Int? = nil,
        maxRolloutPlies: Int? = nil,
        checkpointPath: String? = nil,
        puctExploration: Double? = nil,
        temperature: Double? = nil,
        seed: Int? = nil
    ) {
        self.kind = kind
        self.simulations = simulations
        self.timeLimitSeconds = timeLimitSeconds
        self.nodeBudget = nodeBudget
        self.maxRolloutPlies = maxRolloutPlies
        self.checkpointPath = checkpointPath
        self.puctExploration = puctExploration
        self.temperature = temperature
        self.seed = seed
    }
}

struct BackendSearchMetadata: Codable, Equatable, Sendable {
    var kind: BackendPlayerKind
    var simulations: Int?
    var nodes: Int?
    var elapsedSeconds: Double
    var visitCounts: [String: Int]?
}
