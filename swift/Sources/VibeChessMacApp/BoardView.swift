import SwiftUI

/// Chess board view driven entirely by backend-provided state.
struct BoardView: View {
    @ObservedObject var appState: AppState
    var squareSize: CGFloat

    init(appState: AppState, squareSize: CGFloat = 56) {
        self.appState = appState
        self.squareSize = squareSize
    }

    var body: some View {
        let piecesBySquare = appState.piecesBySquare
        let legalDestinations = Set(appState.legalDestinationsForSelectedSquare)
        let lastMoveSquares = Set(
            BoardLayout.moveSquares(appState.backendState?.lastMove ?? appState.lastAppliedMove)
        )
        let columns = Array(repeating: GridItem(.fixed(squareSize), spacing: 0), count: 8)

        LazyVGrid(columns: columns, spacing: 0) {
            ForEach(BoardLayout.squares(for: appState.boardOrientation), id: \.self) { square in
                SquareView(
                    squareName: square,
                    piece: piecesBySquare[square],
                    isLightSquare: BoardLayout.isLightSquare(square),
                    isSelected: appState.selectedSquare == square,
                    isLegalDestination: legalDestinations.contains(square),
                    isLastMoveSquare: lastMoveSquares.contains(square),
                    squareSize: squareSize,
                    onTap: handleSquareTap
                )
            }
        }
        .frame(width: squareSize * 8, height: squareSize * 8)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Chess board")
    }

    private func handleSquareTap(_ square: String) {
        if appState.legalDestinationsForSelectedSquare.contains(square) {
            Task { await appState.makeSelectedMove(to: square) }
        } else {
            appState.selectSquare(square)
        }
    }
}

/// Pure board-layout helpers kept separate from SwiftUI rendering for focused tests.
enum BoardLayout {
    private static let filesForWhite = ["a", "b", "c", "d", "e", "f", "g", "h"]
    private static let ranksForWhite = ["8", "7", "6", "5", "4", "3", "2", "1"]

    static func squares(for orientation: BackendColor) -> [String] {
        let files: [String]
        let ranks: [String]
        switch orientation {
        case .white:
            files = filesForWhite
            ranks = ranksForWhite
        case .black:
            files = filesForWhite.reversed()
            ranks = ranksForWhite.reversed()
        }
        return ranks.flatMap { rank in files.map { file in "\(file)\(rank)" } }
    }

    static func isLightSquare(_ square: String) -> Bool {
        guard let coordinates = coordinates(for: square) else {
            return false
        }
        return (coordinates.file + coordinates.rank) % 2 == 1
    }

    static func moveSquares(_ move: String?) -> [String] {
        guard let move, move.count >= 4 else {
            return []
        }
        let fromEnd = move.index(move.startIndex, offsetBy: 2)
        let toEnd = move.index(move.startIndex, offsetBy: 4)
        return [String(move[..<fromEnd]), String(move[fromEnd..<toEnd])]
    }

    private static func coordinates(for square: String) -> (file: Int, rank: Int)? {
        guard square.count == 2,
              let fileScalar = square.unicodeScalars.first,
              let rankScalar = square.unicodeScalars.last
        else {
            return nil
        }
        let file = Int(fileScalar.value) - Int(UnicodeScalar("a").value)
        let rank = Int(rankScalar.value) - Int(UnicodeScalar("1").value)
        guard (0..<8).contains(file), (0..<8).contains(rank) else {
            return nil
        }
        return (file, rank)
    }
}
