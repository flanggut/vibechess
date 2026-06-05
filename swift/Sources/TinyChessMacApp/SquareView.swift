import SwiftUI

/// One clickable chess-board square.
struct SquareView: View {
    var squareName: String
    var piece: BackendPiece?
    var isLightSquare: Bool
    var isSelected: Bool
    var isLegalDestination: Bool
    var isLastMoveSquare: Bool
    var squareSize: CGFloat
    var onTap: (String) -> Void

    init(
        squareName: String,
        piece: BackendPiece?,
        isLightSquare: Bool,
        isSelected: Bool,
        isLegalDestination: Bool,
        isLastMoveSquare: Bool,
        squareSize: CGFloat = 56,
        onTap: @escaping (String) -> Void
    ) {
        self.squareName = squareName
        self.piece = piece
        self.isLightSquare = isLightSquare
        self.isSelected = isSelected
        self.isLegalDestination = isLegalDestination
        self.isLastMoveSquare = isLastMoveSquare
        self.squareSize = squareSize
        self.onTap = onTap
    }

    var body: some View {
        Button {
            onTap(squareName)
        } label: {
            ZStack {
                Rectangle()
                    .fill(baseColor)
                if isLastMoveSquare {
                    Rectangle()
                        .fill(Color.yellow.opacity(0.28))
                }
                if isSelected {
                    Rectangle()
                        .strokeBorder(Color.accentColor, lineWidth: 4)
                }
                if isLegalDestination {
                    legalDestinationMarker
                }
                if let glyph = SquareView.unicodeGlyph(forFENSymbol: piece?.piece) {
                    Text(glyph)
                        .font(.system(size: squareSize * 0.68))
                        .minimumScaleFactor(0.5)
                        .frame(width: squareSize, height: squareSize)
                }
            }
            .frame(width: squareSize, height: squareSize)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(accessibilityLabel)
        .accessibilityAddTraits(isSelected ? [.isSelected] : [])
    }

    private var baseColor: Color {
        isLightSquare
            ? Color(red: 0.89, green: 0.82, blue: 0.68)
            : Color(red: 0.54, green: 0.36, blue: 0.24)
    }

    @ViewBuilder
    private var legalDestinationMarker: some View {
        if piece == nil {
            Circle()
                .fill(Color.accentColor.opacity(0.38))
                .frame(width: squareSize * 0.24, height: squareSize * 0.24)
        } else {
            Circle()
                .stroke(Color.accentColor.opacity(0.7), lineWidth: 4)
                .frame(width: squareSize * 0.82, height: squareSize * 0.82)
        }
    }

    private var accessibilityLabel: String {
        if let piece {
            return "\(squareName) \(piece.color.rawValue) \(piece.kind.rawValue)"
        }
        return "\(squareName) empty"
    }

    static func unicodeGlyph(forFENSymbol symbol: String?) -> String? {
        guard let symbol else {
            return nil
        }
        switch symbol {
        case "P": return "♙"
        case "N": return "♘"
        case "B": return "♗"
        case "R": return "♖"
        case "Q": return "♕"
        case "K": return "♔"
        case "p": return "♟"
        case "n": return "♞"
        case "b": return "♝"
        case "r": return "♜"
        case "q": return "♛"
        case "k": return "♚"
        default: return nil
        }
    }
}
