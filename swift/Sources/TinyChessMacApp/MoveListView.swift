import SwiftUI

/// Scrollable UCI move history grouped by full move number.
struct MoveListView: View {
    @ObservedObject var appState: AppState

    var body: some View {
        GroupBox("Moves") {
            if appState.moveHistory.isEmpty {
                Text("No moves yet.")
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 6) {
                        ForEach(MoveListRows.rows(for: appState.moveHistory)) { row in
                            HStack(alignment: .firstTextBaseline, spacing: 8) {
                                Text("\(row.number).")
                                    .font(.body.monospacedDigit())
                                    .foregroundStyle(.secondary)
                                    .frame(width: 32, alignment: .trailing)
                                Text(row.whiteMove)
                                    .font(.body.monospaced())
                                Text(row.blackMove ?? "")
                                    .font(.body.monospaced())
                                    .foregroundStyle(row.blackMove == nil ? .clear : .primary)
                                Spacer(minLength: 0)
                            }
                            .accessibilityElement(children: .combine)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(minHeight: 120, maxHeight: 220)
            }
        }
    }
}

struct MoveListRow: Identifiable, Equatable {
    var number: Int
    var whiteMove: String
    var blackMove: String?

    var id: Int { number }
}

enum MoveListRows {
    static func rows(for moves: [String]) -> [MoveListRow] {
        stride(from: 0, to: moves.count, by: 2).map { index in
            MoveListRow(
                number: index / 2 + 1,
                whiteMove: moves[index],
                blackMove: index + 1 < moves.count ? moves[index + 1] : nil
            )
        }
    }
}
