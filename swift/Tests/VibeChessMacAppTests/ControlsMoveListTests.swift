import Testing
@testable import VibeChessMacApp

@Test func statusPresenterDescribesConnectionThinkingTurnsAndOutcome() {
    #expect(
        AppStatusPresenter.statusText(state: nil, isThinking: false, humanColor: .white)
            == "Start a game to connect to the backend."
    )
    #expect(
        AppStatusPresenter.statusText(state: controlState(sideToMove: .white), isThinking: true, humanColor: .white)
            == "AI thinking…"
    )
    #expect(
        AppStatusPresenter.statusText(state: controlState(sideToMove: .white), isThinking: false, humanColor: .white)
            == "White to move (human)"
    )
    #expect(
        AppStatusPresenter.statusText(state: controlState(sideToMove: .black), isThinking: false, humanColor: .white)
            == "Black to move (AI)"
    )

    let checkmate = BackendOutcome(reason: .checkmate, winner: .black, isDraw: false)
    #expect(AppStatusPresenter.outcomeText(checkmate) == "Checkmate — Black wins")

    let stalemate = BackendOutcome(reason: .stalemate, winner: nil, isDraw: true)
    #expect(AppStatusPresenter.outcomeText(stalemate) == "Draw by stalemate")
}

@Test func controlPresentationParsesOptionalBudgetText() {
    #expect(AppStatusPresenter.optionalNumberText(Optional<Int>.none) == "")
    #expect(AppStatusPresenter.optionalNumberText(25) == "25")
    #expect(AppStatusPresenter.parseOptionalInt("") == nil)
    #expect(AppStatusPresenter.parseOptionalInt(" 100 ") == 100)
    #expect(AppStatusPresenter.parseOptionalInt("0") == nil)
    #expect(AppStatusPresenter.parseOptionalInt("bad") == nil)
    #expect(AppStatusPresenter.parseOptionalDouble("0.5") == 0.5)
    #expect(AppStatusPresenter.parseOptionalDouble("0") == 0)
    #expect(AppStatusPresenter.parseOptionalDouble("-1") == nil)
    #expect(AppStatusPresenter.parseOptionalDouble("nan") == nil)
}

@Test func moveListRowsGroupsUciHistoryByFullMoveNumber() {
    #expect(MoveListRows.rows(for: []) == [])
    #expect(
        MoveListRows.rows(for: ["e2e4"]) == [
            MoveListRow(number: 1, whiteMove: "e2e4", blackMove: nil)
        ]
    )
    #expect(
        MoveListRows.rows(for: ["e2e4", "e7e5", "g1f3"]) == [
            MoveListRow(number: 1, whiteMove: "e2e4", blackMove: "e7e5"),
            MoveListRow(number: 2, whiteMove: "g1f3", blackMove: nil),
        ]
    )
}

private func controlState(sideToMove: BackendColor, outcome: BackendOutcome? = nil) -> BackendState {
    BackendState(
        fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        sideToMove: sideToMove,
        squares: [],
        legalMoves: ["e2e4"],
        legalDestinationsByFrom: ["e2": ["e4"]],
        moves: [],
        lastMove: nil,
        halfmoveClock: 0,
        fullmoveNumber: 1,
        outcome: outcome
    )
}
