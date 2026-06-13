import Testing
@testable import VibeChessMacApp

@Test func boardLayoutOrdersSquaresForWhiteOrientation() {
    let squares = BoardLayout.squares(for: .white)

    #expect(squares.count == 64)
    #expect(Array(squares.prefix(8)) == ["a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8"])
    #expect(Array(squares.suffix(8)) == ["a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1"])
}

@Test func boardLayoutOrdersSquaresForBlackOrientation() {
    let squares = BoardLayout.squares(for: .black)

    #expect(squares.count == 64)
    #expect(Array(squares.prefix(8)) == ["h1", "g1", "f1", "e1", "d1", "c1", "b1", "a1"])
    #expect(Array(squares.suffix(8)) == ["h8", "g8", "f8", "e8", "d8", "c8", "b8", "a8"])
}

@Test func boardLayoutIdentifiesSquareColorsAndLastMoveSquares() {
    #expect(BoardLayout.isLightSquare("a1") == false)
    #expect(BoardLayout.isLightSquare("h1") == true)
    #expect(BoardLayout.isLightSquare("a8") == true)
    #expect(BoardLayout.isLightSquare("h8") == false)
    #expect(BoardLayout.isLightSquare("not-a-square") == false)

    #expect(BoardLayout.moveSquares("e2e4") == ["e2", "e4"])
    #expect(BoardLayout.moveSquares("e7e8q") == ["e7", "e8"])
    #expect(BoardLayout.moveSquares(nil) == [])
    #expect(BoardLayout.moveSquares("bad") == [])
}

@Test func squareViewMapsFENSymbolsToUnicodeGlyphs() {
    #expect(SquareView.unicodeGlyph(forFENSymbol: "P") == "♙")
    #expect(SquareView.unicodeGlyph(forFENSymbol: "N") == "♘")
    #expect(SquareView.unicodeGlyph(forFENSymbol: "B") == "♗")
    #expect(SquareView.unicodeGlyph(forFENSymbol: "R") == "♖")
    #expect(SquareView.unicodeGlyph(forFENSymbol: "Q") == "♕")
    #expect(SquareView.unicodeGlyph(forFENSymbol: "K") == "♔")
    #expect(SquareView.unicodeGlyph(forFENSymbol: "p") == "♟")
    #expect(SquareView.unicodeGlyph(forFENSymbol: "n") == "♞")
    #expect(SquareView.unicodeGlyph(forFENSymbol: "b") == "♝")
    #expect(SquareView.unicodeGlyph(forFENSymbol: "r") == "♜")
    #expect(SquareView.unicodeGlyph(forFENSymbol: "q") == "♛")
    #expect(SquareView.unicodeGlyph(forFENSymbol: "k") == "♚")
    #expect(SquareView.unicodeGlyph(forFENSymbol: nil) == nil)
    #expect(SquareView.unicodeGlyph(forFENSymbol: "x") == nil)
}
