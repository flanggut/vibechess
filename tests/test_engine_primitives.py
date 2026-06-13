from __future__ import annotations

import pytest

from vibechess.engine import Board, Color, Move, Piece, PieceType, parse_square, square_name
from vibechess.engine.board import STARTING_POSITION, board_from_ascii
from vibechess.engine.square import Square, file_index, make_square, rank_index


def test_square_indexing_convention() -> None:
    assert int(parse_square("a1")) == 0
    assert int(parse_square("h1")) == 7
    assert int(parse_square("a2")) == 8
    assert int(parse_square("a8")) == 56
    assert int(parse_square("h8")) == 63
    assert square_name(Square(28)) == "e4"
    assert make_square(4, 3) == parse_square("e4")
    assert file_index(parse_square("e4")) == 4
    assert rank_index(parse_square("e4")) == 3


@pytest.mark.parametrize("name", ["", "a", "i1", "a0", "e9", "abc"])
def test_parse_square_rejects_invalid_names(name: str) -> None:
    with pytest.raises(ValueError):
        parse_square(name)


def test_piece_symbols_round_trip() -> None:
    white_queen = Piece(Color.WHITE, PieceType.QUEEN)
    black_knight = Piece(Color.BLACK, PieceType.KNIGHT)

    assert white_queen.symbol == "Q"
    assert black_knight.symbol == "n"
    assert Piece.from_symbol("Q") == white_queen
    assert Piece.from_symbol("n") == black_knight
    assert Color.WHITE.opposite is Color.BLACK


def test_move_uci_round_trip() -> None:
    move = Move.from_uci("e2e4")
    assert move.from_square == parse_square("e2")
    assert move.to_square == parse_square("e4")
    assert move.promotion is None
    assert move.to_uci() == "e2e4"
    assert str(move) == "e2e4"


def test_move_uci_promotion_round_trip() -> None:
    move = Move.from_uci("e7e8q")
    assert move.promotion is PieceType.QUEEN
    assert move.to_uci() == "e7e8q"


@pytest.mark.parametrize("notation", ["e2", "e2e9", "e2e4k", "e2e4qq"])
def test_move_uci_rejects_invalid_notation(notation: str) -> None:
    with pytest.raises(ValueError):
        Move.from_uci(notation)


def test_starting_position_setup() -> None:
    board = Board.starting_position()

    assert len(board.squares) == 64
    assert len(board.occupied_squares()) == 32
    assert board.side_to_move is Color.WHITE
    assert board.piece_at("a1") == Piece(Color.WHITE, PieceType.ROOK)
    assert board.piece_at("e1") == Piece(Color.WHITE, PieceType.KING)
    assert board.piece_at("d8") == Piece(Color.BLACK, PieceType.QUEEN)
    assert board.piece_at("h7") == Piece(Color.BLACK, PieceType.PAWN)
    assert board.piece_at("e4") is None


@pytest.mark.parametrize("square", [Square(-1), Square(64)])
def test_board_rejects_invalid_square_access(square: Square) -> None:
    board = Board.starting_position()

    with pytest.raises(ValueError):
        board.piece_at(square)

    with pytest.raises(ValueError):
        board.with_piece(square, Piece(Color.WHITE, PieceType.KING))


def test_board_from_ascii_matches_starting_position() -> None:
    board = board_from_ascii(STARTING_POSITION, castling_rights=frozenset("KQkq"))

    assert board == Board.starting_position()


@pytest.mark.parametrize("rows", ["08/8/8/8/8/8/8/8", "9/8/8/8/8/8/8/8"])
def test_board_from_ascii_rejects_invalid_digit_rows(rows: str) -> None:
    with pytest.raises(ValueError):
        board_from_ascii(rows)


def test_board_with_piece_is_immutable() -> None:
    empty = Board.empty()
    with_king = empty.with_piece("e1", Piece(Color.WHITE, PieceType.KING))

    assert empty.piece_at("e1") is None
    assert with_king.piece_at("e1") == Piece(Color.WHITE, PieceType.KING)


def test_board_text_rendering() -> None:
    board = Board.starting_position()

    assert board.render() == "\n".join(
        [
            "8 r n b q k b n r",
            "7 p p p p p p p p",
            "6 . . . . . . . .",
            "5 . . . . . . . .",
            "4 . . . . . . . .",
            "3 . . . . . . . .",
            "2 P P P P P P P P",
            "1 R N B Q K B N R",
            "  a b c d e f g h",
        ]
    )


def test_board_unicode_rendering_without_coordinates() -> None:
    board = Board.empty().with_piece("e1", Piece(Color.WHITE, PieceType.KING))

    assert "♔" in board.render(unicode=True, coordinates=False)
