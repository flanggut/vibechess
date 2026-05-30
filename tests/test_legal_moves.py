from __future__ import annotations

import pytest

from tinychess.engine import (
    Board,
    Color,
    Move,
    Piece,
    PieceType,
    legal_moves,
    perft,
    pseudo_legal_moves,
)
from tinychess.engine.board import board_from_ascii
from tinychess.engine.fen import parse_fen
from tinychess.engine.legal_moves import has_legal_move
from tinychess.engine.square import parse_square


def move_set(moves: tuple[Move, ...]) -> set[str]:
    return {move.to_uci() for move in moves}


def test_start_position_legal_move_count() -> None:
    moves = legal_moves(Board.starting_position())

    assert len(moves) == 20
    assert move_set(moves) >= {"e2e4", "g1f3", "b1c3"}


@pytest.mark.parametrize(
    "fen",
    [
        "startpos",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1",
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1",
    ],
)
def test_has_legal_move_matches_legal_moves_bool(fen: str) -> None:
    board = Board.starting_position() if fen == "startpos" else parse_fen(fen).board

    assert has_legal_move(board) is bool(legal_moves(board))


@pytest.mark.parametrize(("depth", "nodes"), [(1, 20), (2, 400), (3, 8902)])
def test_start_position_perft(depth: int, nodes: int) -> None:
    assert perft(Board.starting_position(), depth) == nodes


def test_complex_castling_position_perft_depths_1_and_2() -> None:
    board = board_from_ascii(
        "r3k2r/p1ppqpb1/bn2pnp1/2pPN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R",
        side_to_move=Color.WHITE,
        castling_rights=frozenset("KQkq"),
    )

    assert perft(board, 1) == 48
    assert perft(board, 2) == 1991


def test_castling_moves_are_generated_when_path_is_clear() -> None:
    board = board_from_ascii(
        "r3k2r/8/8/8/8/8/8/R3K2R",
        side_to_move=Color.WHITE,
        castling_rights=frozenset("KQkq"),
    )

    assert {"e1g1", "e1c1"}.issubset(move_set(legal_moves(board)))


def test_castling_through_attacked_square_is_illegal() -> None:
    board = board_from_ascii(
        "4k3/8/8/8/8/5r2/8/R3K2R",
        side_to_move=Color.WHITE,
        castling_rights=frozenset("KQ"),
    )

    moves = move_set(legal_moves(board))
    assert "e1g1" not in moves
    assert "e1c1" in moves


def test_en_passant_move_is_generated_and_applied() -> None:
    board = board_from_ascii(
        "4k3/8/8/3pP3/8/8/8/4K3",
        side_to_move=Color.WHITE,
        en_passant_target=parse_square("d6"),
    )

    move = Move.from_uci("e5d6")
    assert move in legal_moves(board)

    next_board = board.apply_move(move)
    assert next_board.piece_at("d6") == Piece(Color.WHITE, PieceType.PAWN)
    assert next_board.piece_at("d5") is None
    assert next_board.piece_at("e5") is None
    assert next_board.side_to_move is Color.BLACK


def test_en_passant_requires_capturable_pawn() -> None:
    board = board_from_ascii(
        "4k3/8/8/4P3/8/8/8/4K3",
        side_to_move=Color.WHITE,
        en_passant_target=parse_square("d6"),
    )

    assert "e5d6" not in move_set(legal_moves(board))


def test_black_en_passant_move_is_generated_and_applied() -> None:
    board = board_from_ascii(
        "4k3/8/8/8/3Pp3/8/8/4K3",
        side_to_move=Color.BLACK,
        en_passant_target=parse_square("d3"),
    )

    move = Move.from_uci("e4d3")
    assert move in legal_moves(board)

    next_board = board.apply_move(move)
    assert next_board.piece_at("d3") == Piece(Color.BLACK, PieceType.PAWN)
    assert next_board.piece_at("d4") is None
    assert next_board.piece_at("e4") is None
    assert next_board.side_to_move is Color.WHITE


def test_promotion_moves_include_all_promotion_pieces() -> None:
    board = board_from_ascii("4k3/P7/8/8/8/8/8/4K3", side_to_move=Color.WHITE)

    moves = move_set(legal_moves(board))

    assert {"a7a8q", "a7a8r", "a7a8b", "a7a8n"}.issubset(moves)


def test_black_promotion_moves_include_all_promotion_pieces() -> None:
    board = board_from_ascii("4k3/8/8/8/8/8/p7/4K3", side_to_move=Color.BLACK)

    moves = move_set(legal_moves(board))

    assert {"a2a1q", "a2a1r", "a2a1b", "a2a1n"}.issubset(moves)


def test_apply_move_rejects_invalid_promotion_piece() -> None:
    board = board_from_ascii("4k3/P7/8/8/8/8/8/4K3", side_to_move=Color.WHITE)

    with pytest.raises(ValueError, match="promotion piece"):
        board.apply_move(Move(parse_square("a7"), parse_square("a8"), PieceType.KING))


def test_legal_moves_filter_moves_that_leave_king_in_check() -> None:
    board = board_from_ascii("k3r3/8/8/8/8/8/4R3/4K3", side_to_move=Color.WHITE)

    pseudo = move_set(pseudo_legal_moves(board))
    legal = move_set(legal_moves(board))

    assert "e2d2" in pseudo
    assert "e2d2" not in legal
    assert "e2e8" in legal


def test_apply_move_updates_castling_rights_and_en_passant_target() -> None:
    board = Board.starting_position()

    after_pawn = board.apply_move(Move.from_uci("e2e4"))
    assert after_pawn.en_passant_target == parse_square("e3")
    assert after_pawn.castling_rights == frozenset("KQkq")

    rook_board = board_from_ascii(
        "r3k2r/8/8/8/8/8/8/R3K2R",
        side_to_move=Color.WHITE,
        castling_rights=frozenset("KQkq"),
    )
    after_rook = rook_board.apply_move(Move.from_uci("h1h2"))
    assert after_rook.castling_rights == frozenset("Qkq")
