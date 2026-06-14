from __future__ import annotations

import random

import pytest

from vibechess.engine import (
    Board,
    Color,
    Move,
    Piece,
    PieceType,
    legal_moves,
    perft,
    pseudo_legal_moves,
)
from vibechess.engine.board import board_from_ascii
from vibechess.engine.fen import parse_fen
from vibechess.engine.legal_moves import has_legal_move, is_in_check, is_square_attacked
from vibechess.engine.square import parse_square


def move_set(moves: tuple[Move, ...]) -> set[str]:
    return {move.to_uci() for move in moves}


def reference_legal_moves_by_full_check(board: Board) -> tuple[Move, ...]:
    moving_color = board.side_to_move
    return tuple(
        move
        for move in pseudo_legal_moves(board)
        if not is_in_check(board.apply_move(move), moving_color)
    )


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


@pytest.mark.parametrize(
    "fen",
    [
        "startpos",
        "r3k2r/p1ppqpb1/bn2pnp1/2pPN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "4k3/8/8/r2pP2K/8/8/8/8 w - d6 0 1",
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "rnbq1rk1/ppp2ppp/3bpn2/3p4/3P4/2PBPN2/PP3PPP/RNBQ1RK1 w - - 0 7",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ],
)
def test_legal_moves_matches_full_check_reference_on_fixtures(fen: str) -> None:
    board = Board.starting_position() if fen == "startpos" else parse_fen(fen).board

    assert move_set(legal_moves(board)) == move_set(reference_legal_moves_by_full_check(board))


@pytest.mark.parametrize(
    "fen",
    [
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/8/8/r2pP2K/8/8/8/8 w - d6 0 1",
        "8/8/8/8/R2Pp2k/8/8/4K3 b - d3 0 1",
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/p7/4K3 b - - 0 1",
        "k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1",
    ],
)
def test_legal_moves_matches_reference_on_scratch_restoration_fixtures(fen: str) -> None:
    board = parse_fen(fen).board
    first_moves = move_set(legal_moves(board))
    second_moves = move_set(legal_moves(board))

    assert first_moves == second_moves
    assert first_moves == move_set(reference_legal_moves_by_full_check(board))


@pytest.mark.parametrize(
    ("fen", "en_passant_uci"),
    [
        ("4k3/8/8/r2pP2K/8/8/8/8 w - d6 0 1", "e5d6"),
        ("8/8/8/8/R2Pp2k/8/8/4K3 b - d3 0 1", "e4d3"),
    ],
)
def test_en_passant_discovered_horizontal_check_is_illegal(
    fen: str, en_passant_uci: str
) -> None:
    board = parse_fen(fen).board
    pseudo = move_set(pseudo_legal_moves(board))
    legal = move_set(legal_moves(board))
    reference = move_set(reference_legal_moves_by_full_check(board))

    assert en_passant_uci in pseudo
    assert legal == reference
    assert en_passant_uci not in legal


def test_legal_moves_matches_full_check_reference_on_random_legal_game_states() -> None:
    rng = random.Random(20260603)
    board = Board.starting_position()

    for _ in range(80):
        moves = legal_moves(board)

        assert move_set(moves) == move_set(reference_legal_moves_by_full_check(board))
        if not moves:
            break
        board = board.apply_move(rng.choice(moves))


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


def test_attack_detection_uses_pawn_direction_without_edge_wrap() -> None:
    board = board_from_ascii(
        "4k3/1p6/8/8/8/P7/1P6/4K3",
        side_to_move=Color.WHITE,
    )

    assert is_square_attacked(board, parse_square("a3"), Color.WHITE)
    assert is_square_attacked(board, parse_square("a6"), Color.BLACK)
    assert not is_square_attacked(board, parse_square("a1"), Color.WHITE)
    assert not is_square_attacked(board, parse_square("a8"), Color.BLACK)


def test_attack_detection_uses_leaper_tables_without_edge_wrap() -> None:
    board = board_from_ascii("4k3/8/8/8/8/1n6/8/K7", side_to_move=Color.WHITE)

    assert is_square_attacked(board, parse_square("a1"), Color.BLACK)
    assert is_square_attacked(board, parse_square("d2"), Color.BLACK)
    assert not is_square_attacked(board, parse_square("h1"), Color.BLACK)


def test_attack_detection_respects_slider_blockers() -> None:
    clear_rook = board_from_ascii("4r3/8/8/8/8/8/8/4K2k", side_to_move=Color.WHITE)
    blocked_rook = board_from_ascii("4r3/8/8/8/4P3/8/8/4K2k", side_to_move=Color.WHITE)
    clear_bishop = board_from_ascii("4k3/8/8/8/7b/8/8/4K3", side_to_move=Color.WHITE)
    blocked_bishop = board_from_ascii("4k3/8/8/8/7b/8/5P2/4K3", side_to_move=Color.WHITE)

    assert is_square_attacked(clear_rook, parse_square("e1"), Color.BLACK)
    assert not is_square_attacked(blocked_rook, parse_square("e1"), Color.BLACK)
    assert is_square_attacked(clear_bishop, parse_square("e1"), Color.BLACK)
    assert not is_square_attacked(blocked_bishop, parse_square("e1"), Color.BLACK)


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


def test_apply_move_rejects_own_piece_target() -> None:
    board = Board.starting_position()

    with pytest.raises(ValueError, match="own piece"):
        board.apply_move(Move.from_uci("e1e2"))


@pytest.mark.parametrize("uci", ["e2e4q", "e7e6q", "a8h8q"])
def test_apply_move_rejects_invalid_white_promotion_rank_or_direction(uci: str) -> None:
    board = board_from_ascii("P3k3/4P3/8/8/8/8/4P3/4K3", side_to_move=Color.WHITE)

    with pytest.raises(ValueError, match="promotion"):
        board.apply_move(Move.from_uci(uci))


@pytest.mark.parametrize("uci", ["e7e5q", "e2e3q", "a1h1q"])
def test_apply_move_rejects_invalid_black_promotion_rank_or_direction(uci: str) -> None:
    board = board_from_ascii(
        "4k3/4p3/8/8/8/8/4p3/p3K3",
        side_to_move=Color.BLACK,
    )

    with pytest.raises(ValueError, match="promotion"):
        board.apply_move(Move.from_uci(uci))


def test_apply_move_accepts_legal_white_and_black_promotions() -> None:
    white_board = board_from_ascii("1r2k3/P7/8/8/8/8/8/4K3", side_to_move=Color.WHITE)
    after_white = white_board.apply_move(Move.from_uci("a7b8q"))
    assert after_white.piece_at("b8") == Piece(Color.WHITE, PieceType.QUEEN)
    assert after_white.piece_at("a7") is None
    assert after_white.side_to_move is Color.BLACK

    black_board = board_from_ascii("4k3/8/8/8/8/8/p7/4K3", side_to_move=Color.BLACK)
    after_black = black_board.apply_move(Move.from_uci("a2a1q"))
    assert after_black.piece_at("a1") == Piece(Color.BLACK, PieceType.QUEEN)
    assert after_black.piece_at("a2") is None
    assert after_black.side_to_move is Color.WHITE


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
