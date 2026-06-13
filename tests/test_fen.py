from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from vibechess.engine import (
    STARTING_FEN,
    STARTPOS_FEN,
    Board,
    Color,
    Game,
    Move,
    Piece,
    PieceType,
    board_from_fen,
    board_to_fen,
    legal_moves,
    parse_fen,
)
from vibechess.engine.fen import FenPosition, format_fen
from vibechess.engine.square import parse_square

FIXTURES = Path(__file__).parent / "fixtures" / "fen_positions.json"


def test_starting_position_fen_constant_round_trips() -> None:
    position = parse_fen(STARTING_FEN)

    assert position.board == Board.starting_position()
    assert position.halfmove_clock == 0
    assert position.fullmove_number == 1
    assert format_fen(position) == STARTING_FEN
    assert STARTPOS_FEN == STARTING_FEN
    assert Board.starting_position().to_fen() == STARTING_FEN
    assert Board.from_fen(STARTING_FEN) == Board.starting_position()


@pytest.mark.parametrize(
    "fixture",
    json.loads(FIXTURES.read_text()),
    ids=lambda fixture: fixture["name"],
)
def test_fixture_positions_validate_engine_expectations(fixture: dict[str, object]) -> None:
    position = parse_fen(str(fixture["fen"]))

    assert position.to_fen() == fixture["fen"]
    assert position.board.side_to_move.value == fixture["side_to_move"]
    assert position.board.castling_rights == frozenset(str(fixture["castling_rights"]))
    expected_ep = fixture["en_passant_target"]
    assert position.board.en_passant_target == (
        None if expected_ep is None else parse_square(str(expected_ep))
    )
    assert position.halfmove_clock == fixture["halfmove_clock"]
    assert position.fullmove_number == fixture["fullmove_number"]
    if "legal_move_count" in fixture:
        assert len(legal_moves(position.board)) == fixture["legal_move_count"]
    if "legal_moves_include" in fixture:
        expected_moves = fixture["legal_moves_include"]
        assert isinstance(expected_moves, Sequence)
        move_set = {move.to_uci() for move in legal_moves(position.board)}
        assert {str(move) for move in expected_moves}.issubset(move_set)


def test_parse_fen_sets_board_state_and_counters() -> None:
    fen = "4k3/8/8/3pP3/8/8/8/4K3 w Kq d6 7 12"

    position = parse_fen(fen)

    assert position.board.side_to_move is Color.WHITE
    assert position.board.castling_rights == frozenset("Kq")
    assert position.board.en_passant_target == parse_square("d6")
    assert position.board.piece_at("e5") == Piece(Color.WHITE, PieceType.PAWN)
    assert position.board.piece_at("d5") == Piece(Color.BLACK, PieceType.PAWN)
    assert position.halfmove_clock == 7
    assert position.fullmove_number == 12


def test_board_to_fen_uses_canonical_castling_order() -> None:
    board = Board.empty(castling_rights=frozenset("qKkQ"))

    assert board_to_fen(board) == "8/8/8/8/8/8/8/8 w KQkq - 0 1"


def test_board_from_fen_returns_board_component() -> None:
    board = board_from_fen("8/8/8/8/8/8/8/8 b - - 3 9")

    assert board.side_to_move is Color.BLACK
    assert board.castling_rights == frozenset()


def test_game_from_fen_and_to_fen_preserve_counters() -> None:
    game = Game.from_fen("4k3/8/8/8/8/8/8/4K3 b - - 99 57")

    assert game.halfmove_clock == 99
    assert game.fullmove_number == 57
    assert game.to_fen() == "4k3/8/8/8/8/8/8/4K3 b - - 99 57"


def test_game_to_fen_updates_after_play() -> None:
    game = Game.new().play(Move.from_uci("e2e4"))

    assert game.to_fen() == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"


@pytest.mark.parametrize(
    ("fen", "match"),
    [
        ("8/8/8/8/8/8/8/8 w - - 0", "6 fields"),
        ("8/8/8/8/8/8/8 w - - 0 1", "8 slash-separated"),
        ("8/8/8/8/8/8/8/9 w - - 0 1", "too many files"),
        ("8/8/8/8/8/8/8/11PPPPPP w - - 0 1", "adjacent empty-square counts"),
        ("8/8/8/8/8/8/8/7Z w - - 0 1", "piece symbol"),
        ("8/8/8/8/8/8/8/8 x - - 0 1", "active color"),
        ("8/8/8/8/8/8/8/8 w KK - 0 1", "duplicate"),
        ("8/8/8/8/8/8/8/8 w A - 0 1", "castling"),
        ("8/8/8/8/8/8/8/8 w - e4 0 1", "rank 6 when white is active"),
        ("4k3/8/8/8/8/8/4pP2/4K3 w - e3 0 1", "rank 6 when white is active"),
        ("8/8/8/8/8/8/8/8 w - - -1 1", "halfmove"),
        ("8/8/8/8/8/8/8/8 w - - 0 0", "positive"),
    ],
)
def test_parse_fen_rejects_malformed_input(fen: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_fen(fen)


def test_fen_position_rejects_invalid_counters() -> None:
    with pytest.raises(ValueError, match="halfmove clock"):
        FenPosition(Board.empty(), halfmove_clock=-1)
    with pytest.raises(ValueError, match="fullmove number"):
        FenPosition(Board.empty(), fullmove_number=0)
