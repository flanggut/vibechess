from __future__ import annotations

import pytest

from vibechess.ai.search_state import SearchState
from vibechess.engine import Game, Move, OutcomeReason


def _assert_play_known_legal_matches_game(game: Game, move_uci: str) -> SearchState:
    move = Move.from_uci(move_uci)
    assert move in game.legal_moves
    state = SearchState.from_game(game)

    actual_state = state.play_known_legal(move)
    expected_game = game.play_known_legal(move)

    assert actual_state.to_game() == expected_game
    assert actual_state.board == expected_game.board
    assert actual_state.moves == expected_game.moves
    assert actual_state.halfmove_clock == expected_game.halfmove_clock
    assert actual_state.fullmove_number == expected_game.fullmove_number
    assert dict(actual_state.repetition_counts) == dict(expected_game.repetition_counts)
    assert actual_state.legal_moves == expected_game.legal_moves
    assert actual_state.outcome == expected_game.outcome

    compact_game = actual_state.to_game(include_positions=False)
    assert compact_game.positions == (expected_game.board,)
    assert compact_game.moves == expected_game.moves
    assert compact_game.halfmove_clock == expected_game.halfmove_clock
    assert compact_game.fullmove_number == expected_game.fullmove_number
    return actual_state


@pytest.mark.parametrize(
    ("fen", "move_uci"),
    [
        ("8/7k/8/8/8/8/4P3/4K3 w - - 7 3", "e2e3"),
        ("7k/8/8/3p4/4P3/8/8/4K3 w - - 12 6", "e4d5"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 9 15", "e1g1"),
        ("7k/8/8/3pP3/8/8/8/4K3 w - d6 0 8", "e5d6"),
        ("7k/P7/8/8/8/8/8/4K3 w - - 0 12", "a7a8q"),
    ],
)
def test_search_state_play_known_legal_matches_game_for_core_move_types(
    fen: str,
    move_uci: str,
) -> None:
    _assert_play_known_legal_matches_game(Game.from_fen(fen), move_uci)


def test_search_state_clocks_match_game_for_black_quiet_move() -> None:
    state = _assert_play_known_legal_matches_game(
        Game.from_fen("7k/8/8/8/8/8/8/4K2r b - - 21 4"),
        "h1h2",
    )

    assert state.halfmove_clock == 22
    assert state.fullmove_number == 5


@pytest.mark.parametrize(
    ("fen", "move_uci"),
    [
        ("7k/8/8/3p4/4P3/8/8/4K3 w - - 12 6", "e4d5"),
        ("7k/8/8/3pP3/8/8/8/4K3 w - d6 12 8", "e5d6"),
        ("7k/P7/8/8/8/8/8/4K3 w - - 12 12", "a7a8q"),
    ],
)
def test_search_state_resets_halfmove_clock_like_game(fen: str, move_uci: str) -> None:
    state = _assert_play_known_legal_matches_game(Game.from_fen(fen), move_uci)

    assert state.halfmove_clock == 0


def test_search_state_repetition_sensitive_outcome_matches_game() -> None:
    game = Game.new()
    for move_uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"):
        game = game.play(Move.from_uci(move_uci))

    state = _assert_play_known_legal_matches_game(game, "f6g8")

    assert state.outcome is not None
    assert state.outcome.reason is OutcomeReason.REPETITION
