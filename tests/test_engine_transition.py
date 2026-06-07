from __future__ import annotations

import pytest

from tinychess.ai.search_state import SearchState
from tinychess.engine import Color, Game, Move, Outcome, OutcomeReason, legal_moves
from tinychess.engine.board import board_from_ascii
from tinychess.engine.transition import (
    TransitionState,
    advance_known_legal_state,
    has_insufficient_material,
    is_capture,
    outcome_for_state,
    position_key,
)


def _transition_state(game: Game) -> TransitionState:
    return TransitionState(
        board=game.board,
        halfmove_clock=game.halfmove_clock,
        fullmove_number=game.fullmove_number,
        repetition_counts=game.repetition_counts,
        forced_outcome=game.forced_outcome,
    )


@pytest.mark.parametrize(
    ("fen", "move_uci"),
    [
        ("8/7k/8/8/8/8/4P3/4K3 w - - 7 3", "e2e3"),
        ("7k/8/8/3p4/4P3/8/8/4K3 w - - 12 6", "e4d5"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 9 15", "e1g1"),
        ("7k/8/8/3pP3/8/8/8/4K3 w - d6 12 8", "e5d6"),
        ("7k/P7/8/8/8/8/8/4K3 w - - 12 12", "a7a8q"),
        ("7k/8/8/8/8/8/8/4K2r b - - 21 4", "h1h2"),
    ],
)
def test_advance_known_legal_state_matches_game_and_search_state(
    fen: str,
    move_uci: str,
) -> None:
    game = Game.from_fen(fen)
    move = Move.from_uci(move_uci)
    assert move in game.legal_moves

    result = advance_known_legal_state(_transition_state(game), move)
    expected_game = game.play_known_legal(move)
    expected_state = SearchState.from_game(game).play_known_legal(move)

    assert result.board == expected_game.board == expected_state.board
    assert result.halfmove_clock == expected_game.halfmove_clock == expected_state.halfmove_clock
    assert result.fullmove_number == expected_game.fullmove_number == expected_state.fullmove_number
    assert result.repetition_counts == dict(expected_game.repetition_counts)
    assert result.repetition_counts == dict(expected_state.repetition_counts)


@pytest.mark.parametrize(
    ("fen", "move_uci", "expected_capture"),
    [
        ("8/7k/8/8/8/8/4P3/4K3 w - - 7 3", "e2e3", False),
        ("7k/8/8/3p4/4P3/8/8/4K3 w - - 12 6", "e4d5", True),
        ("7k/8/8/3pP3/8/8/8/4K3 w - d6 12 8", "e5d6", True),
    ],
)
def test_position_key_and_capture_helpers(
    fen: str,
    move_uci: str,
    expected_capture: bool,
) -> None:
    game = Game.from_fen(fen)
    move = Move.from_uci(move_uci)
    moving_piece = game.board.piece_at(move.from_square)
    assert moving_piece is not None

    assert position_key(game.board) in game.repetition_counts
    assert is_capture(game.board, move, moving_piece) is expected_capture


def test_advance_known_legal_state_rejects_empty_source() -> None:
    game = Game.new()

    with pytest.raises(ValueError, match="empty square"):
        advance_known_legal_state(_transition_state(game), Move.from_uci("a3a4"))


@pytest.mark.parametrize(
    ("game", "expected_reason"),
    [
        (
            Game.new(board_from_ascii("7k/5Q2/6K1/8/8/8/8/8", side_to_move=Color.BLACK)),
            OutcomeReason.STALEMATE,
        ),
        (
            Game.new(board_from_ascii("4k3/8/8/8/8/8/8/4K3")),
            OutcomeReason.INSUFFICIENT_MATERIAL,
        ),
    ],
)
def test_outcome_for_state_matches_pragmatic_draws(
    game: Game,
    expected_reason: OutcomeReason,
) -> None:
    outcome = outcome_for_state(_transition_state(game), legal_moves(game.board))

    assert outcome is not None
    assert outcome.reason is expected_reason
    assert outcome == game.outcome


def test_outcome_for_state_matches_checkmate() -> None:
    game = Game.new()
    for move_uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        game = game.play(Move.from_uci(move_uci))

    outcome = outcome_for_state(_transition_state(game), legal_moves(game.board))

    assert outcome is not None
    assert outcome.reason is OutcomeReason.CHECKMATE
    assert outcome.winner is Color.BLACK
    assert outcome == game.outcome


def test_outcome_for_state_preserves_draw_precedence() -> None:
    board = board_from_ascii("4k3/8/8/8/8/8/8/4K3")
    game = Game(
        positions=(board,),
        halfmove_clock=100,
        fullmove_number=1,
        repetition_counts={position_key(board): 3},
    )

    outcome = outcome_for_state(_transition_state(game), legal_moves(game.board))

    assert outcome is not None
    assert outcome.reason is OutcomeReason.FIFTY_MOVE


def test_outcome_for_state_returns_forced_outcome_first() -> None:
    game = Game(
        positions=(Game.new().board,),
        repetition_counts={position_key(Game.new().board): 1},
        forced_outcome=Outcome(OutcomeReason.MAX_PLIES),
    )

    outcome = outcome_for_state(_transition_state(game), legal_moves(game.board))

    assert outcome == Outcome(OutcomeReason.MAX_PLIES)


def test_has_insufficient_material_helper_matches_engine_game_compatibility_export() -> None:
    from tinychess.engine.game import has_insufficient_material as game_has_insufficient_material

    board = board_from_ascii("4k3/8/8/8/8/8/8/3BK3")

    assert has_insufficient_material(board)
    assert game_has_insufficient_material(board) is has_insufficient_material(board)
