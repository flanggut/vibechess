from __future__ import annotations

import pytest

from tinychess.engine import (
    Color,
    Game,
    Move,
    OutcomeReason,
    random_move_selector,
    simulate_game,
)
from tinychess.engine.board import board_from_ascii
from tinychess.engine.game import has_insufficient_material


def test_game_records_history_and_clocks() -> None:
    game = Game.new()

    after_e4 = game.play(Move.from_uci("e2e4"))
    after_e5 = after_e4.play(Move.from_uci("e7e5"))
    after_nf3 = after_e5.play(Move.from_uci("g1f3"))

    assert after_e4.moves == (Move.from_uci("e2e4"),)
    assert len(after_e4.positions) == 2
    assert after_e4.halfmove_clock == 0
    assert after_e4.fullmove_number == 1
    assert after_e5.halfmove_clock == 0
    assert after_e5.fullmove_number == 2
    assert after_nf3.halfmove_clock == 1
    assert after_nf3.board.side_to_move is Color.BLACK


def test_game_rejects_illegal_move() -> None:
    game = Game.new()

    with pytest.raises(ValueError, match="illegal move"):
        game.play(Move.from_uci("e2e5"))


def test_play_known_legal_matches_play_for_representative_moves() -> None:
    quiet = (Game.new(), Move.from_uci("e2e4"))

    capture_game = Game.new().play(Move.from_uci("e2e4")).play(Move.from_uci("d7d5"))
    capture = (capture_game, Move.from_uci("e4d5"))

    castle = (
        Game.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
        Move.from_uci("e1g1"),
    )

    en_passant_game = Game.new()
    for notation in ("e2e4", "h7h5", "e4e5", "d7d5"):
        en_passant_game = en_passant_game.play(Move.from_uci(notation))
    en_passant = (en_passant_game, Move.from_uci("e5d6"))

    promotion = (
        Game.from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"),
        Move.from_uci("a7a8q"),
    )

    repetition_game = Game.new(
        board_from_ascii("r3k3/8/8/8/8/8/8/R3K3", side_to_move=Color.WHITE)
    )
    for notation in ("a1a2", "a8a7", "a2a1", "a7a8"):
        repetition_game = repetition_game.play(Move.from_uci(notation))
    repetition_sensitive = (repetition_game, Move.from_uci("a1a2"))

    for game, move in (
        quiet,
        capture,
        castle,
        en_passant,
        promotion,
        repetition_sensitive,
    ):
        assert move in game.legal_moves
        assert game.play_known_legal(move) == game.play(move)


def test_fools_mate_is_checkmate() -> None:
    game = Game.new()
    for notation in ("f2f3", "e7e5", "g2g4", "d8h4"):
        game = game.play(Move.from_uci(notation))

    assert game.outcome is not None
    assert game.outcome.reason is OutcomeReason.CHECKMATE
    assert game.outcome.winner is Color.BLACK


def test_stalemate_detection() -> None:
    game = Game.new(
        board_from_ascii("7k/5Q2/6K1/8/8/8/8/8", side_to_move=Color.BLACK)
    )

    assert game.outcome is not None
    assert game.outcome.reason is OutcomeReason.STALEMATE
    assert game.outcome.is_draw


def test_halfmove_draw_detection() -> None:
    game = Game.new(
        board_from_ascii("r3k3/8/8/8/8/8/8/R3K3", side_to_move=Color.WHITE)
    )
    game = Game(
        positions=game.positions,
        moves=game.moves,
        halfmove_clock=100,
        fullmove_number=game.fullmove_number,
        repetition_counts=game.repetition_counts,
    )

    assert game.outcome is not None
    assert game.outcome.reason is OutcomeReason.FIFTY_MOVE


def test_repetition_draw_detection() -> None:
    game = Game.new(
        board_from_ascii("r3k3/8/8/8/8/8/8/R3K3", side_to_move=Color.WHITE)
    )
    for notation in (
        "a1a2",
        "a8a7",
        "a2a1",
        "a7a8",
        "a1a2",
        "a8a7",
        "a2a1",
        "a7a8",
    ):
        game = game.play(Move.from_uci(notation))

    assert game.outcome is not None
    assert game.outcome.reason is OutcomeReason.REPETITION


def test_insufficient_material_detection() -> None:
    kings_only = board_from_ascii("4k3/8/8/8/8/8/8/4K3")
    king_and_bishop = board_from_ascii("4k3/8/8/8/8/8/8/3BK3")
    king_and_rook = board_from_ascii("4k3/8/8/8/8/8/8/R3K3")

    assert has_insufficient_material(kings_only)
    assert has_insufficient_material(king_and_bishop)
    assert not has_insufficient_material(king_and_rook)
    outcome = Game.new(kings_only).outcome
    assert outcome is not None
    assert outcome.reason is OutcomeReason.INSUFFICIENT_MATERIAL


def test_game_rejects_play_after_outcome() -> None:
    game = Game.new(board_from_ascii("7k/5Q2/6K1/8/8/8/8/8", side_to_move=Color.BLACK))

    assert game.outcome is not None
    with pytest.raises(ValueError, match="cannot play move after game outcome"):
        game.play(Move.from_uci("h8g8"))


def test_repetition_counts_are_copied_between_game_snapshots() -> None:
    game = Game.new()
    after_e4 = game.play(Move.from_uci("e2e4"))

    assert after_e4.repetition_counts is not game.repetition_counts
    assert len(game.repetition_counts) == 1


def test_seeded_random_simulations_are_deterministic() -> None:
    first = simulate_game(random_move_selector(seed=7), max_plies=40)
    second = simulate_game(random_move_selector(seed=7), max_plies=40)

    assert first.moves == second.moves
    assert first.outcome == second.outcome


def test_simulate_game_completes_with_deterministic_random_selector() -> None:
    game = simulate_game(random_move_selector(seed=7), max_plies=40)

    assert len(game.moves) <= 40
    assert game.outcome is not None
    assert game.outcome.reason in {
        OutcomeReason.CHECKMATE,
        OutcomeReason.STALEMATE,
        OutcomeReason.FIFTY_MOVE,
        OutcomeReason.REPETITION,
        OutcomeReason.INSUFFICIENT_MATERIAL,
        OutcomeReason.MAX_PLIES,
    }


def test_simulate_game_rejects_negative_ply_cap() -> None:
    with pytest.raises(ValueError, match="max_plies"):
        simulate_game(random_move_selector(seed=1), max_plies=-1)
