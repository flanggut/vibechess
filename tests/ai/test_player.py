from __future__ import annotations

import random

import pytest

from vibechess.ai.player import NoLegalMoveError, Player, RandomPlayer, play_game
from vibechess.engine import Game, Move, OutcomeReason
from vibechess.engine.board import board_from_ascii
from vibechess.engine.piece import Color


def test_random_player_conforms_to_player_protocol() -> None:
    player: Player = RandomPlayer(seed=1)

    assert isinstance(player, Player)
    assert player.select_move(Game.new()) in Game.new().legal_moves


def test_random_player_seeded_selection_is_deterministic() -> None:
    first = RandomPlayer(seed=7)
    second = RandomPlayer(seed=7)
    game = Game.new()

    first_moves = [first.select_move(game) for _ in range(5)]
    second_moves = [second.select_move(game) for _ in range(5)]

    assert first_moves == second_moves


def test_random_player_uses_provided_local_rng() -> None:
    player = RandomPlayer(rng=random.Random(3))

    assert player.select_move(Game.new()) in Game.new().legal_moves


def test_random_player_rejects_seed_and_rng_together() -> None:
    with pytest.raises(ValueError, match="either seed or rng"):
        RandomPlayer(seed=1, rng=random.Random(1))


def test_random_player_selects_legal_moves_only_after_position_changes() -> None:
    game = Game.new().play(Move.from_uci("e2e4"))
    player = RandomPlayer(seed=11)

    for _ in range(20):
        assert player.select_move(game) in game.legal_moves


def test_random_player_terminal_position_raises_clear_error() -> None:
    game = Game.new(board_from_ascii("7k/5Q2/6K1/8/8/8/8/8", side_to_move=Color.BLACK))

    with pytest.raises(NoLegalMoveError, match="terminal game: stalemate"):
        RandomPlayer(seed=1).select_move(game)


def test_play_game_returns_already_terminal_game_without_moving() -> None:
    terminal_game = Game.new(
        board_from_ascii("7k/5Q2/6K1/8/8/8/8/8", side_to_move=Color.BLACK)
    )

    result = play_game(RandomPlayer(seed=1), RandomPlayer(seed=2), game=terminal_game)

    assert result is terminal_game
    assert result.moves == ()
    assert result.outcome is not None
    assert result.outcome.reason is OutcomeReason.STALEMATE


def test_play_game_random_vs_random_is_deterministic_with_seeds() -> None:
    first = play_game(RandomPlayer(seed=1), RandomPlayer(seed=2), max_plies=40)
    second = play_game(RandomPlayer(seed=1), RandomPlayer(seed=2), max_plies=40)

    assert first.moves == second.moves
    assert first.outcome == second.outcome


def test_play_game_caps_unfinished_random_game() -> None:
    game = play_game(RandomPlayer(seed=1), RandomPlayer(seed=2), max_plies=2)

    assert len(game.moves) == 2
    assert game.outcome is not None
    assert game.outcome.reason is OutcomeReason.MAX_PLIES


def test_play_game_rejects_illegal_player_move() -> None:
    class BadPlayer:
        def select_move(self, _game: Game) -> Move:
            return Move.from_uci("e2e5")

    with pytest.raises(ValueError, match="player selected illegal move"):
        play_game(BadPlayer(), RandomPlayer(seed=1), max_plies=1)


def test_play_game_rejects_negative_ply_cap() -> None:
    with pytest.raises(ValueError, match="max_plies"):
        play_game(RandomPlayer(seed=1), RandomPlayer(seed=2), max_plies=-1)
