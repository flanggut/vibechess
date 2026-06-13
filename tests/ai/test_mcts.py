from __future__ import annotations

import random
import subprocess
import sys
from io import StringIO

import pytest

from vibechess.ai.mcts import MCTSNode, MCTSPlayer, _static_leaf_value
from vibechess.ai.player import NoLegalMoveError, RandomPlayer, play_game
from vibechess.ai.search_config import MCTSConfig
from vibechess.engine import Game, Move, Outcome, OutcomeReason
from vibechess.engine.board import board_from_ascii
from vibechess.engine.piece import Color
from vibechess.ui.terminal import PlayConfig, play_terminal


def test_mcts_player_selects_legal_move_from_start_position() -> None:
    game = Game.new()
    player = MCTSPlayer(MCTSConfig(simulations=8, seed=1, max_rollout_plies=4))

    move = player.select_move(game)

    assert move in game.legal_moves
    assert player.last_result is not None
    assert player.last_result.simulations == 8


def test_mcts_node_create_caches_legal_moves_and_outcome() -> None:
    game = Game.new()

    node = MCTSNode.create(game, rng=random.Random(1))

    assert node.legal_moves == game.legal_moves
    assert node.outcome == game.outcome
    assert sorted(node.untried_moves, key=Move.to_uci) == sorted(game.legal_moves, key=Move.to_uci)
    assert not node.is_terminal


def test_mcts_node_create_caches_terminal_outcome_without_untried_moves() -> None:
    game = Game.new(board_from_ascii("7k/5Q2/6K1/8/8/8/8/8", side_to_move=Color.BLACK))

    node = MCTSNode.create(game, rng=random.Random(1))

    assert node.outcome == game.outcome
    assert node.legal_moves == ()
    assert node.untried_moves == []
    assert node.is_terminal


def test_mcts_player_terminal_position_raises_clear_error() -> None:
    game = Game.new(board_from_ascii("7k/5Q2/6K1/8/8/8/8/8", side_to_move=Color.BLACK))

    with pytest.raises(NoLegalMoveError, match="terminal game: stalemate"):
        MCTSPlayer(MCTSConfig(simulations=2, seed=1)).select_move(game)


def test_mcts_player_respects_simulation_and_node_budgets() -> None:
    player = MCTSPlayer(MCTSConfig(simulations=6, node_budget=3, seed=2, max_rollout_plies=2))

    result = player.search(Game.new())

    assert result.simulations == 6
    assert result.nodes <= 3
    assert result.move in Game.new().legal_moves


def test_mcts_player_seeded_selection_is_deterministic() -> None:
    game = Game.new()
    config = MCTSConfig(simulations=12, seed=7, max_rollout_plies=4)

    first = MCTSPlayer(config).select_move(game)
    second = MCTSPlayer(config).select_move(game)

    assert first == second


def test_mcts_zero_rollout_plies_uses_static_leaf_mode() -> None:
    game = Game.new()
    node = MCTSNode.create(game, rng=random.Random(1))
    player = MCTSPlayer(MCTSConfig(simulations=5, seed=5, max_rollout_plies=0))

    assert player._rollout_value(
        game,
        Color.WHITE,
        legal_moves=node.legal_moves,
        outcome=node.outcome,
    ) == 0.0
    result = player.search(game)

    assert result.simulations == 5
    assert result.move in game.legal_moves


def test_static_leaf_value_handles_terminal_outcomes_from_root_perspective() -> None:
    game = Game.new()

    assert _static_leaf_value(
        game,
        Color.WHITE,
        outcome=Outcome(OutcomeReason.CHECKMATE, winner=Color.WHITE),
    ) == 1.0
    assert _static_leaf_value(
        game,
        Color.WHITE,
        outcome=Outcome(OutcomeReason.CHECKMATE, winner=Color.BLACK),
    ) == -1.0
    assert _static_leaf_value(game, Color.WHITE, outcome=Outcome(OutcomeReason.STALEMATE)) == 0.0


def test_mcts_opponent_nodes_minimize_root_value() -> None:
    root_color = Color.WHITE
    black_to_move = Game.new().play(Move.from_uci("e2e4"))
    node = MCTSNode.create(black_to_move, rng=random.Random(1))
    favorable = MCTSNode.create(
        black_to_move.play(Move.from_uci("e7e5")),
        rng=random.Random(2),
        parent=node,
        move=Move.from_uci("e7e5"),
    )
    refutation = MCTSNode.create(
        black_to_move.play(Move.from_uci("c7c5")),
        rng=random.Random(3),
        parent=node,
        move=Move.from_uci("c7c5"),
    )
    node.visits = 20
    favorable.visits = refutation.visits = 10
    favorable.total_value = 8.0
    refutation.total_value = -2.0
    assert favorable.move is not None
    assert refutation.move is not None
    node.children = {favorable.move: favorable, refutation.move: refutation}

    assert node.best_child(exploration=0.0, root_color=root_color).move == Move.from_uci("c7c5")


def test_mcts_player_zero_time_budget_returns_legal_fallback_without_searching() -> None:
    game = Game.new()
    player = MCTSPlayer(MCTSConfig(simulations=100, time_limit_seconds=0, seed=3))

    result = player.search(game)

    assert result.simulations == 0
    assert result.nodes == 1
    assert result.move in game.legal_moves


def test_mcts_reuses_selected_child_after_actual_move() -> None:
    game = Game.new()
    player = MCTSPlayer(MCTSConfig(simulations=1, seed=1, max_rollout_plies=0))

    first = player.search(game)
    assert player._tree_root is not None
    saved_child = player._tree_root.children[first.move]

    second = player.search(game.play(first.move))

    assert player._tree_root is saved_child
    assert saved_child.parent is None
    assert second.move in saved_child.game.legal_moves


def test_mcts_reuses_deeper_descendant_and_counts_only_new_nodes() -> None:
    game = Game.new()
    move = Move.from_uci("e2e4")
    reply = Move.from_uci("e7e5")
    next_game = game.play(move)
    target_game = next_game.play(reply)
    root = MCTSNode.create(game, rng=random.Random(1))
    child = MCTSNode.create(next_game, rng=random.Random(2), parent=root, move=move)
    grandchild = MCTSNode.create(target_game, rng=random.Random(3), parent=child, move=reply)
    root.children[move] = child
    child.children[reply] = grandchild
    player = MCTSPlayer(MCTSConfig(simulations=10, time_limit_seconds=0, seed=4))
    player._tree_root = root

    result = player.search(target_game)

    assert player._tree_root is grandchild
    assert grandchild.parent is None
    assert result.simulations == 0
    assert result.nodes == 0
    assert result.move in target_game.legal_moves


def test_mcts_falls_back_when_descendant_path_is_missing() -> None:
    game = Game.new()
    player = MCTSPlayer(MCTSConfig(simulations=1, seed=1, max_rollout_plies=0))

    first = player.search(game)
    assert player._tree_root is not None
    root = player._tree_root
    next_game = game.play(first.move)
    missing_reply = next_game.legal_moves[0]
    target_game = next_game.play(missing_reply)

    player.search(target_game)

    assert player._tree_root is not None
    assert player._tree_root is not root.children[first.move]
    assert player._tree_root.game == target_game
    assert player._tree_root.parent is None


def test_mcts_does_not_reuse_unrelated_game_with_matching_empty_history() -> None:
    game = Game.new()
    unrelated = Game.from_fen(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    )
    root = MCTSNode.create(game, rng=random.Random(1))
    player = MCTSPlayer(MCTSConfig(simulations=10, time_limit_seconds=0, seed=2))
    player._tree_root = root

    result = player.search(unrelated)

    assert player._tree_root is not root
    assert player._tree_root is not None
    assert player._tree_root.game == unrelated
    assert result.nodes == 1
    assert result.move in unrelated.legal_moves


def test_mcts_clear_tree_forces_fresh_root() -> None:
    game = Game.new()
    root = MCTSNode.create(game, rng=random.Random(1))
    player = MCTSPlayer(MCTSConfig(simulations=10, time_limit_seconds=0, seed=2))
    player._tree_root = root

    player.clear_tree()
    result = player.search(game)

    assert player._tree_root is not root
    assert player._tree_root is not None
    assert player._tree_root.game == game
    assert result.nodes == 1


def test_mcts_reuse_tree_false_keeps_fresh_root_behavior() -> None:
    game = Game.new()
    root = MCTSNode.create(game, rng=random.Random(1))
    player = MCTSPlayer(
        MCTSConfig(simulations=10, time_limit_seconds=0, seed=2, reuse_tree=False)
    )
    player._tree_root = root

    result = player.search(game)

    assert player._tree_root is not root
    assert player._tree_root is not None
    assert player._tree_root.game == game
    assert result.nodes == 1


def test_mcts_config_defaults_to_static_leaf_mode() -> None:
    assert MCTSConfig().max_rollout_plies == 0


def test_mcts_config_validates_budgets() -> None:
    with pytest.raises(ValueError, match="simulations"):
        MCTSConfig(simulations=0)
    with pytest.raises(ValueError, match="node_budget"):
        MCTSConfig(node_budget=0)
    with pytest.raises(ValueError, match="time_limit"):
        MCTSConfig(time_limit_seconds=-1)
    with pytest.raises(ValueError, match="max_rollout_plies"):
        MCTSConfig(max_rollout_plies=-1)


def test_mcts_vs_random_smoke_path_reaches_ply_cap() -> None:
    game = play_game(
        MCTSPlayer(MCTSConfig(simulations=3, seed=1, max_rollout_plies=2)),
        RandomPlayer(seed=2),
        max_plies=4,
    )

    assert len(game.moves) == 4
    assert game.outcome is not None
    assert game.outcome.reason is OutcomeReason.MAX_PLIES


def test_terminal_play_supports_mcts_player() -> None:
    output = StringIO()

    game = play_terminal(
        PlayConfig(white="mcts", black="random", max_plies=1, seed=4, mcts_simulations=1),
        stdout=output,
    )

    assert len(game.moves) == 1
    assert "white mcts plays" in output.getvalue()


def test_mcts_benchmark_script_smoke() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/mcts_benchmark.py",
            "--simulations",
            "1",
            "--rollout-plies",
            "0",
            "--seed",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "bestmove=" in result.stdout
    assert "simulations=1" in result.stdout
    assert "sims_per_sec=" in result.stdout


def test_mcts_benchmark_fast_leaf_flag_smoke() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/mcts_benchmark.py",
            "--simulations",
            "1",
            "--fast-leaf",
            "--seed",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "bestmove=" in result.stdout
    assert "simulations=1" in result.stdout


def test_mcts_benchmark_fast_leaf_rejects_explicit_rollout_plies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/mcts_benchmark.py",
            "--simulations",
            "1",
            "--fast-leaf",
            "--rollout-plies",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "cannot be combined" in result.stderr
