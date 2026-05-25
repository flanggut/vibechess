from __future__ import annotations

import random
from dataclasses import dataclass

import mlx.core as mx
import pytest

from tinychess.ai.mcts import MCTSPlayer
from tinychess.ai.neural_mcts import (
    NeuralMCTSConfig,
    NeuralMCTSNode,
    NeuralMCTSPlayer,
    _select_by_temperature,
)
from tinychess.ai.player import NoLegalMoveError, RandomPlayer, play_game
from tinychess.ai.search_config import MCTSConfig
from tinychess.engine import Game, Move, OutcomeReason
from tinychess.engine.board import board_from_ascii
from tinychess.engine.piece import Color
from tinychess.nn.encode import ACTION_SPACE_SIZE, move_to_action_index
from tinychess.nn.model import (
    InferenceResult,
    PolicyValueConfig,
    PolicyValueInference,
    PolicyValueNet,
)


@dataclass(slots=True)
class FakeInference:
    value: float = 0.25
    preferred_move: Move | None = None
    illegal_move: Move | None = None
    calls: int = 0

    def predict(self, game: Game, *, mask_legal_moves: bool = True) -> InferenceResult:
        assert mask_legal_moves is True
        self.calls += 1
        values = [0.0] * ACTION_SPACE_SIZE
        if self.illegal_move is not None:
            values[move_to_action_index(self.illegal_move, game.board)] = 1000.0
        if self.preferred_move in game.legal_moves:
            values[move_to_action_index(self.preferred_move, game.board)] = 1.0
        elif game.legal_moves:
            values[move_to_action_index(game.legal_moves[0], game.board)] = 1.0
        policy = mx.array(values, dtype=mx.float32)
        mx.eval(policy)
        return InferenceResult(policy_logits=policy, policy=policy, value=self.value)


def test_neural_mcts_selects_legal_move_from_start_position() -> None:
    game = Game.new()
    player = NeuralMCTSPlayer(FakeInference(), NeuralMCTSConfig(simulations=8, seed=1))

    move = player.select_move(game)

    assert move in game.legal_moves
    assert player.last_result is not None
    assert player.last_result.simulations == 8


def test_neural_mcts_with_policy_value_net_searches_start_position() -> None:
    game = Game.new()
    tiny_config = PolicyValueConfig(
        residual_channels=4,
        residual_blocks=0,
        policy_channels=1,
        value_channels=1,
        value_hidden_dim=4,
    )
    player = NeuralMCTSPlayer(
        PolicyValueInference(PolicyValueNet(tiny_config)),
        NeuralMCTSConfig(simulations=1, seed=1),
    )

    result = player.search(game)

    assert result.move in game.legal_moves
    assert result.simulations == 1
    assert result.nodes > 1
    assert result.visit_counts


def test_neural_mcts_masks_illegal_policy_actions_before_expansion() -> None:
    game = Game.new()
    illegal = Move.from_uci("e2e5")
    preferred = Move.from_uci("e2e4")
    player = NeuralMCTSPlayer(
        FakeInference(preferred_move=preferred, illegal_move=illegal),
        NeuralMCTSConfig(simulations=4, puct_exploration=0.0, seed=2),
    )

    result = player.search(game)

    assert result.move in game.legal_moves
    assert illegal not in result.visit_counts
    assert set(result.visit_counts).issubset(set(game.legal_moves))


def test_neural_value_backup_flips_side_to_move_perspective() -> None:
    root = NeuralMCTSNode(Game.new())
    move = Move.from_uci("e2e4")
    child = NeuralMCTSNode(root.game.play(move), parent=root, move=move, prior=1.0)
    root.children[move] = child

    NeuralMCTSPlayer._backup(child, 0.7)

    assert child.visits == 1
    assert child.total_value == pytest.approx(0.7)
    assert root.visits == 1
    assert root.total_value == pytest.approx(-0.7)


def test_neural_puct_selection_uses_child_value_from_parent_perspective() -> None:
    root = NeuralMCTSNode(Game.new())
    first_move, second_move = root.game.legal_moves[:2]
    bad_for_root = NeuralMCTSNode(
        root.game.play(first_move),
        parent=root,
        move=first_move,
        prior=0.5,
        visits=5,
        total_value=4.0,
    )
    good_for_root = NeuralMCTSNode(
        root.game.play(second_move),
        parent=root,
        move=second_move,
        prior=0.5,
        visits=5,
        total_value=-1.0,
    )
    root.visits = 10
    root.children = {first_move: bad_for_root, second_move: good_for_root}

    assert root.best_child(exploration=0.0).move == second_move


def test_neural_puct_selection_uses_priors_when_values_and_visits_match() -> None:
    root = NeuralMCTSNode(Game.new(), visits=6)
    low_prior_move, high_prior_move = root.game.legal_moves[:2]
    low_prior = NeuralMCTSNode(
        root.game.play(low_prior_move),
        parent=root,
        move=low_prior_move,
        prior=0.1,
        visits=3,
        total_value=0.0,
    )
    high_prior = NeuralMCTSNode(
        root.game.play(high_prior_move),
        parent=root,
        move=high_prior_move,
        prior=0.9,
        visits=3,
        total_value=0.0,
    )
    root.children = {low_prior_move: low_prior, high_prior_move: high_prior}

    assert root.best_child(exploration=1.5).move == high_prior_move


def test_select_by_temperature_zero_selects_highest_visit_count() -> None:
    root = NeuralMCTSNode(Game.new())
    low_visit_move, high_visit_move = root.game.legal_moves[:2]
    root.children = {
        low_visit_move: NeuralMCTSNode(
            root.game.play(low_visit_move), parent=root, move=low_visit_move, visits=1
        ),
        high_visit_move: NeuralMCTSNode(
            root.game.play(high_visit_move), parent=root, move=high_visit_move, visits=5
        ),
    }

    selected = _select_by_temperature(root, temperature=0.0, rng=random.Random(1))

    assert selected == high_visit_move


def test_select_by_temperature_positive_weights_exclude_zero_visit_children() -> None:
    root = NeuralMCTSNode(Game.new())
    visited_move, zero_visit_move = root.game.legal_moves[:2]
    root.children = {
        visited_move: NeuralMCTSNode(
            root.game.play(visited_move), parent=root, move=visited_move, visits=2
        ),
        zero_visit_move: NeuralMCTSNode(
            root.game.play(zero_visit_move), parent=root, move=zero_visit_move, visits=0
        ),
    }

    selections = {
        _select_by_temperature(root, temperature=1.0, rng=random.Random(seed))
        for seed in range(10)
    }

    assert selections == {visited_move}


def test_select_by_temperature_all_zero_visits_falls_back_safely() -> None:
    root = NeuralMCTSNode(Game.new())
    prior_move, other_move = root.game.legal_moves[:2]
    root.children = {
        prior_move: NeuralMCTSNode(
            root.game.play(prior_move),
            parent=root,
            move=prior_move,
            visits=0,
            prior=1.0,
        ),
        other_move: NeuralMCTSNode(
            root.game.play(other_move),
            parent=root,
            move=other_move,
            visits=0,
            prior=0.0,
        ),
    }

    selected = _select_by_temperature(root, temperature=1.0, rng=random.Random(1))

    assert selected == prior_move


def test_neural_mcts_temperature_zero_is_deterministic() -> None:
    game = Game.new()
    config = NeuralMCTSConfig(simulations=10, temperature=0.0, seed=7)

    first = NeuralMCTSPlayer(FakeInference(), config).select_move(game)
    second = NeuralMCTSPlayer(FakeInference(), config).select_move(game)

    assert first == second
    assert first in game.legal_moves


def test_neural_mcts_positive_temperature_returns_legal_move() -> None:
    game = Game.new()
    player = NeuralMCTSPlayer(
        FakeInference(), NeuralMCTSConfig(simulations=6, temperature=1.0, seed=3)
    )

    move = player.select_move(game)

    assert move in game.legal_moves


def test_neural_mcts_respects_node_budget() -> None:
    player = NeuralMCTSPlayer(
        FakeInference(), NeuralMCTSConfig(simulations=6, node_budget=3, seed=2)
    )

    result = player.search(Game.new())

    assert result.nodes <= 3
    assert result.move in Game.new().legal_moves


def test_neural_mcts_terminal_position_raises_clear_error() -> None:
    game = Game.new(board_from_ascii("7k/5Q2/6K1/8/8/8/8/8", side_to_move=Color.BLACK))

    with pytest.raises(NoLegalMoveError, match="terminal game: stalemate"):
        NeuralMCTSPlayer(FakeInference(), NeuralMCTSConfig(simulations=2)).select_move(game)


def test_neural_mcts_vs_random_smoke_path_reaches_ply_cap() -> None:
    game = play_game(
        NeuralMCTSPlayer(FakeInference(), NeuralMCTSConfig(simulations=3, seed=1)),
        RandomPlayer(seed=2),
        max_plies=4,
    )

    assert len(game.moves) == 4
    assert game.outcome is not None
    assert game.outcome.reason is OutcomeReason.MAX_PLIES


def test_neural_mcts_vs_classical_mcts_smoke_path_is_legal() -> None:
    game = play_game(
        NeuralMCTSPlayer(FakeInference(), NeuralMCTSConfig(simulations=2, seed=1)),
        MCTSPlayer(MCTSConfig(simulations=1, max_rollout_plies=0, seed=2)),
        max_plies=2,
    )

    assert len(game.moves) == 2
    assert game.outcome is not None
    assert game.outcome.reason is OutcomeReason.MAX_PLIES
