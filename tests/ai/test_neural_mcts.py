from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import mlx.core as mx
import pytest

import tinychess.engine.game as game_module
from tinychess.ai.mcts import MCTSPlayer
from tinychess.ai.neural_mcts import (
    NeuralMCTSConfig,
    NeuralMCTSNode,
    NeuralMCTSPlayer,
    _legal_priors,
    _select_by_temperature,
)
from tinychess.ai.player import NoLegalMoveError, RandomPlayer, play_game
from tinychess.ai.search_config import MCTSConfig
from tinychess.engine import Game, Move, OutcomeReason
from tinychess.engine.board import Board, board_from_ascii
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


def _policy_result(values: list[float], *, value: float = 0.0) -> InferenceResult:
    policy = mx.array(values, dtype=mx.float32)
    mx.eval(policy)
    return InferenceResult(policy_logits=policy, policy=policy, value=value)


def test_neural_node_create_caches_legal_moves_and_is_terminal_uses_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_generate = cast(
        Callable[[Board], tuple[Move, ...]], game_module.__dict__["generate_legal_moves"]
    )
    calls = 0

    def counting_generate(board: Board) -> tuple[Move, ...]:
        nonlocal calls
        calls += 1
        return original_generate(board)

    monkeypatch.setattr(game_module, "generate_legal_moves", counting_generate)
    node = NeuralMCTSNode.create(Game.new())

    assert calls == 1
    assert node.legal_moves == original_generate(node.game.board)
    assert node.outcome is None
    assert not node.is_terminal
    assert not node.is_terminal
    assert calls == 1


def test_neural_node_create_caches_terminal_outcome() -> None:
    game = Game.new(board_from_ascii("7k/5Q2/6K1/8/8/8/8/8", side_to_move=Color.BLACK))

    node = NeuralMCTSNode.create(game)

    assert node.legal_moves == ()
    assert node.outcome is not None
    assert node.outcome.reason is OutcomeReason.STALEMATE
    assert node.is_terminal


def test_legal_priors_use_cached_legal_moves_and_filter_illegal_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game = Game.new()
    node = NeuralMCTSNode.create(game)
    preferred = Move.from_uci("e2e4")
    illegal = Move.from_uci("e2e5")
    values = [0.0] * ACTION_SPACE_SIZE
    values[move_to_action_index(illegal, game.board)] = 1000.0
    values[move_to_action_index(preferred, game.board)] = 1.0
    prediction = _policy_result(values)

    def fail_generate(_board: Board) -> tuple[Move, ...]:
        raise AssertionError("legal moves should come from the node cache")

    monkeypatch.setattr(game_module, "generate_legal_moves", fail_generate)

    priors = _legal_priors(node, prediction)

    assert set(priors) == set(node.legal_moves)
    assert illegal not in priors
    assert priors[preferred] == pytest.approx(1.0)


def test_expand_uses_known_legal_transitions_equivalent_to_play(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game = Game.new()
    expected_games = {move: game.play(move) for move in game.legal_moves}
    root = NeuralMCTSNode.create(game)
    player = NeuralMCTSPlayer(FakeInference(), NeuralMCTSConfig(simulations=1, seed=1))

    def fail_play(self: Game, move: Move) -> Game:
        raise AssertionError(f"play() should not validate cached legal move {move}")

    monkeypatch.setattr(Game, "play", fail_play)

    _value, created = player._expand(root)

    assert created == len(root.legal_moves)
    assert set(root.children) == set(root.legal_moves)
    for move, child in root.children.items():
        assert child.game == expected_games[move]
        assert child.parent is root
        assert child.move == move


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
    root = NeuralMCTSNode.create(Game.new())
    move = Move.from_uci("e2e4")
    child = NeuralMCTSNode.create(root.game.play(move), parent=root, move=move, prior=1.0)
    root.children[move] = child

    NeuralMCTSPlayer._backup(child, 0.7)

    assert child.visits == 1
    assert child.total_value == pytest.approx(0.7)
    assert root.visits == 1
    assert root.total_value == pytest.approx(-0.7)


def test_neural_puct_selection_uses_child_value_from_parent_perspective() -> None:
    root = NeuralMCTSNode.create(Game.new())
    first_move, second_move = root.legal_moves[:2]
    bad_for_root = NeuralMCTSNode.create(
        root.game.play(first_move),
        parent=root,
        move=first_move,
        prior=0.5,
    )
    bad_for_root.visits = 5
    bad_for_root.total_value = 4.0
    good_for_root = NeuralMCTSNode.create(
        root.game.play(second_move),
        parent=root,
        move=second_move,
        prior=0.5,
    )
    good_for_root.visits = 5
    good_for_root.total_value = -1.0
    root.visits = 10
    root.children = {first_move: bad_for_root, second_move: good_for_root}

    assert root.best_child(exploration=0.0).move == second_move


def test_neural_puct_selection_uses_priors_when_values_and_visits_match() -> None:
    root = NeuralMCTSNode.create(Game.new())
    root.visits = 6
    low_prior_move, high_prior_move = root.legal_moves[:2]
    low_prior = NeuralMCTSNode.create(
        root.game.play(low_prior_move),
        parent=root,
        move=low_prior_move,
        prior=0.1,
    )
    low_prior.visits = 3
    high_prior = NeuralMCTSNode.create(
        root.game.play(high_prior_move),
        parent=root,
        move=high_prior_move,
        prior=0.9,
    )
    high_prior.visits = 3
    root.children = {low_prior_move: low_prior, high_prior_move: high_prior}

    assert root.best_child(exploration=1.5).move == high_prior_move


def test_select_by_temperature_zero_selects_highest_visit_count() -> None:
    root = NeuralMCTSNode.create(Game.new())
    low_visit_move, high_visit_move = root.legal_moves[:2]
    low_visit = NeuralMCTSNode.create(
        root.game.play(low_visit_move), parent=root, move=low_visit_move
    )
    low_visit.visits = 1
    high_visit = NeuralMCTSNode.create(
        root.game.play(high_visit_move), parent=root, move=high_visit_move
    )
    high_visit.visits = 5
    root.children = {low_visit_move: low_visit, high_visit_move: high_visit}

    selected = _select_by_temperature(root, temperature=0.0, rng=random.Random(1))

    assert selected == high_visit_move


def test_select_by_temperature_positive_weights_exclude_zero_visit_children() -> None:
    root = NeuralMCTSNode.create(Game.new())
    visited_move, zero_visit_move = root.legal_moves[:2]
    visited = NeuralMCTSNode.create(root.game.play(visited_move), parent=root, move=visited_move)
    visited.visits = 2
    zero_visit = NeuralMCTSNode.create(
        root.game.play(zero_visit_move), parent=root, move=zero_visit_move
    )
    root.children = {visited_move: visited, zero_visit_move: zero_visit}

    selections = {
        _select_by_temperature(root, temperature=1.0, rng=random.Random(seed))
        for seed in range(10)
    }

    assert selections == {visited_move}


def test_select_by_temperature_all_zero_visits_falls_back_safely() -> None:
    root = NeuralMCTSNode.create(Game.new())
    prior_move, other_move = root.legal_moves[:2]
    prior_child = NeuralMCTSNode.create(
        root.game.play(prior_move),
        parent=root,
        move=prior_move,
        prior=1.0,
    )
    other_child = NeuralMCTSNode.create(
        root.game.play(other_move),
        parent=root,
        move=other_move,
        prior=0.0,
    )
    root.children = {prior_move: prior_child, other_move: other_child}

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


def test_neural_mcts_node_budget_counts_reused_root() -> None:
    game = Game.new()
    player = NeuralMCTSPlayer(
        FakeInference(), NeuralMCTSConfig(simulations=2, node_budget=1, seed=2)
    )

    first_result = player.search(game)
    first_root = player._tree_root
    second_result = player.search(game)

    assert first_result.nodes == 1
    assert second_result.nodes == 1
    assert player._tree_root is first_root
    assert first_root is not None
    assert not first_root.children
    assert second_result.move in game.legal_moves


def test_neural_tree_reuse_adopts_exact_descendant_and_preserves_visits() -> None:
    game = Game.new()
    player = NeuralMCTSPlayer(FakeInference(), NeuralMCTSConfig(simulations=3, seed=1))

    first_result = player.search(game)
    first_root = player._tree_root
    assert first_root is not None
    reused_child = first_root.children[first_result.move]
    reused_child.visits = 7
    requested = game.play(first_result.move)

    second_result = player.search(requested)

    assert player._tree_root is reused_child
    assert reused_child.game == requested
    assert reused_child.parent is None
    assert reused_child.visits >= 7
    assert second_result.move in requested.legal_moves


def test_neural_clear_tree_discards_reusable_state() -> None:
    game = Game.new()
    player = NeuralMCTSPlayer(FakeInference(), NeuralMCTSConfig(simulations=2, seed=1))

    result = player.search(game)
    first_root = player._tree_root
    assert first_root is not None
    old_child = first_root.children[result.move]
    requested = game.play(result.move)
    player.clear_tree()

    player.search(requested)

    assert player._tree_root is not None
    assert player._tree_root is not old_child
    assert player._tree_root.game == requested
    assert player._tree_root.parent is None


def test_neural_tree_reuse_rejects_same_board_non_descendant_and_absent_child() -> None:
    game = Game.new()
    move = Move.from_uci("e2e4")
    player = NeuralMCTSPlayer(FakeInference(), NeuralMCTSConfig(simulations=2, seed=1))
    player.search(game)
    root = player._tree_root
    assert root is not None
    child = root.children[move]

    same_board_non_descendant = Game.from_fen(child.game.to_fen())

    assert player._adopt_descendant_root(same_board_non_descendant) is None

    capped_player = NeuralMCTSPlayer(
        FakeInference(), NeuralMCTSConfig(simulations=2, node_budget=1, seed=1)
    )
    capped_player.search(game)
    capped_root = capped_player._tree_root
    assert capped_root is not None
    assert not capped_root.children
    descendant_with_absent_child = game.play(move)

    assert capped_player._adopt_descendant_root(descendant_with_absent_child) is None


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
