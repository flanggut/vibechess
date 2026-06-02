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
    NeuralMCTSEdge,
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
from tinychess.nn import model as model_module
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


class RecordingPolicyValueInference(PolicyValueInference):
    legal_calls: int
    predict_calls: int
    seen_legal_moves: list[tuple[Move, ...]]

    def __init__(self) -> None:
        tiny_config = PolicyValueConfig(
            residual_channels=4,
            residual_blocks=0,
            policy_channels=1,
            value_channels=1,
            value_hidden_dim=4,
        )
        super().__init__(PolicyValueNet(tiny_config))
        self.legal_calls = 0
        self.predict_calls = 0
        self.seen_legal_moves = []

    def predict(self, game: Game, *, mask_legal_moves: bool = True) -> InferenceResult:
        self.predict_calls += 1
        return super().predict(game, mask_legal_moves=mask_legal_moves)

    def predict_with_legal_moves(
        self,
        game: Game,
        legal_moves: tuple[Move, ...],
    ) -> InferenceResult:
        self.legal_calls += 1
        self.seen_legal_moves.append(legal_moves)
        return super().predict_with_legal_moves(game, legal_moves)


def _policy_result(values: list[float], *, value: float = 0.0) -> InferenceResult:
    policy = mx.array(values, dtype=mx.float32)
    mx.eval(policy)
    return InferenceResult(policy_logits=policy, policy=policy, value=value)


class PolicyAccessRaises:
    def __getitem__(self, _index: int) -> object:
        raise AssertionError("compact legal priors should not index the full policy")


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


def test_legal_priors_use_compact_legal_policy_without_full_policy_indexing() -> None:
    game = Game.new()
    node = NeuralMCTSNode.create(game)
    legal = node.legal_moves[:2]
    prediction = InferenceResult(
        policy_logits=mx.zeros((ACTION_SPACE_SIZE,), dtype=mx.float32),
        policy=PolicyAccessRaises(),
        value=0.0,
        legal_moves=legal,
        legal_policy=mx.array([0.25, 0.75], dtype=mx.float32),
    )

    priors = _legal_priors(node, prediction, legal_moves=legal)

    assert priors == {legal[0]: pytest.approx(0.25), legal[1]: pytest.approx(0.75)}


def test_legal_priors_return_empty_for_empty_legal_moves() -> None:
    game = Game.from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    node = NeuralMCTSNode.create(game)
    prediction = _policy_result([1.0] * ACTION_SPACE_SIZE)

    assert _legal_priors(node, prediction) == {}


def test_expand_creates_legal_edges_without_materializing_child_games(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game = Game.new()
    root = NeuralMCTSNode.create(game)
    player = NeuralMCTSPlayer(FakeInference(), NeuralMCTSConfig(simulations=1, seed=1))

    def fail_play_known_legal(self: Game, move: Move) -> Game:
        raise AssertionError(f"expansion should not create child game for {move}")

    monkeypatch.setattr(Game, "play_known_legal", fail_play_known_legal)

    value = player._expand(root)

    assert value == pytest.approx(0.25)
    assert root.is_expanded
    assert root.children == {}
    assert set(root.edges) == set(root.legal_moves)
    assert all(edge.child is None for edge in root.edges.values())


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
    assert result.nodes <= 2
    assert result.nodes == 1
    assert set(result.visit_counts) == set(game.legal_moves)


def test_neural_mcts_uses_policy_value_legal_move_inference_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game = Game.new()
    inference = RecordingPolicyValueInference()

    def fail_legal_move_mask(_game: Game) -> object:
        raise AssertionError("neural MCTS should use cached legal moves instead of legal_move_mask")

    monkeypatch.setattr(model_module, "legal_move_mask", fail_legal_move_mask)
    player = NeuralMCTSPlayer(inference, NeuralMCTSConfig(simulations=1, seed=1))

    result = player.search(game)

    assert result.move in game.legal_moves
    assert inference.legal_calls == 1
    assert inference.predict_calls == 0
    assert inference.seen_legal_moves == [game.legal_moves]


def test_neural_mcts_one_simulation_materializes_only_root() -> None:
    game = Game.new()
    player = NeuralMCTSPlayer(FakeInference(), NeuralMCTSConfig(simulations=1, seed=1))

    result = player.search(game)

    assert result.nodes == 1
    assert player._tree_root is not None
    assert player._tree_root.children == {}
    assert set(player._tree_root.edges) == set(game.legal_moves)
    assert set(result.visit_counts) == set(game.legal_moves)
    assert all(visits == 0 for visits in result.visit_counts.values())


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
    assert player._tree_root is not None
    assert illegal not in player._tree_root.edges
    assert illegal not in result.visit_counts
    assert set(player._tree_root.edges) == set(game.legal_moves)
    assert set(result.visit_counts) == set(game.legal_moves)


def test_high_prior_unmaterialized_edge_materializes_on_descent() -> None:
    game = Game.new()
    preferred = Move.from_uci("e2e4")
    player = NeuralMCTSPlayer(
        FakeInference(preferred_move=preferred),
        NeuralMCTSConfig(simulations=2, puct_exploration=1.5, seed=2),
    )

    result = player.search(game)

    assert result.nodes == 2
    assert player._tree_root is not None
    preferred_edge = player._tree_root.edges[preferred]
    assert preferred_edge.child is not None
    assert player._tree_root.children[preferred] is preferred_edge.child
    assert result.visit_counts[preferred] == 1


def test_materialize_child_uses_known_legal_transition_equivalent_to_play(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game = Game.new()
    preferred = Move.from_uci("e2e4")
    expected_game = game.play(preferred)
    root = NeuralMCTSNode.create(game)
    player = NeuralMCTSPlayer(FakeInference(preferred_move=preferred))
    player._expand(root)
    edge = root.edges[preferred]

    def fail_play(self: Game, move: Move) -> Game:
        raise AssertionError(f"materialization should not validate cached legal move {move}")

    monkeypatch.setattr(Game, "play", fail_play)

    child = player._materialize_child(root, edge)

    assert child.game == expected_game
    assert child.parent is root
    assert child.move == preferred
    assert edge.child is child
    assert root.children[preferred] is child


def test_neural_value_backup_flips_side_to_move_perspective() -> None:
    root = NeuralMCTSNode.create(Game.new())
    move = Move.from_uci("e2e4")
    child = NeuralMCTSNode.create(root.game.play(move), parent=root, move=move, prior=1.0)
    root.edges[move] = NeuralMCTSEdge(move=move, prior=1.0, child=child)
    root.children[move] = child

    NeuralMCTSPlayer._backup(child, 0.7)

    assert root.edges[move].visits == 1
    assert root.edges[move].total_value == pytest.approx(0.7)
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
    root.edges = {
        first_move: NeuralMCTSEdge(
            move=first_move,
            prior=0.5,
            child=bad_for_root,
            visits=5,
            total_value=4.0,
        ),
        second_move: NeuralMCTSEdge(
            move=second_move,
            prior=0.5,
            child=good_for_root,
            visits=5,
            total_value=-1.0,
        ),
    }
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
    root.edges = {
        low_prior_move: NeuralMCTSEdge(
            move=low_prior_move,
            prior=0.1,
            child=low_prior,
            visits=3,
        ),
        high_prior_move: NeuralMCTSEdge(
            move=high_prior_move,
            prior=0.9,
            child=high_prior,
            visits=3,
        ),
    }
    root.children = {low_prior_move: low_prior, high_prior_move: high_prior}

    assert root.best_child(exploration=1.5).move == high_prior_move


def test_select_by_temperature_zero_selects_highest_visit_count() -> None:
    root = NeuralMCTSNode.create(Game.new())
    low_visit_move, high_visit_move = root.legal_moves[:2]
    root.edges = {
        low_visit_move: NeuralMCTSEdge(move=low_visit_move, prior=0.5, visits=1),
        high_visit_move: NeuralMCTSEdge(move=high_visit_move, prior=0.5, visits=5),
    }

    selected = _select_by_temperature(root, temperature=0.0, rng=random.Random(1))

    assert selected == high_visit_move


def test_select_by_temperature_positive_weights_exclude_zero_visit_children() -> None:
    root = NeuralMCTSNode.create(Game.new())
    visited_move, zero_visit_move = root.legal_moves[:2]
    root.edges = {
        visited_move: NeuralMCTSEdge(move=visited_move, prior=0.5, visits=2),
        zero_visit_move: NeuralMCTSEdge(move=zero_visit_move, prior=0.5),
    }

    selections = {
        _select_by_temperature(root, temperature=1.0, rng=random.Random(seed))
        for seed in range(10)
    }

    assert selections == {visited_move}


def test_select_by_temperature_all_zero_visits_falls_back_safely() -> None:
    root = NeuralMCTSNode.create(Game.new())
    prior_move, other_move = root.legal_moves[:2]
    root.edges = {
        prior_move: NeuralMCTSEdge(move=prior_move, prior=1.0),
        other_move: NeuralMCTSEdge(move=other_move, prior=0.0),
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


def test_neural_mcts_node_budget_one_expands_edges_without_children() -> None:
    game = Game.new()
    player = NeuralMCTSPlayer(
        FakeInference(), NeuralMCTSConfig(simulations=4, node_budget=1, seed=2)
    )

    result = player.search(game)

    assert result.nodes == 1
    assert result.move in game.legal_moves
    assert player._tree_root is not None
    assert player._tree_root.children == {}
    assert set(result.visit_counts) == set(game.legal_moves)
    assert all(visits == 0 for visits in result.visit_counts.values())


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
    player = NeuralMCTSPlayer(FakeInference(), NeuralMCTSConfig(simulations=2, seed=1))
    result = player.search(game)
    root = player._tree_root
    assert root is not None
    child = root.children[result.move]

    same_board_non_descendant = Game.from_fen(child.game.to_fen())

    assert player._adopt_descendant_root(same_board_non_descendant) is None

    capped_player = NeuralMCTSPlayer(
        FakeInference(), NeuralMCTSConfig(simulations=2, node_budget=1, seed=1)
    )
    capped_player.search(game)
    capped_root = capped_player._tree_root
    assert capped_root is not None
    assert not capped_root.children
    lazy_only_move = next(move for move in capped_root.edges if move not in capped_root.children)
    descendant_with_absent_child = game.play(lazy_only_move)

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
