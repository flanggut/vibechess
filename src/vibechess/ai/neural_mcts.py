"""Neural PUCT Monte Carlo Tree Search player."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from inspect import Parameter, signature
from typing import Any, Protocol, TypeAlias, cast

import mlx.core as mx
import numpy as np

from vibechess.ai.player import NoLegalMoveError
from vibechess.ai.player import simulations_per_second as _simulations_per_second
from vibechess.ai.search_state import SearchState
from vibechess.engine.game import Game
from vibechess.engine.game import determine_outcome as _game_determine_outcome
from vibechess.engine.move import Move
from vibechess.engine.outcome import Outcome
from vibechess.engine.piece import Color, PieceType
from vibechess.nn.encode import encode_board_np, move_to_action_index
from vibechess.nn.inference import (
    InferenceResult,
    LegalPolicyBatchResult,
    LegalPolicyResult,
    PolicyValueInference,
)
from vibechess.profiling import profile_scope, record_counter, record_distribution

# Kept as a module attribute for Work Item 5.1 self-play profiling monkeypatches.
determine_outcome = _game_determine_outcome

# Batched compact-prediction callable used by virtual-loss leaf collection.

_PROMOTION_UCI_ORDER: dict[PieceType | None, int] = {
    None: 0,
    PieceType.BISHOP: 1,
    PieceType.KNIGHT: 2,
    PieceType.QUEEN: 3,
    PieceType.ROOK: 4,
}
_LegalBatchPredict: TypeAlias = Callable[..., LegalPolicyBatchResult]


class NeuralInference(Protocol):
    """Protocol for policy/value inference used by neural MCTS."""

    def predict(self, game: Game, *, mask_legal_moves: bool = True) -> InferenceResult:
        """Return policy probabilities and value for ``game``."""


@dataclass(frozen=True, slots=True)
class NeuralMCTSConfig:
    """Budgets and PUCT settings for neural MCTS.

    ``node_budget`` caps materialized tree nodes, including the root. Legal edge
    priors are cached without creating child nodes; when the cap is reached,
    search keeps simulating but evaluates the current node instead of
    materializing a selected child.

    ``collection_batch_size`` enables virtual-loss leaf collection in ``search()``:
    when greater than ``1`` and the inference backend supports batched legal-move
    prediction, the player gathers up to that many distinct leaves per round
    (temporarily applying ``virtual_loss`` to already-selected paths so selection
    diverges), evaluates them in one batched model call, then unwinds the virtual
    loss and backs up the real values. The default of ``1`` preserves the original
    one-leaf-per-prediction serial behavior exactly.
    """

    simulations: int = 25
    time_limit_seconds: float | None = None
    node_budget: int | None = None
    puct_exploration: float = 1.5
    temperature: float = 0.0
    seed: int | None = None
    collection_batch_size: int = 1
    virtual_loss: int = 1
    reuse_simulation_budget: bool = False
    min_reuse_simulations: int = 16

    def __post_init__(self) -> None:
        if self.simulations < 1:
            msg = f"simulations must be at least 1, got {self.simulations}"
            raise ValueError(msg)
        if self.time_limit_seconds is not None and self.time_limit_seconds < 0:
            msg = f"time_limit_seconds must be non-negative, got {self.time_limit_seconds}"
            raise ValueError(msg)
        if self.node_budget is not None and self.node_budget < 1:
            msg = f"node_budget must be at least 1, got {self.node_budget}"
            raise ValueError(msg)
        if self.puct_exploration < 0:
            msg = f"puct_exploration must be non-negative, got {self.puct_exploration}"
            raise ValueError(msg)
        if self.temperature < 0:
            msg = f"temperature must be non-negative, got {self.temperature}"
            raise ValueError(msg)
        if self.collection_batch_size < 1:
            msg = f"collection_batch_size must be at least 1, got {self.collection_batch_size}"
            raise ValueError(msg)
        if self.virtual_loss < 0:
            msg = f"virtual_loss must be non-negative, got {self.virtual_loss}"
            raise ValueError(msg)
        if self.min_reuse_simulations < 0:
            msg = (
                "min_reuse_simulations must be non-negative, "
                f"got {self.min_reuse_simulations}"
            )
            raise ValueError(msg)
        if (
            self.reuse_simulation_budget
            and self.min_reuse_simulations > self.simulations
        ):
            msg = (
                "min_reuse_simulations must be no greater than simulations when "
                "reuse_simulation_budget is enabled, got "
                f"{self.min_reuse_simulations} > {self.simulations}"
            )
            raise ValueError(msg)


@dataclass(slots=True)
class NeuralMCTSEdge:
    """Selectable legal edge in a neural PUCT search tree.

    Edge statistics store values from the child side-to-move perspective, matching
    the materialized child node when one exists. Unmaterialized edges have a prior
    and zero visits/value until they are selected and a child node can be created.
    """

    move: Move
    prior: float
    child: NeuralMCTSNode | None = None
    visits: int = 0
    total_value: float = 0.0

    @property
    def mean_value(self) -> float:
        """Return the mean value from the child side-to-move perspective."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits


def _edge_puct_score(edge: NeuralMCTSEdge, exploration_bonus: float) -> float:
    # Edge values are from the child/opponent perspective, so negate for the
    # current node's side to move.
    q_value = -(edge.total_value / edge.visits) if edge.visits else 0.0
    u_value = exploration_bonus * edge.prior / (1 + edge.visits)
    return q_value + u_value


@dataclass(slots=True)
class NeuralMCTSNode:
    """One node in a neural PUCT search tree.

    ``total_value`` stores values from this node's side-to-move perspective. During
    backup, signs are flipped at each ply because the side to move alternates.
    Legal move edges are materialized lazily: expansion creates edge priors/stats,
    and a child ``SearchState``/node is created only when an edge is selected for descent.
    """

    state: SearchState
    parent: NeuralMCTSNode | None = None
    move: Move | None = None
    prior: float = 0.0
    legal_moves: tuple[Move, ...] = ()
    outcome: Outcome | None = None
    edges: dict[Move, NeuralMCTSEdge] = field(default_factory=dict)
    children: dict[Move, NeuralMCTSNode] = field(default_factory=dict)
    visits: int = 0
    total_value: float = 0.0
    is_expanded: bool = False
    _legal_action_indices: tuple[int, ...] | None = field(default=None, repr=False)
    _legal_action_index_array: Any | None = field(default=None, repr=False)
    _encoded_input: Any | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.legal_moves and self.outcome is None:
            legal = self.state.legal_moves
            self.legal_moves = legal
            self.outcome = self.state.outcome_with_legal_moves(legal)

    @classmethod
    def create(
        cls,
        game: Game | SearchState,
        *,
        parent: NeuralMCTSNode | None = None,
        move: Move | None = None,
        prior: float = 0.0,
    ) -> NeuralMCTSNode:
        """Create a node with cached legal moves and outcome state."""
        with profile_scope("mcts.node_create"):
            record_counter("mcts.node_create.calls")
            state = SearchState.from_game(game) if isinstance(game, Game) else game
            legal = state.legal_moves
            outcome = state.outcome_with_legal_moves(legal)
            record_distribution("mcts.node_legal_count", len(legal), unit="moves")
            return cls(
                state=state,
                parent=parent,
                move=move,
                prior=prior,
                legal_moves=legal,
                outcome=outcome,
            )

    def cached_legal_action_indices(self) -> tuple[int, ...]:
        """Return legal action indices aligned with ``legal_moves`` for this node."""
        indices = self._legal_action_indices
        if indices is None:
            with profile_scope("policy.legal_indices"):
                record_counter("policy.legal_indices.moves", len(self.legal_moves))
                indices = tuple(
                    move_to_action_index(move, self.state.board) for move in self.legal_moves
                )
            self._legal_action_indices = indices
        return indices

    def cached_legal_action_index_array(self) -> Any:
        """Return an MLX index tensor aligned with ``legal_moves`` for this node."""
        index_array = self._legal_action_index_array
        if index_array is None:
            with profile_scope("policy.legal_index_array"):
                index_array = mx.array(self.cached_legal_action_indices())
            self._legal_action_index_array = index_array
        return index_array

    def cached_encoded_input(self) -> Any:
        """Return an encoded NumPy tensor for this node's position.

        NumPy encoding avoids per-node MLX array construction; callers batch and
        convert to MLX once per inference call (the model receives an identical
        tensor either way).
        """
        encoded = self._encoded_input
        if encoded is None:
            with profile_scope("encode.node_np"):
                encoded = encode_board_np(
                    self.state.board,
                    halfmove_clock=self.state.halfmove_clock,
                    fullmove_number=self.state.fullmove_number,
                )
            self._encoded_input = encoded
        return encoded

    @property
    def game(self) -> Game:
        """Return a reconstructed ``Game`` view for compatibility and boundaries."""
        return self.state.to_game()

    @property
    def is_terminal(self) -> bool:
        """Return whether this node's cached game state is terminal."""
        return self.outcome is not None or not self.legal_moves

    @property
    def mean_value(self) -> float:
        """Return the mean value from this node's side-to-move perspective."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def best_edge(self, exploration: float) -> NeuralMCTSEdge:
        """Return the legal edge with the highest PUCT score for this node's mover."""
        with profile_scope("mcts.best_edge"):
            record_counter("mcts.best_edge.calls")
            if not self.edges:
                msg = "cannot select best edge from an unexpanded leaf"
                raise ValueError(msg)
            parent_visits = max(1, self.visits)
            exploration_bonus = exploration * math.sqrt(parent_visits)

            edge_iter = iter(self.edges.values())
            best = next(edge_iter)
            best_score = _edge_puct_score(best, exploration_bonus)
            best_move_key = _move_uci_order_key(best.move)
            for edge in edge_iter:
                edge_score = _edge_puct_score(edge, exploration_bonus)
                if edge_score > best_score:
                    best = edge
                    best_score = edge_score
                    best_move_key = _move_uci_order_key(edge.move)
                elif edge_score == best_score:
                    edge_move_key = _move_uci_order_key(edge.move)
                    if edge_move_key > best_move_key:
                        best = edge
                        best_move_key = edge_move_key
            return best


    def best_child(self, exploration: float) -> NeuralMCTSNode:
        """Return the materialized child behind the highest-scoring legal edge."""
        edge = self.best_edge(exploration)
        if edge.child is None:
            msg = "best edge has no materialized child"
            raise ValueError(msg)
        return edge.child


@dataclass(frozen=True, slots=True)
class NeuralMCTSResult:
    """Result metadata from a neural MCTS search."""

    move: Move
    simulations: int
    nodes: int
    elapsed_seconds: float
    visit_counts: dict[Move, int]

    @property
    def simulations_per_second(self) -> float:
        """Return completed simulations per second, or infinity for zero elapsed time."""
        return _simulations_per_second(self.simulations, self.elapsed_seconds)


@dataclass(frozen=True, slots=True)
class NeuralMCTSInferenceRequest:
    """Single pending neural inference needed by a serial MCTS search session."""

    session_id: int
    node: NeuralMCTSNode
    game: Game
    legal_moves: tuple[Move, ...]
    budget_blocked: bool
    selection_depth: int
    legal_action_indices: tuple[int, ...] = ()
    legal_action_index_array: Any | None = None
    encoded_input: Any | None = None



@dataclass(frozen=True, slots=True)
class _SerialLeafSelection:
    node: NeuralMCTSNode
    terminal_value: float | None
    budget_blocked: bool
    nodes_created: int
    selection_depth: int


@dataclass(frozen=True, slots=True)
class _PreparedNeuralMCTSSearch:
    root: NeuralMCTSNode
    nodes_created: int
    start_time: float
    deadline: float | None
    target_simulations: int


@dataclass(slots=True)
class NeuralMCTSPlayer:
    """AlphaZero-style PUCT player using neural policy priors and value estimates."""

    inference: NeuralInference | PolicyValueInference
    config: NeuralMCTSConfig = field(default_factory=NeuralMCTSConfig)
    rng: random.Random | None = None
    _rng: random.Random = field(init=False, repr=False)
    last_result: NeuralMCTSResult | None = field(init=False, default=None)
    _tree_root: NeuralMCTSNode | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.config.seed) if self.rng is None else self.rng

    def clear_tree(self) -> None:
        """Discard reusable neural-MCTS search state."""
        self._tree_root = None

    def select_move(self, game: Game) -> Move:
        """Return a neural-MCTS-selected legal move, or raise for terminal positions."""
        return self.search(game).move

    def search(self, game: Game) -> NeuralMCTSResult:
        """Run PUCT search from ``game`` and return the selected move plus metadata."""
        with profile_scope(
            "mcts.search",
            simulations_requested=self.config.simulations,
            moves_played=len(game.moves),
        ):
            return self._search_profiled(game)

    def _search_profiled(self, game: Game) -> NeuralMCTSResult:
        prepared = self._prepare_serial_search(game, start_time=time.perf_counter())
        simulations, nodes_created = self._run_serial_simulations(
            prepared.root,
            prepared.nodes_created,
            prepared.deadline,
            prepared.target_simulations,
        )

        return self._finish_search(
            prepared.root,
            simulations=simulations,
            nodes_created=nodes_created,
            start_time=prepared.start_time,
        )

    def _prepare_serial_search(
        self,
        game: Game,
        *,
        start_time: float,
    ) -> _PreparedNeuralMCTSSearch:
        root, nodes_created, adopted_root = self._root_for_game(game)
        if root.outcome is not None:
            msg = f"cannot select a move from a terminal game: {root.outcome.reason.value}"
            raise NoLegalMoveError(msg)
        legal = root.legal_moves
        record_distribution("mcts.root_legal_count", len(legal), unit="moves")
        record_counter("mcts.root_adoption_hits" if adopted_root else "mcts.root_adoption_misses")
        if not legal:
            msg = "cannot select a move from a position with no legal moves"
            raise NoLegalMoveError(msg)
        if adopted_root:
            self._detach_root(root)
        self._tree_root = root

        target_simulations = self.config.simulations
        if self.config.reuse_simulation_budget:
            if adopted_root:
                remaining = self.config.simulations - root.visits
                target_simulations = min(
                    max(remaining, self.config.min_reuse_simulations),
                    self.config.simulations,
                )
            skipped = self.config.simulations - target_simulations
            record_distribution(
                "mcts.reuse_simulations_skipped",
                skipped,
                unit="simulations",
            )

        deadline = None
        if self.config.time_limit_seconds is not None:
            deadline = start_time + self.config.time_limit_seconds
        return _PreparedNeuralMCTSSearch(
            root=root,
            nodes_created=nodes_created,
            start_time=start_time,
            deadline=deadline,
            target_simulations=target_simulations,
        )

    def _run_serial_simulations(
        self,
        root: NeuralMCTSNode,
        nodes_created: int,
        deadline: float | None,
        target_simulations: int,
    ) -> tuple[int, int]:
        if self.config.collection_batch_size > 1:
            batch_predict = getattr(self.inference, "predict_legal_batch", None)
            if callable(batch_predict):
                return self._run_collected_simulations(
                    root,
                    nodes_created,
                    deadline,
                    target_simulations,
                    cast(_LegalBatchPredict, batch_predict),
                )
        return self._run_single_leaf_simulations(
            root,
            nodes_created,
            deadline,
            target_simulations,
        )

    def _run_single_leaf_simulations(
        self,
        root: NeuralMCTSNode,
        nodes_created: int,
        deadline: float | None,
        target_simulations: int,
    ) -> tuple[int, int]:
        simulations = 0
        while simulations < target_simulations:
            if deadline is not None and simulations > 0 and time.perf_counter() >= deadline:
                break
            with profile_scope("mcts.simulation", simulation_index=simulations):
                if deadline is not None and time.perf_counter() >= deadline:
                    break

                selection = self._select_serial_leaf(root, nodes_created)
                nodes_created = selection.nodes_created
                if selection.terminal_value is not None:
                    value = selection.terminal_value
                elif selection.budget_blocked:
                    value = self._evaluate(selection.node)
                else:
                    value = self._expand(selection.node)
                    record_counter("mcts.expanded_simulations")
                self._backup(selection.node, value)
                simulations += 1
        return simulations, nodes_created

    def _run_collected_simulations(
        self,
        root: NeuralMCTSNode,
        nodes_created: int,
        deadline: float | None,
        target_simulations: int,
        batch_predict: _LegalBatchPredict,
    ) -> tuple[int, int]:
        """Run simulations using virtual-loss leaf collection and batched inference.

        Each round selects up to ``collection_batch_size`` distinct leaves, applying
        ``virtual_loss`` along each selected path so subsequent selections diverge.
        Terminal leaves are backed up immediately (no model call needed). The remaining
        leaves are evaluated in one batched prediction; their virtual loss is then
        removed and the real network value is backed up.
        """
        width = self.config.collection_batch_size
        simulations = 0
        while simulations < target_simulations:
            if deadline is not None and simulations > 0 and time.perf_counter() >= deadline:
                break

            target = min(width, target_simulations - simulations)
            pending: list[_SerialLeafSelection] = []
            pending_ids: set[int] = set()
            with profile_scope("mcts.collect", target=target):
                for _ in range(target):
                    selection = self._select_serial_leaf(root, nodes_created)
                    nodes_created = selection.nodes_created
                    if selection.terminal_value is not None:
                        self._backup(selection.node, selection.terminal_value)
                        simulations += 1
                        continue
                    if id(selection.node) in pending_ids:
                        # Selection re-converged on an in-flight leaf; flush what we have
                        # rather than evaluating or expanding the same node twice.
                        break
                    self._apply_virtual_loss(selection.node)
                    pending.append(selection)
                    pending_ids.add(id(selection.node))

            if not pending:
                continue

            with profile_scope("mcts.collected_predict", batch_size=len(pending)):
                games = tuple(
                    selection.node.state.to_game(include_positions=False) for selection in pending
                )
                legal_by_game = tuple(selection.node.legal_moves for selection in pending)
                legal_indices_by_game = tuple(
                    selection.node.cached_legal_action_indices() for selection in pending
                )
                legal_index_arrays = tuple(
                    selection.node.cached_legal_action_index_array() for selection in pending
                )
                encoded_inputs = tuple(
                    selection.node.cached_encoded_input() for selection in pending
                )
                batch = _call_legal_batch_predict(
                    batch_predict,
                    games,
                    legal_by_game,
                    legal_indices_by_game,
                    legal_index_arrays,
                    encoded_inputs,
                )

            for row_index, selection in enumerate(pending):
                self._remove_virtual_loss(selection.node)
                prediction = batch.result_at(row_index)
                if selection.budget_blocked:
                    record_counter("mcts.evaluate.calls")
                    value = prediction.value
                else:
                    value = self._expand_from_prediction(selection.node, prediction)
                    record_counter("mcts.expanded_simulations")
                self._backup(selection.node, value)
                simulations += 1
        return simulations, nodes_created

    def _apply_virtual_loss(self, node: NeuralMCTSNode) -> None:
        amount = self.config.virtual_loss
        if amount:
            _add_path_value(node, float(-amount), visit_delta=amount)

    def _remove_virtual_loss(self, node: NeuralMCTSNode) -> None:
        amount = self.config.virtual_loss
        if amount:
            _add_path_value(node, float(amount), visit_delta=-amount)

    def _select_serial_leaf(
        self,
        root: NeuralMCTSNode,
        nodes_created: int,
    ) -> _SerialLeafSelection:
        node = root
        budget_blocked = False
        selection_depth = 0
        with profile_scope("mcts.selection"):
            while node.is_expanded and node.edges and not node.is_terminal:
                edge = node.best_edge(self.config.puct_exploration)
                if edge.child is None:
                    budget_reached = (
                        self.config.node_budget is not None
                        and nodes_created >= self.config.node_budget
                    )
                    if budget_reached:
                        budget_blocked = True
                        record_counter("mcts.node_budget_blocked")
                        break
                    node = self._materialize_child(node, edge)
                    nodes_created += 1
                    record_counter("mcts.materialized_nodes")
                else:
                    node = edge.child
                selection_depth += 1
        record_distribution("mcts.selection_depth", selection_depth, unit="edges")

        terminal_value = None
        if node.is_terminal:
            with profile_scope("mcts.terminal_value"):
                record_counter("mcts.terminal_simulations")
                terminal_value = _terminal_value(node.outcome, node.state.board.side_to_move)
        return _SerialLeafSelection(
            node=node,
            terminal_value=terminal_value,
            budget_blocked=budget_blocked,
            nodes_created=nodes_created,
            selection_depth=selection_depth,
        )

    def _finish_search(
        self,
        root: NeuralMCTSNode,
        *,
        simulations: int,
        nodes_created: int,
        start_time: float,
    ) -> NeuralMCTSResult:
        record_counter("mcts.completed_simulations", simulations)
        record_distribution("mcts.simulations_per_search", simulations, unit="simulations")
        record_distribution("mcts.nodes_per_search", nodes_created, unit="nodes")
        selected_move = _select_by_temperature(root, self.config.temperature, self._rng)
        if selected_move is None:
            selected_move = self._rng.choice(root.legal_moves)
        elapsed = time.perf_counter() - start_time
        result = NeuralMCTSResult(
            move=selected_move,
            simulations=simulations,
            nodes=nodes_created,
            elapsed_seconds=elapsed,
            visit_counts={move: edge.visits for move, edge in root.edges.items()},
        )
        self.last_result = result
        self._tree_root = root
        return result

    def _root_for_game(self, game: Game) -> tuple[NeuralMCTSNode, int, bool]:
        with profile_scope("mcts.root_for_game"):
            adopted = self._adopt_descendant_root(game)
            if adopted is not None:
                return adopted, 1, True
            record_counter("mcts.materialized_nodes")
            return NeuralMCTSNode.create(game), 1, False

    def _adopt_descendant_root(self, game: Game) -> NeuralMCTSNode | None:
        with profile_scope("mcts.adopt_descendant_root"):
            return self._adopt_descendant_root_impl(game)

    def _adopt_descendant_root_impl(self, game: Game) -> NeuralMCTSNode | None:
        root = self._tree_root
        if root is None:
            return None
        root_moves = root.state.moves
        requested_moves = game.moves
        if len(root_moves) > len(requested_moves):
            return None
        if requested_moves[: len(root_moves)] != root_moves:
            return None

        current = root
        for move in requested_moves[len(root_moves) :]:
            child = current.children.get(move)
            if child is None:
                return None
            current = child
        if current.state.to_game() != game:
            return None
        return current

    @staticmethod
    def _detach_root(root: NeuralMCTSNode) -> None:
        root.parent = None

    def _expand(self, node: NeuralMCTSNode) -> float:
        with profile_scope("mcts.expand"):
            prediction = self._predict(node)
            return self._expand_from_prediction(node, prediction)

    @staticmethod
    def _expand_from_prediction(
        node: NeuralMCTSNode,
        prediction: InferenceResult | LegalPolicyResult,
    ) -> float:
        record_counter("mcts.expand.calls")
        priors = _legal_priors(node, prediction)
        record_distribution("mcts.edge_count", len(priors), unit="edges")
        with profile_scope("mcts.edge_create"):
            record_counter("mcts.edge_create.edges", len(priors))
            node.edges = {
                move: NeuralMCTSEdge(move=move, prior=prior, child=node.children.get(move))
                for move, prior in priors.items()
            }
        for move, child in node.children.items():
            edge = node.edges.get(move)
            if edge is not None:
                edge.visits = child.visits
                edge.total_value = child.total_value
        node.is_expanded = True
        return prediction.value

    @staticmethod
    def _materialize_child(node: NeuralMCTSNode, edge: NeuralMCTSEdge) -> NeuralMCTSNode:
        with profile_scope("mcts.materialize_child"):
            child = NeuralMCTSNode.create(
                node.state.play_known_legal(edge.move),
                parent=node,
                move=edge.move,
                prior=edge.prior,
            )
            edge.child = child
            node.children[edge.move] = child
            return child

    def _predict(self, node: NeuralMCTSNode) -> InferenceResult | LegalPolicyResult:
        with profile_scope("mcts.predict"):
            return self._predict_profiled(node)

    def _predict_profiled(self, node: NeuralMCTSNode) -> InferenceResult | LegalPolicyResult:
        predict_with_legal_moves = getattr(self.inference, "predict_with_legal_moves", None)
        if callable(predict_with_legal_moves):
            typed_predict = cast(Callable[..., InferenceResult], predict_with_legal_moves)
            return _call_predict_with_cached_node(typed_predict, node)
        return self.inference.predict(
            node.state.to_game(include_positions=False),
            mask_legal_moves=True,
        )

    def _evaluate(self, node: NeuralMCTSNode) -> float:
        with profile_scope("mcts.evaluate"):
            record_counter("mcts.evaluate.calls")
            return self._predict(node).value

    @staticmethod
    def _backup(node: NeuralMCTSNode, value: float) -> None:
        with profile_scope("mcts.backup"):
            depth = _add_path_value(node, value, visit_delta=1)
            record_distribution("mcts.backup_depth", depth, unit="nodes")


@dataclass(slots=True, init=False)
class NeuralMCTSSearchSession:
    """Cooperative serial neural-MCTS search session.

    ``advance()`` runs serial simulations until the session either completes or
    needs one neural prediction for the selected leaf. The caller must provide that
    prediction with ``resume()`` before the same session can select another leaf.
    """

    player: NeuralMCTSPlayer
    game: Game
    session_id: int
    root: NeuralMCTSNode
    simulations: int
    nodes_created: int
    _start_time: float
    _deadline: float | None
    _target_simulations: int
    _pending_selection: _SerialLeafSelection | None
    _pending_request: NeuralMCTSInferenceRequest | None
    _result: NeuralMCTSResult | None

    def __init__(
        self,
        player: NeuralMCTSPlayer,
        game: Game,
        *,
        session_id: int = 0,
    ) -> None:
        self.player = player
        self.game = game
        self.session_id = session_id
        prepared = player._prepare_serial_search(game, start_time=time.perf_counter())
        self.root = prepared.root
        self.nodes_created = prepared.nodes_created
        self._start_time = prepared.start_time
        self._deadline = prepared.deadline
        self._target_simulations = prepared.target_simulations
        self.simulations = 0
        self._pending_selection = None
        self._pending_request = None
        self._result = None

    @property
    def pending_request(self) -> NeuralMCTSInferenceRequest | None:
        """Return the request awaiting ``resume()``, if any."""
        return self._pending_request

    @property
    def result(self) -> NeuralMCTSResult | None:
        """Return the completed result, if the session has finished."""
        return self._result

    @property
    def is_complete(self) -> bool:
        """Return whether the session has completed and updated its player."""
        return self._result is not None

    def advance(self) -> NeuralMCTSInferenceRequest | NeuralMCTSResult:
        """Advance until completion or the next required neural inference request."""
        if self._result is not None:
            return self._result
        if self._pending_request is not None:
            return self._pending_request

        while self.simulations < self._target_simulations:
            if (
                self._deadline is not None
                and self.simulations > 0
                and time.perf_counter() >= self._deadline
            ):
                return self._finish()
            with profile_scope("mcts.simulation", simulation_index=self.simulations):
                if self._deadline is not None and time.perf_counter() >= self._deadline:
                    return self._finish()

                selection = self.player._select_serial_leaf(self.root, self.nodes_created)
                self.nodes_created = selection.nodes_created
                if selection.terminal_value is not None:
                    self.player._backup(selection.node, selection.terminal_value)
                    self.simulations += 1
                    continue

                request = NeuralMCTSInferenceRequest(
                    session_id=self.session_id,
                    node=selection.node,
                    game=selection.node.state.to_game(include_positions=False),
                    legal_moves=selection.node.legal_moves,
                    budget_blocked=selection.budget_blocked,
                    selection_depth=selection.selection_depth,
                    legal_action_indices=selection.node.cached_legal_action_indices(),
                    legal_action_index_array=selection.node.cached_legal_action_index_array(),
                    encoded_input=selection.node.cached_encoded_input(),
                )
                self._pending_selection = selection
                self._pending_request = request
                return request

        return self._finish()

    def resume(self, prediction: InferenceResult | LegalPolicyResult) -> None:
        """Resume a pending simulation with its neural prediction and back it up."""
        if self._result is not None:
            msg = "cannot resume a completed neural MCTS search session"
            raise RuntimeError(msg)
        selection = self._pending_selection
        if selection is None or self._pending_request is None:
            msg = "cannot resume neural MCTS search session without a pending request"
            raise RuntimeError(msg)

        if selection.budget_blocked:
            record_counter("mcts.evaluate.calls")
            value = prediction.value
        else:
            value = self.player._expand_from_prediction(selection.node, prediction)
            record_counter("mcts.expanded_simulations")
        self.player._backup(selection.node, value)
        self.simulations += 1
        self._pending_selection = None
        self._pending_request = None

    def _finish(self) -> NeuralMCTSResult:
        if self._result is None:
            self._result = self.player._finish_search(
                self.root,
                simulations=self.simulations,
                nodes_created=self.nodes_created,
                start_time=self._start_time,
            )
        return self._result


def resolve_active_limit(total_games: int, batch_size: int, active_games: int | None) -> int:
    """Return how many games may run concurrently in a batched scheduler.

    Defaults to ``batch_size`` concurrent games when ``active_games`` is unset, and
    never exceeds the total number of games to play. Shared by self-play generation
    and the evaluation harness so both bound concurrency identically.
    """
    requested = batch_size if active_games is None else active_games
    return min(total_games, requested)


@dataclass(frozen=True, slots=True)
class BatchedSessionProfile:
    """Profile-scope names for :func:`run_batched_sessions`.

    Supplying this enables the named profiling zones (used by self-play to keep its
    benchmark zone contract). Leaving it ``None`` runs the scheduler without
    profiling, which is what the evaluation harness wants.
    """

    queue_scope: str
    predict_scope: str
    advance_scope: str


def run_batched_sessions(
    sessions: Sequence[NeuralMCTSSearchSession],
    batch_predict: _LegalBatchPredict,
    *,
    batch_size: int,
    profile: BatchedSessionProfile | None = None,
) -> list[NeuralMCTSResult]:
    """Drive cooperative serial MCTS sessions to completion via batched inference.

    Each session is advanced until it either completes or yields one pending neural
    inference request. Up to ``batch_size`` pending requests (chosen in ascending
    session order for determinism) are evaluated in a single batched call and resumed.
    Returns one :class:`NeuralMCTSResult` per input session, in the input order.

    This is the shared engine behind self-play's central inference queue and the
    evaluation harness's batched neural decisions; both must stay behaviorally
    identical, so the scheduling logic lives here once.
    """
    results: list[NeuralMCTSResult | None] = [None] * len(sessions)
    completed_count = 0
    pending: list[tuple[int, NeuralMCTSInferenceRequest]] = []

    def drain_pending() -> None:
        del pending_batch[:]
        pending_batch.extend(pending[:batch_size])
        del pending[: len(pending_batch)]
        requests = [request for _index, request in pending_batch]
        games = tuple(request.game for request in requests)
        legal_by_game = tuple(request.legal_moves for request in requests)
        legal_indices_by_game = tuple(request.legal_action_indices for request in requests)
        index_arrays = tuple(request.legal_action_index_array for request in requests)
        cached_index_arrays = None if any(item is None for item in index_arrays) else index_arrays
        encoded_inputs = tuple(request.encoded_input for request in requests)
        cached_encoded = None if any(item is None for item in encoded_inputs) else encoded_inputs
        if profile is None:
            batch = _call_legal_batch_predict(
                batch_predict,
                games,
                legal_by_game,
                legal_indices_by_game,
                cached_index_arrays,
                cached_encoded,
            )
        else:
            with profile_scope(profile.predict_scope, batch_size=len(requests)):
                batch = _call_legal_batch_predict(
                    batch_predict,
                    games,
                    legal_by_game,
                    legal_indices_by_game,
                    cached_index_arrays,
                    cached_encoded,
                )
        for row_index, (search_index, _request) in enumerate(pending_batch):
            sessions[search_index].resume(batch.result_at(row_index))

    def advance_idle() -> bool:
        nonlocal completed_count
        progressed = False
        for search_index, session in enumerate(sessions):
            if results[search_index] is not None or session.pending_request is not None:
                continue
            if profile is None:
                advanced = session.advance()
            else:
                with profile_scope(profile.advance_scope, game_index=session.session_id):
                    advanced = session.advance()
            progressed = True
            if isinstance(advanced, NeuralMCTSResult):
                results[search_index] = advanced
                completed_count += 1
            else:
                pending.append((search_index, advanced))
        return progressed

    def run_loop() -> None:
        while completed_count < len(sessions):
            if pending:
                drain_pending()
            elif not advance_idle():
                raise RuntimeError("batched neural session scheduler made no progress")

    pending_batch: list[tuple[int, NeuralMCTSInferenceRequest]] = []
    if profile is None:
        run_loop()
    else:
        with profile_scope(profile.queue_scope, sessions=len(sessions)):
            run_loop()

    completed_results: list[NeuralMCTSResult] = []
    for result in results:
        if result is None:  # pragma: no cover - completed_count guards this invariant.
            raise RuntimeError("batched neural session scheduler returned incomplete results")
        completed_results.append(result)
    return completed_results


def _supported_cached_kwargs(
    fn: Callable[..., object],
    candidates: dict[str, Callable[[], object]],
) -> dict[str, object]:
    """Return cached-input kwargs that ``fn`` accepts, computed only when supported.

    Custom inference implementations may omit any of the optional cached-tensor
    keyword arguments. Each candidate is keyed by name to a zero-arg builder so the
    (sometimes costly) cached value is materialized only for kwargs ``fn`` declares.
    """
    supported = _supported_keyword_parameters(fn)
    return {name: build() for name, build in candidates.items() if name in supported}


def _call_predict_with_cached_node(
    predict_with_legal_moves: Callable[..., InferenceResult],
    node: NeuralMCTSNode,
) -> InferenceResult:
    game = node.state.to_game(include_positions=False)
    kwargs = _supported_cached_kwargs(
        predict_with_legal_moves,
        {
            "legal_action_indices": node.cached_legal_action_indices,
            "legal_action_index_array": node.cached_legal_action_index_array,
            "encoded_input": node.cached_encoded_input,
        },
    )
    return predict_with_legal_moves(game, node.legal_moves, **kwargs)


def _call_legal_batch_predict(
    batch_predict: _LegalBatchPredict,
    games: Sequence[Game],
    legal_moves: Sequence[Sequence[Move]],
    legal_action_indices: Sequence[Sequence[int]],
    legal_action_index_arrays: Sequence[Any] | None,
    encoded_inputs: Sequence[Any] | None,
) -> LegalPolicyBatchResult:
    kwargs = _supported_cached_kwargs(
        batch_predict,
        {
            "legal_action_indices": lambda: legal_action_indices,
            "legal_action_index_arrays": lambda: legal_action_index_arrays,
            "encoded_inputs": lambda: encoded_inputs,
        },
    )
    return batch_predict(games, legal_moves, **kwargs)


# Memoized per underlying function so search hot paths avoid repeated
# ``inspect.signature`` calls on custom inference implementations.
_keyword_parameter_cache: dict[object, frozenset[str]] = {}


def _supported_keyword_parameters(callable_object: Callable[..., object]) -> frozenset[str]:
    """Return the keyword-argument names ``callable_object`` accepts, memoized."""
    key = getattr(callable_object, "__func__", callable_object)
    try:
        cached = _keyword_parameter_cache.get(key)
    except TypeError:
        # Unhashable callable; fall back to uncached introspection.
        return _introspect_keyword_parameters(callable_object)
    if cached is None:
        cached = _introspect_keyword_parameters(callable_object)
        _keyword_parameter_cache[key] = cached
    return cached


def _introspect_keyword_parameters(callable_object: Callable[..., object]) -> frozenset[str]:
    try:
        parameters = signature(callable_object).parameters
    except (TypeError, ValueError):
        return frozenset()
    if any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return frozenset(
            {
                "legal_action_indices",
                "legal_action_index_array",
                "legal_action_index_arrays",
                "encoded_input",
                "encoded_inputs",
            }
        )
    return frozenset(
        name
        for name, parameter in parameters.items()
        if parameter.kind in (Parameter.KEYWORD_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    )


def _add_path_value(node: NeuralMCTSNode, value: float, *, visit_delta: int) -> int:
    current: NeuralMCTSNode | None = node
    current_value = value
    depth = 0
    while current is not None:
        current.visits += visit_delta
        if current.visits < 0:
            msg = "neural MCTS virtual visit reconciliation made node visits negative"
            raise RuntimeError(msg)
        current.total_value += current_value
        parent = current.parent
        if parent is not None and current.move is not None:
            edge = parent.edges.get(current.move)
            if edge is not None:
                edge.visits += visit_delta
                if edge.visits < 0:
                    msg = "neural MCTS virtual visit reconciliation made edge visits negative"
                    raise RuntimeError(msg)
                edge.total_value += current_value
        depth += 1
        current_value = -current_value
        current = parent
    return depth


def _move_uci_order_key(move: Move) -> int:
    """Return an allocation-free key with the same ordering as ``Move.to_uci()``."""
    from_square = move.from_square
    to_square = move.to_square
    return (
        (((from_square % 8) * 8 + (from_square // 8)) * 8 + (to_square % 8)) * 8
        + (to_square // 8)
    ) * 5 + _PROMOTION_UCI_ORDER[move.promotion]


def _legal_priors(
    position: NeuralMCTSNode | Game,
    prediction: InferenceResult | LegalPolicyResult,
    *,
    legal_moves: Iterable[Move] | None = None,
) -> dict[Move, float]:
    with profile_scope("mcts.legal_priors"):
        return _legal_priors_impl(position, prediction, legal_moves=legal_moves)


def _legal_priors_impl(
    position: NeuralMCTSNode | Game,
    prediction: InferenceResult | LegalPolicyResult,
    *,
    legal_moves: Iterable[Move] | None = None,
) -> dict[Move, float]:
    if isinstance(position, NeuralMCTSNode):
        board = position.state.board
        legal = position.legal_moves if legal_moves is None else tuple(legal_moves)
    else:
        if legal_moves is None:
            msg = "legal_moves must be provided when extracting priors from a Game"
            raise ValueError(msg)
        board = position.board
        legal = tuple(legal_moves)
    if not legal:
        return {}

    if prediction.legal_policy is not None and prediction.legal_moves == legal:
        record_counter("mcts.legal_priors.compact")
        compact_priors = np.asarray(prediction.legal_policy, dtype=np.float32).reshape(-1)
        if compact_priors.shape == (len(legal),):
            compact_raw_priors = {
                move: max(0.0, float(prior))
                for move, prior in zip(legal, compact_priors, strict=True)
            }
            return _normalize_priors(compact_raw_priors)

    if isinstance(prediction, LegalPolicyResult):
        msg = "compact legal prediction must match the node's legal move tuple"
        raise ValueError(msg)

    record_counter("mcts.legal_priors.dense_fallback")
    if prediction.legal_moves == legal and prediction.legal_action_indices:
        record_counter("mcts.legal_priors.dense_cached_indices")
        indices = prediction.legal_action_indices
    elif isinstance(position, NeuralMCTSNode):
        record_counter("mcts.legal_priors.dense_cached_indices")
        indices = position.cached_legal_action_indices()
    else:
        indices = tuple(move_to_action_index(move, board) for move in legal)

    full_raw_priors: dict[Move, float] = {}
    for move, index in zip(legal, indices, strict=True):
        full_raw_priors[move] = max(0.0, float(prediction.policy[index].item()))
    return _normalize_priors(full_raw_priors)


def _normalize_priors(raw_priors: dict[Move, float]) -> dict[Move, float]:
    total = sum(raw_priors.values())
    if total <= 0.0 or not math.isfinite(total):
        uniform = 1.0 / len(raw_priors)
        for move in raw_priors:
            raw_priors[move] = uniform
        return raw_priors
    inverse_total = 1.0 / total
    for move, prior in raw_priors.items():
        raw_priors[move] = prior * inverse_total
    return raw_priors


def _terminal_value(outcome: Outcome | None, side_to_move: Color) -> float:
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner is side_to_move else -1.0


def _select_by_temperature(
    root: NeuralMCTSNode,
    temperature: float,
    rng: random.Random,
) -> Move | None:
    with profile_scope("mcts.select_temperature"):
        record_counter("mcts.select_temperature.calls")
        if not root.edges:
            return None
        edges = list(root.edges.values())
        if temperature == 0.0:
            return max(
                edges,
                key=lambda edge: (
                    edge.visits,
                    -edge.mean_value,
                    _move_uci_order_key(edge.move),
                ),
            ).move
        if temperature == 1.0:
            weights = [float(edge.visits) for edge in edges]
        else:
            inverse_temperature = 1.0 / temperature
            weights = [float(edge.visits) ** inverse_temperature for edge in edges]
        total_weight = sum(weights)
        if total_weight <= 0.0:
            weights = [edge.prior for edge in edges]
            total_weight = sum(weights)
        if total_weight <= 0.0:
            weights = [1.0 for _edge in edges]
        return rng.choices([edge.move for edge in edges], weights=weights, k=1)[0]
