"""Neural PUCT Monte Carlo Tree Search player."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol, cast

import numpy as np

from tinychess.ai.player import NoLegalMoveError
from tinychess.ai.search_state import SearchState
from tinychess.engine.game import Game
from tinychess.engine.game import determine_outcome as _game_determine_outcome
from tinychess.engine.move import Move
from tinychess.engine.outcome import Outcome
from tinychess.engine.piece import Color
from tinychess.nn.encode import move_to_action_index
from tinychess.nn.model import (
    InferenceResult,
    LegalPolicyResult,
    PolicyValueInference,
)
from tinychess.profiling import profile_scope, record_counter, record_distribution

# Kept as a module attribute for Work Item 5.1 self-play profiling monkeypatches.
determine_outcome = _game_determine_outcome


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
    """

    simulations: int = 25
    time_limit_seconds: float | None = None
    node_budget: int | None = None
    puct_exploration: float = 1.5
    temperature: float = 0.0
    seed: int | None = None

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
            candidates = tuple(self.edges.values())
            parent_visits = max(1, self.visits)

            def score(edge: NeuralMCTSEdge) -> float:
                # Edge values are from the child/opponent perspective, so negate for
                # the current node's side to move.
                q_value = -edge.mean_value
                u_value = exploration * edge.prior * math.sqrt(parent_visits) / (1 + edge.visits)
                return q_value + u_value

            return max(
                candidates,
                key=lambda edge: (score(edge), edge.move.to_uci()),
            )

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
        if self.elapsed_seconds == 0:
            return math.inf
        return self.simulations / self.elapsed_seconds


@dataclass(frozen=True, slots=True)
class NeuralMCTSInferenceRequest:
    """Single pending neural inference needed by a serial MCTS search session."""

    session_id: int
    node: NeuralMCTSNode
    game: Game
    legal_moves: tuple[Move, ...]
    budget_blocked: bool
    selection_depth: int


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

        deadline = None
        if self.config.time_limit_seconds is not None:
            deadline = start_time + self.config.time_limit_seconds
        return _PreparedNeuralMCTSSearch(
            root=root,
            nodes_created=nodes_created,
            start_time=start_time,
            deadline=deadline,
        )

    def _run_serial_simulations(
        self,
        root: NeuralMCTSNode,
        nodes_created: int,
        deadline: float | None,
    ) -> tuple[int, int]:
        simulations = 0
        while simulations < self.config.simulations:
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

    def _predict(self, node: NeuralMCTSNode) -> InferenceResult:
        with profile_scope("mcts.predict"):
            return self._predict_profiled(node)

    def _predict_profiled(self, node: NeuralMCTSNode) -> InferenceResult:
        predict_with_legal_moves = getattr(self.inference, "predict_with_legal_moves", None)
        if callable(predict_with_legal_moves):
            typed_predict = cast(
                Callable[[Game, tuple[Move, ...]], InferenceResult],
                predict_with_legal_moves,
            )
            return typed_predict(node.state.to_game(include_positions=False), node.legal_moves)
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

        while self.simulations < self.player.config.simulations:
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
    full_raw_priors: dict[Move, float] = {}
    for move in legal:
        index = move_to_action_index(move, board)
        full_raw_priors[move] = max(0.0, float(prediction.policy[index].item()))
    return _normalize_priors(full_raw_priors)


def _normalize_priors(raw_priors: dict[Move, float]) -> dict[Move, float]:
    total = sum(raw_priors.values())
    if total <= 0.0 or not math.isfinite(total):
        uniform = 1.0 / len(raw_priors)
        return {move: uniform for move in raw_priors}
    return {move: prior / total for move, prior in raw_priors.items()}


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
                    edge.move.to_uci(),
                ),
            ).move
        weights = [float(edge.visits) ** (1.0 / temperature) for edge in edges]
        if sum(weights) <= 0.0:
            weights = [edge.prior for edge in edges]
        if sum(weights) <= 0.0:
            weights = [1.0 for _edge in edges]
        return rng.choices([edge.move for edge in edges], weights=weights, k=1)[0]
