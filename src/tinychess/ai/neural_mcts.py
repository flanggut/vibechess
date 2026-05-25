"""Neural PUCT Monte Carlo Tree Search player."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Protocol

from tinychess.ai.player import NoLegalMoveError
from tinychess.engine.game import Game
from tinychess.engine.move import Move
from tinychess.engine.outcome import Outcome
from tinychess.engine.piece import Color
from tinychess.nn.encode import move_to_action_index
from tinychess.nn.model import InferenceResult, PolicyValueInference


class NeuralInference(Protocol):
    """Protocol for policy/value inference used by neural MCTS."""

    def predict(self, game: Game, *, mask_legal_moves: bool = True) -> InferenceResult:
        """Return policy probabilities and value for ``game``."""


@dataclass(frozen=True, slots=True)
class NeuralMCTSConfig:
    """Budgets and PUCT settings for neural MCTS.

    ``node_budget`` caps the total number of tree nodes, including the root. When
    the cap is reached, search keeps simulating but evaluates selected leaves
    without expanding additional children.
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
class NeuralMCTSNode:
    """One node in a neural PUCT search tree.

    ``total_value`` stores values from this node's side-to-move perspective. During
    backup, signs are flipped at each ply because the side to move alternates.
    """

    game: Game
    parent: NeuralMCTSNode | None = None
    move: Move | None = None
    prior: float = 0.0
    children: dict[Move, NeuralMCTSNode] = field(default_factory=dict)
    visits: int = 0
    total_value: float = 0.0
    is_expanded: bool = False

    @property
    def is_terminal(self) -> bool:
        """Return whether this node's game is terminal."""
        return self.game.outcome is not None or not self.game.legal_moves

    @property
    def mean_value(self) -> float:
        """Return the mean value from this node's side-to-move perspective."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def best_child(self, exploration: float) -> NeuralMCTSNode:
        """Return the child with the highest PUCT score for this node's mover."""
        if not self.children:
            msg = "cannot select best child from an unexpanded leaf"
            raise ValueError(msg)
        parent_visits = max(1, self.visits)

        def score(child: NeuralMCTSNode) -> float:
            # Child values are from the opponent's perspective, so negate for the
            # current node's side to move.
            q_value = -child.mean_value
            u_value = exploration * child.prior * math.sqrt(parent_visits) / (1 + child.visits)
            return q_value + u_value

        return max(
            self.children.values(),
            key=lambda child: (score(child), child.move.to_uci() if child.move else ""),
        )


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


@dataclass(slots=True)
class NeuralMCTSPlayer:
    """AlphaZero-style PUCT player using neural policy priors and value estimates."""

    inference: NeuralInference | PolicyValueInference
    config: NeuralMCTSConfig = field(default_factory=NeuralMCTSConfig)
    rng: random.Random | None = None
    _rng: random.Random = field(init=False, repr=False)
    last_result: NeuralMCTSResult | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.config.seed) if self.rng is None else self.rng

    def select_move(self, game: Game) -> Move:
        """Return a neural-MCTS-selected legal move, or raise for terminal positions."""
        return self.search(game).move

    def search(self, game: Game) -> NeuralMCTSResult:
        """Run PUCT search from ``game`` and return the selected move plus metadata."""
        outcome = game.outcome
        if outcome is not None:
            msg = f"cannot select a move from a terminal game: {outcome.reason.value}"
            raise NoLegalMoveError(msg)
        legal = game.legal_moves
        if not legal:
            msg = "cannot select a move from a position with no legal moves"
            raise NoLegalMoveError(msg)

        start = time.perf_counter()
        deadline = None
        if self.config.time_limit_seconds is not None:
            deadline = start + self.config.time_limit_seconds
        root = NeuralMCTSNode(game=game)
        nodes_created = 1
        simulations = 0

        while simulations < self.config.simulations:
            if deadline is not None and simulations > 0 and time.perf_counter() >= deadline:
                break
            node = root
            if deadline is not None and time.perf_counter() >= deadline:
                break

            while node.is_expanded and node.children and not node.is_terminal:
                node = node.best_child(self.config.puct_exploration)

            if node.is_terminal:
                value = _terminal_value(node.game.outcome, node.game.board.side_to_move)
            else:
                remaining_nodes = None
                if self.config.node_budget is not None:
                    remaining_nodes = max(0, self.config.node_budget - nodes_created)
                if remaining_nodes == 0:
                    value = self._evaluate(node.game)
                else:
                    value, created = self._expand(node, max_children=remaining_nodes)
                    nodes_created += created
            self._backup(node, value)
            simulations += 1

        selected_move = _select_by_temperature(root, self.config.temperature, self._rng)
        if selected_move is None:
            selected_move = self._rng.choice(legal)
        elapsed = time.perf_counter() - start
        result = NeuralMCTSResult(
            move=selected_move,
            simulations=simulations,
            nodes=nodes_created,
            elapsed_seconds=elapsed,
            visit_counts={move: child.visits for move, child in root.children.items()},
        )
        self.last_result = result
        return result

    def _expand(
        self,
        node: NeuralMCTSNode,
        *,
        max_children: int | None = None,
    ) -> tuple[float, int]:
        prediction = self.inference.predict(node.game, mask_legal_moves=True)
        priors = _legal_priors(node.game, prediction)
        created = 0
        for move, prior in priors.items():
            if max_children is not None and created >= max_children:
                break
            child = NeuralMCTSNode(
                game=node.game.play(move),
                parent=node,
                move=move,
                prior=prior,
            )
            node.children[move] = child
            created += 1
        node.is_expanded = True
        return prediction.value, created

    def _evaluate(self, game: Game) -> float:
        return self.inference.predict(game, mask_legal_moves=True).value

    @staticmethod
    def _backup(node: NeuralMCTSNode, value: float) -> None:
        current: NeuralMCTSNode | None = node
        current_value = value
        while current is not None:
            current.visits += 1
            current.total_value += current_value
            current_value = -current_value
            current = current.parent


def _legal_priors(game: Game, prediction: InferenceResult) -> dict[Move, float]:
    legal = game.legal_moves
    raw_priors: dict[Move, float] = {}
    total = 0.0
    for move in legal:
        index = move_to_action_index(move, game.board)
        prior = max(0.0, float(prediction.policy[index].item()))
        raw_priors[move] = prior
        total += prior
    if total <= 0.0 or not math.isfinite(total):
        uniform = 1.0 / len(legal)
        return {move: uniform for move in legal}
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
    if not root.children:
        return None
    children = list(root.children.values())
    if temperature == 0.0:
        return max(
            children,
            key=lambda child: (
                child.visits,
                -child.mean_value,
                child.move.to_uci() if child.move else "",
            ),
        ).move
    weights = [float(child.visits) ** (1.0 / temperature) for child in children]
    if sum(weights) <= 0.0:
        weights = [child.prior for child in children]
    if sum(weights) <= 0.0:
        weights = [1.0 for _child in children]
    return rng.choices([child.move for child in children], weights=weights, k=1)[0]
