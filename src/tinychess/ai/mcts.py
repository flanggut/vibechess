"""Classical Monte Carlo Tree Search baseline."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from tinychess.ai.player import NoLegalMoveError
from tinychess.ai.search_config import MCTSConfig
from tinychess.engine.game import Game, determine_outcome
from tinychess.engine.move import Move
from tinychess.engine.outcome import Outcome
from tinychess.engine.piece import Color, PieceType


@dataclass(slots=True)
class MCTSNode:
    """One node in a classical MCTS game tree."""

    game: Game
    parent: MCTSNode | None = None
    move: Move | None = None
    legal_moves: tuple[Move, ...] = ()
    outcome: Outcome | None = None
    untried_moves: list[Move] = field(default_factory=list)
    children: dict[Move, MCTSNode] = field(default_factory=dict)
    visits: int = 0
    total_value: float = 0.0

    @classmethod
    def create(
        cls,
        game: Game,
        *,
        rng: random.Random,
        parent: MCTSNode | None = None,
        move: Move | None = None,
    ) -> MCTSNode:
        """Create a node with cached position state and shuffled expansion order."""
        legal, outcome = _position_info(game)
        untried = list(legal) if outcome is None else []
        rng.shuffle(untried)
        return cls(
            game=game,
            parent=parent,
            move=move,
            legal_moves=legal,
            outcome=outcome,
            untried_moves=untried,
        )

    @property
    def is_terminal(self) -> bool:
        """Return whether this node's game is terminal."""
        return self.outcome is not None or not self.legal_moves

    @property
    def is_fully_expanded(self) -> bool:
        """Return whether every legal move has been expanded."""
        return not self.untried_moves

    def best_child(self, exploration: float, root_color: Color) -> MCTSNode:
        """Return the child with the highest adversarial UCB1 score.

        Values are stored from the root player's perspective. Root-side nodes maximize
        that value, while opponent-to-move nodes minimize it.
        """
        if not self.children:
            msg = "cannot select best child from a leaf"
            raise ValueError(msg)
        log_parent = math.log(max(1, self.visits))
        maximizing = self.game.board.side_to_move is root_color

        def score(child: MCTSNode) -> float:
            if child.visits == 0:
                return math.inf
            exploitation = child.total_value / child.visits
            if not maximizing:
                exploitation = -exploitation
            exploration_term = exploration * math.sqrt(log_parent / child.visits)
            return exploitation + exploration_term

        return max(
            self.children.values(),
            key=lambda child: (score(child), child.move.to_uci() if child.move else ""),
        )


@dataclass(frozen=True, slots=True)
class MCTSResult:
    """Result metadata from an MCTS search."""

    move: Move
    simulations: int
    nodes: int
    elapsed_seconds: float
    visit_counts: dict[Move, int] = field(default_factory=dict)

    @property
    def simulations_per_second(self) -> float:
        """Return completed simulations per second, or infinity for zero elapsed time."""
        if self.elapsed_seconds == 0:
            return math.inf
        return self.simulations / self.elapsed_seconds


@dataclass(slots=True)
class MCTSPlayer:
    """Classical MCTS player using random rollouts or static leaf evaluation."""

    config: MCTSConfig = field(default_factory=MCTSConfig)
    rng: random.Random | None = None
    _rng: random.Random = field(init=False, repr=False)
    last_result: MCTSResult | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.config.seed) if self.rng is None else self.rng

    def select_move(self, game: Game) -> Move:
        """Return an MCTS-selected legal move, or raise for terminal/no-legal positions."""
        return self.search(game).move

    def search(self, game: Game) -> MCTSResult:
        """Run MCTS from ``game`` and return the selected move plus budget metadata."""
        start = time.perf_counter()
        root = MCTSNode.create(game, rng=self._rng)
        if root.outcome is not None:
            msg = f"cannot select a move from a terminal game: {root.outcome.reason.value}"
            raise NoLegalMoveError(msg)
        legal = root.legal_moves
        if not legal:
            msg = "cannot select a move from a position with no legal moves"
            raise NoLegalMoveError(msg)

        root_color = game.board.side_to_move
        deadline = None
        if self.config.time_limit_seconds is not None:
            deadline = start + self.config.time_limit_seconds
        node_budget = self.config.node_budget
        nodes_created = 1
        simulations = 0

        while simulations < self.config.simulations:
            if deadline is not None and simulations > 0 and time.perf_counter() >= deadline:
                break
            node = root
            if deadline is not None and time.perf_counter() >= deadline:
                break
            while not node.is_terminal and node.is_fully_expanded and node.children:
                node = node.best_child(self.config.exploration, root_color)

            may_create_node = node_budget is None or nodes_created < node_budget
            if not node.is_terminal and node.untried_moves and may_create_node:
                move = node.untried_moves.pop()
                child_game = node.game.play_known_legal(move)
                child = MCTSNode.create(child_game, rng=self._rng, parent=node, move=move)
                node.children[move] = child
                node = child
                nodes_created += 1

            value = self._rollout_value(
                node.game,
                root_color,
                legal_moves=node.legal_moves,
                outcome=node.outcome,
            )
            self._backup(node, value)
            simulations += 1

        selected_move = _most_visited_move(root)
        if selected_move is None:
            selected_move = self._rng.choice(legal)
        elapsed = time.perf_counter() - start
        result = MCTSResult(
            move=selected_move,
            simulations=simulations,
            nodes=nodes_created,
            elapsed_seconds=elapsed,
            visit_counts={move: child.visits for move, child in root.children.items()},
        )
        self.last_result = result
        return result

    def _rollout_value(
        self,
        game: Game,
        root_color: Color,
        *,
        legal_moves: tuple[Move, ...] | None = None,
        outcome: Outcome | None = None,
    ) -> float:
        current = game
        current_legal = legal_moves
        current_outcome = outcome
        for _ in range(self.config.max_rollout_plies):
            if current_outcome is not None:
                return _outcome_value(current_outcome, root_color)
            if current_legal is None:
                current_legal, current_outcome = _position_info(current)
                if current_outcome is not None:
                    return _outcome_value(current_outcome, root_color)
            if not current_legal:
                return 0.0
            current = current.play_known_legal(self._rng.choice(current_legal))
            current_legal = None
            current_outcome = None
        if current_outcome is None and current_legal is None:
            current_legal, current_outcome = _position_info(current)
        return _static_leaf_value(current, root_color, outcome=current_outcome)

    @staticmethod
    def _backup(node: MCTSNode, value: float) -> None:
        current: MCTSNode | None = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent


def _position_info(game: Game) -> tuple[tuple[Move, ...], Outcome | None]:
    legal = game.legal_moves
    return legal, determine_outcome(game, legal_moves=legal)


def _most_visited_move(root: MCTSNode) -> Move | None:
    if not root.children:
        return None
    return max(
        root.children.values(),
        key=lambda child: (
            child.visits,
            child.total_value / child.visits if child.visits else -math.inf,
            child.move.to_uci() if child.move else "",
        ),
    ).move


def _outcome_value(outcome: Outcome, root_color: Color) -> float:
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner is root_color else -1.0


_PIECE_VALUES = {
    PieceType.PAWN: 1.0,
    PieceType.KNIGHT: 3.0,
    PieceType.BISHOP: 3.0,
    PieceType.ROOK: 5.0,
    PieceType.QUEEN: 9.0,
    PieceType.KING: 0.0,
}


def _static_leaf_value(
    game: Game,
    root_color: Color,
    *,
    outcome: Outcome | None = None,
) -> float:
    """Return a bounded value for a selected leaf from the root side's perspective.

    Terminal outcomes are exact. Ongoing positions use material only, which keeps
    ``max_rollout_plies=0`` cheap and avoids extra legal-move generation.
    """
    if outcome is not None:
        return _outcome_value(outcome, root_color)
    return _material_value(game, root_color)


def _material_value(game: Game, root_color: Color) -> float:
    score = 0.0
    for _square, piece in game.board.occupied_squares():
        value = _PIECE_VALUES[piece.kind]
        score += value if piece.color is root_color else -value
    return math.tanh(score / 10.0)
