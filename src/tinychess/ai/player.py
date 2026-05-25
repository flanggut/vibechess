"""Common player interface and baseline random player."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from tinychess.engine.game import Game
from tinychess.engine.move import Move
from tinychess.engine.outcome import Outcome, OutcomeReason
from tinychess.engine.piece import Color


class NoLegalMoveError(ValueError):
    """Raised when a player is asked to move in a terminal/no-legal-move position."""


@runtime_checkable
class Player(Protocol):
    """Interface shared by human, random, MCTS, and neural-MCTS players."""

    def select_move(self, game: Game) -> Move:
        """Return one legal move for the current game position."""


@dataclass(slots=True)
class RandomPlayer:
    """Player that selects uniformly from the current legal moves.

    The player owns a local ``random.Random`` instance by default. Passing ``seed`` makes
    selections reproducible; passing ``rng`` allows callers to provide an already
    configured local random generator. The global RNG is never used.
    """

    seed: int | None = None
    rng: random.Random | None = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.seed is not None and self.rng is not None:
            msg = "pass either seed or rng, not both"
            raise ValueError(msg)
        self._rng = random.Random(self.seed) if self.rng is None else self.rng

    def select_move(self, game: Game) -> Move:
        """Select a legal move from ``game`` or raise ``NoLegalMoveError`` clearly."""
        outcome = game.outcome
        if outcome is not None:
            msg = f"cannot select a move from a terminal game: {outcome.reason.value}"
            raise NoLegalMoveError(msg)
        legal = game.legal_moves
        if not legal:
            msg = "cannot select a move from a position with no legal moves"
            raise NoLegalMoveError(msg)
        return self._rng.choice(legal)


def play_game(
    white: Player,
    black: Player,
    *,
    game: Game | None = None,
    max_plies: int = 512,
) -> Game:
    """Play white-vs-black players until an outcome or a maximum-ply cap."""
    if max_plies < 0:
        msg = f"max_plies must be non-negative, got {max_plies}"
        raise ValueError(msg)
    current = Game.new() if game is None else game
    players = {Color.WHITE: white, Color.BLACK: black}
    for _ in range(max_plies):
        if current.outcome is not None:
            return current
        legal = current.legal_moves
        if not legal:
            return current
        move = players[current.board.side_to_move].select_move(current)
        if move not in legal:
            msg = f"player selected illegal move: {move}"
            raise ValueError(msg)
        current = current.play(move)
    if current.outcome is None:
        return Game(
            positions=current.positions,
            moves=current.moves,
            halfmove_clock=current.halfmove_clock,
            fullmove_number=current.fullmove_number,
            repetition_counts=dict(current.repetition_counts),
            forced_outcome=Outcome(OutcomeReason.MAX_PLIES),
        )
    return current
