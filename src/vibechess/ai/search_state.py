"""Lightweight game state for neural MCTS search trees.

``SearchState`` mirrors the parts of :class:`vibechess.engine.game.Game` needed by
search while avoiding full history snapshots on speculative child nodes. It keeps
only the current board, clocks, repetition map, forced outcome, and a compact move
path. A full ``Game`` can be reconstructed at boundaries that still require that
API.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field

from vibechess.engine.board import Board
from vibechess.engine.game import Game
from vibechess.engine.legal_moves import legal_moves as generate_legal_moves
from vibechess.engine.move import Move
from vibechess.engine.outcome import Outcome
from vibechess.engine.transition import (
    PositionKey,
    TransitionState,
    advance_known_legal_state,
    outcome_for_state,
    position_key,
)


def _profile_scope(name: str, **tags: object) -> AbstractContextManager[None]:
    from vibechess.profiling import profile_scope

    return profile_scope(name, **tags)


def _record_counter(name: str, amount: int | float = 1, **tags: object) -> None:
    from vibechess.profiling import record_counter

    record_counter(name, amount, **tags)


def _record_distribution(name: str, value: int | float, *, unit: str, **tags: object) -> None:
    from vibechess.profiling import record_distribution

    record_distribution(name, value, unit=unit, **tags)


@dataclass(frozen=True, slots=True)
class SearchState:
    """Search-only state with ``Game``-equivalent transition semantics for MCTS.

    Speculative children share the immutable base history from ``from_game()`` and
    append only suffix moves in ``move_path``. They do not store a copied
    ``Game.positions`` tuple, full ``Game.moves`` tuple, or full ``Game`` object per node.
    """

    board: Board
    halfmove_clock: int = 0
    fullmove_number: int = 1
    repetition_counts: Mapping[PositionKey, int] = field(default_factory=dict)
    move_path: tuple[Move, ...] = ()
    forced_outcome: Outcome | None = None
    _base_positions: tuple[Board, ...] = field(default_factory=tuple, repr=False, compare=False)
    _base_moves: tuple[Move, ...] = field(default_factory=tuple, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.repetition_counts:
            object.__setattr__(self, "repetition_counts", {position_key(self.board): 1})
        if self._base_positions and len(self._base_positions) != len(self._base_moves) + 1:
            msg = "base positions must contain exactly one more entry than base moves"
            raise ValueError(msg)

    @classmethod
    def from_game(cls, game: Game) -> SearchState:
        """Create a search state from ``game`` without copying its position history."""
        with _profile_scope("search_state.from_game"):
            return cls(
                board=game.board,
                halfmove_clock=game.halfmove_clock,
                fullmove_number=game.fullmove_number,
                repetition_counts=dict(game.repetition_counts),
                move_path=(),
                forced_outcome=game.forced_outcome,
                _base_positions=game.positions,
                _base_moves=game.moves,
            )

    def to_game(self, *, include_positions: bool = True) -> Game:
        """Return a ``Game`` view of this search state.

        ``include_positions=True`` reconstructs full position history from the shared
        base plus speculative suffix moves, so equality and history-sensitive boundary
        code match ``Game.play_known_legal()``. ``include_positions=False`` returns a
        compact single-position ``Game`` for inference encoding, where only the current
        board, clocks, repetition counts, outcome, and move path are needed.
        """
        with _profile_scope("search_state.to_game", include_positions=include_positions):
            _record_counter("search_state.to_game.calls")
            _record_distribution("search_state.move_path_length", len(self.move_path), unit="moves")
            positions = self._positions() if include_positions else (self.board,)
            return Game(
                positions=positions,
                moves=self.moves,
                halfmove_clock=self.halfmove_clock,
                fullmove_number=self.fullmove_number,
                repetition_counts=dict(self.repetition_counts),
                forced_outcome=self.forced_outcome,
            )

    @property
    def moves(self) -> tuple[Move, ...]:
        """Return the full move path as a boundary-compatible tuple."""
        return (*self._base_moves, *self.move_path)

    @property
    def legal_moves(self) -> tuple[Move, ...]:
        """Return legal moves in the current search position."""
        with _profile_scope("search_state.legal_moves"):
            legal = generate_legal_moves(self.board)
            _record_distribution("search_state.legal_moves", len(legal), unit="moves")
            return legal

    @property
    def outcome(self) -> Outcome | None:
        """Return the current outcome using the same pragmatic rules as ``Game``."""
        with _profile_scope("search_state.outcome"):
            if self.forced_outcome is not None:
                return self.forced_outcome
            legal = self.legal_moves
            return self.outcome_with_legal_moves(legal)

    def outcome_with_legal_moves(self, legal_moves: tuple[Move, ...]) -> Outcome | None:
        """Return the current outcome using a caller-supplied legal-move cache."""
        with _profile_scope("search_state.outcome_with_legal_moves"):
            return self._outcome_with_legal_moves_impl(legal_moves)

    def _outcome_with_legal_moves_impl(self, legal_moves: tuple[Move, ...]) -> Outcome | None:
        return outcome_for_state(
            TransitionState(
                board=self.board,
                halfmove_clock=self.halfmove_clock,
                fullmove_number=self.fullmove_number,
                repetition_counts=self.repetition_counts,
                forced_outcome=self.forced_outcome,
            ),
            legal_moves,
        )

    def play_known_legal(self, move: Move) -> SearchState:
        """Return the state after applying a move already known to be legal."""
        with _profile_scope("search_state.play_known_legal"):
            _record_counter("search_state.play_known_legal.calls")
            return self._play_known_legal_impl(move)

    def _play_known_legal_impl(self, move: Move) -> SearchState:
        result = advance_known_legal_state(
            TransitionState(
                board=self.board,
                halfmove_clock=self.halfmove_clock,
                fullmove_number=self.fullmove_number,
                repetition_counts=self.repetition_counts,
                forced_outcome=self.forced_outcome,
            ),
            move,
        )

        return SearchState(
            board=result.board,
            halfmove_clock=result.halfmove_clock,
            fullmove_number=result.fullmove_number,
            repetition_counts=result.repetition_counts,
            move_path=(*self.move_path, move),
            forced_outcome=None,
            _base_positions=self._base_positions,
            _base_moves=self._base_moves,
        )

    def _positions(self) -> tuple[Board, ...]:
        with _profile_scope("search_state.positions_reconstruct"):
            if self._base_positions:
                positions = list(self._base_positions)
                board = positions[-1]
                suffix = self.move_path
            else:
                positions = [self.board]
                board = self.board
                suffix = ()

            for move in suffix:
                board = board.apply_move(move)
                positions.append(board)
            if positions[-1] != self.board:
                positions[-1] = self.board
            return tuple(positions)
