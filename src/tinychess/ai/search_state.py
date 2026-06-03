"""Lightweight game state for neural MCTS search trees.

``SearchState`` mirrors the parts of :class:`tinychess.engine.game.Game` needed by
search while avoiding full history snapshots on speculative child nodes. It keeps
only the current board, clocks, repetition map, forced outcome, and a compact move
path. A full ``Game`` can be reconstructed at boundaries that still require that
API.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from tinychess.engine.board import Board
from tinychess.engine.game import (
    Game,
    PositionKey,
    _is_capture,
    _position_key,
    has_insufficient_material,
)
from tinychess.engine.legal_moves import is_in_check
from tinychess.engine.legal_moves import legal_moves as generate_legal_moves
from tinychess.engine.move import Move
from tinychess.engine.outcome import Outcome, OutcomeReason
from tinychess.engine.piece import Color, PieceType


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
            object.__setattr__(self, "repetition_counts", {_position_key(self.board): 1})
        if self._base_positions and len(self._base_positions) != len(self._base_moves) + 1:
            msg = "base positions must contain exactly one more entry than base moves"
            raise ValueError(msg)

    @classmethod
    def from_game(cls, game: Game) -> SearchState:
        """Create a search state from ``game`` without copying its position history."""
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
        return generate_legal_moves(self.board)

    @property
    def outcome(self) -> Outcome | None:
        """Return the current outcome using the same pragmatic rules as ``Game``."""
        if self.forced_outcome is not None:
            return self.forced_outcome
        legal = self.legal_moves
        return self.outcome_with_legal_moves(legal)

    def outcome_with_legal_moves(self, legal_moves: tuple[Move, ...]) -> Outcome | None:
        """Return the current outcome using a caller-supplied legal-move cache."""
        if self.forced_outcome is not None:
            return self.forced_outcome
        if not legal_moves:
            if is_in_check(self.board, self.board.side_to_move):
                return Outcome(
                    reason=OutcomeReason.CHECKMATE,
                    winner=self.board.side_to_move.opposite,
                )
            return Outcome(reason=OutcomeReason.STALEMATE)

        if self.halfmove_clock >= 100:
            return Outcome(reason=OutcomeReason.FIFTY_MOVE)
        if self.repetition_counts.get(_position_key(self.board), 0) >= 3:
            return Outcome(reason=OutcomeReason.REPETITION)
        if has_insufficient_material(self.board):
            return Outcome(reason=OutcomeReason.INSUFFICIENT_MATERIAL)
        return None

    def play_known_legal(self, move: Move) -> SearchState:
        """Return the state after applying a move already known to be legal."""
        moving_piece = self.board.piece_at(move.from_square)
        if moving_piece is None:  # defensive; legal membership should prevent this
            msg = f"cannot move from empty square {move.from_square}"
            raise ValueError(msg)
        is_capture = _is_capture(self.board, move, moving_piece)
        next_board = self.board.apply_move(move)
        next_key = _position_key(next_board)
        next_repetitions = dict(self.repetition_counts)
        next_repetitions[next_key] = next_repetitions.get(next_key, 0) + 1

        next_halfmove_clock = (
            0 if moving_piece.kind is PieceType.PAWN or is_capture else self.halfmove_clock + 1
        )
        next_fullmove_number = self.fullmove_number + (
            1 if self.board.side_to_move is Color.BLACK else 0
        )

        return SearchState(
            board=next_board,
            halfmove_clock=next_halfmove_clock,
            fullmove_number=next_fullmove_number,
            repetition_counts=next_repetitions,
            move_path=(*self.move_path, move),
            forced_outcome=None,
            _base_positions=self._base_positions,
            _base_moves=self._base_moves,
        )

    def _positions(self) -> tuple[Board, ...]:
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
