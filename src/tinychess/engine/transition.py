"""Shared engine state transition and outcome primitives.

This module centralizes the board-key, capture, clock, repetition, and pragmatic
outcome rules used by lightweight engine state containers. It intentionally does
not validate full move legality; callers must pass moves already known to be legal
when advancing state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from tinychess.engine.board import Board
from tinychess.engine.legal_moves import is_in_check
from tinychess.engine.move import Move
from tinychess.engine.outcome import Outcome, OutcomeReason
from tinychess.engine.piece import Color, Piece, PieceType
from tinychess.engine.square import Square, file_index, rank_index

PositionKey = tuple[tuple[Piece | None, ...], Color, frozenset[str], Square | None]


@dataclass(frozen=True, slots=True)
class TransitionState:
    """Minimal game-state data needed to advance clocks and outcomes."""

    board: Board
    halfmove_clock: int
    fullmove_number: int
    repetition_counts: Mapping[PositionKey, int]
    forced_outcome: Outcome | None = None


@dataclass(frozen=True, slots=True)
class TransitionResult:
    """Result of applying one already-legal move to a transition state."""

    board: Board
    halfmove_clock: int
    fullmove_number: int
    repetition_counts: dict[PositionKey, int]
    moving_piece: Piece
    is_capture: bool


def position_key(board: Board) -> PositionKey:
    """Return the repetition-relevant key for ``board``."""
    return (board.squares, board.side_to_move, board.castling_rights, board.en_passant_target)


def is_capture(board: Board, move: Move, moving_piece: Piece) -> bool:
    """Return whether ``move`` captures under current pseudo-legal assumptions."""
    if board.piece_at(move.to_square) is not None:
        return True
    return (
        moving_piece.kind is PieceType.PAWN
        and board.en_passant_target == move.to_square
        and abs(int(move.to_square) - int(move.from_square)) in {7, 9}
    )


def advance_known_legal_state(state: TransitionState, move: Move) -> TransitionResult:
    """Advance ``state`` by a move already known to be legal.

    The helper updates only board placement, side-to-move, clocks, and repetition
    counts. It preserves the existing copy-on-apply board architecture and relies
    on ``Board.apply_move()`` for board-state updates.
    """
    board = state.board
    moving_piece = board.piece_at(move.from_square)
    if moving_piece is None:
        msg = f"cannot move from empty square {move.from_square}"
        raise ValueError(msg)

    capture = is_capture(board, move, moving_piece)
    next_board = board.apply_move(move)
    next_key = position_key(next_board)
    next_repetitions = dict(state.repetition_counts)
    next_repetitions[next_key] = next_repetitions.get(next_key, 0) + 1

    next_halfmove_clock = (
        0 if moving_piece.kind is PieceType.PAWN or capture else state.halfmove_clock + 1
    )
    next_fullmove_number = state.fullmove_number + (1 if board.side_to_move is Color.BLACK else 0)

    return TransitionResult(
        board=next_board,
        halfmove_clock=next_halfmove_clock,
        fullmove_number=next_fullmove_number,
        repetition_counts=next_repetitions,
        moving_piece=moving_piece,
        is_capture=capture,
    )


def outcome_for_state(
    state: TransitionState,
    legal_moves: tuple[Move, ...],
) -> Outcome | None:
    """Return the pragmatic outcome for ``state``, or ``None`` if ongoing."""
    if state.forced_outcome is not None:
        return state.forced_outcome
    board = state.board
    if not legal_moves:
        if is_in_check(board, board.side_to_move):
            return Outcome(reason=OutcomeReason.CHECKMATE, winner=board.side_to_move.opposite)
        return Outcome(reason=OutcomeReason.STALEMATE)

    if state.halfmove_clock >= 100:
        return Outcome(reason=OutcomeReason.FIFTY_MOVE)
    if state.repetition_counts.get(position_key(board), 0) >= 3:
        return Outcome(reason=OutcomeReason.REPETITION)
    if has_insufficient_material(board):
        return Outcome(reason=OutcomeReason.INSUFFICIENT_MATERIAL)
    return None


def has_insufficient_material(board: Board) -> bool:
    """Return whether material is insufficient for a pragmatic checkmate possibility."""
    pieces = tuple(piece for _square, piece in board.occupied_squares())
    non_kings = [piece for piece in pieces if piece.kind is not PieceType.KING]
    if not non_kings:
        return True
    if any(piece.kind in {PieceType.PAWN, PieceType.ROOK, PieceType.QUEEN} for piece in non_kings):
        return False
    if len(non_kings) == 1:
        return non_kings[0].kind in {PieceType.BISHOP, PieceType.KNIGHT}
    if all(piece.kind is PieceType.BISHOP for piece in non_kings):
        bishop_colors = {
            square_color(square)
            for square, piece in board.occupied_squares()
            if piece.kind is PieceType.BISHOP
        }
        return len(bishop_colors) == 1
    return False


def square_color(square: Square) -> int:
    """Return the color index of a board square: 0 for dark parity, 1 for light parity."""
    return (file_index(square) + rank_index(square)) % 2
