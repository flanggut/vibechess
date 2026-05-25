"""Position encoding and AlphaZero-style policy action mapping.

WP11 intentionally keeps this module independent of model/training code.  It uses
plain Python numeric tensors so tests and engine integrations do not require MLX to
be installed; callers on Apple Silicon can convert with :func:`to_mlx`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

from tinychess.engine.board import Board
from tinychess.engine.game import Game
from tinychess.engine.move import Move
from tinychess.engine.piece import Color, PieceType
from tinychess.engine.square import BOARD_SIZE, Square, file_index, rank_index, validate_square

ACTION_SPACE_VERSION = "az-8x8x73-v1"
ENCODER_VERSION = "tinychess-board-v1"
ACTION_PLANES = 73
ACTION_SPACE_SIZE = BOARD_SIZE * ACTION_PLANES
POLICY_SHAPE = (BOARD_SIZE, ACTION_PLANES)

# 12 piece planes + side + 4 castling + en-passant + halfmove + fullmove.
ENCODER_CHANNELS = 20
TENSOR_SHAPE = (ENCODER_CHANNELS, 8, 8)

Tensor3D: TypeAlias = list[list[list[float]]]

_PIECE_CHANNELS: dict[tuple[Color, PieceType], int] = {
    (Color.WHITE, PieceType.PAWN): 0,
    (Color.WHITE, PieceType.KNIGHT): 1,
    (Color.WHITE, PieceType.BISHOP): 2,
    (Color.WHITE, PieceType.ROOK): 3,
    (Color.WHITE, PieceType.QUEEN): 4,
    (Color.WHITE, PieceType.KING): 5,
    (Color.BLACK, PieceType.PAWN): 6,
    (Color.BLACK, PieceType.KNIGHT): 7,
    (Color.BLACK, PieceType.BISHOP): 8,
    (Color.BLACK, PieceType.ROOK): 9,
    (Color.BLACK, PieceType.QUEEN): 10,
    (Color.BLACK, PieceType.KING): 11,
}
_CASTLING_CHANNELS = {"K": 13, "Q": 14, "k": 15, "q": 16}

# Queen-like planes: 8 compass directions, distances 1..7.
_QUEEN_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)
_KNIGHT_DELTAS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)
_UNDERPROMOTION_TYPES = (PieceType.KNIGHT, PieceType.BISHOP, PieceType.ROOK)
_UNDERPROMOTION_FILES = (0, -1, 1)  # forward, capture-left, capture-right from mover's view.
_UNDERPROMOTION_OFFSET = 64


def encode_game(game: Game) -> Tensor3D:
    """Encode the current game state as a ``[20][8][8]`` numeric tensor.

    Channels 0..11 are one-hot piece planes ordered white PNBRQK then black
    pnbrqk. Channel 12 is all ones when black is to move and zeros for white.
    Channels 13..16 are castling rights KQkq. Channel 17 marks the en-passant
    target square. Channels 18 and 19 are full-board scalar planes containing
    ``halfmove_clock / 100`` and ``fullmove_number / 100`` respectively.
    """
    return encode_board(
        game.board,
        halfmove_clock=game.halfmove_clock,
        fullmove_number=game.fullmove_number,
    )


def encode_board(
    board: Board,
    *,
    halfmove_clock: int = 0,
    fullmove_number: int = 1,
) -> Tensor3D:
    """Encode a board and optional clocks as a deterministic Python tensor."""
    tensor = [
        [[0.0 for _file in range(8)] for _rank in range(8)]
        for _channel in range(ENCODER_CHANNELS)
    ]

    for square, piece in board.occupied_squares():
        channel = _PIECE_CHANNELS[(piece.color, piece.kind)]
        tensor[channel][rank_index(square)][file_index(square)] = 1.0

    if board.side_to_move is Color.BLACK:
        _fill_plane(tensor[12], 1.0)
    for right, channel in _CASTLING_CHANNELS.items():
        if right in board.castling_rights:
            _fill_plane(tensor[channel], 1.0)
    if board.en_passant_target is not None:
        tensor[17][rank_index(board.en_passant_target)][file_index(board.en_passant_target)] = 1.0
    _fill_plane(tensor[18], halfmove_clock / 100.0)
    _fill_plane(tensor[19], fullmove_number / 100.0)
    return tensor


def tensor_shape(tensor: Sequence[Sequence[Sequence[float]]]) -> tuple[int, int, int]:
    """Return the shape of a nested Python position tensor."""
    channels = len(tensor)
    ranks = len(tensor[0]) if channels else 0
    files = len(tensor[0][0]) if ranks else 0
    return (channels, ranks, files)


def to_mlx(tensor: Tensor3D) -> object:
    """Convert a Python tensor to an MLX array when MLX is installed."""
    try:
        import mlx.core as mx  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local platform env
        msg = "MLX is not installed; use the plain Python tensor or install mlx"
        raise RuntimeError(msg) from exc
    return mx.array(tensor)


def move_to_action_index(move: Move, board: Board | None = None) -> int:
    """Return the fixed-policy action index for a representable move.

    Queen promotions use the normal queen-like movement plane. Knight, bishop,
    and rook underpromotions use planes 64..72 and require ``board`` so the
    mover's forward direction is unambiguous. When ``board`` is supplied, any
    promotion annotation is validated against a side-to-move pawn reaching the
    final rank.
    """
    from_square = validate_square(move.from_square)
    to_square = validate_square(move.to_square)
    from_file = file_index(from_square)
    from_rank = rank_index(from_square)
    df = file_index(to_square) - from_file
    dr = rank_index(to_square) - from_rank

    if move.promotion is not None and board is not None:
        _validate_promotion_move(move, board, df, dr)

    if move.promotion in _UNDERPROMOTION_TYPES:
        if board is None:
            raise ValueError("underpromotion action mapping requires board state")
        plane = _underpromotion_plane(move, board, df, dr)
    else:
        plane = _queen_or_knight_plane(df, dr)
    return int(from_square) * ACTION_PLANES + plane


def action_index_to_move(index: int, board: Board | None = None) -> Move:
    """Decode an action index into a move.

    Underpromotion action planes require ``board`` because their direction is
    stored relative to the side to move. Queen promotions also require ``board``
    to decode with the promotion annotation instead of as an ordinary queen-like
    move. Decoded moves are not guaranteed legal; callers should intersect with
    ``Game.legal_moves`` or use :func:`legal_move_mask`.
    """
    if not 0 <= index < ACTION_SPACE_SIZE:
        msg = f"action index must be in 0..{ACTION_SPACE_SIZE - 1}, got {index}"
        raise ValueError(msg)
    from_square = Square(index // ACTION_PLANES)
    plane = index % ACTION_PLANES
    from_file = file_index(from_square)
    from_rank = rank_index(from_square)

    promotion = None
    if plane < 56:
        direction = _QUEEN_DIRECTIONS[plane // 7]
        distance = plane % 7 + 1
        df = direction[0] * distance
        dr = direction[1] * distance
    elif plane < 64:
        df, dr = _KNIGHT_DELTAS[plane - 56]
    else:
        if board is None:
            raise ValueError("underpromotion action decoding requires board state")
        under_index = plane - _UNDERPROMOTION_OFFSET
        promotion = _UNDERPROMOTION_TYPES[under_index // 3]
        file_delta = _UNDERPROMOTION_FILES[under_index % 3]
        rank_delta = 1 if board.side_to_move is Color.WHITE else -1
        df = file_delta
        dr = rank_delta

    to_file = from_file + df
    to_rank = from_rank + dr
    if not (0 <= to_file < 8 and 0 <= to_rank < 8):
        msg = f"action index {index} decodes off-board from square {int(from_square)}"
        raise ValueError(msg)
    to_square = Square(to_rank * 8 + to_file)
    if promotion is None and board is not None:
        piece = board.piece_at(from_square)
        if (
            piece is not None
            and piece.kind is PieceType.PAWN
            and piece.color is board.side_to_move
            and to_rank == (7 if piece.color is Color.WHITE else 0)
        ):
            promotion = PieceType.QUEEN
    return Move(from_square=from_square, to_square=to_square, promotion=promotion)


def legal_move_mask(game: Game) -> list[float]:
    """Return a length-4672 mask with ``1.0`` for legal actions and ``0.0`` elsewhere."""
    mask = [0.0] * ACTION_SPACE_SIZE
    for move in game.legal_moves:
        mask[move_to_action_index(move, game.board)] = 1.0
    return mask


def _queen_or_knight_plane(df: int, dr: int) -> int:
    if (df, dr) in _KNIGHT_DELTAS:
        return 56 + _KNIGHT_DELTAS.index((df, dr))
    distance = max(abs(df), abs(dr))
    if distance < 1 or distance > 7:
        raise ValueError(f"move delta ({df}, {dr}) is not representable")
    step = (0 if df == 0 else df // abs(df), 0 if dr == 0 else dr // abs(dr))
    is_straight_or_diagonal = abs(df) in {0, distance} and abs(dr) in {0, distance}
    if step not in _QUEEN_DIRECTIONS or not is_straight_or_diagonal:
        raise ValueError(f"move delta ({df}, {dr}) is not representable")
    return _QUEEN_DIRECTIONS.index(step) * 7 + (distance - 1)


def _underpromotion_plane(move: Move, board: Board, df: int, dr: int) -> int:
    _validate_promotion_move(move, board, df, dr)
    assert move.promotion is not None
    return (
        _UNDERPROMOTION_OFFSET
        + _UNDERPROMOTION_TYPES.index(move.promotion) * 3
        + _UNDERPROMOTION_FILES.index(df)
    )


def _validate_promotion_move(move: Move, board: Board, df: int, dr: int) -> None:
    if move.promotion not in (*_UNDERPROMOTION_TYPES, PieceType.QUEEN):
        raise ValueError("promotion piece must be queen, rook, bishop, or knight")
    piece = board.piece_at(move.from_square)
    if piece is None or piece.kind is not PieceType.PAWN or piece.color is not board.side_to_move:
        raise ValueError("promotion move must be by the side-to-move pawn")
    expected_rank_delta = 1 if piece.color is Color.WHITE else -1
    if dr != expected_rank_delta or df not in _UNDERPROMOTION_FILES:
        raise ValueError(f"promotion delta ({df}, {dr}) is not representable")
    target_rank = rank_index(move.to_square)
    if target_rank != (7 if piece.color is Color.WHITE else 0):
        raise ValueError("promotion target must be the final rank")


def _fill_plane(plane: list[list[float]], value: float) -> None:
    for rank in range(8):
        for file_ in range(8):
            plane[rank][file_] = value
