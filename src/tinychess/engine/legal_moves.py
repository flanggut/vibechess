"""Pseudo-legal and legal chess move generation."""

from __future__ import annotations

from collections.abc import Iterable

from tinychess.engine.board import Board
from tinychess.engine.move import Move
from tinychess.engine.piece import Color, Piece, PieceType
from tinychess.engine.square import Square, file_index, make_square, rank_index

PROMOTION_PIECES = (PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT)

_KNIGHT_DELTAS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
_KING_DELTAS = ((1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1))
_BISHOP_DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_ROOK_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_QUEEN_DIRECTIONS = _BISHOP_DIRECTIONS + _ROOK_DIRECTIONS


def pseudo_legal_moves(board: Board) -> tuple[Move, ...]:
    """Return pseudo-legal moves for the side to move.

    Pseudo-legal moves follow piece movement rules, including castling and en passant,
    but may leave the moving side's king in check.
    """
    moves: list[Move] = []
    for square, piece in board.occupied_squares():
        if piece.color is not board.side_to_move:
            continue
        if piece.kind is PieceType.PAWN:
            moves.extend(_pawn_moves(board, square, piece.color))
        elif piece.kind is PieceType.KNIGHT:
            moves.extend(_leaper_moves(board, square, piece.color, _KNIGHT_DELTAS))
        elif piece.kind is PieceType.BISHOP:
            moves.extend(_slider_moves(board, square, piece.color, _BISHOP_DIRECTIONS))
        elif piece.kind is PieceType.ROOK:
            moves.extend(_slider_moves(board, square, piece.color, _ROOK_DIRECTIONS))
        elif piece.kind is PieceType.QUEEN:
            moves.extend(_slider_moves(board, square, piece.color, _QUEEN_DIRECTIONS))
        elif piece.kind is PieceType.KING:
            moves.extend(_king_moves(board, square, piece.color))
    return tuple(moves)


def legal_moves(board: Board) -> tuple[Move, ...]:
    """Return legal moves for the side to move."""
    legal: list[Move] = []
    moving_color = board.side_to_move
    for move in pseudo_legal_moves(board):
        next_board = board.apply_move(move)
        if not is_in_check(next_board, moving_color):
            legal.append(move)
    return tuple(legal)


def perft(board: Board, depth: int) -> int:
    """Return the number of legal move leaf nodes at ``depth``."""
    if depth < 0:
        msg = f"perft depth must be non-negative, got {depth}"
        raise ValueError(msg)
    if depth == 0:
        return 1
    return sum(perft(board.apply_move(move), depth - 1) for move in legal_moves(board))


def is_in_check(board: Board, color: Color) -> bool:
    """Return whether ``color``'s king is attacked."""
    king_square = _king_square(board, color)
    return is_square_attacked(board, king_square, color.opposite)


def is_square_attacked(board: Board, square: Square, by_color: Color) -> bool:
    """Return whether ``square`` is attacked by ``by_color``."""
    target_file = file_index(square)
    target_rank = rank_index(square)

    pawn_rank_delta = -1 if by_color is Color.WHITE else 1
    for file_delta in (-1, 1):
        attacker = _offset_square(target_file, target_rank, file_delta, pawn_rank_delta)
        if _has_piece(board, attacker, by_color, PieceType.PAWN):
            return True

    for file_delta, rank_delta in _KNIGHT_DELTAS:
        attacker = _offset_square(target_file, target_rank, file_delta, rank_delta)
        if _has_piece(board, attacker, by_color, PieceType.KNIGHT):
            return True

    bishop_attackers = {PieceType.BISHOP, PieceType.QUEEN}
    for file_delta, rank_delta in _BISHOP_DIRECTIONS:
        if _ray_attacked(
            board, target_file, target_rank, file_delta, rank_delta, by_color, bishop_attackers
        ):
            return True

    rook_attackers = {PieceType.ROOK, PieceType.QUEEN}
    for file_delta, rank_delta in _ROOK_DIRECTIONS:
        if _ray_attacked(
            board, target_file, target_rank, file_delta, rank_delta, by_color, rook_attackers
        ):
            return True

    for file_delta, rank_delta in _KING_DELTAS:
        attacker = _offset_square(target_file, target_rank, file_delta, rank_delta)
        if _has_piece(board, attacker, by_color, PieceType.KING):
            return True

    return False


def _pawn_moves(board: Board, square: Square, color: Color) -> Iterable[Move]:
    moves: list[Move] = []
    start_rank = 1 if color is Color.WHITE else 6
    promotion_rank = 7 if color is Color.WHITE else 0
    direction = 1 if color is Color.WHITE else -1
    from_file = file_index(square)
    from_rank = rank_index(square)

    one_step = _offset_square(from_file, from_rank, 0, direction)
    if one_step is not None and board.piece_at(one_step) is None:
        moves.extend(_promotion_or_normal(square, one_step, rank_index(one_step) == promotion_rank))
        two_step = _offset_square(from_file, from_rank, 0, direction * 2)
        if from_rank == start_rank and two_step is not None and board.piece_at(two_step) is None:
            moves.append(Move(square, two_step))

    for file_delta in (-1, 1):
        target = _offset_square(from_file, from_rank, file_delta, direction)
        if target is None:
            continue
        target_piece = board.piece_at(target)
        if target_piece is not None and target_piece.color is not color:
            moves.extend(_promotion_or_normal(square, target, rank_index(target) == promotion_rank))
        elif board.en_passant_target == target:
            capture_square = _en_passant_capture_square(target, color)
            if _has_piece(board, capture_square, color.opposite, PieceType.PAWN):
                moves.append(Move(square, target))
    return moves


def _leaper_moves(
    board: Board, square: Square, color: Color, deltas: Iterable[tuple[int, int]]
) -> Iterable[Move]:
    moves: list[Move] = []
    from_file = file_index(square)
    from_rank = rank_index(square)
    for file_delta, rank_delta in deltas:
        target = _offset_square(from_file, from_rank, file_delta, rank_delta)
        if target is None:
            continue
        target_piece = board.piece_at(target)
        if target_piece is None or target_piece.color is not color:
            moves.append(Move(square, target))
    return moves


def _slider_moves(
    board: Board, square: Square, color: Color, directions: Iterable[tuple[int, int]]
) -> Iterable[Move]:
    moves: list[Move] = []
    from_file = file_index(square)
    from_rank = rank_index(square)
    for file_delta, rank_delta in directions:
        target_file = from_file + file_delta
        target_rank = from_rank + rank_delta
        while _is_on_board(target_file, target_rank):
            target = make_square(target_file, target_rank)
            target_piece = board.piece_at(target)
            if target_piece is None:
                moves.append(Move(square, target))
            else:
                if target_piece.color is not color:
                    moves.append(Move(square, target))
                break
            target_file += file_delta
            target_rank += rank_delta
    return moves


def _king_moves(board: Board, square: Square, color: Color) -> Iterable[Move]:
    moves = list(_leaper_moves(board, square, color, _KING_DELTAS))
    moves.extend(_castling_moves(board, square, color))
    return moves


def _castling_moves(board: Board, square: Square, color: Color) -> Iterable[Move]:
    if is_in_check(board, color):
        return ()
    rank = 0 if color is Color.WHITE else 7
    expected_king_square = make_square(4, rank)
    if square != expected_king_square:
        return ()

    moves: list[Move] = []
    opponent = color.opposite
    king_side = "K" if color is Color.WHITE else "k"
    queen_side = "Q" if color is Color.WHITE else "q"

    if king_side in board.castling_rights:
        rook_square = make_square(7, rank)
        if (
            _has_piece(board, rook_square, color, PieceType.ROOK)
            and board.piece_at(make_square(5, rank)) is None
            and board.piece_at(make_square(6, rank)) is None
            and not is_square_attacked(board, make_square(5, rank), opponent)
            and not is_square_attacked(board, make_square(6, rank), opponent)
        ):
            moves.append(Move(square, make_square(6, rank)))

    if queen_side in board.castling_rights:
        rook_square = make_square(0, rank)
        if (
            _has_piece(board, rook_square, color, PieceType.ROOK)
            and board.piece_at(make_square(1, rank)) is None
            and board.piece_at(make_square(2, rank)) is None
            and board.piece_at(make_square(3, rank)) is None
            and not is_square_attacked(board, make_square(2, rank), opponent)
            and not is_square_attacked(board, make_square(3, rank), opponent)
        ):
            moves.append(Move(square, make_square(2, rank)))
    return moves


def _promotion_or_normal(from_square: Square, to_square: Square, is_promotion: bool) -> list[Move]:
    if not is_promotion:
        return [Move(from_square, to_square)]
    return [Move(from_square, to_square, promotion) for promotion in PROMOTION_PIECES]


def _offset_square(file_idx: int, rank_idx: int, file_delta: int, rank_delta: int) -> Square | None:
    target_file = file_idx + file_delta
    target_rank = rank_idx + rank_delta
    if not _is_on_board(target_file, target_rank):
        return None
    return make_square(target_file, target_rank)


def _is_on_board(file_idx: int, rank_idx: int) -> bool:
    return 0 <= file_idx < 8 and 0 <= rank_idx < 8


def _has_piece(board: Board, square: Square | None, color: Color, kind: PieceType) -> bool:
    if square is None:
        return False
    piece = board.piece_at(square)
    return piece == Piece(color, kind)


def _en_passant_capture_square(target: Square, capturing_color: Color) -> Square | None:
    offset = -8 if capturing_color is Color.WHITE else 8
    capture_index = int(target) + offset
    if not 0 <= capture_index < 64:
        return None
    return Square(capture_index)


def _ray_attacked(
    board: Board,
    target_file: int,
    target_rank: int,
    file_delta: int,
    rank_delta: int,
    by_color: Color,
    attacking_kinds: set[PieceType],
) -> bool:
    current_file = target_file + file_delta
    current_rank = target_rank + rank_delta
    while _is_on_board(current_file, current_rank):
        square = make_square(current_file, current_rank)
        piece = board.piece_at(square)
        if piece is None:
            current_file += file_delta
            current_rank += rank_delta
            continue
        return piece.color is by_color and piece.kind in attacking_kinds
    return False


def _king_square(board: Board, color: Color) -> Square:
    for square, piece in board.occupied_squares():
        if piece == Piece(color, PieceType.KING):
            return square
    msg = f"board has no {color.value} king"
    raise ValueError(msg)
