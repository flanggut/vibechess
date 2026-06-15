"""Pseudo-legal and legal chess move generation."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager

from vibechess.engine.board import Board
from vibechess.engine.move import Move
from vibechess.engine.piece import Color, Piece, PieceType
from vibechess.engine.square import Square, validate_square


def _profile_scope(name: str, **tags: object) -> AbstractContextManager[None]:
    from vibechess.profiling import profile_scope

    return profile_scope(name, **tags)


def _record_counter(name: str, amount: int | float = 1, **tags: object) -> None:
    from vibechess.profiling import record_counter

    record_counter(name, amount, **tags)


def _record_distribution(name: str, value: int | float, *, unit: str, **tags: object) -> None:
    from vibechess.profiling import record_distribution

    record_distribution(name, value, unit=unit, **tags)


PROMOTION_PIECES = (PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT)

_KNIGHT_DELTAS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
_KING_DELTAS = ((1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1))
_BISHOP_DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_ROOK_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_QUEEN_DIRECTIONS = _BISHOP_DIRECTIONS + _ROOK_DIRECTIONS


def _build_leaper_attack_table(
    deltas: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, ...], ...]:
    table: list[tuple[int, ...]] = []
    for target_index in range(64):
        target_file = target_index % 8
        target_rank = target_index // 8
        attackers: list[int] = []
        for file_delta, rank_delta in deltas:
            attacker_file = target_file + file_delta
            attacker_rank = target_rank + rank_delta
            if 0 <= attacker_file < 8 and 0 <= attacker_rank < 8:
                attackers.append(attacker_rank * 8 + attacker_file)
        table.append(tuple(attackers))
    return tuple(table)


def _build_pawn_attack_table(color: Color) -> tuple[tuple[int, ...], ...]:
    table: list[tuple[int, ...]] = []
    rank_delta = -1 if color is Color.WHITE else 1
    for target_index in range(64):
        target_file = target_index % 8
        target_rank = target_index // 8
        attackers: list[int] = []
        for file_delta in (-1, 1):
            attacker_file = target_file + file_delta
            attacker_rank = target_rank + rank_delta
            if 0 <= attacker_file < 8 and 0 <= attacker_rank < 8:
                attackers.append(attacker_rank * 8 + attacker_file)
        table.append(tuple(attackers))
    return tuple(table)


def _build_ray_attack_table(
    directions: tuple[tuple[int, int], ...],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    table: list[tuple[tuple[int, ...], ...]] = []
    for target_index in range(64):
        target_file = target_index % 8
        target_rank = target_index // 8
        rays: list[tuple[int, ...]] = []
        for file_delta, rank_delta in directions:
            ray: list[int] = []
            current_file = target_file + file_delta
            current_rank = target_rank + rank_delta
            while 0 <= current_file < 8 and 0 <= current_rank < 8:
                ray.append(current_rank * 8 + current_file)
                current_file += file_delta
                current_rank += rank_delta
            rays.append(tuple(ray))
        table.append(tuple(rays))
    return tuple(table)


_KNIGHT_ATTACKERS_BY_TARGET = _build_leaper_attack_table(_KNIGHT_DELTAS)
_KING_ATTACKERS_BY_TARGET = _build_leaper_attack_table(_KING_DELTAS)
_WHITE_PAWN_ATTACKERS_BY_TARGET = _build_pawn_attack_table(Color.WHITE)
_BLACK_PAWN_ATTACKERS_BY_TARGET = _build_pawn_attack_table(Color.BLACK)
_BISHOP_ATTACK_RAYS_BY_TARGET = _build_ray_attack_table(_BISHOP_DIRECTIONS)
_ROOK_ATTACK_RAYS_BY_TARGET = _build_ray_attack_table(_ROOK_DIRECTIONS)


def pseudo_legal_moves(board: Board) -> tuple[Move, ...]:
    """Return pseudo-legal moves for the side to move.

    Pseudo-legal moves follow piece movement rules, including castling and en passant,
    but may leave the moving side's king in check.
    """
    with _profile_scope("legal.pseudo"):
        moves: list[Move] = []
        side_to_move = board.side_to_move
        for square_index, piece in enumerate(board.squares):
            if piece is None or piece.color is not side_to_move:
                continue
            if piece.kind is PieceType.PAWN:
                _append_pawn_moves(moves, board, square_index, piece.color)
            elif piece.kind is PieceType.KNIGHT:
                _append_leaper_moves(
                    moves, board.squares, square_index, piece.color, _KNIGHT_DELTAS
                )
            elif piece.kind is PieceType.BISHOP:
                _append_slider_moves(
                    moves, board.squares, square_index, piece.color, _BISHOP_DIRECTIONS
                )
            elif piece.kind is PieceType.ROOK:
                _append_slider_moves(
                    moves, board.squares, square_index, piece.color, _ROOK_DIRECTIONS
                )
            elif piece.kind is PieceType.QUEEN:
                _append_slider_moves(
                    moves, board.squares, square_index, piece.color, _QUEEN_DIRECTIONS
                )
            elif piece.kind is PieceType.KING:
                _append_king_moves(moves, board, square_index, piece.color)
        result = tuple(moves)
        _record_distribution("legal.pseudo_moves", len(result), unit="moves")
        return result


def legal_moves(board: Board) -> tuple[Move, ...]:
    """Return legal moves for the side to move."""
    with _profile_scope("legal.legal_moves"):
        pseudo_moves = pseudo_legal_moves(board)
        if not pseudo_moves:
            _record_distribution("legal.legal_moves", 0, unit="moves")
            return ()

        legal: list[Move] = []
        squares = board.squares
        scratch = list(squares)
        en_passant_target = board.en_passant_target
        moving_color = board.side_to_move
        opponent = moving_color.opposite
        king_index = int(_king_square(board, moving_color))
        with _profile_scope("legal.filter"):
            _record_counter("legal.filter_candidates", len(pseudo_moves))
            _record_distribution("legal.filter_candidates", len(pseudo_moves), unit="moves")
            for move in pseudo_moves:
                moving_piece = squares[int(move.from_square)]
                if _king_safe_after_move(
                    scratch, squares, move, moving_piece, king_index, opponent, en_passant_target
                ):
                    legal.append(move)
        result = tuple(legal)
        _record_distribution("legal.legal_moves", len(result), unit="moves")
        return result


def has_legal_move(board: Board) -> bool:
    """Return whether the side to move has at least one legal move."""
    with _profile_scope("legal.has_legal_move"):
        pseudo_moves = pseudo_legal_moves(board)
        if not pseudo_moves:
            return False

        squares = board.squares
        scratch = list(squares)
        en_passant_target = board.en_passant_target
        moving_color = board.side_to_move
        opponent = moving_color.opposite
        king_index = int(_king_square(board, moving_color))
        with _profile_scope("legal.filter"):
            _record_counter("legal.filter_candidates", len(pseudo_moves))
            for move in pseudo_moves:
                moving_piece = squares[int(move.from_square)]
                if _king_safe_after_move(
                    scratch, squares, move, moving_piece, king_index, opponent, en_passant_target
                ):
                    return True
        return False


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
    with _profile_scope("legal.is_in_check"):
        _record_counter("legal.in_check_calls")
        king_square = _king_square(board, color)
        return is_square_attacked(board, king_square, color.opposite)


def is_square_attacked(board: Board, square: Square, by_color: Color) -> bool:
    """Return whether ``square`` is attacked by ``by_color``."""
    with _profile_scope("legal.is_square_attacked"):
        _record_counter("legal.is_square_attacked.calls")
        return _is_square_attacked_index(board.squares, int(validate_square(square)), by_color)


def _is_square_attacked_index(
    squares: Sequence[Piece | None], target_index: int, by_color: Color
) -> bool:
    """Return whether ``target_index`` is attacked, operating on a raw squares array.

    This is the hot legality-check core. It avoids constructing a :class:`Board` and
    skips ``Square`` validation, accepting an in-range board index and a piece array
    (tuple or scratch list) directly.
    """
    pawn_attackers = (
        _WHITE_PAWN_ATTACKERS_BY_TARGET[target_index]
        if by_color is Color.WHITE
        else _BLACK_PAWN_ATTACKERS_BY_TARGET[target_index]
    )
    for square_index in pawn_attackers:
        piece = squares[square_index]
        if piece is not None and piece.color is by_color and piece.kind is PieceType.PAWN:
            return True

    for square_index in _KNIGHT_ATTACKERS_BY_TARGET[target_index]:
        piece = squares[square_index]
        if piece is not None and piece.color is by_color and piece.kind is PieceType.KNIGHT:
            return True

    for ray in _BISHOP_ATTACK_RAYS_BY_TARGET[target_index]:
        for square_index in ray:
            piece = squares[square_index]
            if piece is None:
                continue
            if piece.color is by_color and (
                piece.kind is PieceType.BISHOP or piece.kind is PieceType.QUEEN
            ):
                return True
            break

    for ray in _ROOK_ATTACK_RAYS_BY_TARGET[target_index]:
        for square_index in ray:
            piece = squares[square_index]
            if piece is None:
                continue
            if piece.color is by_color and (
                piece.kind is PieceType.ROOK or piece.kind is PieceType.QUEEN
            ):
                return True
            break

    for square_index in _KING_ATTACKERS_BY_TARGET[target_index]:
        piece = squares[square_index]
        if piece is not None and piece.color is by_color and piece.kind is PieceType.KING:
            return True

    return False


def _king_safe_after_move(
    scratch: list[Piece | None],
    squares: tuple[Piece | None, ...],
    move: Move,
    moving_piece: Piece | None,
    king_index: int,
    opponent: Color,
    en_passant_target: Square | None,
) -> bool:
    """Return whether the mover's king is safe after a pseudo-legal move.

    Uses make/unmake on a shared ``scratch`` array to test king safety without
    constructing a new :class:`Board`. Only the squares the move actually changes are
    mutated and then restored, so each call costs O(changed squares) rather than a full
    64-entry board copy. ``scratch`` must equal ``squares`` on entry and is restored to
    that state before returning. Castling rights, en-passant target, clocks, and side to
    move do not affect attack detection and are intentionally ignored here.
    """
    if moving_piece is None:
        return False
    from_index = int(move.from_square)
    to_index = int(move.to_square)
    color = moving_piece.color
    is_king = moving_piece.kind is PieceType.KING

    from_original = squares[from_index]
    to_original = squares[to_index]
    scratch[from_index] = None

    delta_abs = abs(to_index - from_index)
    en_passant_capture_index = -1
    en_passant_capture_original: Piece | None = None
    if (
        moving_piece.kind is PieceType.PAWN
        and en_passant_target == move.to_square
        and to_original is None
        and (delta_abs == 7 or delta_abs == 9)
    ):
        capture_index = to_index + (-8 if color is Color.WHITE else 8)
        if 0 <= capture_index < 64:
            en_passant_capture_index = capture_index
            en_passant_capture_original = squares[capture_index]
            scratch[capture_index] = None

    scratch[to_index] = Piece(color, move.promotion) if move.promotion is not None else moving_piece

    rook_from = -1
    rook_to = -1
    rook_from_original: Piece | None = None
    rook_to_original: Piece | None = None
    if is_king and delta_abs == 2:
        rank_offset = 0 if color is Color.WHITE else 56
        if to_index == rank_offset + 6:
            rook_from, rook_to = rank_offset + 7, rank_offset + 5
        else:
            rook_from, rook_to = rank_offset, rank_offset + 3
        rook_from_original = squares[rook_from]
        rook_to_original = squares[rook_to]
        scratch[rook_to] = rook_from_original
        scratch[rook_from] = None

    target_index = to_index if is_king else king_index
    safe = not _is_square_attacked_index(scratch, target_index, opponent)

    # Restore scratch to the original board placement for the next candidate.
    if rook_from != -1:
        scratch[rook_to] = rook_to_original
        scratch[rook_from] = rook_from_original
    if en_passant_capture_index != -1:
        scratch[en_passant_capture_index] = en_passant_capture_original
    scratch[to_index] = to_original
    scratch[from_index] = from_original
    return safe


def _append_pawn_moves(
    moves: list[Move],
    board: Board,
    from_index: int,
    color: Color,
) -> None:
    squares = board.squares
    start_rank = 1 if color is Color.WHITE else 6
    promotion_rank = 7 if color is Color.WHITE else 0
    direction = 1 if color is Color.WHITE else -1
    from_file = from_index & 7
    from_rank = from_index >> 3
    from_square = Square(from_index)

    one_step_index = from_index + direction * 8
    if 0 <= one_step_index < 64 and squares[one_step_index] is None:
        _append_promotion_or_normal(
            moves,
            from_square,
            one_step_index,
            one_step_index >> 3 == promotion_rank,
        )
        two_step_index = from_index + direction * 16
        if (
            from_rank == start_rank
            and 0 <= two_step_index < 64
            and squares[two_step_index] is None
        ):
            moves.append(Move(from_square, Square(two_step_index)))

    en_passant_target = (
        -1 if board.en_passant_target is None else int(board.en_passant_target)
    )
    opponent = color.opposite
    for file_delta in (-1, 1):
        target_file = from_file + file_delta
        if not 0 <= target_file < 8:
            continue
        target_index = from_index + direction * 8 + file_delta
        if not 0 <= target_index < 64:
            continue
        target_piece = squares[target_index]
        if target_piece is not None and target_piece.color is not color:
            _append_promotion_or_normal(
                moves,
                from_square,
                target_index,
                target_index >> 3 == promotion_rank,
            )
        elif target_index == en_passant_target:
            capture_index = target_index + (-8 if color is Color.WHITE else 8)
            if (
                0 <= capture_index < 64
                and _has_piece_at_index(squares, capture_index, opponent, PieceType.PAWN)
            ):
                moves.append(Move(from_square, Square(target_index)))


def _append_leaper_moves(
    moves: list[Move],
    squares: Sequence[Piece | None],
    from_index: int,
    color: Color,
    deltas: Sequence[tuple[int, int]],
) -> None:
    from_file = from_index & 7
    from_rank = from_index >> 3
    from_square = Square(from_index)
    for file_delta, rank_delta in deltas:
        target_file = from_file + file_delta
        target_rank = from_rank + rank_delta
        if not (0 <= target_file < 8 and 0 <= target_rank < 8):
            continue
        target_index = target_rank * 8 + target_file
        target_piece = squares[target_index]
        if target_piece is None or target_piece.color is not color:
            moves.append(Move(from_square, Square(target_index)))


def _append_slider_moves(
    moves: list[Move],
    squares: Sequence[Piece | None],
    from_index: int,
    color: Color,
    directions: Sequence[tuple[int, int]],
) -> None:
    from_file = from_index & 7
    from_rank = from_index >> 3
    from_square = Square(from_index)
    for file_delta, rank_delta in directions:
        target_file = from_file + file_delta
        target_rank = from_rank + rank_delta
        while 0 <= target_file < 8 and 0 <= target_rank < 8:
            target_index = target_rank * 8 + target_file
            target_piece = squares[target_index]
            if target_piece is None:
                moves.append(Move(from_square, Square(target_index)))
            else:
                if target_piece.color is not color:
                    moves.append(Move(from_square, Square(target_index)))
                break
            target_file += file_delta
            target_rank += rank_delta


def _append_king_moves(
    moves: list[Move],
    board: Board,
    from_index: int,
    color: Color,
) -> None:
    _append_leaper_moves(moves, board.squares, from_index, color, _KING_DELTAS)
    _append_castling_moves(moves, board, from_index, color)


def _append_castling_moves(
    moves: list[Move],
    board: Board,
    from_index: int,
    color: Color,
) -> None:
    rank_offset = 0 if color is Color.WHITE else 56
    expected_king_index = rank_offset + 4
    if from_index != expected_king_index:
        return

    opponent = color.opposite
    squares = board.squares
    if _is_square_attacked_index(squares, from_index, opponent):
        return

    from_square = Square(from_index)
    king_side = "K" if color is Color.WHITE else "k"
    queen_side = "Q" if color is Color.WHITE else "q"

    if (
        king_side in board.castling_rights
        and _has_piece_at_index(squares, rank_offset + 7, color, PieceType.ROOK)
        and squares[rank_offset + 5] is None
        and squares[rank_offset + 6] is None
        and not _is_square_attacked_index(squares, rank_offset + 5, opponent)
        and not _is_square_attacked_index(squares, rank_offset + 6, opponent)
    ):
        moves.append(Move(from_square, Square(rank_offset + 6)))

    if (
        queen_side in board.castling_rights
        and _has_piece_at_index(squares, rank_offset, color, PieceType.ROOK)
        and squares[rank_offset + 1] is None
        and squares[rank_offset + 2] is None
        and squares[rank_offset + 3] is None
        and not _is_square_attacked_index(squares, rank_offset + 2, opponent)
        and not _is_square_attacked_index(squares, rank_offset + 3, opponent)
    ):
        moves.append(Move(from_square, Square(rank_offset + 2)))


def _append_promotion_or_normal(
    moves: list[Move],
    from_square: Square,
    to_index: int,
    is_promotion: bool,
) -> None:
    to_square = Square(to_index)
    if not is_promotion:
        moves.append(Move(from_square, to_square))
        return
    for promotion in PROMOTION_PIECES:
        moves.append(Move(from_square, to_square, promotion))


def _has_piece_at_index(
    squares: Sequence[Piece | None], square_index: int, color: Color, kind: PieceType
) -> bool:
    piece = squares[square_index]
    return piece is not None and piece.color is color and piece.kind is kind


def _king_square(board: Board, color: Color) -> Square:
    with _profile_scope("legal.king_square"):
        for square_index, piece in enumerate(board.squares):
            if piece is not None and piece.color is color and piece.kind is PieceType.KING:
                return Square(square_index)
    msg = f"board has no {color.value} king"
    raise ValueError(msg)
