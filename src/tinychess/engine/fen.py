"""FEN parsing and serialization helpers.

FEN strings contain six fields: piece placement, active color, castling rights,
en-passant target, halfmove clock, and fullmove number. ``Board`` stores the
first four fields directly; ``FenPosition`` carries the two move counters as
well so callers can initialize game state without losing information.
"""

from __future__ import annotations

from dataclasses import dataclass

from tinychess.engine.board import Board
from tinychess.engine.piece import Color, Piece
from tinychess.engine.square import BOARD_SIZE, Square, parse_square, square_name

STANDARD_STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
STARTING_FEN = STANDARD_STARTING_FEN
_CASTLING_ORDER = "KQkq"


@dataclass(frozen=True, slots=True)
class FenPosition:
    """A full FEN position.

    ``board`` carries placement, side to move, castling rights, and en-passant
    target. The counters are kept here because they are game-level state in the
    tinychess model.
    """

    board: Board
    halfmove_clock: int = 0
    fullmove_number: int = 1

    def __post_init__(self) -> None:
        if self.halfmove_clock < 0:
            msg = f"halfmove clock must be non-negative, got {self.halfmove_clock}"
            raise ValueError(msg)
        if self.fullmove_number < 1:
            msg = f"fullmove number must be positive, got {self.fullmove_number}"
            raise ValueError(msg)

    def to_fen(self) -> str:
        """Serialize this position to canonical FEN."""
        return format_fen(self)


# Backward-compatible alias for the standard chess initial FEN.
STARTPOS_FEN = STANDARD_STARTING_FEN


def parse_fen(fen: str) -> FenPosition:
    """Parse a full six-field FEN string.

    Raises ``ValueError`` with a field-specific message when the string is
    malformed. Semantic legality beyond basic FEN field validation is left to
    engine move-generation checks.
    """
    fields = fen.split()
    if len(fields) != 6:
        msg = f"FEN must contain exactly 6 fields, got {len(fields)}"
        raise ValueError(msg)

    placement, active_color, castling, en_passant, halfmove, fullmove = fields
    board = Board(
        squares=_parse_placement(placement),
        side_to_move=_parse_active_color(active_color),
        castling_rights=_parse_castling_rights(castling),
        en_passant_target=_parse_en_passant(en_passant, _parse_active_color(active_color)),
    )
    return FenPosition(
        board=board,
        halfmove_clock=_parse_halfmove_clock(halfmove),
        fullmove_number=_parse_fullmove_number(fullmove),
    )


def board_from_fen(fen: str) -> Board:
    """Parse FEN and return its board component."""
    return parse_fen(fen).board


def format_fen(position: FenPosition) -> str:
    """Serialize a ``FenPosition`` to canonical full FEN."""
    return board_to_fen(
        position.board,
        halfmove_clock=position.halfmove_clock,
        fullmove_number=position.fullmove_number,
    )


def board_to_fen(board: Board, *, halfmove_clock: int = 0, fullmove_number: int = 1) -> str:
    """Serialize a board plus move counters to canonical full FEN."""
    if halfmove_clock < 0:
        msg = f"halfmove clock must be non-negative, got {halfmove_clock}"
        raise ValueError(msg)
    if fullmove_number < 1:
        msg = f"fullmove number must be positive, got {fullmove_number}"
        raise ValueError(msg)

    placement = _format_placement(board)
    active_color = "w" if board.side_to_move is Color.WHITE else "b"
    castling = _format_castling_rights(board.castling_rights)
    en_passant = "-" if board.en_passant_target is None else square_name(board.en_passant_target)
    return f"{placement} {active_color} {castling} {en_passant} {halfmove_clock} {fullmove_number}"


def _parse_placement(placement: str) -> tuple[Piece | None, ...]:
    rows = placement.split("/")
    if len(rows) != 8:
        msg = "FEN placement must contain 8 slash-separated ranks"
        raise ValueError(msg)

    squares: list[Piece | None] = [None] * BOARD_SIZE
    for row_index, row in enumerate(rows):
        if not row:
            msg = f"FEN placement rank {8 - row_index} is empty"
            raise ValueError(msg)
        rank = 7 - row_index
        file_index = 0
        previous_was_digit = False
        for char in row:
            if char.isdigit():
                if previous_was_digit:
                    msg = f"adjacent empty-square counts in FEN rank {8 - row_index}"
                    raise ValueError(msg)
                if char == "0":
                    msg = f"invalid zero empty-square count in FEN rank {8 - row_index}"
                    raise ValueError(msg)
                empty_count = int(char)
                file_index += empty_count
                if file_index > 8:
                    msg = f"too many files in FEN rank {8 - row_index}"
                    raise ValueError(msg)
                previous_was_digit = True
                continue
            previous_was_digit = False
            if file_index >= 8:
                msg = f"too many files in FEN rank {8 - row_index}"
                raise ValueError(msg)
            try:
                piece = Piece.from_symbol(char)
            except ValueError as error:
                msg = f"invalid FEN piece symbol {char!r} in rank {8 - row_index}"
                raise ValueError(msg) from error
            squares[rank * 8 + file_index] = piece
            file_index += 1
        if file_index != 8:
            msg = f"FEN rank {8 - row_index} covers {file_index} files, expected 8"
            raise ValueError(msg)
    return tuple(squares)


def _parse_active_color(active_color: str) -> Color:
    if active_color == "w":
        return Color.WHITE
    if active_color == "b":
        return Color.BLACK
    msg = f"invalid FEN active color: {active_color!r}"
    raise ValueError(msg)


def _parse_castling_rights(castling: str) -> frozenset[str]:
    if castling == "-":
        return frozenset()
    if not castling:
        msg = "FEN castling rights field is empty"
        raise ValueError(msg)
    rights: set[str] = set()
    for char in castling:
        if char not in _CASTLING_ORDER:
            msg = f"invalid FEN castling right: {char!r}"
            raise ValueError(msg)
        if char in rights:
            msg = f"duplicate FEN castling right: {char!r}"
            raise ValueError(msg)
        rights.add(char)
    return frozenset(rights)


def _parse_en_passant(en_passant: str, active_color: Color) -> Square | None:
    if en_passant == "-":
        return None
    try:
        square = parse_square(en_passant)
    except ValueError as error:
        msg = f"invalid FEN en passant target: {en_passant!r}"
        raise ValueError(msg) from error
    rank_char = en_passant[1]
    expected_rank = "6" if active_color is Color.WHITE else "3"
    if rank_char != expected_rank:
        msg = (
            "FEN en passant target must be on rank 6 when white is active "
            f"or rank 3 when black is active, got {en_passant!r} with active color "
            f"{'w' if active_color is Color.WHITE else 'b'}"
        )
        raise ValueError(msg)
    return square


def _parse_halfmove_clock(halfmove: str) -> int:
    if not halfmove.isdecimal():
        msg = f"invalid FEN halfmove clock: {halfmove!r}"
        raise ValueError(msg)
    return int(halfmove)


def _parse_fullmove_number(fullmove: str) -> int:
    if not fullmove.isdecimal():
        msg = f"invalid FEN fullmove number: {fullmove!r}"
        raise ValueError(msg)
    value = int(fullmove)
    if value < 1:
        msg = f"FEN fullmove number must be positive, got {value}"
        raise ValueError(msg)
    return value


def _format_placement(board: Board) -> str:
    rows: list[str] = []
    for rank in range(7, -1, -1):
        row_parts: list[str] = []
        empty_count = 0
        for file_index in range(8):
            piece = board.squares[rank * 8 + file_index]
            if piece is None:
                empty_count += 1
                continue
            if empty_count:
                row_parts.append(str(empty_count))
                empty_count = 0
            row_parts.append(piece.symbol)
        if empty_count:
            row_parts.append(str(empty_count))
        rows.append("".join(row_parts))
    return "/".join(rows)


def _format_castling_rights(castling_rights: frozenset[str]) -> str:
    ordered = "".join(right for right in _CASTLING_ORDER if right in castling_rights)
    return ordered or "-"
