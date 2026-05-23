"""Board representation and starting position setup."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from tinychess.engine.piece import Color, Piece, PieceType
from tinychess.engine.square import BOARD_SIZE, Square, parse_square, square_name, validate_square

STARTING_POSITION = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


@dataclass(frozen=True, slots=True)
class Board:
    """A compact array/mailbox-style chess board.

    The board stores 64 entries using the project square convention: ``a1 == 0`` and
    ``h8 == 63``. Empty squares contain ``None``. This type models placement and side
    to move only; castling rights, en passant, clocks, move application, and legality
    belong to later work packages.
    """

    squares: tuple[Piece | None, ...]
    side_to_move: Color = Color.WHITE

    def __post_init__(self) -> None:
        if len(self.squares) != BOARD_SIZE:
            msg = f"board must contain {BOARD_SIZE} squares, got {len(self.squares)}"
            raise ValueError(msg)

    @classmethod
    def empty(cls, *, side_to_move: Color = Color.WHITE) -> Board:
        """Return an empty board."""
        return cls(squares=(None,) * BOARD_SIZE, side_to_move=side_to_move)

    @classmethod
    def starting_position(cls) -> Board:
        """Return the standard chess starting position."""
        board = cls.empty(side_to_move=Color.WHITE)
        placements: dict[str, Piece] = {}

        back_rank = [
            PieceType.ROOK,
            PieceType.KNIGHT,
            PieceType.BISHOP,
            PieceType.QUEEN,
            PieceType.KING,
            PieceType.BISHOP,
            PieceType.KNIGHT,
            PieceType.ROOK,
        ]
        for file_index, kind in enumerate(back_rank):
            file_name = chr(ord("a") + file_index)
            placements[f"{file_name}1"] = Piece(Color.WHITE, kind)
            placements[f"{file_name}2"] = Piece(Color.WHITE, PieceType.PAWN)
            placements[f"{file_name}7"] = Piece(Color.BLACK, PieceType.PAWN)
            placements[f"{file_name}8"] = Piece(Color.BLACK, kind)

        return board.with_pieces(placements.items())

    def piece_at(self, square: Square | str) -> Piece | None:
        """Return the piece at a square, or ``None`` if it is empty."""
        index = parse_square(square) if isinstance(square, str) else validate_square(square)
        return self.squares[int(index)]

    def with_piece(self, square: Square | str, piece: Piece | None) -> Board:
        """Return a new board with one square changed."""
        index = parse_square(square) if isinstance(square, str) else validate_square(square)
        squares = list(self.squares)
        squares[int(index)] = piece
        return Board(squares=tuple(squares), side_to_move=self.side_to_move)

    def with_pieces(self, pieces: Iterable[tuple[str | Square, Piece | None]]) -> Board:
        """Return a new board with multiple square placements changed."""
        board = self
        for square, piece in pieces:
            board = board.with_piece(square, piece)
        return board

    def occupied_squares(self) -> tuple[tuple[Square, Piece], ...]:
        """Return occupied squares and pieces in ascending square-index order."""
        return tuple(
            (Square(index), piece) for index, piece in enumerate(self.squares) if piece is not None
        )

    def render(self, *, unicode: bool = False, coordinates: bool = True) -> str:
        """Render the board as text for a simple terminal UI."""
        symbols = _UNICODE_SYMBOLS if unicode else None
        lines: list[str] = []
        for rank in range(7, -1, -1):
            cells: list[str] = []
            for file_index in range(8):
                square = Square(rank * 8 + file_index)
                piece = self.piece_at(square)
                if piece is None:
                    cells.append(".")
                elif symbols is None:
                    cells.append(piece.symbol)
                else:
                    cells.append(symbols[piece])
            line = " ".join(cells)
            if coordinates:
                line = f"{rank + 1} {line}"
            lines.append(line)
        if coordinates:
            lines.append("  a b c d e f g h")
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return the coordinate text rendering."""
        return self.render()


_UNICODE_SYMBOLS: dict[Piece, str] = {
    Piece(Color.WHITE, PieceType.KING): "♔",
    Piece(Color.WHITE, PieceType.QUEEN): "♕",
    Piece(Color.WHITE, PieceType.ROOK): "♖",
    Piece(Color.WHITE, PieceType.BISHOP): "♗",
    Piece(Color.WHITE, PieceType.KNIGHT): "♘",
    Piece(Color.WHITE, PieceType.PAWN): "♙",
    Piece(Color.BLACK, PieceType.KING): "♚",
    Piece(Color.BLACK, PieceType.QUEEN): "♛",
    Piece(Color.BLACK, PieceType.ROOK): "♜",
    Piece(Color.BLACK, PieceType.BISHOP): "♝",
    Piece(Color.BLACK, PieceType.KNIGHT): "♞",
    Piece(Color.BLACK, PieceType.PAWN): "♟",
}


def board_from_ascii(rows: str, *, side_to_move: Color = Color.WHITE) -> Board:
    """Build a board from eight slash-separated ranks using FEN-style symbols.

    This helper is intentionally placement-only; it does not parse full FEN state.
    """
    rank_rows = rows.split("/")
    if len(rank_rows) != 8:
        msg = "board rows must contain eight slash-separated ranks"
        raise ValueError(msg)

    board = Board.empty(side_to_move=side_to_move)
    for rank_offset, row in enumerate(rank_rows):
        rank = 7 - rank_offset
        file_index = 0
        for char in row:
            if char.isdigit():
                empty_count = int(char)
                if empty_count == 0:
                    msg = f"invalid zero empty-square count in rank row {row!r}"
                    raise ValueError(msg)
                file_index += empty_count
                if file_index > 8:
                    msg = f"too many files in rank row {row!r}"
                    raise ValueError(msg)
                continue
            if file_index >= 8:
                msg = f"too many files in rank row {row!r}"
                raise ValueError(msg)
            board = board.with_piece(
                square_name(Square(rank * 8 + file_index)), Piece.from_symbol(char)
            )
            file_index += 1
        if file_index != 8:
            msg = f"rank row {row!r} covers {file_index} files, expected 8"
            raise ValueError(msg)
    return board
