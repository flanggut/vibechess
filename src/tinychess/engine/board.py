"""Board representation and starting position setup."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass

from tinychess.engine.move import Move
from tinychess.engine.piece import Color, Piece, PieceType
from tinychess.engine.square import BOARD_SIZE, Square, parse_square, square_name, validate_square

STARTING_POSITION = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def _profile_scope(name: str, **tags: object) -> AbstractContextManager[None]:
    from tinychess.nn.self_play_profile import profile_scope

    return profile_scope(name, **tags)


def _record_counter(name: str, amount: int | float = 1, **tags: object) -> None:
    from tinychess.nn.self_play_profile import record_counter

    record_counter(name, amount, **tags)


@dataclass(frozen=True, slots=True)
class Board:
    """A compact array/mailbox-style chess board.

    The board stores 64 entries using the project square convention: ``a1 == 0`` and
    ``h8 == 63``. Empty squares contain ``None``. WP03 also tracks the minimum state
    required for legal move generation: side to move, castling rights, and en passant
    target square.
    """

    squares: tuple[Piece | None, ...]
    side_to_move: Color = Color.WHITE
    castling_rights: frozenset[str] = frozenset()
    en_passant_target: Square | None = None

    def __post_init__(self) -> None:
        if len(self.squares) != BOARD_SIZE:
            msg = f"board must contain {BOARD_SIZE} squares, got {len(self.squares)}"
            raise ValueError(msg)
        invalid_rights = self.castling_rights - frozenset("KQkq")
        if invalid_rights:
            msg = f"invalid castling rights: {sorted(invalid_rights)!r}"
            raise ValueError(msg)
        if self.en_passant_target is not None:
            validate_square(self.en_passant_target)

    @classmethod
    def empty(
        cls,
        *,
        side_to_move: Color = Color.WHITE,
        castling_rights: frozenset[str] = frozenset(),
        en_passant_target: Square | None = None,
    ) -> Board:
        """Return an empty board."""
        return cls(
            squares=(None,) * BOARD_SIZE,
            side_to_move=side_to_move,
            castling_rights=castling_rights,
            en_passant_target=en_passant_target,
        )

    @classmethod
    def starting_position(cls) -> Board:
        """Return the standard chess starting position."""
        board = cls.empty(side_to_move=Color.WHITE, castling_rights=frozenset("KQkq"))
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

    @classmethod
    def from_fen(cls, fen: str) -> Board:
        """Parse a full six-field FEN string and return its board component."""
        from tinychess.engine.fen import board_from_fen

        return board_from_fen(fen)

    def to_fen(self, *, halfmove_clock: int = 0, fullmove_number: int = 1) -> str:
        """Serialize this board plus move counters to a full six-field FEN string."""
        from tinychess.engine.fen import board_to_fen

        return board_to_fen(
            self,
            halfmove_clock=halfmove_clock,
            fullmove_number=fullmove_number,
        )

    def piece_at(self, square: Square | str) -> Piece | None:
        """Return the piece at a square, or ``None`` if it is empty."""
        index = parse_square(square) if isinstance(square, str) else validate_square(square)
        return self.squares[int(index)]

    def with_piece(self, square: Square | str, piece: Piece | None) -> Board:
        """Return a new board with one square changed."""
        index = parse_square(square) if isinstance(square, str) else validate_square(square)
        squares = list(self.squares)
        squares[int(index)] = piece
        return Board(
            squares=tuple(squares),
            side_to_move=self.side_to_move,
            castling_rights=self.castling_rights,
            en_passant_target=self.en_passant_target,
        )

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

    def apply_move(self, move: Move) -> Board:
        """Return the board after applying a pseudo-legal move.

        This method is intentionally small and rule-focused for WP03 legal move
        generation and perft. It does not track game history, clocks, or outcomes.
        """
        with _profile_scope("board.apply_move"):
            _record_counter("board.apply_move.calls")
            moving_piece = self.piece_at(move.from_square)
            if moving_piece is None:
                msg = f"cannot move from empty square {move.from_square}"
                raise ValueError(msg)
            if moving_piece.color is not self.side_to_move:
                msg = "cannot move a piece that is not side_to_move"
                raise ValueError(msg)

            captured_piece = self.piece_at(move.to_square)
            squares = list(self.squares)
            squares[int(move.from_square)] = None

            if _is_en_passant_capture(self, move, moving_piece):
                capture_square = Square(
                    int(move.to_square) + (-8 if moving_piece.color is Color.WHITE else 8)
                )
                captured_piece = self.piece_at(capture_square)
                squares[int(capture_square)] = None

            placed_piece = moving_piece
            if move.promotion is not None:
                if moving_piece.kind is not PieceType.PAWN:
                    msg = "only pawns can promote"
                    raise ValueError(msg)
                if move.promotion not in _PROMOTION_PIECES:
                    msg = "promotion piece must be queen, rook, bishop, or knight"
                    raise ValueError(msg)
                placed_piece = Piece(moving_piece.color, move.promotion)
            squares[int(move.to_square)] = placed_piece

            is_castling = (
                moving_piece.kind is PieceType.KING
                and abs(int(move.to_square) - int(move.from_square)) == 2
            )
            if is_castling:
                _move_castling_rook(squares, moving_piece.color, move.to_square)

            return Board(
                squares=tuple(squares),
                side_to_move=self.side_to_move.opposite,
                castling_rights=_updated_castling_rights(self, move, moving_piece, captured_piece),
                en_passant_target=_next_en_passant_target(move, moving_piece),
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


def board_from_ascii(
    rows: str,
    *,
    side_to_move: Color = Color.WHITE,
    castling_rights: frozenset[str] = frozenset(),
    en_passant_target: Square | None = None,
) -> Board:
    """Build a board from eight slash-separated ranks using FEN-style symbols.

    This helper is intentionally placement-only; it does not parse full FEN state.
    """
    rank_rows = rows.split("/")
    if len(rank_rows) != 8:
        msg = "board rows must contain eight slash-separated ranks"
        raise ValueError(msg)

    board = Board.empty(
        side_to_move=side_to_move,
        castling_rights=castling_rights,
        en_passant_target=en_passant_target,
    )
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


_PROMOTION_PIECES = frozenset(
    {PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT}
)


def _is_en_passant_capture(board: Board, move: Move, moving_piece: Piece) -> bool:
    if (
        moving_piece.kind is not PieceType.PAWN
        or board.en_passant_target != move.to_square
        or board.piece_at(move.to_square) is not None
        or abs(int(move.to_square) - int(move.from_square)) not in {7, 9}
    ):
        return False
    capture_index = int(move.to_square) + (-8 if moving_piece.color is Color.WHITE else 8)
    if not 0 <= capture_index < BOARD_SIZE:
        return False
    capture_square = Square(capture_index)
    return board.piece_at(capture_square) == Piece(moving_piece.color.opposite, PieceType.PAWN)


def _move_castling_rook(squares: list[Piece | None], color: Color, king_target: Square) -> None:
    rank_offset = 0 if color is Color.WHITE else 56
    if int(king_target) == rank_offset + 6:
        rook_from = rank_offset + 7
        rook_to = rank_offset + 5
    else:
        rook_from = rank_offset
        rook_to = rank_offset + 3
    rook = squares[rook_from]
    squares[rook_from] = None
    squares[rook_to] = rook


def _updated_castling_rights(
    board: Board, move: Move, moving_piece: Piece, captured_piece: Piece | None
) -> frozenset[str]:
    rights = set(board.castling_rights)
    if moving_piece.kind is PieceType.KING:
        rights.difference_update({"K", "Q"} if moving_piece.color is Color.WHITE else {"k", "q"})
    elif moving_piece.kind is PieceType.ROOK:
        if int(move.from_square) == 0:
            rights.discard("Q")
        elif int(move.from_square) == 7:
            rights.discard("K")
        elif int(move.from_square) == 56:
            rights.discard("q")
        elif int(move.from_square) == 63:
            rights.discard("k")

    if captured_piece is not None and captured_piece.kind is PieceType.ROOK:
        if int(move.to_square) == 0:
            rights.discard("Q")
        elif int(move.to_square) == 7:
            rights.discard("K")
        elif int(move.to_square) == 56:
            rights.discard("q")
        elif int(move.to_square) == 63:
            rights.discard("k")
    return frozenset(rights)


def _next_en_passant_target(move: Move, moving_piece: Piece) -> Square | None:
    if moving_piece.kind is not PieceType.PAWN:
        return None
    distance = int(move.to_square) - int(move.from_square)
    if abs(distance) != 16:
        return None
    return Square(int(move.from_square) + distance // 2)
