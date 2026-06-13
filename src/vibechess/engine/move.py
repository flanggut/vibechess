"""Move representation and UCI long algebraic conversion."""

from __future__ import annotations

from dataclasses import dataclass

from vibechess.engine.piece import PieceType
from vibechess.engine.square import Square, parse_square, square_name

_PROMOTION_TO_CHAR: dict[PieceType, str] = {
    PieceType.KNIGHT: "n",
    PieceType.BISHOP: "b",
    PieceType.ROOK: "r",
    PieceType.QUEEN: "q",
}
_CHAR_TO_PROMOTION: dict[str, PieceType] = {value: key for key, value in _PROMOTION_TO_CHAR.items()}


@dataclass(frozen=True, slots=True)
class Move:
    """A chess move from one square to another, with optional promotion.

    Legality is intentionally not checked here. Legal move generation belongs to WP03.
    """

    from_square: Square
    to_square: Square
    promotion: PieceType | None = None

    def to_uci(self) -> str:
        """Return UCI long algebraic notation, such as ``e2e4`` or ``e7e8q``."""
        notation = f"{square_name(self.from_square)}{square_name(self.to_square)}"
        if self.promotion is not None:
            if self.promotion not in _PROMOTION_TO_CHAR:
                msg = "promotion piece must be queen, rook, bishop, or knight"
                raise ValueError(msg)
            notation += _PROMOTION_TO_CHAR[self.promotion]
        return notation

    @classmethod
    def from_uci(cls, notation: str) -> Move:
        """Parse UCI long algebraic notation into a move."""
        if len(notation) not in {4, 5}:
            msg = f"UCI move must have length 4 or 5, got {notation!r}"
            raise ValueError(msg)
        promotion = None
        if len(notation) == 5:
            promotion_char = notation[4]
            if promotion_char not in _CHAR_TO_PROMOTION:
                msg = f"invalid UCI promotion piece: {promotion_char!r}"
                raise ValueError(msg)
            promotion = _CHAR_TO_PROMOTION[promotion_char]
        return cls(
            from_square=parse_square(notation[:2]),
            to_square=parse_square(notation[2:4]),
            promotion=promotion,
        )

    def __str__(self) -> str:
        """Return UCI notation for display/debugging."""
        return self.to_uci()
