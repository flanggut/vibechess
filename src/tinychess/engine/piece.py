"""Piece and color types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Color(Enum):
    """Chess side to move or piece color."""

    WHITE = "white"
    BLACK = "black"

    @property
    def opposite(self) -> Color:
        """Return the opposite color."""
        return Color.BLACK if self is Color.WHITE else Color.WHITE


class PieceType(Enum):
    """Chess piece kinds."""

    PAWN = "pawn"
    KNIGHT = "knight"
    BISHOP = "bishop"
    ROOK = "rook"
    QUEEN = "queen"
    KING = "king"


_WHITE_SYMBOLS: dict[PieceType, str] = {
    PieceType.PAWN: "P",
    PieceType.KNIGHT: "N",
    PieceType.BISHOP: "B",
    PieceType.ROOK: "R",
    PieceType.QUEEN: "Q",
    PieceType.KING: "K",
}

_SYMBOL_TO_TYPE: dict[str, PieceType] = {symbol: kind for kind, symbol in _WHITE_SYMBOLS.items()}


@dataclass(frozen=True, slots=True)
class Piece:
    """A chess piece with color and type."""

    color: Color
    kind: PieceType

    @property
    def symbol(self) -> str:
        """Return the FEN-style single-character piece symbol."""
        symbol = _WHITE_SYMBOLS[self.kind]
        return symbol if self.color is Color.WHITE else symbol.lower()

    @classmethod
    def from_symbol(cls, symbol: str) -> Piece:
        """Create a piece from a FEN-style single-character piece symbol."""
        if len(symbol) != 1 or symbol.upper() not in _SYMBOL_TO_TYPE:
            msg = f"invalid piece symbol: {symbol!r}"
            raise ValueError(msg)
        color = Color.WHITE if symbol.isupper() else Color.BLACK
        return cls(color=color, kind=_SYMBOL_TO_TYPE[symbol.upper()])
