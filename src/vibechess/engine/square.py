"""Square helpers for the vibechess 0..63 board convention.

Squares are indexed from White's perspective with ``a1 == 0`` and ``h8 == 63``.
Files increase left-to-right from ``a`` to ``h``. Ranks increase upward from
White's home rank: ``a2 == 8``, ``a8 == 56``, and ``h8 == 63``.
"""

from __future__ import annotations

from typing import NewType

Square = NewType("Square", int)

FILES = "abcdefgh"
RANKS = "12345678"
BOARD_SIZE = 64


def make_square(file_index: int, rank_index: int) -> Square:
    """Return the square for zero-based file and rank indexes."""
    if not 0 <= file_index < 8:
        msg = f"file index must be in 0..7, got {file_index}"
        raise ValueError(msg)
    if not 0 <= rank_index < 8:
        msg = f"rank index must be in 0..7, got {rank_index}"
        raise ValueError(msg)
    return Square(rank_index * 8 + file_index)


def file_index(square: Square) -> int:
    """Return a square's zero-based file index."""
    validate_square(square)
    return int(square) % 8


def rank_index(square: Square) -> int:
    """Return a square's zero-based rank index."""
    validate_square(square)
    return int(square) // 8


def square_name(square: Square) -> str:
    """Return algebraic coordinate notation for a square, such as ``e4``."""
    return f"{FILES[file_index(square)]}{RANKS[rank_index(square)]}"


def parse_square(name: str) -> Square:
    """Parse algebraic coordinate notation into a square index."""
    if len(name) != 2:
        msg = f"square must have length 2, got {name!r}"
        raise ValueError(msg)
    file_char, rank_char = name[0], name[1]
    if file_char not in FILES or rank_char not in RANKS:
        msg = f"invalid square name: {name!r}"
        raise ValueError(msg)
    return make_square(FILES.index(file_char), RANKS.index(rank_char))


def validate_square(square: Square) -> Square:
    """Validate and return a square index.

    ``Square`` is a static typing aid over ``int``. Runtime callers can still pass
    invalid values, so public APIs should validate before indexing board arrays.
    """
    if not 0 <= int(square) < BOARD_SIZE:
        msg = f"square must be in 0..63, got {int(square)}"
        raise ValueError(msg)
    return square
