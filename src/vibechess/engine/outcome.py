"""Game outcome types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from vibechess.engine.piece import Color


class OutcomeReason(Enum):
    """Reasons a game can end."""

    CHECKMATE = "checkmate"
    STALEMATE = "stalemate"
    FIFTY_MOVE = "fifty_move"
    REPETITION = "repetition"
    INSUFFICIENT_MATERIAL = "insufficient_material"
    MAX_PLIES = "max_plies"


@dataclass(frozen=True, slots=True)
class Outcome:
    """A completed game outcome.

    ``winner`` is ``None`` for draws.
    """

    reason: OutcomeReason
    winner: Color | None = None

    @property
    def is_draw(self) -> bool:
        """Return whether the outcome is a draw."""
        return self.winner is None
