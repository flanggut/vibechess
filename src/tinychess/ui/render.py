"""Terminal board and game-state rendering helpers."""

from __future__ import annotations

from tinychess.engine.game import Game
from tinychess.engine.move import Move
from tinychess.engine.square import square_name


def render_game(
    game: Game,
    *,
    last_move: Move | None = None,
    unicode: bool = False,
    coordinates: bool = True,
) -> str:
    """Render a game position with useful terminal status lines."""
    board = game.board
    lines = [board.render(unicode=unicode, coordinates=coordinates)]
    lines.append(f"Side to move: {board.side_to_move.value}")
    castling = "".join(right for right in "KQkq" if right in board.castling_rights) or "-"
    ep_target = "-" if board.en_passant_target is None else square_name(board.en_passant_target)
    lines.append(f"Castling: {castling}  En passant: {ep_target}")
    lines.append(f"Halfmove: {game.halfmove_clock}  Fullmove: {game.fullmove_number}")
    lines.append(f"Last move: {last_move.to_uci() if last_move is not None else '-'}")
    outcome = game.outcome
    if outcome is None:
        lines.append("Status: ongoing")
    elif outcome.winner is None:
        lines.append(f"Status: draw by {outcome.reason.value}")
    else:
        lines.append(f"Status: {outcome.winner.value} wins by {outcome.reason.value}")
    return "\n".join(lines)
