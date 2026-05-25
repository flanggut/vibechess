"""Simple terminal play loop for tinychess."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TextIO

from tinychess.ai.player import RandomPlayer
from tinychess.engine.game import Game
from tinychess.engine.move import Move
from tinychess.ui.render import render_game

PlayerKind = str


class HumanQuit(Exception):
    """Raised when a human player intentionally exits the terminal game."""


@dataclass(frozen=True, slots=True)
class PlayConfig:
    """Configuration for the terminal play loop."""

    white: PlayerKind = "human"
    black: PlayerKind = "human"
    max_plies: int = 512
    seed: int | None = None
    unicode: bool = False
    coordinates: bool = True


def parse_legal_uci_move(text: str, game: Game) -> Move:
    """Parse a UCI move string and require it to be legal in ``game``."""
    notation = text.strip().lower()
    if not notation:
        msg = "empty move; enter a UCI move such as e2e4"
        raise ValueError(msg)
    try:
        move = Move.from_uci(notation)
    except ValueError as exc:
        msg = f"invalid UCI move {text.strip()!r}: {exc}"
        raise ValueError(msg) from exc
    legal = game.legal_moves
    if move not in legal:
        legal_examples = ", ".join(candidate.to_uci() for candidate in legal[:8])
        suffix = f" Examples: {legal_examples}" if legal_examples else ""
        msg = f"illegal move {notation!r} for the current position.{suffix}"
        raise ValueError(msg)
    return move


def play_terminal(
    config: PlayConfig,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> Game:
    """Run a human/random terminal game and return the final game state."""
    if config.max_plies < 0:
        msg = f"max_plies must be non-negative, got {config.max_plies}"
        raise ValueError(msg)
    input_stream = sys.stdin if stdin is None else stdin
    output_stream = sys.stdout if stdout is None else stdout
    random_player = RandomPlayer(seed=config.seed)
    players = {"white": config.white, "black": config.black}
    game = Game.new()
    last_move: Move | None = None
    plies_played = 0

    while game.outcome is None and plies_played < config.max_plies:
        print(
            render_game(
                game,
                last_move=last_move,
                unicode=config.unicode,
                coordinates=config.coordinates,
            ),
            file=output_stream,
        )
        side = game.board.side_to_move.value
        player = players[side]
        legal = game.legal_moves
        if not legal:
            break
        if player == "random":
            move = random_player.select_move(game)
            print(f"{side} random plays {move.to_uci()}", file=output_stream)
        elif player == "human":
            move = _read_human_move(
                game=game,
                input_stream=input_stream,
                output_stream=output_stream,
            )
        else:  # defensive for direct API callers; argparse also constrains values.
            msg = f"unsupported player kind: {player!r}"
            raise ValueError(msg)
        game = game.play(move)
        last_move = move
        plies_played += 1
        print("", file=output_stream)

    print(
        render_game(
            game,
            last_move=last_move,
            unicode=config.unicode,
            coordinates=config.coordinates,
        ),
        file=output_stream,
    )
    if game.outcome is None and plies_played >= config.max_plies:
        print(
            f"Stopped after max plies ({config.max_plies}) without an outcome.",
            file=output_stream,
        )
    return game


def _read_human_move(*, game: Game, input_stream: TextIO, output_stream: TextIO) -> Move:
    side = game.board.side_to_move.value
    while True:
        print(f"{side} move (UCI, e.g. e2e4; 'quit' to exit): ", end="", file=output_stream)
        output_stream.flush()
        line = input_stream.readline()
        if line == "":
            raise EOFError("input ended while waiting for a human move")
        command = line.strip()
        if command.lower() in {"quit", "exit"}:
            raise HumanQuit("human player quit")
        try:
            return parse_legal_uci_move(command, game)
        except ValueError as exc:
            print(f"Invalid move: {exc}", file=output_stream)
