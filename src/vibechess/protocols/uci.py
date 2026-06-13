"""Bounded Universal Chess Interface (UCI) protocol support."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TextIO

from vibechess import __version__
from vibechess.ai.player import RandomPlayer
from vibechess.engine import STARTING_FEN, Game, Move

NO_MOVE = "0000"


@dataclass(frozen=True, slots=True)
class UciConfig:
    """Configuration for the bounded UCI loop."""

    seed: int | None = None
    engine_name: str = f"vibechess {__version__}"
    author: str = "vibechess"


class UciSession:
    """Small stateful UCI command handler.

    The initial WP08 implementation is intentionally synchronous: ``go`` immediately
    selects one legal move from the current position and writes ``bestmove``. Search
    budgets are parsed for bounded UCI compatibility but do not affect random move
    selection yet.
    """

    def __init__(self, config: UciConfig | None = None) -> None:
        self.config = UciConfig() if config is None else config
        self.game = Game.new()
        self._player = RandomPlayer(seed=self.config.seed)
        self._quit = False

    @property
    def should_quit(self) -> bool:
        """Return whether the session has received ``quit``."""
        return self._quit

    def handle_line(self, line: str, output: TextIO) -> None:
        """Handle one UCI input line and write responses to ``output``."""
        stripped = line.strip()
        if not stripped:
            return
        parts = stripped.split()
        command = parts[0].lower()
        args = parts[1:]

        if command == "uci":
            self._write_uci(output)
        elif command == "isready":
            print("readyok", file=output)
        elif command == "ucinewgame":
            self.game = Game.new()
        elif command == "position":
            self._handle_position(args, output)
        elif command == "go":
            self._handle_go(args, output)
        elif command == "stop":
            # Search is synchronous and completes during ``go``; there is nothing to stop.
            print("info string no search in progress", file=output)
        elif command == "quit":
            self._quit = True
        else:
            print(f"info string unsupported command: {command}", file=output)
        output.flush()

    def _write_uci(self, output: TextIO) -> None:
        print(f"id name {self.config.engine_name}", file=output)
        print(f"id author {self.config.author}", file=output)
        print("uciok", file=output)

    def _handle_position(self, args: list[str], output: TextIO) -> None:
        try:
            self.game = parse_position_command(args)
        except ValueError as exc:
            print(f"info string error: {exc}", file=output)

    def _handle_go(self, args: list[str], output: TextIO) -> None:
        try:
            parse_go_command(args)
        except ValueError as exc:
            print(f"info string error: {exc}", file=output)

        legal = self.game.legal_moves
        if self.game.outcome is not None or not legal:
            print(f"bestmove {NO_MOVE}", file=output)
            return
        move = self._player.select_move(self.game)
        print(f"bestmove {move.to_uci()}", file=output)


def run_uci_loop(
    config: UciConfig | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> UciSession:
    """Run a bounded UCI command loop and return the final session state."""
    input_stream = sys.stdin if stdin is None else stdin
    output_stream = sys.stdout if stdout is None else stdout
    session = UciSession(config)
    for line in input_stream:
        session.handle_line(line, output_stream)
        if session.should_quit:
            break
    return session


def parse_position_command(args: list[str]) -> Game:
    """Parse UCI ``position`` arguments into a ``Game`` with applied legal moves."""
    if not args:
        msg = "position requires 'startpos' or 'fen'"
        raise ValueError(msg)

    move_tokens: list[str]
    if args[0] == "startpos":
        game = Game.from_fen(STARTING_FEN)
        move_tokens = _tokens_after_optional_moves(args[1:])
    elif args[0] == "fen":
        if "moves" in args[1:]:
            moves_index = args.index("moves")
            fen_fields = args[1:moves_index]
            move_tokens = args[moves_index + 1 :]
        else:
            fen_fields = args[1:]
            move_tokens = []
        if len(fen_fields) != 6:
            msg = f"position fen requires 6 FEN fields, got {len(fen_fields)}"
            raise ValueError(msg)
        game = Game.from_fen(" ".join(fen_fields))
    else:
        msg = f"unsupported position form: {args[0]!r}"
        raise ValueError(msg)

    for token in move_tokens:
        try:
            move = Move.from_uci(token.lower())
        except ValueError as exc:
            msg = f"invalid move in position command {token!r}: {exc}"
            raise ValueError(msg) from exc
        if move not in game.legal_moves:
            msg = f"illegal move in position command: {token}"
            raise ValueError(msg)
        game = game.play(move)
    return game


def parse_go_command(args: list[str]) -> None:
    """Validate bounded UCI ``go`` arguments.

    Supported budget tokens are accepted for compatibility but are not used by the
    current random move selector: ``depth``, ``nodes``, ``movetime``, ``wtime``,
    ``btime``, ``winc``, ``binc``, and ``mate``. ``infinite`` is accepted as a no-op.
    """
    value_options = {"depth", "nodes", "movetime", "wtime", "btime", "winc", "binc", "mate"}
    flag_options = {"infinite"}
    index = 0
    while index < len(args):
        token = args[index]
        if token in flag_options:
            index += 1
            continue
        if token not in value_options:
            msg = f"unsupported go option: {token}"
            raise ValueError(msg)
        if index + 1 >= len(args):
            msg = f"go option {token!r} requires a value"
            raise ValueError(msg)
        value_text = args[index + 1]
        try:
            value = int(value_text)
        except ValueError as exc:
            msg = f"go option {token!r} requires an integer value, got {value_text!r}"
            raise ValueError(msg) from exc
        if value < 0:
            msg = f"go option {token!r} must be non-negative, got {value}"
            raise ValueError(msg)
        index += 2


def _tokens_after_optional_moves(args: list[str]) -> list[str]:
    if not args:
        return []
    if args[0] != "moves":
        msg = f"unexpected token after startpos: {args[0]!r}"
        raise ValueError(msg)
    return args[1:]
