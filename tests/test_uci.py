from __future__ import annotations

from io import StringIO

import pytest

from tinychess.cli import main
from tinychess.engine import Game, Move
from tinychess.protocols.uci import UciConfig, UciSession, parse_position_command, run_uci_loop


def _bestmove(text: str) -> str:
    lines = [line for line in text.splitlines() if line.startswith("bestmove ")]
    assert lines
    return lines[-1].split()[1]


def test_uci_handshake_and_isready() -> None:
    output = StringIO()

    run_uci_loop(stdin=StringIO("uci\nisready\nquit\n"), stdout=output)

    text = output.getvalue()
    assert "id name tinychess" in text
    assert "id author" in text
    assert "uciok" in text
    assert "readyok" in text


def test_position_startpos_applies_optional_moves() -> None:
    game = parse_position_command(["startpos", "moves", "e2e4", "e7e5", "g1f3"])

    assert [move.to_uci() for move in game.moves] == ["e2e4", "e7e5", "g1f3"]
    assert game.board.side_to_move.value == "black"


def test_position_fen_applies_optional_moves_and_counters() -> None:
    args = [
        "fen",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR",
        "w",
        "KQkq",
        "-",
        "0",
        "2",
        "moves",
        "g1f3",
    ]

    game = parse_position_command(args)

    assert (
        game.to_fen()
        == "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
    )


def test_go_returns_legal_bestmove() -> None:
    output = StringIO()
    session = UciSession(UciConfig(seed=7))
    session.handle_line("go depth 1", output)

    move = Move.from_uci(_bestmove(output.getvalue()))
    assert move in Game.new().legal_moves


def test_go_returns_legal_bestmove_after_position_moves() -> None:
    output = StringIO()
    commands = "position startpos moves e2e4\ngo depth 1\nquit\n"
    run_uci_loop(UciConfig(seed=7), stdin=StringIO(commands), stdout=output)

    move = Move.from_uci(_bestmove(output.getvalue()))
    assert move in parse_position_command(["startpos", "moves", "e2e4"]).legal_moves


@pytest.mark.parametrize(
    "command,error",
    [
        ("go searchmoves e2e4", "unsupported go option: searchmoves"),
        ("go depth", "go option 'depth' requires a value"),
        ("go depth many", "go option 'depth' requires an integer value, got 'many'"),
        ("go depth -1", "go option 'depth' must be non-negative, got -1"),
    ],
)
def test_invalid_go_reports_error_but_still_returns_legal_move(command: str, error: str) -> None:
    output = StringIO()
    session = UciSession(UciConfig(seed=7))

    session.handle_line(command, output)

    text = output.getvalue()
    assert f"info string error: {error}" in text
    move = Move.from_uci(_bestmove(text))
    assert move in Game.new().legal_moves


def test_terminal_position_returns_no_move() -> None:
    output = StringIO()
    run_uci_loop(
        stdin=StringIO("position fen 7k/5Q2/7K/8/8/8/8/8 b - - 0 1\ngo\nquit\n"),
        stdout=output,
    )

    assert "bestmove 0000" in output.getvalue()


def test_terminal_position_with_invalid_go_still_returns_no_move() -> None:
    output = StringIO()
    run_uci_loop(
        stdin=StringIO("position fen 7k/5Q2/7K/8/8/8/8/8 b - - 0 1\ngo depth nope\nquit\n"),
        stdout=output,
    )

    text = output.getvalue()
    assert "info string error: go option 'depth' requires an integer value, got 'nope'" in text
    assert "bestmove 0000" in text


def test_ucinewgame_resets_position() -> None:
    output = StringIO()
    session = UciSession()

    session.handle_line("position startpos moves e2e4", output)
    session.handle_line("ucinewgame", output)

    assert session.game.to_fen() == Game.new().to_fen()


def test_stop_reports_no_search_in_progress() -> None:
    output = StringIO()

    UciSession().handle_line("stop", output)

    assert output.getvalue() == "info string no search in progress\n"


def test_quit_stops_command_loop() -> None:
    output = StringIO()

    session = run_uci_loop(stdin=StringIO("quit\nuci\n"), stdout=output)

    assert session.should_quit
    assert "uciok" not in output.getvalue()


def test_invalid_position_and_unsupported_commands_are_reported() -> None:
    output = StringIO()
    commands = "position startpos moves e2e5\nsetoption name Hash value 16\nquit\n"
    run_uci_loop(stdin=StringIO(commands), stdout=output)

    text = output.getvalue()
    assert "info string error: illegal move in position command: e2e5" in text
    assert "info string unsupported command: setoption" in text


def test_cli_uci_invocation() -> None:
    output = StringIO()

    code = main(["uci", "--seed", "1"], stdin=StringIO("uci\ngo movetime 1\nquit\n"), stdout=output)

    assert code == 0
    text = output.getvalue()
    assert "uciok" in text
    assert _bestmove(text) in {move.to_uci() for move in Game.new().legal_moves}
