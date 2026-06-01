from __future__ import annotations

from io import StringIO

import pytest

from tinychess.cli import main
from tinychess.engine.game import Game
from tinychess.ui.render import render_game
from tinychess.ui.terminal import PlayConfig, parse_legal_uci_move, play_terminal


def test_render_game_shows_board_and_status() -> None:
    text = render_game(Game.new())

    assert "8 r n b q k b n r" in text
    assert "Side to move: white" in text
    assert "Castling: KQkq" in text
    assert "Status: ongoing" in text


def test_parse_legal_uci_move_accepts_and_rejects() -> None:
    game = Game.new()

    assert parse_legal_uci_move("e2e4", game).to_uci() == "e2e4"
    assert parse_legal_uci_move(" E2E4 ", game).to_uci() == "e2e4"

    with pytest.raises(ValueError, match="invalid UCI move"):
        parse_legal_uci_move("not-a-move", game)
    with pytest.raises(ValueError, match="illegal move"):
        parse_legal_uci_move("e2e5", game)


def test_play_terminal_random_vs_random_respects_max_plies() -> None:
    output = StringIO()

    game = play_terminal(
        PlayConfig(white="random", black="random", max_plies=2, seed=7),
        stdout=output,
    )

    assert len(game.moves) == 2
    text = output.getvalue()
    assert "white random plays" in text
    assert "black random plays" in text
    assert "Stopped after max plies (2)" in text


def test_cli_play_random_vs_random() -> None:
    output = StringIO()

    code = main(
        ["play", "--white", "random", "--black", "random", "--max-plies", "1", "--seed", "1"],
        stdout=output,
    )

    assert code == 0
    assert "random plays" in output.getvalue()


def test_cli_play_mcts_supports_static_leaf_rollout_plies() -> None:
    output = StringIO()

    code = main(
        [
            "play",
            "--white",
            "mcts",
            "--black",
            "random",
            "--max-plies",
            "1",
            "--mcts-simulations",
            "1",
            "--mcts-rollout-plies",
            "0",
            "--seed",
            "1",
        ],
        stdout=output,
    )

    assert code == 0
    assert "white mcts plays" in output.getvalue()


def test_cli_play_unicode_without_coordinates() -> None:
    output = StringIO()

    code = main(
        [
            "play",
            "--white",
            "random",
            "--black",
            "random",
            "--max-plies",
            "0",
            "--unicode",
            "--no-coordinates",
        ],
        stdout=output,
    )

    text = output.getvalue()
    assert code == 0
    assert "♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜" in text
    assert "8 ♜" not in text
    assert "  a b c d e f g h" not in text


def test_cli_play_rejects_negative_max_plies() -> None:
    error = StringIO()

    code = main(["play", "--max-plies", "-1"], stderr=error)

    assert code == 2
    assert "max_plies must be non-negative" in error.getvalue()


def test_cli_human_quit_exits_cleanly() -> None:
    output = StringIO()
    error = StringIO()

    code = main(["play"], stdin=StringIO("quit\n"), stdout=output, stderr=error)

    assert code == 0
    assert "white move" in output.getvalue()
    assert error.getvalue() == ""


def test_cli_human_eof_returns_error() -> None:
    error = StringIO()

    code = main(["play"], stdin=StringIO(""), stdout=StringIO(), stderr=error)

    assert code == 1
    assert "input ended while waiting for a human move" in error.getvalue()


def test_cli_human_vs_human_uses_uci_input() -> None:
    output = StringIO()
    code = main(
        ["play", "--max-plies", "2"],
        stdin=StringIO("bad\ne2e4\ne7e5\n"),
        stdout=output,
    )

    assert code == 0
    text = output.getvalue()
    assert "Invalid move:" in text
    assert "Last move: e7e5" in text
