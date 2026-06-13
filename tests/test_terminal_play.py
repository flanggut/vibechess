from __future__ import annotations

import json
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest

from vibechess.cli import main
from vibechess.engine.game import Game
from vibechess.engine.move import Move
from vibechess.ui import terminal
from vibechess.ui.render import render_game
from vibechess.ui.terminal import PlayConfig, parse_legal_uci_move, play_terminal


class FirstLegalPlayer:
    def select_move(self, game: Game) -> Move:
        return game.legal_moves[0]


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


def test_play_config_defaults_to_static_leaf_mcts() -> None:
    config = PlayConfig()

    assert config.mcts_rollout_plies == 0
    assert config.ai_checkpoint is None
    assert config.ai_simulations == 25


def test_play_terminal_ai_uses_configured_player_and_prints_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = StringIO()
    seen_checkpoint: Path | None = None

    def fake_create_ai_players(
        config: PlayConfig,
        players: dict[str, terminal.PlayerKind],
    ) -> dict[str, FirstLegalPlayer]:
        nonlocal seen_checkpoint
        seen_checkpoint = config.ai_checkpoint
        assert players == {"white": "random", "black": "ai"}
        return {"black": FirstLegalPlayer()}

    monkeypatch.setattr(terminal, "_create_ai_players", fake_create_ai_players)

    game = play_terminal(
        PlayConfig(
            white="random",
            black="ai",
            max_plies=2,
            seed=7,
            ai_checkpoint=Path("checkpoint-dir"),
        ),
        stdout=output,
    )

    assert seen_checkpoint == Path("checkpoint-dir")
    assert len(game.moves) == 2
    text = output.getvalue()
    assert "white random plays" in text
    assert "black ai plays" in text


def test_cli_play_ai_accepts_neural_options(monkeypatch: pytest.MonkeyPatch) -> None:
    output = StringIO()
    seen_config: PlayConfig | None = None

    def fake_create_ai_players(
        config: PlayConfig,
        players: dict[str, terminal.PlayerKind],
    ) -> dict[str, FirstLegalPlayer]:
        nonlocal seen_config
        seen_config = config
        assert players == {"white": "ai", "black": "random"}
        return {"white": FirstLegalPlayer()}

    monkeypatch.setattr(terminal, "_create_ai_players", fake_create_ai_players)

    code = main(
        [
            "play",
            "--white",
            "ai",
            "--black",
            "random",
            "--max-plies",
            "1",
            "--seed",
            "1",
            "--ai-checkpoint",
            "checkpoint-dir",
            "--ai-simulations",
            "3",
            "--ai-node-budget",
            "7",
            "--ai-time-limit-seconds",
            "0.5",
            "--ai-temperature",
            "0.25",
            "--ai-puct-exploration",
            "1.25",
        ],
        stdout=output,
    )

    assert code == 0
    assert seen_config is not None
    assert str(seen_config.ai_checkpoint) == "checkpoint-dir"
    assert seen_config.ai_simulations == 3
    assert seen_config.ai_node_budget == 7
    assert seen_config.ai_time_limit_seconds == 0.5
    assert seen_config.ai_temperature == 0.25
    assert seen_config.ai_puct_exploration == 1.25
    assert "white ai plays" in output.getvalue()


def test_cli_play_rejects_removed_ai_parallel_batch_option() -> None:
    removed_option = "--ai-" + "leaf" + "-parallelism"
    with pytest.raises(SystemExit):
        main(["play", removed_option, "2"])


def test_cli_import_does_not_import_neural_modules() -> None:
    modules = [
        "vibechess.ai.neural_mcts",
        "vibechess.nn.checkpoint",
        "vibechess.nn.model",
        "mlx.core",
    ]
    code = (
        "import json\n"
        "import sys\n"
        "import vibechess.cli\n"
        f"modules = {modules!r}\n"
        "print(json.dumps({name: name in sys.modules for name in modules}, sort_keys=True))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {name: False for name in modules}


def test_cli_play_ai_requires_checkpoint() -> None:
    error = StringIO()

    code = main(["play", "--black", "ai", "--max-plies", "0"], stderr=error)

    assert code == 2
    assert "ai player requires --ai-checkpoint" in error.getvalue()


def test_cli_play_ai_invalid_checkpoint_path_returns_usage_error(tmp_path: Path) -> None:
    error = StringIO()

    code = main(
        ["play", "--black", "ai", "--ai-checkpoint", str(tmp_path / "missing")],
        stdout=StringIO(),
        stderr=error,
    )

    text = error.getvalue()
    assert code == 2
    assert "failed to load ai checkpoint" in text
    assert "Traceback" not in text


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
