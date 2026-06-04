from __future__ import annotations

import json
from io import StringIO
from typing import Any, cast

from tinychess.cli import main


def test_cli_help_smoke(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main([]) == 0

    captured = capsys.readouterr()
    assert "tinychess" in captured.out
    assert "--version" in captured.out
    assert "gui-server" in captured.out


def test_cli_version_smoke(capsys) -> None:  # type: ignore[no-untyped-def]
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0

    captured = capsys.readouterr()
    assert "tinychess" in captured.out


def test_gui_server_help_lists_protocol_options(capsys) -> None:  # type: ignore[no-untyped-def]
    try:
        main(["gui-server", "--help"])
    except SystemExit as exc:
        assert exc.code == 0

    captured = capsys.readouterr()
    assert "GUI backend" in captured.out
    assert "--seed" in captured.out
    assert "--ai-kind" in captured.out
    assert "--ai-simulations" in captured.out
    assert "--ai-checkpoint" in captured.out


def test_gui_server_invalid_default_config_returns_usage_error() -> None:
    stderr = StringIO()

    code = main(
        ["gui-server", "--ai-kind", "mcts", "--ai-simulations", "0"],
        stdin=StringIO(),
        stderr=stderr,
    )

    assert code == 2
    assert "tinychess gui-server" in stderr.getvalue()
    assert "simulations" in stderr.getvalue()


def test_gui_server_cli_responds_to_hello_state_and_quit() -> None:
    stdin = StringIO(
        "\n".join(
            (
                json.dumps({"id": 1, "cmd": "hello"}),
                json.dumps({"id": 2, "cmd": "state"}),
                json.dumps({"id": 3, "cmd": "quit"}),
                "",
            )
        )
    )
    stdout = StringIO()

    code = main(
        [
            "gui-server",
            "--seed",
            "7",
            "--ai-kind",
            "mcts",
            "--ai-simulations",
            "3",
            "--ai-node-budget",
            "5",
        ],
        stdin=stdin,
        stdout=stdout,
    )

    assert code == 0
    responses = [cast(dict[str, Any], json.loads(line)) for line in stdout.getvalue().splitlines()]
    assert [response["id"] for response in responses] == [1, 2, 3]
    assert all(response["ok"] is True for response in responses)
    assert responses[0]["protocol"] == "tinychess-gui-v1"
    assert responses[1]["state"]["moves"] == []
    assert responses[2]["state"]["sideToMove"] == "white"
