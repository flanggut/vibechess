from __future__ import annotations

from tinychess.cli import main


def test_cli_help_smoke(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main([]) == 0

    captured = capsys.readouterr()
    assert "tinychess" in captured.out
    assert "--version" in captured.out


def test_cli_version_smoke(capsys) -> None:  # type: ignore[no-untyped-def]
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0

    captured = capsys.readouterr()
    assert "tinychess" in captured.out
