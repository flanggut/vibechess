from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_engine_move_application_does_not_import_nn_package() -> None:
    code = """
import sys
from tinychess.engine import Board, Move
Board.starting_position().apply_move(Move.from_uci("e2e4"))
if "tinychess.nn" in sys.modules:
    raise SystemExit("tinychess.nn was imported by engine move application")
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
