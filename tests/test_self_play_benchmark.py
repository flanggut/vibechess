from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_self_play_benchmark_json_smoke_removes_outputs(tmp_path: Path) -> None:
    output_root = tmp_path / "benchmark-output"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/self_play_benchmark.py",
            "--games",
            "1",
            "--max-plies",
            "2",
            "--simulations",
            "1",
            "--workers",
            "1",
            "--repeat",
            "1",
            "--format",
            "json",
            "--output-root",
            str(output_root),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    data = json.loads(completed.stdout)
    assert data["benchmark"] == "self_play_generation"
    assert data["format_version"] == 1
    assert data["games"] == 1
    assert data["max_plies"] == 2
    assert data["simulations"] == 1
    assert data["temperature"] == 1.0
    assert data["workers"] == 1
    assert data["effective_workers"] == 1
    assert data["sample_count"] == 2
    assert data["sample_count_min"] == 2
    assert data["sample_count_max"] == 2
    assert data["game_count"] == 1
    assert data["game_count_min"] == 1
    assert data["game_count_max"] == 1
    assert data["ply_count"] == 2
    assert data["ply_count_min"] == 2
    assert data["ply_count_max"] == 2
    assert data["elapsed_seconds"] > 0.0
    assert data["samples_per_second"] > 0.0
    assert data["games_per_second"] > 0.0
    assert data["output_bytes"] > 0
    assert data["model_config"] == {"blocks": 0, "channels": 4}
    assert data["chunks"] == [{"games": 1, "seed": 1, "start_game": 0}]
    repeat = _single_repeat(data)
    assert repeat["sample_count"] == 2
    assert repeat["game_count"] == 1
    assert repeat["ply_count"] == 2
    assert repeat["output_bytes"] == data["output_bytes"]
    assert "scripts/self_play.py" in repeat["command"]
    assert "schema=tinychess-selfplay-v1" in repeat["stdout"]
    assert not output_root.exists()


def test_self_play_benchmark_keep_output_preserves_dataset(tmp_path: Path) -> None:
    output_root = tmp_path / "benchmark-output"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/self_play_benchmark.py",
            "--games",
            "1",
            "--max-plies",
            "1",
            "--simulations",
            "1",
            "--workers",
            "1",
            "--repeat",
            "1",
            "--format",
            "json",
            "--output-root",
            str(output_root),
            "--keep-output",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    data = json.loads(completed.stdout)
    output_dir = Path(_single_repeat(data)["output_directory"])
    assert output_dir.is_dir()
    assert output_dir == output_root / "repeat-001"
    assert (output_dir / "metadata.json").is_file()
    assert (output_dir / "samples.npz").is_file()
    assert (output_dir / "games.jsonl").is_file()


def _single_repeat(data: dict[str, Any]) -> dict[str, Any]:
    repeats = data["repeat_results"]
    assert isinstance(repeats, list)
    assert len(repeats) == 1
    repeat = repeats[0]
    assert isinstance(repeat, dict)
    return repeat
