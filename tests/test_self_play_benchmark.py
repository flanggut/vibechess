from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from vibechess.nn.checkpoint import save_checkpoint
from vibechess.nn.model import PolicyValueConfig, PolicyValueNet

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
    assert data["format_version"] == 2
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
    assert data["config"]["profile"] is True
    assert data["config"]["profile_level"] == "detailed"
    profile = data["profile"]
    assert profile["format_version"] == 2
    assert profile["repeat_count"] == 1
    assert profile["bottleneck_summary"]
    assert profile["percent_of_elapsed"]["game_legal_moves"] >= 0.0
    assert profile["stats"]["timers"]["search"]["completed_simulations"] == 2
    assert profile["stats"]["timers"]["model_single"]["calls"] == 2
    assert "mcts.search" in profile["stats"]["zones"]
    repeat = _single_repeat(data)
    assert repeat["sample_count"] == 2
    assert repeat["game_count"] == 1
    assert repeat["ply_count"] == 2
    assert repeat["output_bytes"] == data["output_bytes"]
    assert repeat["profile"]["stats"]["timers"]["game_legal_moves"]["calls"] > 0
    assert "scripts/self_play.py" in repeat["command"]
    assert "schema=vibechess-selfplay-v1" in repeat["stdout"]
    assert not output_root.exists()


def test_self_play_benchmark_central_queue_profile_counters(tmp_path: Path) -> None:
    output_root = tmp_path / "benchmark-central"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/self_play_benchmark.py",
            "--games",
            "2",
            "--max-plies",
            "1",
            "--simulations",
            "1",
            "--batch-size",
            "2",
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
    assert data["batching_mode"] == "central_inference_queue"
    assert data["inference_batch_size"] == 2
    repeat = _single_repeat(data)
    assert repeat["batching_mode"] == "central_inference_queue"
    assert repeat["inference_batch_size"] == 2
    stats = data["profile"]["stats"]
    assert stats["counters"]["inference.predict_legal_batch.calls"] >= 1
    assert stats["counters"]["inference.legal_batch_positions"] == 2
    legal_batch_size = stats["distributions"]["inference.legal_batch_size"]
    assert legal_batch_size["count"] >= 1
    assert legal_batch_size["max"] == 2.0
    derived = data["profile"]["derived"]
    assert derived["predict_legal_batch_calls"] >= 1
    assert derived["predict_legal_batch_positions"] == 2
    assert data["profile"]["stats"]["timers"]["model_legal_batch"]["positions"] == 2
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
    assert (output_dir / "profile.json").is_file()


def test_self_play_benchmark_no_profile_clears_inherited_profile_env(tmp_path: Path) -> None:
    output_root = tmp_path / "benchmark-output"
    env = os.environ.copy()
    env["VIBECHESS_SELF_PLAY_PROFILE"] = "1"

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
            "--no-profile",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    data = json.loads(completed.stdout)
    assert data["config"]["profile"] is False
    assert data["config"]["profile_level"] == "none"
    assert data["profile"] is None
    repeat = _single_repeat(data)
    assert repeat["profile"] is None
    output_dir = Path(repeat["output_directory"])
    assert output_dir.is_dir()
    assert not (output_dir / "profile.json").exists()


def test_self_play_benchmark_markdown_profile_sections(tmp_path: Path) -> None:
    output_root = tmp_path / "benchmark-markdown"

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
            "markdown",
            "--output-root",
            str(output_root),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "## Bottleneck Summary" in completed.stdout
    assert "## MCTS Breakdown" in completed.stdout
    assert "## Inference / MLX Breakdown" in completed.stdout
    assert "## Legal and Transition Breakdown" in completed.stdout
    assert "## Dataset and Serialization Breakdown" in completed.stdout
    assert "## Worker Breakdown" in completed.stdout
    assert "batching_label: central_inference_queue" not in completed.stdout
    assert "## Central Queue Batching" not in completed.stdout


def test_self_play_benchmark_profile_overhead_check_reports_pair(tmp_path: Path) -> None:
    output_root = tmp_path / "overhead"
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
            "--profile-level",
            "detailed",
            "--profile-overhead-check",
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
    overhead = data["profile_overhead"]
    assert overhead["enabled"] is True
    assert isinstance(overhead["overhead_percent"], float)
    assert overhead["counts_match"] is True
    assert overhead["deterministic_games_match"] is True


def test_self_play_benchmark_profile_does_not_change_generated_moves(tmp_path: Path) -> None:
    none_root = tmp_path / "none"
    profiled_root = tmp_path / "profiled"
    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    common_args = [
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
        "--seed",
        "7",
        "--checkpoint",
        str(checkpoint),
        "--format",
        "json",
        "--keep-output",
    ]

    none = subprocess.run(
        [*common_args, "--no-profile", "--output-root", str(none_root)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    profiled = subprocess.run(
        [*common_args, "--profile-level", "detailed", "--output-root", str(profiled_root)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert none.returncode == 0, none.stderr
    assert profiled.returncode == 0, profiled.stderr
    none_data = json.loads(none.stdout)
    profiled_data = json.loads(profiled.stdout)
    none_dir = Path(_single_repeat(none_data)["output_directory"])
    profiled_dir = Path(_single_repeat(profiled_data)["output_directory"])
    assert (none_dir / "games.jsonl").read_text() == (profiled_dir / "games.jsonl").read_text()
    assert not (none_dir / "profile.json").exists()
    assert (profiled_dir / "profile.json").is_file()


def _write_tiny_checkpoint(path: Path) -> Path:
    config = PolicyValueConfig(
        residual_channels=4,
        residual_blocks=0,
        policy_channels=1,
        value_channels=1,
        value_hidden_dim=4,
    )
    save_checkpoint(PolicyValueNet(config), path)
    return path


def _single_repeat(data: dict[str, Any]) -> dict[str, Any]:
    repeats = data["repeat_results"]
    assert isinstance(repeats, list)
    assert len(repeats) == 1
    repeat = repeats[0]
    assert isinstance(repeat, dict)
    return repeat
