from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tinychess.nn.checkpoint import DEFAULT_WEIGHTS_FILENAME, load_checkpoint_metadata
from tinychess.nn.pgn_dataset import (
    DEFAULT_MANIFEST_FILENAME,
    PgnIngestConfig,
    PgnIngestProgress,
    ingest_pgn_dataset,
)
from tinychess.nn.self_play import load_self_play_dataset
from tinychess.nn.train import TrainingConfig, train_from_directory

PGN_TEXT = """[Event "Tiny"]
[Result "1-0"]

1. e4 e5 2. Nf3 1-0

[Event "Fen"]
[SetUp "1"]
[FEN "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"]
[Result "*"]

1. a8=Q+ *

[Event "Annotated"]
[Result "0-1"]

1. d4! {good} d5?! (1... Nf6) 2. c4 0-1

[Event "Unfinished"]
[Result "*"]

1. e4 e5 *
"""


def test_ingest_pgn_dataset_writes_loadable_shards_and_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "games.pgn"
    output_dir = tmp_path / "dataset"
    input_path.write_text(PGN_TEXT)

    result = ingest_pgn_dataset(
        PgnIngestConfig(input_path=input_path, output_dir=output_dir, shard_samples=3)
    )

    manifest = json.loads((output_dir / DEFAULT_MANIFEST_FILENAME).read_text())
    assert result.games_read == 4
    assert result.games_written == 2
    assert result.games_skipped == 2
    assert result.samples == 6
    assert manifest["shard_count"] == 2
    assert "git_commit" in manifest

    first = load_self_play_dataset(output_dir / "shard-00000")
    second = load_self_play_dataset(output_dir / "shard-00001")
    assert first.metadata.generation_settings["label_source"] == "pgn"
    assert first.metadata.sample_count == 3
    assert second.metadata.sample_count == 3
    assert np.allclose(first.mcts_policies.sum(axis=1), 1.0)
    assert first.outcomes.tolist() == [1.0, -1.0, 1.0]
    assert second.outcomes.tolist() == [-1.0, 1.0, -1.0]


def test_ingest_pgn_dataset_skips_empty_games_without_polluting_shards(tmp_path: Path) -> None:
    input_path = tmp_path / "games.pgn"
    output_dir = tmp_path / "dataset"
    input_path.write_text(
        """[Event "Empty"]
[Result "1-0"]

1-0

[Event "Tiny"]
[Result "1-0"]

1. e4 e5 1-0
"""
    )

    result = ingest_pgn_dataset(PgnIngestConfig(input_path=input_path, output_dir=output_dir))

    dataset = load_self_play_dataset(output_dir / "shard-00000")
    assert result.games_read == 2
    assert result.games_written == 1
    assert result.games_skipped == 1
    assert dataset.metadata.game_count == 1
    assert [record.game_index for record in dataset.games] == [0]
    assert len(dataset.games) == 1
    assert dataset.games[0].moves_uci == ["e2e4", "e7e5"]


def test_ingest_pgn_dataset_reports_progress(tmp_path: Path) -> None:
    input_path = tmp_path / "games.pgn"
    output_dir = tmp_path / "dataset"
    input_path.write_text(PGN_TEXT)
    updates: list[PgnIngestProgress] = []

    result = ingest_pgn_dataset(
        PgnIngestConfig(input_path=input_path, output_dir=output_dir, shard_samples=3),
        progress=updates.append,
        progress_every_games=1,
    )

    assert updates
    assert updates[-1].games_read == result.games_read
    assert updates[-1].games_written == result.games_written
    assert updates[-1].games_skipped == result.games_skipped
    assert updates[-1].samples == result.samples
    assert updates[-1].shards == result.shards


def test_ingest_pgn_dataset_progress_interval_emits_periodic_and_final(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "games.pgn"
    output_dir = tmp_path / "dataset"
    input_path.write_text(
        """[Event "One"]
[Result "1-0"]

1. e4 e5 1-0

[Event "Two"]
[Result "1-0"]

1. d4 d5 1-0

[Event "Three"]
[Result "1-0"]

1. c4 c5 1-0
"""
    )
    updates: list[PgnIngestProgress] = []

    result = ingest_pgn_dataset(
        PgnIngestConfig(input_path=input_path, output_dir=output_dir),
        progress=updates.append,
        progress_every_games=2,
    )

    assert result.games_written == 3
    assert [update.games_written for update in updates] == [2, 3]
    assert updates[-1].samples == result.samples


def test_pgn_ingest_config_rejects_fen_ingestion() -> None:
    try:
        PgnIngestConfig(input_path=Path("games.pgn"), output_dir=Path("dataset"), skip_fen=False)
    except ValueError as exc:
        assert "FEN/SetUp" in str(exc)
    else:  # pragma: no cover - defensive assertion path
        raise AssertionError("expected FEN ingestion to be rejected")


def test_pgn_ingest_script_creates_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "games.pgn"
    output_dir = tmp_path / "script-dataset"
    input_path.write_text(PGN_TEXT)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/pgn_ingest.py",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--max-games",
            "1",
            "--shard-samples",
            "2",
        ],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "games_written=1" in result.stdout
    assert "games_written=1" in result.stderr
    assert "samples=" in result.stderr
    assert (output_dir / DEFAULT_MANIFEST_FILENAME).is_file()


def test_pgn_ingest_script_progress_can_be_disabled(tmp_path: Path) -> None:
    input_path = tmp_path / "games.pgn"
    output_dir = tmp_path / "script-dataset-no-progress"
    input_path.write_text(PGN_TEXT)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/pgn_ingest.py",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--max-games",
            "1",
            "--shard-samples",
            "2",
            "--progress-every-games",
            "0",
        ],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "games_written=1" in result.stdout
    assert "games_written=" not in result.stderr
    assert (output_dir / DEFAULT_MANIFEST_FILENAME).is_file()


def test_train_from_directory_auto_detects_pgn_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "games.pgn"
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "train"
    input_path.write_text(PGN_TEXT)
    ingest_pgn_dataset(
        PgnIngestConfig(input_path=input_path, output_dir=dataset_dir, shard_samples=3)
    )

    result = train_from_directory(
        dataset_dir,
        output_dir,
        config=TrainingConfig(epochs=1, batch_size=2, learning_rate=1.0e-3),
    )

    assert result.steps == 4
    assert result.samples == 6
    assert (output_dir / "checkpoint-final" / DEFAULT_WEIGHTS_FILENAME).is_file()
    assert (output_dir / "metrics.jsonl").read_text().count("\n") == 4
    assert load_checkpoint_metadata(output_dir / "checkpoint-final").training_step == 4


def test_train_script_consumes_pgn_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "games.pgn"
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "train-script"
    input_path.write_text(PGN_TEXT)
    ingest_pgn_dataset(
        PgnIngestConfig(input_path=input_path, output_dir=dataset_dir, shard_samples=3)
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--dataset",
            str(dataset_dir),
            "--output",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--residual-channels",
            "8",
            "--residual-blocks",
            "1",
            "--policy-channels",
            "2",
            "--value-channels",
            "1",
            "--value-hidden-dim",
            "8",
        ],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "training complete" in result.stdout
    assert "samples=6" in result.stdout
    assert (output_dir / "checkpoint-final" / DEFAULT_WEIGHTS_FILENAME).is_file()
