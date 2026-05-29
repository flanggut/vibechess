from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

ROOT = Path(__file__).parents[1]
BENCHMARK_SCRIPT = ROOT / "scripts" / "pgn_ingest_benchmark.py"


def _load_benchmark_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pgn_ingest_benchmark", BENCHMARK_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark_module = _load_benchmark_module()
benchmark_pgn_ingest = cast(Any, benchmark_module).benchmark_pgn_ingest
benchmark_pgn_ingest_full_write = cast(Any, benchmark_module).benchmark_pgn_ingest_full_write

PGN_FIXTURE = """[Event "One"]
[Result "1-0"]

1. e4 e5 1-0

[Event "Two"]
[Result "0-1"]

1. d4 d5 0-1

[Event "Unfinished"]
[Result "*"]

1. c4 c5 *
"""

STABLE_JSON_KEYS = {
    "mode",
    "input_path",
    "output_dir",
    "output_bytes",
    "output_files",
    "strict",
    "skip_fen",
    "max_records",
    "max_games",
    "limits",
    "shard_samples",
    "elapsed_seconds",
    "records_per_second",
    "samples_per_second",
    "records_read",
    "games_accepted",
    "games_skipped",
    "samples",
    "shards",
    "counters",
    "timings",
    "timing_shares",
}


def test_dry_run_benchmark_json_has_stable_fields(tmp_path: Path) -> None:
    input_path = _write_fixture(tmp_path)

    report = benchmark_pgn_ingest(input_path=input_path, max_records=2)
    data = report.to_dict()

    assert data.keys() >= STABLE_JSON_KEYS
    assert data["mode"] == "dry-run"
    assert data["output_dir"] is None
    assert data["output_bytes"] is None
    assert data["records_read"] == 2
    assert data["games_accepted"] == 2
    assert data["games_skipped"] == 0
    assert data["samples"] == 4
    assert data["shards"] == 0
    assert "parse_sanitize" in _dict(data["timings"])
    assert "advance_replay_state" in _dict(data["timings"])


def test_full_write_benchmark_json_has_stable_fields_and_output_size(tmp_path: Path) -> None:
    input_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "benchmark-dataset"

    report = benchmark_pgn_ingest_full_write(
        input_path=input_path,
        output_dir=output_dir,
        max_records=2,
        shard_samples=2,
    )
    data = report.to_dict()

    assert data.keys() >= STABLE_JSON_KEYS
    assert data["mode"] == "full-write"
    assert data["output_dir"] == str(output_dir)
    assert _int(data["output_bytes"]) > 0
    assert _int(data["output_files"]) >= 4
    assert data["records_read"] == 2
    assert data["games_accepted"] == 2
    assert data["games_skipped"] == 0
    assert data["samples"] == 4
    assert data["shards"] == 2
    assert (output_dir / "manifest.json").is_file()
    assert "ingest_pgn_dataset" in _dict(data["timings"])


def test_benchmark_cli_json_smoke_for_dry_run_and_full_write(tmp_path: Path) -> None:
    input_path = _write_fixture(tmp_path)
    dry = _run_benchmark_cli(
        tmp_path,
        "--input",
        str(input_path),
        "--max-records",
        "2",
        "--format",
        "json",
    )
    full_output_dir = tmp_path / "cli-full-write-dataset"
    full = _run_benchmark_cli(
        tmp_path,
        "--input",
        str(input_path),
        "--max-records",
        "2",
        "--mode",
        "full-write",
        "--dataset-output-dir",
        str(full_output_dir),
        "--shard-samples",
        "2",
        "--format",
        "json",
    )

    assert dry["mode"] == "dry-run"
    assert dry["records_read"] == 2
    assert dry["games_accepted"] == 2
    assert full["mode"] == "full-write"
    assert full["records_read"] == 2
    assert full["games_accepted"] == 2
    assert _int(full["output_bytes"]) > 0
    assert (full_output_dir / "manifest.json").is_file()


def _write_fixture(tmp_path: Path) -> Path:
    input_path = tmp_path / "games.pgn"
    input_path.write_text(PGN_FIXTURE)
    return input_path


def _run_benchmark_cli(tmp_path: Path, *args: str) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "scripts/pgn_ingest_benchmark.py", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert isinstance(data, dict)
    return data


def _dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _int(value: object) -> int:
    assert isinstance(value, int)
    return value
