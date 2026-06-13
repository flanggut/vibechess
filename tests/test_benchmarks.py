import json
import subprocess
import sys
from pathlib import Path

import pytest

from vibechess.benchmarks import (
    BenchmarkReport,
    BenchmarkResult,
    benchmark_move_generation,
    format_report,
    recommend_swift_acceleration,
    run_benchmark_suite,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_move_generation_benchmark_smoke_metrics() -> None:
    result = benchmark_move_generation(iterations=1)

    assert result.name == "move_generation"
    assert result.metrics["positions"] == 2
    assert result.metrics["calls"] == 2
    calls_per_second = result.metrics["calls_per_second"]
    assert result.metrics["generated_moves"] == 58
    assert isinstance(calls_per_second, float)
    assert calls_per_second > 0.0
    assert result.metrics["position_move_counts"] == "startpos:20,kiwipete:38"


def test_benchmark_report_formats_markdown_and_json() -> None:
    report = BenchmarkReport(
        results=(BenchmarkResult("example", {"elapsed_seconds": 0.01, "count": 2}),),
        recommendation="measure first",
    )

    markdown = format_report(report)
    assert "# vibechess Benchmark Report" in markdown
    assert "## example" in markdown
    assert "measure first" in markdown

    data = json.loads(format_report(report, output_format="json"))
    assert data["results"][0]["name"] == "example"
    assert data["recommendation"] == "measure first"


def test_benchmark_report_rejects_unknown_format() -> None:
    report = BenchmarkReport(results=(), recommendation="none")

    with pytest.raises(ValueError, match="unsupported report format"):
        format_report(report, output_format="xml")  # type: ignore[arg-type]


def test_smoke_suite_includes_recommendation_and_optional_batched_result() -> None:
    report = run_benchmark_suite(smoke=True, include_batched=True)

    names = {result.name for result in report.results}
    assert {
        "move_generation",
        "complete_game_simulation",
        "mcts_simulations",
        "mlx_inference",
        "mlx_batched_inference",
    } <= names
    assert "Do not justify Swift acceleration" in report.recommendation
    assert "suite-time heuristic" in report.recommendation


def test_benchmark_script_writes_output_and_respects_no_batched(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "benchmark.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark.py",
            "--smoke",
            "--no-batched",
            "--format",
            "json",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"wrote benchmark report to {output}" in completed.stdout
    data = json.loads(output.read_text())
    names = {result["name"] for result in data["results"]}
    assert "mlx_batched_inference" not in names
    assert "mlx_inference" in names


def test_recommendation_is_conservative_without_dominant_component() -> None:
    results = (
        BenchmarkResult("move_generation", {"elapsed_seconds": 1.0}),
        BenchmarkResult("complete_game_simulation", {"elapsed_seconds": 1.0}),
        BenchmarkResult("mcts_simulations", {"elapsed_seconds": 1.0}),
        BenchmarkResult("mlx_inference", {"elapsed_seconds": 1.0}),
    )

    recommendation = recommend_swift_acceleration(results)

    assert "heuristic" in recommendation
    assert "does not justify" in recommendation
