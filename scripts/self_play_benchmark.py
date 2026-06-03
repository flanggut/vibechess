#!/usr/bin/env python3
"""Benchmark tinychess self-play generation throughput."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path("/tmp/tinychess-self-play-benchmark")
SELF_PLAY_PROFILE_ENV = "TINYCHESS_SELF_PLAY_PROFILE"
SELF_PLAY_PROFILE_FILENAME = "profile.json"
ReportFormat = Literal["json", "markdown"]


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """One game chunk assigned to a self-play worker."""

    start_game: int
    games: int
    seed: int


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Self-play benchmark settings."""

    games: int
    max_plies: int
    simulations: int
    temperature: float
    workers: int
    seed: int
    batch_size: int
    label_source: str
    checkpoint: str | None
    checkpoint_id: str | None
    classical_exploration: float
    classical_max_rollout_plies: int
    channels: int
    blocks: int
    repeat: int
    output_root: str
    keep_output: bool
    profile: bool


@dataclass(frozen=True, slots=True)
class RepeatResult:
    """Measured result for one benchmark repeat."""

    repeat_index: int
    elapsed_seconds: float
    samples_per_second: float
    games_per_second: float
    sample_count: int
    game_count: int
    ply_count: int
    output_bytes: int
    output_directory: str
    command: list[str]
    stdout: str
    profile: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Aggregate self-play benchmark report."""

    benchmark: str
    format_version: int
    config: BenchmarkConfig
    elapsed_seconds: float
    elapsed_seconds_min: float
    elapsed_seconds_max: float
    samples_per_second: float
    samples_per_second_min: float
    samples_per_second_max: float
    games_per_second: float
    games_per_second_min: float
    games_per_second_max: float
    sample_count: float
    sample_count_min: int
    sample_count_max: int
    game_count: float
    game_count_min: int
    game_count_max: int
    ply_count: float
    ply_count_min: int
    ply_count_max: int
    output_bytes: int
    workers: int
    effective_workers: int
    chunks: list[ChunkConfig]
    model_config: dict[str, int]
    profile: dict[str, object] | None
    repeat_results: list[RepeatResult]


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    config = _benchmark_config(args)
    report = run_benchmark(config)
    rendered = format_report(report, output_format=args.format)
    print(rendered)
    return 0


def run_benchmark(config: BenchmarkConfig) -> BenchmarkReport:
    """Run the configured self-play benchmark and return aggregate metrics."""
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[RepeatResult] = []
    try:
        for repeat_index in range(1, config.repeat + 1):
            output_dir = output_root / f"repeat-{repeat_index:03d}"
            if output_dir.exists():
                shutil.rmtree(output_dir)
            result = _run_repeat(config, repeat_index, output_dir)
            results.append(result)
            if not config.keep_output:
                shutil.rmtree(output_dir, ignore_errors=True)
    finally:
        if not config.keep_output:
            _remove_empty_directory(output_root)

    elapsed_values = [result.elapsed_seconds for result in results]
    sample_rates = [result.samples_per_second for result in results]
    game_rates = [result.games_per_second for result in results]
    sample_counts = [result.sample_count for result in results]
    game_counts = [result.game_count for result in results]
    ply_counts = [result.ply_count for result in results]
    output_sizes = [result.output_bytes for result in results]
    profile = _aggregate_profiles(results)
    return BenchmarkReport(
        benchmark="self_play_generation",
        format_version=1,
        config=config,
        elapsed_seconds=float(median(elapsed_values)),
        elapsed_seconds_min=min(elapsed_values),
        elapsed_seconds_max=max(elapsed_values),
        samples_per_second=float(median(sample_rates)),
        samples_per_second_min=min(sample_rates),
        samples_per_second_max=max(sample_rates),
        games_per_second=float(median(game_rates)),
        games_per_second_min=min(game_rates),
        games_per_second_max=max(game_rates),
        sample_count=float(median(sample_counts)),
        sample_count_min=min(sample_counts),
        sample_count_max=max(sample_counts),
        game_count=float(median(game_counts)),
        game_count_min=min(game_counts),
        game_count_max=max(game_counts),
        ply_count=float(median(ply_counts)),
        ply_count_min=min(ply_counts),
        ply_count_max=max(ply_counts),
        output_bytes=int(median(output_sizes)),
        workers=config.workers,
        effective_workers=_effective_workers(config.games, config.workers),
        chunks=_chunks(config.games, config.workers, config.seed),
        model_config={"channels": config.channels, "blocks": config.blocks},
        profile=profile,
        repeat_results=results,
    )


def format_report(report: BenchmarkReport, *, output_format: ReportFormat) -> str:
    """Render a benchmark report as JSON or Markdown."""
    if output_format == "json":
        return json.dumps(_report_to_dict(report), indent=2, sort_keys=True)
    if output_format == "markdown":
        return _format_markdown(report)
    raise ValueError(f"unsupported report format: {output_format}")


def _run_repeat(config: BenchmarkConfig, repeat_index: int, output_dir: Path) -> RepeatResult:
    command = _self_play_command(config, output_dir)
    start = time.perf_counter()
    env = os.environ.copy()
    if config.profile:
        env[SELF_PLAY_PROFILE_ENV] = "1"
    else:
        env.pop(SELF_PLAY_PROFILE_ENV, None)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(
            "self-play generation failed "
            f"with exit code {completed.returncode}\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    metadata = _read_metadata(output_dir)
    sample_count = _expect_int(metadata, "sample_count")
    game_count = _expect_int(metadata, "game_count")
    ply_count = _read_ply_count(output_dir)
    profile = _read_profile(output_dir) if config.profile else None
    output_bytes = _directory_size(output_dir)
    return RepeatResult(
        repeat_index=repeat_index,
        elapsed_seconds=elapsed,
        samples_per_second=_rate(sample_count, elapsed),
        games_per_second=_rate(game_count, elapsed),
        sample_count=sample_count,
        game_count=game_count,
        ply_count=ply_count,
        output_bytes=output_bytes,
        output_directory=str(output_dir),
        command=command,
        stdout=completed.stdout.strip(),
        profile=profile,
    )


def _self_play_command(config: BenchmarkConfig, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/self_play.py",
        "--output",
        str(output_dir),
        "--label-source",
        config.label_source,
        "--games",
        str(config.games),
        "--max-plies",
        str(config.max_plies),
        "--simulations",
        str(config.simulations),
        "--temperature",
        str(config.temperature),
        "--classical-exploration",
        str(config.classical_exploration),
        "--classical-max-rollout-plies",
        str(config.classical_max_rollout_plies),
        "--seed",
        str(config.seed),
        "--channels",
        str(config.channels),
        "--blocks",
        str(config.blocks),
        "--batch-size",
        str(config.batch_size),
        "--workers",
        str(config.workers),
    ]
    if config.checkpoint is not None:
        command.extend(["--checkpoint", config.checkpoint])
    if config.checkpoint_id is not None:
        command.extend(["--checkpoint-id", config.checkpoint_id])
    return command


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark tinychess self-play generation throughput."
    )
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--max-plies", type=int, default=192)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--label-source",
        choices=("neural", "classical"),
        default="neural",
        help="search source for policy labels and self-play moves",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-id", default=None)
    parser.add_argument("--classical-exploration", type=float, default=1.41421356237)
    parser.add_argument(
        "--classical-max-rollout-plies",
        type=int,
        default=0,
        help="classical MCTS rollout cap; default 0 uses static leaf evaluation",
    )
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--keep-output", action="store_true")
    parser.add_argument(
        "--no-profile",
        dest="profile",
        action="store_false",
        help="disable benchmark profile counters in the self-play subprocess",
    )
    parser.set_defaults(profile=True)
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="markdown",
        help="benchmark report output format",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("games", "simulations", "workers", "batch_size", "channels"):
        value = getattr(args, name)
        if value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if args.max_plies < 0:
        raise ValueError("--max-plies must be non-negative")
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")
    if args.blocks < 0:
        raise ValueError("--blocks must be non-negative")
    if args.label_source == "classical":
        if args.checkpoint is not None:
            raise ValueError("--checkpoint is only supported with --label-source neural")
        if args.checkpoint_id is not None:
            raise ValueError("--checkpoint-id is only supported with --label-source neural")


def _benchmark_config(args: argparse.Namespace) -> BenchmarkConfig:
    checkpoint = None if args.checkpoint is None else str(args.checkpoint)
    return BenchmarkConfig(
        games=args.games,
        max_plies=args.max_plies,
        simulations=args.simulations,
        temperature=args.temperature,
        workers=args.workers,
        seed=args.seed,
        batch_size=args.batch_size,
        label_source=args.label_source,
        checkpoint=checkpoint,
        checkpoint_id=args.checkpoint_id,
        classical_exploration=args.classical_exploration,
        classical_max_rollout_plies=args.classical_max_rollout_plies,
        channels=args.channels,
        blocks=args.blocks,
        repeat=args.repeat,
        output_root=str(args.output_root),
        keep_output=args.keep_output,
        profile=args.profile,
    )


def _report_to_dict(report: BenchmarkReport) -> dict[str, object]:
    return {
        "benchmark": report.benchmark,
        "format_version": report.format_version,
        "config": asdict(report.config),
        "games": report.config.games,
        "max_plies": report.config.max_plies,
        "simulations": report.config.simulations,
        "temperature": report.config.temperature,
        "seed": report.config.seed,
        "batch_size": report.config.batch_size,
        "label_source": report.config.label_source,
        "channels": report.config.channels,
        "blocks": report.config.blocks,
        "elapsed_seconds": report.elapsed_seconds,
        "elapsed_seconds_min": report.elapsed_seconds_min,
        "elapsed_seconds_max": report.elapsed_seconds_max,
        "samples_per_second": report.samples_per_second,
        "samples_per_second_min": report.samples_per_second_min,
        "samples_per_second_max": report.samples_per_second_max,
        "games_per_second": report.games_per_second,
        "games_per_second_min": report.games_per_second_min,
        "games_per_second_max": report.games_per_second_max,
        "sample_count": report.sample_count,
        "sample_count_min": report.sample_count_min,
        "sample_count_max": report.sample_count_max,
        "game_count": report.game_count,
        "game_count_min": report.game_count_min,
        "game_count_max": report.game_count_max,
        "ply_count": report.ply_count,
        "ply_count_min": report.ply_count_min,
        "ply_count_max": report.ply_count_max,
        "output_bytes": report.output_bytes,
        "workers": report.workers,
        "effective_workers": report.effective_workers,
        "chunks": [asdict(chunk) for chunk in report.chunks],
        "model_config": report.model_config,
        "profile": report.profile,
        "repeat_results": [asdict(result) for result in report.repeat_results],
    }


def _format_markdown(report: BenchmarkReport) -> str:
    lines = [
        "# tinychess Self-Play Benchmark",
        "",
        "## Summary",
        "",
        f"- elapsed_seconds_median: {report.elapsed_seconds:.6f}",
        f"- elapsed_seconds_min: {report.elapsed_seconds_min:.6f}",
        f"- elapsed_seconds_max: {report.elapsed_seconds_max:.6f}",
        f"- samples_per_second_median: {report.samples_per_second:.2f}",
        f"- games_per_second_median: {report.games_per_second:.2f}",
        f"- sample_count_median: {report.sample_count:g}",
        f"- sample_count_min: {report.sample_count_min}",
        f"- sample_count_max: {report.sample_count_max}",
        f"- game_count_median: {report.game_count:g}",
        f"- game_count_min: {report.game_count_min}",
        f"- game_count_max: {report.game_count_max}",
        f"- ply_count_median: {report.ply_count:g}",
        f"- ply_count_min: {report.ply_count_min}",
        f"- ply_count_max: {report.ply_count_max}",
        f"- output_bytes_median: {report.output_bytes}",
        f"- workers: {report.workers}",
        f"- effective_workers: {report.effective_workers}",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(asdict(report.config), indent=2, sort_keys=True),
        "```",
        "",
        "## Profile",
        "",
        *_format_profile_lines(report.profile),
        "",
        "## Chunks",
        "",
    ]
    for chunk in report.chunks:
        lines.append(
            f"- start_game={chunk.start_game} games={chunk.games} seed={chunk.seed}"
        )
    lines.extend(["", "## Repeats", ""])
    for result in report.repeat_results:
        lines.extend(
            [
                f"### Repeat {result.repeat_index}",
                "",
                f"- elapsed_seconds: {result.elapsed_seconds:.6f}",
                f"- samples_per_second: {result.samples_per_second:.2f}",
                f"- games_per_second: {result.games_per_second:.2f}",
                f"- sample_count: {result.sample_count}",
                f"- game_count: {result.game_count}",
                f"- ply_count: {result.ply_count}",
                f"- output_bytes: {result.output_bytes}",
                f"- output_directory: {result.output_directory}",
                f"- command: `{' '.join(result.command)}`",
                "",
            ]
        )
    return "\n".join(lines)


def _format_profile_lines(profile: dict[str, object] | None) -> list[str]:
    if profile is None:
        return ["- profile: disabled or unavailable"]
    percentages = _expect_mapping(profile, "percent_of_elapsed")
    stats = _expect_mapping(profile, "stats")
    timers = _expect_mapping(stats, "timers")
    lines = [
        f"- repeat_count: {_expect_int(profile, 'repeat_count')}",
        "- timer percentages are diagnostic and may overlap",
    ]
    for name in (
        "game_legal_moves",
        "determine_outcome",
        "game_play_known_legal",
        "board_apply_move",
        "model_single",
        "model_batch",
        "search",
    ):
        timer = _expect_mapping(timers, name)
        percent = _expect_number(percentages, name)
        seconds = _expect_number(timer, "seconds")
        calls = _expect_int(timer, "calls")
        lines.append(
            f"- {name}: calls={calls} seconds={seconds:.6f} "
            f"elapsed_pct={percent:.2f}"
        )
    search = _expect_mapping(timers, "search")
    model_batch = _expect_mapping(timers, "model_batch")
    lines.extend(
        [
            f"- completed_simulations: {_expect_int(search, 'completed_simulations')}",
            f"- materialized_nodes: {_expect_int(search, 'materialized_nodes')}",
            f"- model_batch_positions: {_expect_int(model_batch, 'positions')}",
            f"- model_batch_size_mean: {_expect_number(model_batch, 'batch_size_mean'):.2f}",
        ]
    )
    return lines


def _read_profile(output_dir: Path) -> dict[str, object]:
    profile_path = output_dir / SELF_PLAY_PROFILE_FILENAME
    data = json.loads(profile_path.read_text())
    if not isinstance(data, dict):
        raise TypeError("self-play profile must be a JSON object")
    return data


def _aggregate_profiles(results: list[RepeatResult]) -> dict[str, object] | None:
    profiled_results = [result for result in results if result.profile is not None]
    if not profiled_results:
        return None
    stats = _sum_profile_stats([result.profile for result in profiled_results])
    elapsed = sum(result.elapsed_seconds for result in profiled_results)
    percentages = _profile_percentages(stats, elapsed)
    return {
        "format_version": 1,
        "repeat_count": len(profiled_results),
        "elapsed_seconds": elapsed,
        "stats": stats,
        "percent_of_elapsed": percentages,
        "limitations": [
            "Timer categories are diagnostic and can overlap; do not sum percentages.",
            "Worker runs aggregate child-process profile reports written by self_play.py.",
        ],
    }


def _sum_profile_stats(profiles: list[dict[str, object] | None]) -> dict[str, object]:
    timer_names = (
        "game_legal_moves",
        "determine_outcome",
        "game_play_known_legal",
        "board_apply_move",
        "model_single",
        "model_batch",
        "search",
    )
    totals: dict[str, dict[str, object]] = {
        name: {"calls": 0, "seconds": 0.0} for name in timer_names
    }
    totals["model_batch"].update(
        {
            "positions": 0,
            "batch_size_min": None,
            "batch_size_max": None,
            "batch_size_mean": 0.0,
        }
    )
    totals["search"].update({"materialized_nodes": 0, "completed_simulations": 0})

    for profile in profiles:
        if profile is None:
            continue
        stats = _profile_stats(profile)
        timers = _expect_mapping(stats, "timers")
        for name in timer_names:
            source = _expect_mapping(timers, name)
            target = totals[name]
            target["calls"] = _expect_int(target, "calls") + _expect_int(source, "calls")
            target["seconds"] = _expect_number(target, "seconds") + _expect_number(
                source, "seconds"
            )
        source_batch = _expect_mapping(timers, "model_batch")
        target_batch = totals["model_batch"]
        target_batch["positions"] = _expect_int(target_batch, "positions") + _expect_int(
            source_batch, "positions"
        )
        _merge_optional_min(target_batch, source_batch, "batch_size_min")
        _merge_optional_max(target_batch, source_batch, "batch_size_max")
        source_search = _expect_mapping(timers, "search")
        target_search = totals["search"]
        target_search["materialized_nodes"] = _expect_int(
            target_search, "materialized_nodes"
        ) + _expect_int(source_search, "materialized_nodes")
        target_search["completed_simulations"] = _expect_int(
            target_search, "completed_simulations"
        ) + _expect_int(source_search, "completed_simulations")

    batch = totals["model_batch"]
    batch_calls = _expect_int(batch, "calls")
    if batch_calls > 0:
        batch["batch_size_mean"] = _expect_int(batch, "positions") / batch_calls
    return {"format_version": 1, "timers": totals}


def _profile_stats(profile: dict[str, object]) -> dict[str, object]:
    stats = profile.get("stats")
    if isinstance(stats, dict):
        return dict(stats)
    return profile


def _profile_percentages(stats: dict[str, object], elapsed_seconds: float) -> dict[str, float]:
    timers = _expect_mapping(stats, "timers")
    percentages: dict[str, float] = {}
    for name, value in timers.items():
        if isinstance(name, str) and isinstance(value, dict):
            seconds = _expect_number(value, "seconds")
            percentages[name] = 0.0 if elapsed_seconds == 0.0 else seconds / elapsed_seconds * 100.0
    return percentages


def _merge_optional_min(
    target: dict[str, object], source: dict[str, object], key: str
) -> None:
    source_value = _optional_int(source, key)
    if source_value is None:
        return
    target_value = _optional_int(target, key)
    if target_value is None or source_value < target_value:
        target[key] = source_value


def _merge_optional_max(
    target: dict[str, object], source: dict[str, object], key: str
) -> None:
    source_value = _optional_int(source, key)
    if source_value is None:
        return
    target_value = _optional_int(target, key)
    if target_value is None or source_value > target_value:
        target[key] = source_value


def _chunks(games: int, workers: int, seed: int) -> list[ChunkConfig]:
    effective_workers = _effective_workers(games, workers)
    chunks: list[ChunkConfig] = []
    start = 0
    for worker_index in range(effective_workers):
        count = games // effective_workers + (
            1 if worker_index < games % effective_workers else 0
        )
        if count > 0:
            chunks.append(ChunkConfig(start_game=start, games=count, seed=seed + start))
            start += count
    return chunks


def _effective_workers(games: int, workers: int) -> int:
    if workers == 1 or games == 1:
        return 1
    return min(workers, games)


def _read_metadata(output_dir: Path) -> dict[str, object]:
    metadata_path = output_dir / "metadata.json"
    data = json.loads(metadata_path.read_text())
    if not isinstance(data, dict):
        raise TypeError("self-play metadata must be a JSON object")
    return data


def _read_ply_count(output_dir: Path) -> int:
    games_path = output_dir / "games.jsonl"
    total = 0
    with games_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError("self-play game record must be a JSON object")
            total += _expect_int(record, "plies")
    return total


def _directory_size(directory: Path) -> int:
    total = 0
    for path in directory.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def _rate(count: int, elapsed_seconds: float) -> float:
    if elapsed_seconds == 0.0:
        return float("inf")
    return count / elapsed_seconds


def _expect_int(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"metadata field {key!r} must be an integer")
    return value


def _optional_int(data: dict[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"profile field {key!r} must be an integer or null")
    return value


def _expect_number(data: dict[str, object], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"profile field {key!r} must be a number")
    return float(value)


def _expect_mapping(data: dict[str, object], key: str) -> dict[str, object]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"profile field {key!r} must be an object")
    return dict(value)


def _remove_empty_directory(directory: Path) -> None:
    try:
        directory.rmdir()
    except OSError:
        return


if __name__ == "__main__":
    raise SystemExit(main())
