#!/usr/bin/env python3
"""Benchmark vibechess self-play generation throughput."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from typing import Any, Literal

from vibechess import _jsonio, _scriptutil
from vibechess.nn.checkpoint import save_checkpoint
from vibechess.nn.model import PolicyValueConfig, PolicyValueNet
from vibechess.nn.self_play_profile import (
    ProfileStats,
    profile_limitations,
    stats_from_profile_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path("/tmp/vibechess-self-play-benchmark")
SELF_PLAY_PROFILE_ENV = "VIBECHESS_SELF_PLAY_PROFILE"
SELF_PLAY_PROFILE_FILENAME = "profile.json"
ReportFormat = Literal["json", "markdown"]
ProfileLevel = Literal["none", "summary", "detailed"]


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
    active_games: int | None
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
    profile_level: ProfileLevel
    profile_overhead_check: bool


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
    batching_mode: str | None
    inference_batch_size: int | None
    profile: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Aggregate self-play benchmark report."""

    benchmark: str
    format_version: int
    config: BenchmarkConfig
    batching_mode: str
    inference_batch_size: int | None
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
    profile_overhead: dict[str, object]
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
    overhead_pairs: list[dict[str, object]] = []
    no_profile_config = replace(config, profile=False, profile_level="none")
    try:
        for repeat_index in range(1, config.repeat + 1):
            output_dir = output_root / f"repeat-{repeat_index:03d}"
            if output_dir.exists():
                shutil.rmtree(output_dir)
            no_profile_result: RepeatResult | None = None
            no_profile_dir = output_root / f"repeat-{repeat_index:03d}-noprofile"
            repeat_config = config
            repeat_no_profile_config = no_profile_config
            with _shared_overhead_checkpoint(config) as checkpoint_path:
                if checkpoint_path is not None:
                    repeat_config = _config_with_checkpoint(config, checkpoint_path)
                    repeat_no_profile_config = _config_with_checkpoint(
                        no_profile_config,
                        checkpoint_path,
                    )
                if config.profile_overhead_check and config.profile:
                    if no_profile_dir.exists():
                        shutil.rmtree(no_profile_dir)
                    no_profile_result = _run_repeat(
                        repeat_no_profile_config,
                        repeat_index,
                        no_profile_dir,
                    )
                result = _run_repeat(repeat_config, repeat_index, output_dir)
            results.append(result)
            if no_profile_result is not None:
                overhead_pairs.append(
                    _profile_overhead_pair(no_profile_result, result)
                )
            if not config.keep_output:
                shutil.rmtree(output_dir, ignore_errors=True)
                shutil.rmtree(no_profile_dir, ignore_errors=True)
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
    profile_overhead = _aggregate_profile_overhead(overhead_pairs, config)
    return BenchmarkReport(
        benchmark="self_play_generation",
        format_version=2,
        config=config,
        batching_mode=_aggregate_batching_mode(results),
        inference_batch_size=_aggregate_inference_batch_size(results),
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
        profile_overhead=profile_overhead,
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
    if config.profile and config.profile_level != "none":
        env[SELF_PLAY_PROFILE_ENV] = config.profile_level
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
    batching_mode, inference_batch_size = _read_batching_settings(metadata)
    profile = (
        _read_profile(output_dir)
        if config.profile and config.profile_level != "none"
        else None
    )
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
        batching_mode=batching_mode,
        inference_batch_size=inference_batch_size,
        profile=profile,
    )


@contextmanager
def _shared_overhead_checkpoint(config: BenchmarkConfig) -> Iterator[Path | None]:
    """Create a shared temporary model for paired no-profile/profile comparisons.

    Without an explicit checkpoint, each self-play subprocess initializes a fresh random
    neural model.  The overhead check compares two subprocesses, so using one temporary
    checkpoint isolates profiling effects from unrelated model-initialization randomness.
    Normal benchmark runs keep their existing no-checkpoint behavior.
    """
    if (
        not config.profile_overhead_check
        or not config.profile
        or config.checkpoint is not None
        or config.label_source != "neural"
    ):
        yield None
        return
    with TemporaryDirectory(prefix="vibechess-profile-overhead-") as temp_dir:
        checkpoint_path = Path(temp_dir)
        save_checkpoint(PolicyValueNet(_policy_value_config(config)), checkpoint_path)
        yield checkpoint_path


def _config_with_checkpoint(config: BenchmarkConfig, checkpoint: Path) -> BenchmarkConfig:
    return replace(
        config,
        checkpoint=str(checkpoint),
        checkpoint_id=config.checkpoint_id or "profile-overhead-check-shared-model",
    )


def _policy_value_config(config: BenchmarkConfig) -> PolicyValueConfig:
    return PolicyValueConfig(
        residual_channels=config.channels,
        residual_blocks=config.blocks,
        policy_channels=1,
        value_channels=1,
        value_hidden_dim=4,
    )


def _profile_overhead_pair(
    no_profile: RepeatResult,
    profiled: RepeatResult,
) -> dict[str, object]:
    deterministic_games_match = _games_text(no_profile.output_directory) == _games_text(
        profiled.output_directory
    )
    elapsed_delta = profiled.elapsed_seconds - no_profile.elapsed_seconds
    overhead_percent = _pct(elapsed_delta, no_profile.elapsed_seconds)
    counts_match = (
        no_profile.sample_count == profiled.sample_count
        and no_profile.game_count == profiled.game_count
        and no_profile.ply_count == profiled.ply_count
    )
    return {
        "repeat_index": profiled.repeat_index,
        "no_profile_elapsed_seconds": no_profile.elapsed_seconds,
        "profiled_elapsed_seconds": profiled.elapsed_seconds,
        "overhead_seconds": elapsed_delta,
        "overhead_percent": overhead_percent,
        "counts_match": counts_match,
        "deterministic_games_match": deterministic_games_match,
    }


def _aggregate_profile_overhead(
    pairs: list[dict[str, object]],
    config: BenchmarkConfig,
) -> dict[str, object]:
    if not config.profile_overhead_check:
        return {"enabled": False}
    if not pairs:
        return {"enabled": True, "pairs": []}
    no_profile_elapsed = [
        _expect_number(pair, "no_profile_elapsed_seconds") for pair in pairs
    ]
    profiled_elapsed = [_expect_number(pair, "profiled_elapsed_seconds") for pair in pairs]
    no_profile_median = float(median(no_profile_elapsed))
    profiled_median = float(median(profiled_elapsed))
    return {
        "enabled": True,
        "pairs": pairs,
        "no_profile_elapsed_seconds": no_profile_median,
        "profiled_elapsed_seconds": profiled_median,
        "overhead_seconds": profiled_median - no_profile_median,
        "overhead_percent": _pct(profiled_median - no_profile_median, no_profile_median),
        "counts_match": all(bool(pair.get("counts_match")) for pair in pairs),
        "deterministic_games_match": all(
            bool(pair.get("deterministic_games_match")) for pair in pairs
        ),
    }


def _games_text(output_directory: str) -> str:
    path = Path(output_directory) / "games.jsonl"
    return path.read_text() if path.exists() else ""


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
    if config.active_games is not None:
        command.extend(["--active-games", str(config.active_games)])
    if config.checkpoint is not None:
        command.extend(["--checkpoint", config.checkpoint])
    if config.checkpoint_id is not None:
        command.extend(["--checkpoint-id", config.checkpoint_id])
    return command


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark vibechess self-play generation throughput."
    )
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--max-plies", type=int, default=192)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--active-games",
        type=int,
        default=None,
        help=(
            "maximum in-process active neural self-play games; defaults to --batch-size; "
            "inference calls are still capped by --batch-size"
        ),
    )
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
        "--profile-level",
        choices=("none", "summary", "detailed"),
        default="detailed",
        help="self-play profiling detail level for benchmark subprocesses",
    )
    parser.add_argument(
        "--profile-overhead-check",
        action="store_true",
        help="include paired no-profile/profile overhead metadata in the report",
    )
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
    parsed = parser.parse_args()
    if not parsed.profile:
        parsed.profile_level = "none"
    return parsed


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("games", "simulations", "workers", "batch_size", "channels"):
        value = getattr(args, name)
        if value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if args.max_plies < 0:
        raise ValueError("--max-plies must be non-negative")
    if args.active_games is not None and args.active_games < 1:
        raise ValueError("--active-games must be at least 1")
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
        active_games=args.active_games,
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
        profile=args.profile and args.profile_level != "none",
        profile_level=args.profile_level,
        profile_overhead_check=args.profile_overhead_check,
    )


def _report_to_dict(report: BenchmarkReport) -> dict[str, object]:
    return {
        "benchmark": report.benchmark,
        "format_version": report.format_version,
        "config": asdict(report.config),
        "batching_mode": report.batching_mode,
        "inference_batch_size": report.inference_batch_size,
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
        "profile_overhead": report.profile_overhead,
        "repeat_results": [asdict(result) for result in report.repeat_results],
    }


def _format_markdown(report: BenchmarkReport) -> str:
    lines = [
        "# vibechess Self-Play Benchmark",
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
        f"- batching_mode: {report.batching_mode}",
        f"- inference_batch_size: {report.inference_batch_size}",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(asdict(report.config), indent=2, sort_keys=True),
        "```",
        "",
        "## Profile",
        "",
        *_format_profile_lines(
            report.profile,
            report.batching_mode,
            report.inference_batch_size,
        ),
        "",
        "## Profiling Overhead",
        "",
        *_format_profile_overhead_lines(report.profile_overhead),
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
                f"- batching_mode: {result.batching_mode}",
                f"- inference_batch_size: {result.inference_batch_size}",
                f"- command: `{' '.join(result.command)}`",
                "",
            ]
        )
    return "\n".join(lines)


def _format_profile_lines(
    profile: dict[str, object] | None,
    batching_mode: str,
    inference_batch_size: int | None,
) -> list[str]:
    if profile is None:
        return ["- profile: disabled or unavailable"]
    stats = _expect_mapping(profile, "stats")
    timers = _expect_mapping(stats, "timers")
    lines = [
        f"- repeat_count: {_expect_int(profile, 'repeat_count')}",
        f"- profile_level: {profile.get('profile_level', 'unknown')}",
        "- bottleneck rankings use exclusive stack-attributed time",
        "",
        "## Bottleneck Summary",
        "",
        "| rank | zone | exclusive s | inclusive s | % generation | calls | interpretation |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    bottlenecks = profile.get("bottleneck_summary", [])
    if isinstance(bottlenecks, list):
        for row in bottlenecks:
            if isinstance(row, dict):
                lines.append(
                    "| {rank} | {zone} | {exclusive:.6f} | {inclusive:.6f} | "
                    "{pct:.2f} | {calls} | {reason} |".format(
                        rank=row.get("rank", ""),
                        zone=row.get("zone", ""),
                        exclusive=float(row.get("exclusive_seconds", 0.0)),
                        inclusive=float(row.get("inclusive_seconds", 0.0)),
                        pct=float(row.get("percent_of_generation", 0.0)),
                        calls=row.get("calls", ""),
                        reason=row.get("reason", ""),
                    )
                )
    lines.extend(["", "## MCTS Breakdown", ""])
    lines.extend(
        _zone_table(
            stats,
            (
                "mcts.search",
                "mcts.simulation",
                "mcts.selection",
                "mcts.materialize_child",
                "mcts.expand",
                "mcts.predict",
                "mcts.backup",
                "mcts.select_temperature",
            ),
        )
    )
    lines.extend(["", "## Inference / MLX Breakdown", ""])
    lines.extend(
        _zone_table(
            stats,
            (
                "inference.predict",
                "inference.predict_with_legal_moves",
                "inference.predict_batch",
                "inference.predict_legal_batch",
                "self_play.central_inference_queue",
                "self_play.central_predict_legal_batch",
                "encode.game_mlx",
                "encode.batch_stack",
                "model.forward",
                "mlx.sync.value_item",
                "mlx.sync.policy_eval",
                "mlx.sync.legal_batch_eval",
                "policy.legal_indices",
                "policy.legal_mask_mlx",
            ),
        )
    )
    lines.extend(["", "## Legal and Transition Breakdown", ""])
    lines.extend(
        _zone_table(
            stats,
            (
                "game.legal_moves",
                "search_state.legal_moves",
                "legal.legal_moves",
                "legal.pseudo",
                "legal.filter",
                "legal.is_square_attacked",
                "board.apply_move",
                "game.play_known_legal",
                "search_state.play_known_legal",
            ),
        )
    )
    lines.extend(["", "## Dataset and Serialization Breakdown", ""])
    lines.extend(
        _zone_table(
            stats,
            (
                "record.position_encode_np",
                "record.legal_mask_np",
                "record.policy_target",
                "dataset.stack_positions",
                "dataset.save",
                "dataset.save_npz_compressed",
                "dataset.write_metadata",
                "dataset.write_games_jsonl",
            ),
        )
    )
    lines.extend(["", "## Worker Breakdown", ""])
    workers = profile.get("workers", [])
    if isinstance(workers, list) and workers:
        lines.append(f"- worker_profiles: {len(workers)}")
    else:
        lines.append("- worker_profiles: none")
    lines.extend(["", "### Compatibility Timers", ""])
    percentages = _expect_mapping(profile, "percent_of_elapsed")
    for name in (
        "game_legal_moves",
        "determine_outcome",
        "game_play_known_legal",
        "board_apply_move",
        "model_single",
        "model_batch",
        "model_legal_batch",
        "search",
    ):
        timer = _expect_mapping(timers, name)
        percent = _expect_number(percentages, name)
        seconds = _expect_number(timer, "seconds")
        calls = _expect_int(timer, "calls")
        lines.append(
            f"- {name}: calls={calls} seconds={seconds:.6f} elapsed_pct={percent:.2f}"
        )
    search = _expect_mapping(timers, "search")
    model_batch = _expect_mapping(timers, "model_batch")
    model_legal_batch = _expect_mapping(timers, "model_legal_batch")
    lines.extend(
        [
            f"- completed_simulations: {_expect_int(search, 'completed_simulations')}",
            f"- materialized_nodes: {_expect_int(search, 'materialized_nodes')}",
            f"- model_batch_positions: {_expect_int(model_batch, 'positions')}",
            f"- model_batch_size_mean: {_expect_number(model_batch, 'batch_size_mean'):.2f}",
            f"- model_legal_batch_calls: {_expect_int(model_legal_batch, 'calls')}",
            f"- model_legal_batch_positions: {_expect_int(model_legal_batch, 'positions')}",
            f"- model_legal_batch_size_mean: "
            f"{_expect_number(model_legal_batch, 'batch_size_mean'):.2f}",
        ]
    )
    if _has_central_queue_profile(stats, batching_mode):
        lines.extend(["", "## Central Queue Batching", ""])
        lines.extend(_central_queue_profile_lines(stats, batching_mode, inference_batch_size))
    return lines


def _has_central_queue_profile(stats: dict[str, object], batching_mode: str) -> bool:
    if batching_mode == "central_inference_queue":
        return True
    counters = _expect_mapping(stats, "counters")
    distributions = _expect_mapping(stats, "distributions")
    return (
        _counter_int(counters, "inference.predict_legal_batch.calls") > 0
        or _counter_int(counters, "inference.legal_batch_positions") > 0
        or isinstance(distributions.get("inference.legal_batch_size"), dict)
    )


def _central_queue_profile_lines(
    stats: dict[str, object],
    batching_mode: str,
    inference_batch_size: int | None,
) -> list[str]:
    counters = _expect_mapping(stats, "counters")
    distributions = _expect_mapping(stats, "distributions")
    legal_batch = distributions.get("inference.legal_batch_size")
    lines = [
        f"- batching_mode: {batching_mode}",
        f"- inference_batch_size: {inference_batch_size}",
        "- inference.predict_legal_batch.calls: "
        f"{_counter_int(counters, 'inference.predict_legal_batch.calls')}",
        "- inference.legal_batch_positions: "
        f"{_counter_int(counters, 'inference.legal_batch_positions')}",
    ]
    if isinstance(legal_batch, dict):
        lines.extend(
            [
                "- inference.legal_batch_size.count: "
                f"{_expect_int(legal_batch, 'count')}",
                f"- inference.legal_batch_size.min: {_expect_number(legal_batch, 'min'):.0f}",
                f"- inference.legal_batch_size.max: {_expect_number(legal_batch, 'max'):.0f}",
                f"- inference.legal_batch_size.mean: {_expect_number(legal_batch, 'mean'):.2f}",
            ]
        )
    else:
        lines.append("- inference.legal_batch_size: none recorded")
    return lines


def _zone_table(stats: dict[str, object], names: tuple[str, ...]) -> list[str]:
    zones = _expect_mapping(stats, "zones")
    lines = ["| zone | inclusive s | exclusive s | calls |", "|---|---:|---:|---:|"]
    for name in names:
        raw = zones.get(name)
        if not isinstance(raw, dict):
            continue
        lines.append(
            f"| {name} | {_expect_number(raw, 'inclusive_seconds'):.6f} | "
            f"{_expect_number(raw, 'exclusive_seconds'):.6f} | {_expect_int(raw, 'calls')} |"
        )
    if len(lines) == 2:
        lines.append("| _none recorded_ | 0 | 0 | 0 |")
    return lines



def _format_profile_overhead_lines(profile_overhead: dict[str, object]) -> list[str]:
    if not profile_overhead.get("enabled", False):
        return ["- enabled: false"]
    return [f"- {key}: {value}" for key, value in sorted(profile_overhead.items())]


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
    merged = ProfileStats.merged(
        [stats_from_profile_report(result.profile or {}) for result in profiled_results]
    )
    stats = merged.to_dict()
    elapsed = sum(result.elapsed_seconds for result in profiled_results)
    percentages = _profile_percentages(stats, elapsed)
    zones = _expect_mapping(stats, "zones")
    workers = _profile_workers(profiled_results)
    derived = _profile_derived(merged, elapsed, workers=workers)
    return {
        "format_version": 2,
        "profile_level": _profile_level_from_results(profiled_results),
        "repeat_count": len(profiled_results),
        "elapsed_seconds": elapsed,
        "stats": stats,
        "percent_of_elapsed": percentages,
        "bottleneck_summary": _bottleneck_summary(merged),
        "derived": derived,
        "workers": workers,
        "slowest_plies": merged.slowest_plies,
        "slowest_searches": merged.slowest_searches,
        "top_exclusive_zones": _top_exclusive_zones(zones),
        "limitations": profile_limitations(),
    }


def _profile_level_from_results(results: list[RepeatResult]) -> str:
    for result in results:
        profile = result.profile or {}
        level = profile.get("profile_level")
        if isinstance(level, str):
            return level
    return "detailed"


def _profile_workers(results: list[RepeatResult]) -> list[dict[str, object]]:
    workers: list[dict[str, object]] = []
    for result in results:
        profile = result.profile or {}
        raw_workers = profile.get("workers", profile.get("worker_profiles", []))
        if isinstance(raw_workers, list):
            for worker in raw_workers:
                if isinstance(worker, dict):
                    worker_copy = dict(worker)
                    worker_copy.setdefault("repeat_index", result.repeat_index)
                    workers.append(worker_copy)
    return workers


def _profile_derived(
    stats: ProfileStats,
    elapsed_seconds: float,
    *,
    workers: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    zones = stats.zones
    generation = _zone_seconds(zones, "self_play.generate_dataset")
    main = _zone_seconds(zones, "self_play.main")
    mcts = _zone_seconds(zones, "mcts.search")
    inference = sum(
        _zone_seconds(zones, name)
        for name in (
            "inference.predict",
            "inference.predict_with_legal_moves",
            "inference.predict_batch",
            "inference.predict_legal_batch",
        )
    )
    legal_transition = sum(
        _zone_seconds(zones, name)
        for name in (
            "game.legal_moves",
            "search_state.legal_moves",
            "legal.legal_moves",
            "board.apply_move",
            "game.play_known_legal",
            "search_state.play_known_legal",
        )
    )
    save = _zone_seconds(zones, "dataset.save")
    central_queue = _zone_seconds(zones, "self_play.central_inference_queue")
    central_predict = _zone_seconds(zones, "self_play.central_predict_legal_batch")
    legal_batch_distribution = stats.distributions.get("inference.legal_batch_size")
    derived: dict[str, object] = {
        "self_play_main_seconds": main,
        "generation_seconds": generation,
        "mcts_search_seconds": mcts,
        "mcts_search_percent_of_generation": _pct(mcts, generation),
        "inference_percent_of_mcts_search": _pct(inference, mcts),
        "legal_transition_percent_of_mcts_search": _pct(legal_transition, mcts),
        "dataset_save_percent_of_self_play_main": _pct(save, main),
        "central_queue_seconds": central_queue,
        "central_queue_predict_seconds": central_predict,
        "predict_legal_batch_calls": _stats_counter_int(
            stats,
            "inference.predict_legal_batch.calls",
        ),
        "predict_legal_batch_positions": _stats_counter_int(
            stats,
            "inference.legal_batch_positions",
        ),
        "repeat_wall_seconds": elapsed_seconds,
    }
    if legal_batch_distribution is not None:
        derived["predict_legal_batch_size"] = legal_batch_distribution.to_dict()
    pool = _zone_seconds(zones, "worker.pool_elapsed")
    worker = _zone_seconds(zones, "worker.chunk_elapsed")
    if pool > 0:
        derived["worker_pool_elapsed_seconds"] = pool
        derived["worker_time_sum_seconds"] = worker
    if workers:
        worker_elapsed_by_repeat: dict[int, list[float]] = {}
        for worker_profile in workers:
            elapsed = _worker_elapsed_seconds(worker_profile)
            if elapsed is None:
                continue
            repeat_index = _object_int(worker_profile.get("repeat_index")) or 0
            worker_elapsed_by_repeat.setdefault(repeat_index, []).append(elapsed)
        worker_elapsed = [
            elapsed
            for repeat_workers in worker_elapsed_by_repeat.values()
            for elapsed in repeat_workers
        ]
        if worker_elapsed:
            worker_sum = sum(worker_elapsed)
            repeat_max_sum = sum(
                max(repeat_workers) for repeat_workers in worker_elapsed_by_repeat.values()
            )
            average_workers = len(worker_elapsed) / max(1, len(worker_elapsed_by_repeat))
            derived.update(
                {
                    "worker_count": len(worker_elapsed),
                    "worker_time_max_seconds": max(worker_elapsed),
                    "worker_time_sum_seconds": worker_sum,
                    "worker_ipc_merge_gap_seconds": max(0.0, pool - repeat_max_sum)
                    if pool > 0
                    else 0.0,
                    "worker_parallel_efficiency": worker_sum / (pool * average_workers)
                    if pool > 0 and average_workers > 0
                    else 0.0,
                }
            )
    return derived


def _worker_elapsed_seconds(worker_profile: dict[str, object]) -> float | None:
    try:
        stats = stats_from_profile_report(worker_profile)
    except (TypeError, ValueError):
        return None
    timer = stats.zones.get("worker.chunk_elapsed")
    if timer is None:
        return None
    return timer.inclusive_ns / 1_000_000_000.0


def _object_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _zone_seconds(zones: dict[str, object] | dict[str, Any], name: str) -> float:
    raw = zones.get(name)
    if isinstance(raw, dict):
        return _expect_number(raw, "inclusive_seconds")
    if hasattr(raw, "inclusive_ns"):
        return float(raw.inclusive_ns) / 1_000_000_000.0
    return 0.0


def _stats_counter_int(stats: ProfileStats, name: str) -> int:
    counter = stats.counters.get(name)
    return 0 if counter is None else int(counter.value)


def _pct(value: float, denominator: float) -> float:
    return 0.0 if denominator == 0.0 else value / denominator * 100.0


def _top_exclusive_zones(zones: dict[str, object], limit: int = 12) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, raw in zones.items():
        if not isinstance(raw, dict):
            continue
        rows.append(
            {
                "zone": name,
                "calls": _expect_int(raw, "calls"),
                "inclusive_seconds": _expect_number(raw, "inclusive_seconds"),
                "exclusive_seconds": _expect_number(raw, "exclusive_seconds"),
            }
        )
    rows.sort(key=lambda row: _object_number(row["exclusive_seconds"]), reverse=True)
    return rows[:limit]


def _bottleneck_summary(stats: ProfileStats, limit: int = 8) -> list[dict[str, object]]:
    generation = _zone_seconds(stats.zones, "self_play.generate_dataset")
    summary: list[dict[str, object]] = []
    rows = sorted(
        stats.zones.items(),
        key=lambda item: item[1].exclusive_ns,
        reverse=True,
    )
    for rank, (zone, aggregate) in enumerate(rows[:limit], start=1):
        summary.append(
            {
                "rank": rank,
                "zone": zone,
                "calls": aggregate.calls,
                "inclusive_seconds": aggregate.inclusive_ns / 1_000_000_000.0,
                "exclusive_seconds": aggregate.exclusive_ns / 1_000_000_000.0,
                "percent_of_generation": _pct(aggregate.exclusive_ns / 1_000_000_000.0, generation),
                "reason": _zone_reason(zone),
            }
        )
    return summary


def _zone_reason(zone: str) -> str:
    if zone.startswith("legal") or zone in {"board.apply_move", "search_state.legal_moves"}:
        return "legal move generation or transition cost"
    if zone.startswith("inference") or zone.startswith("mlx") or zone == "model.forward":
        return "model inference / MLX synchronization cost"
    if zone.startswith("mcts"):
        return "MCTS search phase cost"
    if zone.startswith("dataset") or zone.startswith("record"):
        return "dataset recording or serialization cost"
    if zone.startswith("worker"):
        return "worker/process overhead"
    return "profiled exclusive time"



def _profile_percentages(stats: dict[str, object], elapsed_seconds: float) -> dict[str, float]:
    timers = _expect_mapping(stats, "timers")
    percentages: dict[str, float] = {}
    for name, value in timers.items():
        if isinstance(name, str) and isinstance(value, dict):
            seconds = _expect_number(value, "seconds")
            percentages[name] = _pct(seconds, elapsed_seconds)
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


def _read_batching_settings(metadata: dict[str, object]) -> tuple[str | None, int | None]:
    settings = metadata.get("generation_settings")
    if not isinstance(settings, dict):
        return None, None
    batching_mode = settings.get("batching_mode")
    inference_batch_size = settings.get("inference_batch_size")
    resolved_mode = batching_mode if isinstance(batching_mode, str) else None
    resolved_size = (
        inference_batch_size
        if isinstance(inference_batch_size, int) and not isinstance(inference_batch_size, bool)
        else None
    )
    return resolved_mode, resolved_size


def _aggregate_batching_mode(results: list[RepeatResult]) -> str:
    modes = {result.batching_mode for result in results if result.batching_mode is not None}
    if not modes:
        return "unknown"
    if len(modes) == 1:
        return modes.pop()
    return "mixed"


def _aggregate_inference_batch_size(results: list[RepeatResult]) -> int | None:
    sizes = {
        result.inference_batch_size
        for result in results
        if result.inference_batch_size is not None
    }
    if len(sizes) == 1:
        return sizes.pop()
    return None


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
    return _scriptutil.directory_size(directory)


def _rate(count: int, elapsed_seconds: float) -> float:
    return _scriptutil.rate(count, elapsed_seconds)


def _object_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return 0.0
    return float(value)


def _expect_int(data: dict[str, object], key: str) -> int:
    return _jsonio.expect_int(data, key, label="metadata field")


def _optional_int(data: dict[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"profile field {key!r} must be an integer or null")
    return value


def _expect_number(data: dict[str, object], key: str) -> float:
    return _jsonio.expect_number(data, key, label="profile field")


def _expect_mapping(data: dict[str, object], key: str) -> dict[str, object]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"profile field {key!r} must be an object")
    return dict(value)


def _counter_int(data: dict[str, object], key: str) -> int:
    value = data.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"profile counter {key!r} must be numeric")
    return int(value)


def _remove_empty_directory(directory: Path) -> None:
    try:
        directory.rmdir()
    except OSError:
        return


if __name__ == "__main__":
    raise SystemExit(main())
