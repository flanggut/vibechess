#!/usr/bin/env python3
"""Generate a small tinychess MCTS self-play dataset."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from tinychess.ai.neural_mcts import NeuralMCTSConfig
from tinychess.ai.search_config import MCTSConfig
from tinychess.nn.checkpoint import load_checkpoint, save_checkpoint
from tinychess.nn.model import PolicyValueConfig, PolicyValueInference, PolicyValueNet
from tinychess.nn.self_play import (
    DEFAULT_PROFILE_FILENAME,
    LABEL_SOURCE_CLASSICAL,
    LABEL_SOURCE_NEURAL,
    LABEL_SOURCES,
    SelfPlayConfig,
    SelfPlayDataset,
    generate_self_play_dataset,
    merge_self_play_datasets,
    save_self_play_dataset,
    self_play_profile,
)
from tinychess.nn.self_play_profile import (
    ProfileStats,
    profile_level_from_env,
    profile_scope,
    stats_from_profile_report,
)
from tinychess.nn.self_play_profile import (
    profile_report as build_profile_report,
)

PROFILE_ENV_VAR = "TINYCHESS_SELF_PLAY_PROFILE"


@dataclass(frozen=True, slots=True)
class GenerationArgs:
    checkpoint: Path | None
    checkpoint_id: str | None
    label_source: str
    max_plies: int
    simulations: int
    temperature: float
    classical_exploration: float
    classical_max_rollout_plies: int
    seed: int
    channels: int
    blocks: int
    batch_size: int
    leaf_parallelism: int


@dataclass(frozen=True, slots=True)
class ChunkTask:
    generation_args: GenerationArgs
    worker_id: int
    start_game: int
    games: int
    parent_pool_start_ns: int
    profile_level: str


@dataclass(frozen=True, slots=True)
class ChunkGenerationResult:
    dataset: SelfPlayDataset
    profile: dict[str, object] | None = None


def _build_inference(args: GenerationArgs) -> PolicyValueInference:
    with profile_scope("model.build_inference"):
        if args.checkpoint is None:
            model_config = PolicyValueConfig(
                residual_channels=args.channels,
                residual_blocks=args.blocks,
                policy_channels=1,
                value_channels=1,
                value_hidden_dim=4,
            )
            return PolicyValueInference(PolicyValueNet(model_config))

        loaded = load_checkpoint(args.checkpoint)
        return PolicyValueInference(loaded.model)


def _self_play_config(args: GenerationArgs, *, games: int, seed: int) -> SelfPlayConfig:
    return SelfPlayConfig(
        games=games,
        max_plies=args.max_plies,
        mcts=NeuralMCTSConfig(
            simulations=args.simulations,
            temperature=args.temperature,
            seed=seed,
            leaf_parallelism=args.leaf_parallelism,
        ),
        classical_mcts=MCTSConfig(
            simulations=args.simulations,
            exploration=args.classical_exploration,
            max_rollout_plies=args.classical_max_rollout_plies,
            seed=seed,
        ),
        label_source=args.label_source,
        model_checkpoint_id=args.checkpoint_id,
        seed=seed,
        batch_size=args.batch_size,
    )


def _generate_chunk(task: ChunkTask) -> ChunkGenerationResult:
    metadata = {
        "worker_id": task.worker_id,
        "start_game": task.start_game,
        "games": task.games,
        "seed": task.generation_args.seed + task.start_game,
    }
    with self_play_profile(task.profile_level) as profiler:
        if profiler.enabled:
            profiler.stats.metadata.update(metadata)
            profiler.stats.add_counter(
                "worker.start_lag_ns",
                time.perf_counter_ns() - task.parent_pool_start_ns,
            )
        with profile_scope("worker.chunk_elapsed", **metadata):
            inference = None
            if task.generation_args.label_source == LABEL_SOURCE_NEURAL:
                inference = _build_inference(task.generation_args)
            config = _self_play_config(
                task.generation_args,
                games=task.games,
                seed=task.generation_args.seed + task.start_game,
            )
            dataset = generate_self_play_dataset(inference, config=config)
        if not profiler.enabled:
            return ChunkGenerationResult(dataset)
        profile = build_profile_report(
            profiler.stats,
            scope="self_play_worker",
            profile_level=profiler.level,
            metadata={
                **metadata,
                "pid": os.getpid(),
                "samples": dataset.metadata.sample_count,
                "plies": sum(record.plies for record in dataset.games),
            },
        )
        return ChunkGenerationResult(dataset, profile)


def _profile_level() -> str:
    return profile_level_from_env(PROFILE_ENV_VAR)


def _profile_from_report(report: dict[str, object]) -> ProfileStats:
    return stats_from_profile_report(report)


def _write_profile(
    output: Path,
    profile: ProfileStats,
    *,
    profile_level: str,
    worker_profiles: list[dict[str, object]] | None = None,
    derived: dict[str, object] | None = None,
) -> None:
    with profile_scope("profile.write_sidecar"):
        report = build_profile_report(
            profile,
            scope="self_play_generation",
            profile_level=profile_level,  # type: ignore[arg-type]
            worker_profiles=worker_profiles,
            derived=derived,
        )
        (output / DEFAULT_PROFILE_FILENAME).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )


def _temporary_checkpoint(args: GenerationArgs) -> TemporaryDirectory[str]:
    with profile_scope("self_play.temp_checkpoint_save"):
        temp_dir = TemporaryDirectory(prefix="tinychess-self-play-")
        model_config = PolicyValueConfig(
            residual_channels=args.channels,
            residual_blocks=args.blocks,
            policy_channels=1,
            value_channels=1,
            value_hidden_dim=4,
        )
        save_checkpoint(PolicyValueNet(model_config), temp_dir.name)
        return temp_dir


def _split_games(games: int, workers: int) -> list[tuple[int, int]]:
    chunks: list[tuple[int, int]] = []
    start = 0
    for worker_index in range(workers):
        count = games // workers + (1 if worker_index < games % workers else 0)
        if count > 0:
            chunks.append((start, count))
            start += count
    return chunks


def _worker_derived_profile(worker_reports: list[dict[str, object]]) -> dict[str, object]:
    worker_elapsed: list[float] = []
    for report in worker_reports:
        stats = stats_from_profile_report(report)
        chunk_timer = stats.zones.get("worker.chunk_elapsed")
        if chunk_timer is not None:
            worker_elapsed.append(chunk_timer.inclusive_ns / 1_000_000_000.0)
    derived: dict[str, object] = {"worker_count": len(worker_reports)}
    if worker_elapsed:
        derived.update(
            {
                "worker_time_max_seconds": max(worker_elapsed),
                "worker_time_sum_seconds": sum(worker_elapsed),
            }
        )
    return derived


def _directory_size(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate tinychess self-play samples.")
    parser.add_argument("--output", type=Path, default=Path("data/selfplay/smoke"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-id", default=None)
    parser.add_argument(
        "--label-source",
        choices=LABEL_SOURCES,
        default=LABEL_SOURCE_NEURAL,
        help="search source for policy labels and self-play moves",
    )
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--classical-exploration", type=float, default=1.41421356237)
    parser.add_argument(
        "--classical-max-rollout-plies",
        type=int,
        default=0,
        help="classical MCTS rollout cap; default 0 uses static leaf evaluation",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="neural self-play root inference batch size; default 1 preserves serial behavior",
    )
    parser.add_argument(
        "--leaf-parallelism",
        type=int,
        default=1,
        help=(
            "opt-in leaf-parallel neural MCTS batch width within one game; "
            "default 1 preserves serial search semantics"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of worker processes for parallel self-play generation",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.leaf_parallelism < 1:
        parser.error("--leaf-parallelism must be at least 1")

    if args.label_source == LABEL_SOURCE_CLASSICAL:
        if args.checkpoint is not None:
            parser.error("--checkpoint is only supported with --label-source neural")
        if args.checkpoint_id is not None:
            parser.error("--checkpoint-id is only supported with --label-source neural")

    checkpoint_id = args.checkpoint_id
    if args.checkpoint is not None:
        checkpoint_id = checkpoint_id or str(args.checkpoint)
    generation_args = GenerationArgs(
        checkpoint=args.checkpoint,
        checkpoint_id=checkpoint_id,
        label_source=args.label_source,
        max_plies=args.max_plies,
        simulations=args.simulations,
        temperature=args.temperature,
        classical_exploration=args.classical_exploration,
        classical_max_rollout_plies=args.classical_max_rollout_plies,
        seed=args.seed,
        channels=args.channels,
        blocks=args.blocks,
        batch_size=args.batch_size,
        leaf_parallelism=args.leaf_parallelism,
    )
    full_config = _self_play_config(generation_args, games=args.games, seed=args.seed)

    profile_level = _profile_level()
    worker_reports: list[dict[str, object]] = []
    derived_profile: dict[str, object] = {}
    with self_play_profile(profile_level) as main_profiler, profile_scope("self_play.main"):
        with profile_scope("self_play.setup"):
            pass
        chunk_results: list[ChunkGenerationResult] = []
        if args.workers == 1 or args.games == 1:
            inference = None
            if generation_args.label_source == LABEL_SOURCE_NEURAL:
                inference = _build_inference(generation_args)
            dataset = generate_self_play_dataset(inference, config=full_config)
        else:
            workers = min(args.workers, args.games)
            chunks = _split_games(args.games, workers)
            temp_checkpoint: TemporaryDirectory[str] | None = None
            if (
                generation_args.label_source == LABEL_SOURCE_NEURAL
                and generation_args.checkpoint is None
            ):
                temp_checkpoint = _temporary_checkpoint(generation_args)
                generation_args = replace(
                    generation_args,
                    checkpoint=Path(temp_checkpoint.name),
                )
            pool_start_ns = time.perf_counter_ns()
            try:
                with (
                    profile_scope("worker.pool_elapsed", workers=workers),
                    ProcessPoolExecutor(max_workers=workers) as executor,
                ):
                    chunk_results = list(
                        executor.map(
                            _generate_chunk,
                            (
                                ChunkTask(
                                    generation_args=generation_args,
                                    worker_id=worker_id,
                                    start_game=start_game,
                                    games=games,
                                    parent_pool_start_ns=pool_start_ns,
                                    profile_level=profile_level,
                                )
                                for worker_id, (start_game, games) in enumerate(chunks)
                            ),
                        )
                    )
            finally:
                if temp_checkpoint is not None:
                    temp_checkpoint.cleanup()
            worker_reports = [
                result.profile for result in chunk_results if result.profile is not None
            ]
            worker_stats = [_profile_from_report(profile) for profile in worker_reports]
            if main_profiler.enabled:
                for stats in worker_stats:
                    main_profiler.stats.merge(stats)
                derived_profile.update(_worker_derived_profile(worker_reports))
            parallel_settings: dict[str, object] = {
                "parallel": {
                    "workers": workers,
                    "chunks": [
                        {
                            "start_game": start_game,
                            "games": games,
                            "seed": args.seed + start_game,
                        }
                        for start_game, games in chunks
                    ],
                }
            }
            dataset = merge_self_play_datasets(
                [result.dataset for result in chunk_results],
                config=full_config,
                generation_settings_extra=parallel_settings,
            )
        save_self_play_dataset(dataset, args.output)
        if main_profiler.enabled:
            main_profiler.stats.add_counter(
                "dataset.output_bytes",
                _directory_size(args.output),
            )
            _write_profile(
                args.output,
                main_profiler.stats,
                profile_level=main_profiler.level,
                worker_profiles=worker_reports,
                derived=derived_profile,
            )
    print(
        " ".join(
            [
                f"output={args.output}",
                f"games={dataset.metadata.game_count}",
                f"samples={dataset.metadata.sample_count}",
                f"schema={dataset.metadata.schema_version}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
