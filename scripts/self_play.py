#!/usr/bin/env python3
"""Generate a small tinychess MCTS self-play dataset."""

from __future__ import annotations

import argparse
import json
import os
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
    SelfPlayProfileStats,
    generate_self_play_dataset,
    merge_self_play_datasets,
    save_self_play_dataset,
    self_play_profile,
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


@dataclass(frozen=True, slots=True)
class ChunkGenerationResult:
    dataset: SelfPlayDataset
    profile: dict[str, object] | None = None


def _build_inference(args: GenerationArgs) -> PolicyValueInference:
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


def _generate_chunk(args: tuple[GenerationArgs, int, int]) -> ChunkGenerationResult:
    generation_args, start_game, games = args
    inference = None
    if generation_args.label_source == LABEL_SOURCE_NEURAL:
        inference = _build_inference(generation_args)
    config = _self_play_config(generation_args, games=games, seed=generation_args.seed + start_game)
    return _generate_dataset_with_optional_profile(inference, config)


def _generate_dataset_with_optional_profile(
    inference: PolicyValueInference | None,
    config: SelfPlayConfig,
) -> ChunkGenerationResult:
    if not _profiling_enabled():
        return ChunkGenerationResult(generate_self_play_dataset(inference, config=config))
    with self_play_profile() as profile:
        dataset = generate_self_play_dataset(inference, config=config)
    return ChunkGenerationResult(dataset, profile.to_dict())


def _profiling_enabled() -> bool:
    return os.environ.get(PROFILE_ENV_VAR) == "1"


def _profile_from_report(report: dict[str, object]) -> SelfPlayProfileStats:
    return SelfPlayProfileStats.from_dict(report)


def _profile_report(
    profile: SelfPlayProfileStats,
    *,
    worker_profiles: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "scope": "self_play_generation",
        "limitations": [
            "Counters are collected by benchmark-only in-process monkeypatching.",
            "Timer categories are diagnostic and can overlap; percentages in the benchmark "
            "report should not be summed as exclusive CPU time.",
        ],
        "stats": profile.to_dict(),
        "worker_profiles": worker_profiles or [],
    }


def _write_profile(
    output: Path,
    profile: SelfPlayProfileStats,
    *,
    worker_profiles: list[dict[str, object]] | None = None,
) -> None:
    report = _profile_report(profile, worker_profiles=worker_profiles)
    (output / DEFAULT_PROFILE_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


def _temporary_checkpoint(args: GenerationArgs) -> TemporaryDirectory[str]:
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
    )
    full_config = _self_play_config(generation_args, games=args.games, seed=args.seed)

    profile_report: dict[str, object] | None = None
    chunk_results: list[ChunkGenerationResult] = []
    if args.workers == 1 or args.games == 1:
        inference = None
        if generation_args.label_source == LABEL_SOURCE_NEURAL:
            inference = _build_inference(generation_args)
        result = _generate_dataset_with_optional_profile(inference, full_config)
        dataset = result.dataset
        profile_report = result.profile
    else:
        workers = min(args.workers, args.games)
        chunks = _split_games(args.games, workers)
        temp_checkpoint: TemporaryDirectory[str] | None = None
        if (
            generation_args.label_source == LABEL_SOURCE_NEURAL
            and generation_args.checkpoint is None
        ):
            temp_checkpoint = _temporary_checkpoint(generation_args)
            generation_args = replace(generation_args, checkpoint=Path(temp_checkpoint.name))
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                chunk_results = list(
                    executor.map(
                        _generate_chunk,
                        ((generation_args, start_game, games) for start_game, games in chunks),
                    )
                )
        finally:
            if temp_checkpoint is not None:
                temp_checkpoint.cleanup()
        worker_profiles = [
            result.profile for result in chunk_results if result.profile is not None
        ]
        if worker_profiles:
            profile_report = SelfPlayProfileStats.merge(
                [_profile_from_report(profile) for profile in worker_profiles]
            ).to_dict()
        parallel_settings: dict[str, object] = {
            "parallel": {
                "workers": workers,
                "chunks": [
                    {"start_game": start_game, "games": games, "seed": args.seed + start_game}
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
    if profile_report is not None:
        profile = _profile_from_report(profile_report)
        worker_reports = None
        if chunk_results:
            worker_reports = [
                result.profile for result in chunk_results if result.profile is not None
            ]
        _write_profile(args.output, profile, worker_profiles=worker_reports)
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
