#!/usr/bin/env python3
"""Generate a small tinychess neural-MCTS self-play dataset."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from tinychess.ai.neural_mcts import NeuralMCTSConfig
from tinychess.nn.checkpoint import load_checkpoint, save_checkpoint
from tinychess.nn.model import PolicyValueConfig, PolicyValueInference, PolicyValueNet
from tinychess.nn.self_play import (
    SelfPlayConfig,
    SelfPlayDataset,
    generate_self_play_dataset,
    merge_self_play_datasets,
    save_self_play_dataset,
)


@dataclass(frozen=True, slots=True)
class GenerationArgs:
    checkpoint: Path | None
    checkpoint_id: str | None
    max_plies: int
    simulations: int
    temperature: float
    seed: int
    channels: int
    blocks: int


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
        model_checkpoint_id=args.checkpoint_id,
        seed=seed,
    )


def _generate_chunk(args: tuple[GenerationArgs, int, int]) -> SelfPlayDataset:
    generation_args, start_game, games = args
    inference = _build_inference(generation_args)
    config = _self_play_config(generation_args, games=games, seed=generation_args.seed + start_game)
    return generate_self_play_dataset(inference, config=config)


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
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of worker processes for parallel self-play generation",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    checkpoint_id = args.checkpoint_id
    if args.checkpoint is not None:
        checkpoint_id = checkpoint_id or str(args.checkpoint)
    generation_args = GenerationArgs(
        checkpoint=args.checkpoint,
        checkpoint_id=checkpoint_id,
        max_plies=args.max_plies,
        simulations=args.simulations,
        temperature=args.temperature,
        seed=args.seed,
        channels=args.channels,
        blocks=args.blocks,
    )
    full_config = _self_play_config(generation_args, games=args.games, seed=args.seed)

    if args.workers == 1 or args.games == 1:
        inference = _build_inference(generation_args)
        dataset = generate_self_play_dataset(inference, config=full_config)
    else:
        workers = min(args.workers, args.games)
        chunks = _split_games(args.games, workers)
        temp_checkpoint: TemporaryDirectory[str] | None = None
        if generation_args.checkpoint is None:
            temp_checkpoint = _temporary_checkpoint(generation_args)
            generation_args = GenerationArgs(
                checkpoint=Path(temp_checkpoint.name),
                checkpoint_id=checkpoint_id,
                max_plies=args.max_plies,
                simulations=args.simulations,
                temperature=args.temperature,
                seed=args.seed,
                channels=args.channels,
                blocks=args.blocks,
            )
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                shards = list(
                    executor.map(
                        _generate_chunk,
                        ((generation_args, start_game, games) for start_game, games in chunks),
                    )
                )
        finally:
            if temp_checkpoint is not None:
                temp_checkpoint.cleanup()
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
            shards,
            config=full_config,
            generation_settings_extra=parallel_settings,
        )
    save_self_play_dataset(dataset, args.output)
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
