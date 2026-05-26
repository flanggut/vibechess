#!/usr/bin/env python3
"""Generate a small tinychess neural-MCTS self-play dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from tinychess.ai.neural_mcts import NeuralMCTSConfig
from tinychess.nn.checkpoint import load_checkpoint
from tinychess.nn.model import PolicyValueConfig, PolicyValueInference, PolicyValueNet
from tinychess.nn.self_play import (
    SelfPlayConfig,
    generate_self_play_dataset,
    save_self_play_dataset,
)


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
    args = parser.parse_args()

    checkpoint_id = args.checkpoint_id
    if args.checkpoint is None:
        model_config = PolicyValueConfig(
            residual_channels=args.channels,
            residual_blocks=args.blocks,
            policy_channels=1,
            value_channels=1,
            value_hidden_dim=4,
        )
        inference = PolicyValueInference(PolicyValueNet(model_config))
    else:
        loaded = load_checkpoint(args.checkpoint)
        checkpoint_id = checkpoint_id or str(args.checkpoint)
        inference = PolicyValueInference(loaded.model)

    dataset = generate_self_play_dataset(
        inference,
        config=SelfPlayConfig(
            games=args.games,
            max_plies=args.max_plies,
            mcts=NeuralMCTSConfig(
                simulations=args.simulations,
                temperature=args.temperature,
                seed=args.seed,
            ),
            model_checkpoint_id=checkpoint_id,
            seed=args.seed,
        ),
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
