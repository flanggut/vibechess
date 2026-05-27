#!/usr/bin/env python3
"""Train a tiny MLX policy/value checkpoint from self-play or PGN datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from tinychess.nn.checkpoint import load_checkpoint
from tinychess.nn.model import PolicyValueConfig
from tinychess.nn.pgn_dataset import DEFAULT_MANIFEST_FILENAME
from tinychess.nn.self_play import load_self_play_dataset
from tinychess.nn.train import (
    TrainingConfig,
    train_from_directory,
    train_from_sharded_directory,
    train_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        help="self-play dataset directory or PGN ingestion manifest directory",
    )
    parser.add_argument(
        "--output",
        default="data/checkpoints/train-smoke",
        help="directory for metrics and checkpoint artifacts",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="also write checkpoint-step-N every N optimizer steps; 0 disables interim checkpoints",
    )
    parser.add_argument(
        "--input-checkpoint",
        help="optional existing checkpoint directory to continue model weights/config from",
    )
    parser.add_argument("--residual-channels", type=int, default=8)
    parser.add_argument("--residual-blocks", type=int, default=1)
    parser.add_argument("--policy-channels", type=int, default=2)
    parser.add_argument("--value-channels", type=int, default=1)
    parser.add_argument("--value-hidden-dim", type=int, default=8)
    args = parser.parse_args()

    train_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
    )
    notes = "tinychess WP15 training run"

    dataset_path = Path(args.dataset)
    is_sharded = (dataset_path / DEFAULT_MANIFEST_FILENAME).is_file()

    if args.input_checkpoint:
        loaded = load_checkpoint(args.input_checkpoint)
        if is_sharded:
            result = train_from_sharded_directory(
                dataset_path,
                args.output,
                model=loaded.model,
                config=train_config,
                notes=notes,
                initial_step=loaded.metadata.training_step,
            )
        else:
            dataset = load_self_play_dataset(dataset_path)
            result = train_model(
                dataset,
                args.output,
                model=loaded.model,
                config=train_config,
                notes=notes,
                initial_step=loaded.metadata.training_step,
            )
    else:
        model_config = PolicyValueConfig(
            residual_channels=args.residual_channels,
            residual_blocks=args.residual_blocks,
            policy_channels=args.policy_channels,
            value_channels=args.value_channels,
            value_hidden_dim=args.value_hidden_dim,
        )
        result = train_from_directory(
            args.dataset,
            args.output,
            model_config=model_config,
            config=train_config,
            notes=notes,
        )

    print(
        "training complete: "
        f"steps={result.steps} samples={result.samples} "
        f"loss={result.final_metrics.loss:.6f} "
        f"policy_loss={result.final_metrics.policy_loss:.6f} "
        f"value_loss={result.final_metrics.value_loss:.6f} "
        f"checkpoint={Path(result.checkpoint_dir)} "
        f"metrics={Path(result.metrics_path)}"
    )


if __name__ == "__main__":
    main()
