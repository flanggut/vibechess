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
    EpochMetrics,
    TrainingConfig,
    train_from_directory,
    train_from_sharded_directory,
    train_model,
)


def _print_epoch_metrics(metrics: EpochMetrics) -> None:
    validation_loss = (
        "n/a" if metrics.validation_loss is None else f"{metrics.validation_loss:.6f}"
    )
    print(
        f"epoch {metrics.epoch}: "
        f"training_loss={metrics.training_loss:.6f} "
        f"validation_loss={validation_loss} "
        f"training_samples={metrics.training_samples} "
        f"validation_samples={metrics.validation_samples}"
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
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="fraction of samples to reserve for validation; 0 disables validation",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="also write checkpoint-step-N every N optimizer steps; 0 disables interim checkpoints",
    )
    parser.add_argument(
        "--metrics-every",
        type=int,
        default=1,
        help="write per-step metrics every N optimizer steps and always on the final step",
    )
    parser.add_argument(
        "--skip-shard-checkpoints",
        action="store_true",
        help=(
            "skip per-shard checkpoint writes for manifest training; "
            "top-level final checkpoint is still written"
        ),
    )
    parser.add_argument(
        "--carry-optimizer-state-across-shards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "carry Adam optimizer state in memory across manifest shards; "
            "use --no-carry-optimizer-state-across-shards to reset per shard; "
            "checkpoint metadata still reports no persisted optimizer state"
        ),
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
        metrics_every=args.metrics_every,
        write_shard_checkpoints=not args.skip_shard_checkpoints,
        carry_optimizer_state_across_shards=args.carry_optimizer_state_across_shards,
        validation_fraction=args.validation_fraction,
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
                epoch_callback=_print_epoch_metrics,
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
                epoch_callback=_print_epoch_metrics,
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
            epoch_callback=_print_epoch_metrics,
        )

    print(
        "training complete: "
        f"steps={result.steps} samples={result.samples} "
        f"training_samples={result.training_samples} "
        f"validation_samples={result.validation_samples} "
        f"loss={result.final_metrics.loss:.6f} "
        f"policy_loss={result.final_metrics.policy_loss:.6f} "
        f"value_loss={result.final_metrics.value_loss:.6f} "
        f"checkpoint={Path(result.checkpoint_dir)} "
        f"metrics={Path(result.metrics_path)}"
    )


if __name__ == "__main__":
    main()
