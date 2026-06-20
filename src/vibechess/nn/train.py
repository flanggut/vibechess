"""Small MLX training loop for self-play policy/value datasets."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import mlx.core as mx
import mlx.nn as _nn
import mlx.optimizers as optim
import numpy as np
import numpy.typing as npt

from vibechess.nn.checkpoint import CheckpointMetadata, save_checkpoint
from vibechess.nn.encode import ACTION_SPACE_SIZE, TENSOR_SHAPE
from vibechess.nn.model import PolicyValueConfig, PolicyValueNet

if TYPE_CHECKING:
    from vibechess.nn.self_play_dataset import SelfPlayDataset

MLXArray: TypeAlias = Any
nn: Any = _nn

DEFAULT_EPOCH_METRICS_FILENAME = "epoch_metrics.jsonl"


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    """Policy/value training losses for one batch."""

    total: MLXArray
    policy: MLXArray
    value: MLXArray


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Smoke-friendly settings for local MLX training."""

    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 1.0e-3
    warmup_steps: int = 0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    seed: int | None = 0
    checkpoint_every: int = 0
    write_shard_checkpoints: bool = True
    carry_optimizer_state_across_shards: bool = True
    validation_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.policy_loss_weight < 0.0:
            raise ValueError("policy_loss_weight must be non-negative")
        if self.value_loss_weight < 0.0:
            raise ValueError("value_loss_weight must be non-negative")
        if self.policy_loss_weight == 0.0 and self.value_loss_weight == 0.0:
            raise ValueError("at least one loss weight must be positive")
        if self.checkpoint_every < 0:
            raise ValueError("checkpoint_every must be non-negative")
        if self.validation_fraction < 0.0 or self.validation_fraction >= 1.0:
            raise ValueError("validation_fraction must be in [0.0, 1.0)")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable training settings."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Serializable average training/validation metrics for one epoch."""

    epoch: int
    training_samples: int
    validation_samples: int
    training_loss: float
    training_policy_loss: float
    training_value_loss: float
    validation_loss: float | None
    validation_policy_loss: float | None
    validation_value_loss: float | None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable epoch metrics."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Summary of a training run."""

    steps: int
    samples: int
    training_samples: int
    validation_samples: int
    final_training_step: int
    epoch_metrics: tuple[EpochMetrics, ...]
    checkpoint_dir: Path
    epoch_metrics_path: Path


def compute_policy_value_loss(
    model: PolicyValueNet,
    positions: MLXArray,
    legal_masks: MLXArray,
    policy_targets: MLXArray,
    value_targets: MLXArray,
    *,
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = 1.0,
) -> LossBreakdown:
    """Compute masked cross-entropy policy loss plus MSE value loss.

    ``policy_targets`` are expected to be normalized over legal actions. Illegal
    logits are excluded from the policy softmax by ``legal_masks``.
    """
    output = model(positions)
    masked_logits = mx.where(
        legal_masks > 0.0,
        output.policy_logits,
        mx.full(output.policy_logits.shape, -1.0e9, dtype=output.policy_logits.dtype),
    )
    log_probs = masked_logits - mx.logsumexp(masked_logits, axis=1, keepdims=True)
    policy_loss = -mx.mean(mx.sum(policy_targets * log_probs, axis=1))
    value_loss = mx.mean(mx.square(output.value - value_targets))
    total = policy_loss_weight * policy_loss + value_loss_weight * value_loss
    return LossBreakdown(total=total, policy=policy_loss, value=value_loss)


def train_model(
    dataset: SelfPlayDataset,
    output_dir: str | Path,
    *,
    model: PolicyValueNet | None = None,
    config: TrainingConfig | None = None,
    notes: str | None = None,
    initial_step: int = 0,
    epoch_callback: Callable[[EpochMetrics], None] | None = None,
) -> TrainingResult:
    """Train a policy/value model on a loaded self-play dataset and save outputs."""
    return _train_loaded_dataset(
        dataset,
        output_dir,
        model=model,
        config=config,
        notes=notes,
        initial_step=initial_step,
        epoch_callback=epoch_callback,
        write_checkpoints=True,
    )


def _train_loaded_dataset(
    dataset: SelfPlayDataset,
    output_dir: str | Path,
    *,
    model: PolicyValueNet | None = None,
    config: TrainingConfig | None = None,
    notes: str | None = None,
    initial_step: int = 0,
    epoch_callback: Callable[[EpochMetrics], None] | None = None,
    write_checkpoints: bool = True,
    optimizer: Any | None = None,
    optimizer_step_offset: int = 0,
) -> TrainingResult:
    """Train a loaded dataset with optional checkpoint writes for internal sharded use."""
    if initial_step < 0:
        raise ValueError("initial_step must be non-negative")
    if optimizer_step_offset < 0:
        raise ValueError("optimizer_step_offset must be non-negative")
    resolved_config = TrainingConfig() if config is None else config
    _validate_dataset_has_samples(dataset)
    train_dir = Path(output_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "metrics.jsonl").unlink(missing_ok=True)
    epoch_metrics_path = train_dir / DEFAULT_EPOCH_METRICS_FILENAME
    epoch_metrics_path.write_text("")

    resolved_model = PolicyValueNet() if model is None else model
    resolved_optimizer = (
        optim.Adam(learning_rate=resolved_config.learning_rate) if optimizer is None else optimizer
    )
    rng = np.random.default_rng(resolved_config.seed)
    train_indices, validation_indices = _split_train_validation_indices(
        dataset.metadata.sample_count,
        validation_fraction=resolved_config.validation_fraction,
        rng=rng,
    )
    run_step = 0
    epoch_metric_values: list[EpochMetrics] = []

    for epoch in range(1, resolved_config.epochs + 1):
        order = np.array(train_indices, copy=True)
        rng.shuffle(order)
        batches = _batch_indices(order, resolved_config.batch_size)
        for indices in batches:
            batch = _Batch.from_dataset(dataset, indices)
            loss_value, gradients = nn.value_and_grad(resolved_model, _loss_for_grad)(
                resolved_model,
                batch.positions,
                batch.legal_masks,
                batch.policy_targets,
                batch.value_targets,
                resolved_config.policy_loss_weight,
                resolved_config.value_loss_weight,
            )
            _set_optimizer_learning_rate(
                resolved_optimizer,
                _learning_rate_for_step(
                    resolved_config.learning_rate,
                    warmup_steps=resolved_config.warmup_steps,
                    optimizer_step=optimizer_step_offset + run_step + 1,
                ),
            )
            resolved_optimizer.update(resolved_model, gradients)
            mx.eval(resolved_model.parameters(), resolved_optimizer.state, loss_value)

            run_step += 1
            current_step = initial_step + run_step
            checkpoint_due = (
                write_checkpoints
                and resolved_config.checkpoint_every
                and run_step % resolved_config.checkpoint_every == 0
            )
            if checkpoint_due:
                _save_training_checkpoint(
                    resolved_model,
                    train_dir / f"checkpoint-step-{current_step}",
                    step=current_step,
                    notes=notes,
                )

        train_loss = _evaluate_loss(
            resolved_model,
            dataset,
            train_indices,
            batch_size=resolved_config.batch_size,
            config=resolved_config,
        )
        validation_loss = (
            _evaluate_loss(
                resolved_model,
                dataset,
                validation_indices,
                batch_size=resolved_config.batch_size,
                config=resolved_config,
            )
            if len(validation_indices) > 0
            else None
        )
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            training_samples=len(train_indices),
            validation_samples=len(validation_indices),
            training_loss=train_loss.loss,
            training_policy_loss=train_loss.policy_loss,
            training_value_loss=train_loss.value_loss,
            validation_loss=None if validation_loss is None else validation_loss.loss,
            validation_policy_loss=None if validation_loss is None else validation_loss.policy_loss,
            validation_value_loss=None if validation_loss is None else validation_loss.value_loss,
        )
        epoch_metric_values.append(epoch_metrics)
        _append_epoch_metrics(epoch_metrics_path, epoch_metrics)
        if epoch_callback is not None:
            epoch_callback(epoch_metrics)

    final_training_step = initial_step + run_step
    checkpoint_dir = train_dir / "checkpoint-final"
    if write_checkpoints:
        _save_training_checkpoint(
            resolved_model,
            checkpoint_dir,
            step=final_training_step,
            notes=notes,
        )
    (train_dir / "training.json").write_text(
        json.dumps(
            {
                "training_config": resolved_config.to_dict(),
                "initial_training_step": initial_step,
                "dataset_metadata": dataset.metadata.to_dict(),
                "training_samples": len(train_indices),
                "validation_samples": len(validation_indices),
                "final_training_step": final_training_step,
                "epoch_metrics": [metrics.to_dict() for metrics in epoch_metric_values],
                "checkpoint_dir": str(checkpoint_dir),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return TrainingResult(
        steps=run_step,
        samples=dataset.metadata.sample_count,
        training_samples=len(train_indices),
        validation_samples=len(validation_indices),
        final_training_step=final_training_step,
        epoch_metrics=tuple(epoch_metric_values),
        checkpoint_dir=checkpoint_dir,
        epoch_metrics_path=epoch_metrics_path,
    )


def train_from_directory(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    model_config: PolicyValueConfig | None = None,
    config: TrainingConfig | None = None,
    notes: str | None = None,
    epoch_callback: Callable[[EpochMetrics], None] | None = None,
) -> TrainingResult:
    """Load a dataset directory or PGN shard manifest and train a model."""
    from vibechess.nn.pgn_dataset import DEFAULT_MANIFEST_FILENAME
    from vibechess.nn.self_play_dataset import load_self_play_dataset

    input_dir = Path(dataset_dir)
    if (input_dir / DEFAULT_MANIFEST_FILENAME).is_file():
        return train_from_sharded_directory(
            input_dir,
            output_dir,
            model_config=model_config,
            config=config,
            notes=notes,
            epoch_callback=epoch_callback,
        )

    model = PolicyValueNet(model_config)
    return train_model(
        load_self_play_dataset(input_dir),
        output_dir,
        model=model,
        config=config,
        notes=notes,
        epoch_callback=epoch_callback,
    )


def train_from_sharded_directory(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    model_config: PolicyValueConfig | None = None,
    model: PolicyValueNet | None = None,
    config: TrainingConfig | None = None,
    notes: str | None = None,
    initial_step: int = 0,
    epoch_callback: Callable[[EpochMetrics], None] | None = None,
) -> TrainingResult:
    """Train over PGN ingestion shards one shard at a time to bound memory use.

    Adam optimizer state is carried in memory across shards by default for this
    process only, unless configured to reset per shard. The current checkpoint
    format persists model weights only. Training step and model weights are
    carried forward, and per-shard epoch metrics are aggregated into the
    top-level ``epoch_metrics.jsonl``.
    """
    from vibechess.nn.pgn_dataset import shard_directories
    from vibechess.nn.self_play_dataset import load_self_play_dataset

    if initial_step < 0:
        raise ValueError("initial_step must be non-negative")
    resolved_config = TrainingConfig() if config is None else config
    train_dir = Path(output_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "metrics.jsonl").unlink(missing_ok=True)
    epoch_metrics_path = train_dir / DEFAULT_EPOCH_METRICS_FILENAME
    epoch_metrics_path.write_text("")

    model = PolicyValueNet(model_config) if model is None else model
    total_steps = initial_step
    total_samples = 0
    total_training_samples = 0
    total_validation_samples = 0
    epoch_metric_values: list[EpochMetrics] = []
    shard_summaries: list[dict[str, object]] = []
    shards = shard_directories(dataset_dir)
    if not shards:
        raise ValueError("sharded training requires at least one shard")
    shared_optimizer = (
        optim.Adam(learning_rate=resolved_config.learning_rate)
        if resolved_config.carry_optimizer_state_across_shards
        else None
    )

    for shard_index, shard_dir in enumerate(shards):
        shard_output = train_dir / f"shard-train-{shard_index:05d}"
        result = _train_loaded_dataset(
            load_self_play_dataset(shard_dir),
            shard_output,
            model=model,
            config=resolved_config,
            notes=notes,
            initial_step=total_steps,
            epoch_callback=epoch_callback,
            write_checkpoints=resolved_config.write_shard_checkpoints,
            optimizer=shared_optimizer,
            optimizer_step_offset=total_steps - initial_step,
        )
        total_steps += result.steps
        total_samples += result.samples
        total_training_samples += result.training_samples
        total_validation_samples += result.validation_samples
        epoch_metric_values.extend(result.epoch_metrics)
        _append_file(result.epoch_metrics_path, epoch_metrics_path)
        shard_summaries.append(
            {
                "dataset_shard": str(shard_dir),
                "training_output": str(shard_output),
                "steps": result.steps,
                "samples": result.samples,
                "training_samples": result.training_samples,
                "validation_samples": result.validation_samples,
                "final_step": result.final_training_step,
                "checkpoint_written": resolved_config.write_shard_checkpoints,
            }
        )

    checkpoint_dir = train_dir / "checkpoint-final"
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    if resolved_config.write_shard_checkpoints:
        shutil.copytree(shard_output / "checkpoint-final", checkpoint_dir)
    else:
        _save_training_checkpoint(model, checkpoint_dir, step=total_steps, notes=notes)
    (train_dir / "training.json").write_text(
        json.dumps(
            {
                "training_config": resolved_config.to_dict(),
                "initial_training_step": initial_step,
                "dataset_manifest_dir": str(Path(dataset_dir)),
                "sharded": True,
                "shards": shard_summaries,
                "training_samples": total_training_samples,
                "validation_samples": total_validation_samples,
                "final_training_step": total_steps,
                "epoch_metrics": [metrics.to_dict() for metrics in epoch_metric_values],
                "checkpoint_dir": str(checkpoint_dir),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return TrainingResult(
        steps=total_steps - initial_step,
        samples=total_samples,
        training_samples=total_training_samples,
        validation_samples=total_validation_samples,
        final_training_step=total_steps,
        epoch_metrics=tuple(epoch_metric_values),
        checkpoint_dir=checkpoint_dir,
        epoch_metrics_path=epoch_metrics_path,
    )


@dataclass(frozen=True, slots=True)
class _LossSummary:
    loss: float
    policy_loss: float
    value_loss: float


def _split_train_validation_indices(
    sample_count: int,
    *,
    validation_fraction: float,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    indices = np.arange(sample_count, dtype=np.int64)
    if sample_count < 2 or validation_fraction == 0.0:
        return indices, np.array([], dtype=np.int64)

    shuffled = np.array(indices, copy=True)
    rng.shuffle(shuffled)
    validation_count = int(sample_count * validation_fraction + 0.5)
    validation_count = max(1, min(validation_count, sample_count - 1))
    return shuffled[validation_count:], shuffled[:validation_count]


def _evaluate_loss(
    model: PolicyValueNet,
    dataset: SelfPlayDataset,
    indices: npt.NDArray[np.int64],
    *,
    batch_size: int,
    config: TrainingConfig,
) -> _LossSummary:
    if len(indices) < 1:
        raise ValueError("loss evaluation requires at least one sample")

    weighted_loss = 0.0
    weighted_policy_loss = 0.0
    weighted_value_loss = 0.0
    total_samples = 0
    for batch_indices in _batch_indices(indices, batch_size):
        batch = _Batch.from_dataset(dataset, batch_indices)
        losses = compute_policy_value_loss(
            model,
            batch.positions,
            batch.legal_masks,
            batch.policy_targets,
            batch.value_targets,
            policy_loss_weight=config.policy_loss_weight,
            value_loss_weight=config.value_loss_weight,
        )
        mx.eval(losses.total, losses.policy, losses.value)
        samples = len(batch_indices)
        weighted_loss += _scalar(losses.total) * samples
        weighted_policy_loss += _scalar(losses.policy) * samples
        weighted_value_loss += _scalar(losses.value) * samples
        total_samples += samples

    return _LossSummary(
        loss=weighted_loss / total_samples,
        policy_loss=weighted_policy_loss / total_samples,
        value_loss=weighted_value_loss / total_samples,
    )


def _learning_rate_for_step(
    learning_rate: float,
    *,
    warmup_steps: int,
    optimizer_step: int,
) -> float:
    if warmup_steps <= 0 or optimizer_step >= warmup_steps:
        return learning_rate
    return learning_rate * optimizer_step / warmup_steps


def _set_optimizer_learning_rate(optimizer: Any, learning_rate: float) -> None:
    optimizer.learning_rate = learning_rate


@dataclass(frozen=True, slots=True)
class _Batch:
    positions: MLXArray
    legal_masks: MLXArray
    policy_targets: MLXArray
    value_targets: MLXArray

    @classmethod
    def from_dataset(cls, dataset: SelfPlayDataset, indices: npt.NDArray[np.int64]) -> _Batch:
        return cls(
            positions=mx.array(dataset.positions[indices], dtype=mx.float32),
            legal_masks=mx.array(dataset.legal_masks[indices], dtype=mx.float32),
            policy_targets=mx.array(dataset.policy_targets.dense_rows(indices), dtype=mx.float32),
            value_targets=mx.array(dataset.outcomes[indices], dtype=mx.float32),
        )


def _loss_for_grad(
    model: PolicyValueNet,
    positions: MLXArray,
    legal_masks: MLXArray,
    policy_targets: MLXArray,
    value_targets: MLXArray,
    policy_loss_weight: float,
    value_loss_weight: float,
) -> MLXArray:
    return compute_policy_value_loss(
        model,
        positions,
        legal_masks,
        policy_targets,
        value_targets,
        policy_loss_weight=policy_loss_weight,
        value_loss_weight=value_loss_weight,
    ).total


def _batch_indices(
    indices: npt.NDArray[np.int64],
    batch_size: int,
) -> Iterator[npt.NDArray[np.int64]]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]



def _validate_dataset_has_samples(dataset: SelfPlayDataset) -> None:
    if dataset.metadata.sample_count < 1:
        raise ValueError("training requires at least one self-play sample")
    if dataset.positions.shape != (dataset.metadata.sample_count, *TENSOR_SHAPE):
        raise ValueError("dataset positions shape does not match metadata")
    if dataset.legal_masks.shape != (dataset.metadata.sample_count, ACTION_SPACE_SIZE):
        raise ValueError("dataset legal_masks shape does not match metadata")
    if dataset.policy_targets.sample_count != dataset.metadata.sample_count:
        raise ValueError("dataset mcts_policies shape does not match metadata")
    if dataset.outcomes.shape != (dataset.metadata.sample_count,):
        raise ValueError("dataset outcomes shape does not match metadata")
    if not np.isfinite(dataset.positions).all():
        raise ValueError("dataset positions must be finite")
    if not np.isfinite(dataset.legal_masks).all():
        raise ValueError("dataset legal_masks must be finite")
    _validate_policy_targets(dataset)
    if not np.isfinite(dataset.outcomes).all():
        raise ValueError("dataset outcomes must be finite")
    if not np.all((dataset.legal_masks == 0.0) | (dataset.legal_masks == 1.0)):
        raise ValueError("dataset legal_masks must be binary")
    if np.any(dataset.outcomes < -1.0) or np.any(dataset.outcomes > 1.0):
        raise ValueError("dataset outcomes must be in [-1, 1]")


def _validate_policy_targets(dataset: SelfPlayDataset) -> None:
    for row_index in range(dataset.metadata.sample_count):
        indices, probabilities = dataset.policy_targets.row(row_index)
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("dataset mcts_policies must be finite")
        if np.any(probabilities < 0.0):
            raise ValueError("dataset mcts_policies must be non-negative")
        if np.any(indices < 0) or np.any(indices >= ACTION_SPACE_SIZE):
            raise ValueError("dataset mcts_policies action index out of range")
        if np.unique(indices).shape[0] != indices.shape[0]:
            raise ValueError("dataset mcts_policies contains duplicate action indices")
        if indices.size and np.any(dataset.legal_masks[row_index, indices] <= 0.0):
            raise ValueError("dataset mcts_policies must put probability only on legal actions")
        if not np.isclose(float(probabilities.sum()), 1.0, rtol=1.0e-5, atol=1.0e-6):
            raise ValueError("dataset mcts_policies rows must sum to 1")


def _save_training_checkpoint(
    model: PolicyValueNet,
    directory: Path,
    *,
    step: int,
    notes: str | None,
) -> None:
    save_checkpoint(
        model,
        directory,
        metadata=CheckpointMetadata.initial(
            model.config,
            training_step=step,
            optimizer_state_available=False,
            notes=notes,
        ),
    )


def _append_epoch_metrics(path: Path, metrics: EpochMetrics) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(metrics.to_dict(), sort_keys=True) + "\n")


def _append_file(source: Path, destination: Path) -> None:
    with source.open("rb") as source_handle, destination.open("ab") as destination_handle:
        shutil.copyfileobj(source_handle, destination_handle)


def _scalar(value: MLXArray) -> float:
    return float(value.item())


__all__ = [
    "DEFAULT_EPOCH_METRICS_FILENAME",
    "EpochMetrics",
    "LossBreakdown",
    "TrainingConfig",
    "TrainingResult",
    "compute_policy_value_loss",
    "train_from_directory",
    "train_from_sharded_directory",
    "train_model",
]
