from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from vibechess.engine import Game
from vibechess.engine.outcome import OutcomeReason
from vibechess.nn.checkpoint import (
    DEFAULT_METADATA_FILENAME,
    DEFAULT_WEIGHTS_FILENAME,
    load_checkpoint,
    load_checkpoint_metadata,
)
from vibechess.nn.encode import (
    ACTION_SPACE_SIZE,
    encode_game,
    legal_move_mask,
    move_to_action_index,
)
from vibechess.nn.model import PolicyValueConfig, PolicyValueNet
from vibechess.nn.self_play import SelfPlayConfig
from vibechess.nn.self_play_dataset import (
    SelfPlayDataset,
    SelfPlayGameRecord,
    SelfPlayMetadata,
    load_self_play_dataset,
    save_self_play_dataset,
)
from vibechess.nn.train import (
    DEFAULT_EPOCH_METRICS_FILENAME,
    EpochMetrics,
    TrainingConfig,
    _learning_rate_for_step,
    compute_policy_value_loss,
    train_model,
)


def tiny_config() -> PolicyValueConfig:
    return PolicyValueConfig(
        residual_channels=8,
        residual_blocks=1,
        policy_channels=2,
        value_channels=1,
        value_hidden_dim=8,
    )


def tiny_dataset(sample_count: int = 2) -> SelfPlayDataset:
    game = Game.new()
    move = game.legal_moves[0]
    policy = np.zeros((ACTION_SPACE_SIZE,), dtype=np.float32)
    policy[move_to_action_index(move, game.board)] = 1.0
    next_game = game.play(move)
    positions = np.stack(
        [np.asarray(encode_game(game), dtype=np.float32) for _ in range(sample_count)]
    )
    masks = np.stack(
        [np.asarray(legal_move_mask(game), dtype=np.float32) for _ in range(sample_count)]
    )
    policies = np.stack([policy for _ in range(sample_count)])
    outcomes = np.zeros((sample_count,), dtype=np.float32)
    metadata = SelfPlayMetadata.create(
        SelfPlayConfig(games=1, max_plies=1),
        sample_count=sample_count,
    )
    return SelfPlayDataset(
        positions=positions,
        legal_masks=masks,
        mcts_policies=policies,
        outcomes=outcomes,
        metadata=metadata,
        games=[
            SelfPlayGameRecord(
                game_index=0,
                plies=1,
                outcome_reason=OutcomeReason.MAX_PLIES.value,
                winner=None,
                final_fen=next_game.to_fen(),
                moves_uci=[move.to_uci()],
            )
        ],
    )


def _parameter_arrays(model: PolicyValueNet) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        else:
            arrays.append(np.array(value))

    visit(model.parameters())
    return arrays


def test_policy_value_loss_computes_finite_components() -> None:
    dataset = tiny_dataset(sample_count=2)
    model = PolicyValueNet(tiny_config())

    losses = compute_policy_value_loss(
        model,
        mx.array(dataset.positions, dtype=mx.float32),
        mx.array(dataset.legal_masks, dtype=mx.float32),
        mx.array(dataset.mcts_policies, dtype=mx.float32),
        mx.array(dataset.outcomes, dtype=mx.float32),
    )
    mx.eval(losses.total, losses.policy, losses.value)

    assert float(losses.total.item()) > 0.0
    assert float(losses.policy.item()) > 0.0
    assert float(losses.value.item()) >= 0.0
    assert np.isfinite(float(losses.total.item()))


def test_train_model_public_signature_does_not_expose_internal_checkpoint_toggle() -> None:
    assert "_write_checkpoints" not in inspect.signature(train_model).parameters
    assert "write_checkpoints" not in inspect.signature(train_model).parameters


def test_train_model_writes_epoch_metrics_and_checkpoint(tmp_path: Path) -> None:
    dataset = tiny_dataset(sample_count=4)
    model = PolicyValueNet(tiny_config())
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "metrics.jsonl").write_text("stale\n")
    before = np.array(model(mx.array(dataset.positions[:1], dtype=mx.float32)).policy_logits)

    result = train_model(
        dataset,
        tmp_path,
        model=model,
        config=TrainingConfig(epochs=1, batch_size=1, learning_rate=1.0e-3, seed=7),
        notes="unit test",
    )

    epoch_metrics_lines = (tmp_path / DEFAULT_EPOCH_METRICS_FILENAME).read_text().splitlines()
    metadata = load_checkpoint_metadata(result.checkpoint_dir)
    loaded = load_checkpoint(result.checkpoint_dir)
    after = np.array(model(mx.array(dataset.positions[:1], dtype=mx.float32)).policy_logits)
    losses = compute_policy_value_loss(
        loaded.model,
        mx.array(dataset.positions, dtype=mx.float32),
        mx.array(dataset.legal_masks, dtype=mx.float32),
        mx.array(dataset.mcts_policies, dtype=mx.float32),
        mx.array(dataset.outcomes, dtype=mx.float32),
    )
    mx.eval(losses.total)

    assert result.steps == 3
    assert result.samples == 4
    assert result.training_samples == 3
    assert result.validation_samples == 1
    assert result.final_training_step == 3
    assert result.epoch_metrics_path == tmp_path / DEFAULT_EPOCH_METRICS_FILENAME
    assert len(epoch_metrics_lines) == 1
    assert not (tmp_path / "metrics.jsonl").exists()
    assert json.loads(epoch_metrics_lines[-1])["validation_loss"] is not None
    assert not np.allclose(before, after)
    assert np.isfinite(float(losses.total.item()))
    assert (result.checkpoint_dir / DEFAULT_WEIGHTS_FILENAME).is_file()
    assert (result.checkpoint_dir / DEFAULT_METADATA_FILENAME).is_file()
    assert metadata.training_step == 3
    assert loaded.metadata.training_step == 3
    assert metadata.optimizer_state_available is False
    assert metadata.notes == "unit test"
    assert (tmp_path / "training.json").is_file()


def test_train_model_trains_from_saved_sparse_dataset(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "train"
    save_self_play_dataset(tiny_dataset(sample_count=1), dataset_dir)
    loaded = load_self_play_dataset(dataset_dir)

    result = train_model(
        loaded,
        output_dir,
        model=PolicyValueNet(tiny_config()),
        config=TrainingConfig(epochs=1, batch_size=1, learning_rate=1.0e-3),
    )

    assert result.steps == 1
    assert result.samples == 1
    assert (output_dir / "checkpoint-final" / DEFAULT_WEIGHTS_FILENAME).is_file()


def test_train_model_reserves_validation_split_and_reports_epoch_losses(tmp_path: Path) -> None:
    dataset = tiny_dataset(sample_count=10)
    callbacks: list[EpochMetrics] = []

    result = train_model(
        dataset,
        tmp_path,
        model=PolicyValueNet(tiny_config()),
        config=TrainingConfig(
            epochs=2,
            batch_size=3,
            learning_rate=1.0e-3,
            seed=7,
            validation_fraction=0.2,
        ),
        epoch_callback=callbacks.append,
    )

    assert result.samples == 10
    assert result.training_samples == 8
    assert result.validation_samples == 2
    assert result.steps == 6
    assert len(result.epoch_metrics) == 2
    assert callbacks == list(result.epoch_metrics)
    assert all(metrics.validation_loss is not None for metrics in result.epoch_metrics)
    assert (tmp_path / DEFAULT_EPOCH_METRICS_FILENAME).read_text().count("\n") == 2


def test_training_config_serializes_sharded_training_settings() -> None:
    assert TrainingConfig().to_dict()["write_shard_checkpoints"] is True
    assert TrainingConfig().to_dict()["carry_optimizer_state_across_shards"] is True
    assert TrainingConfig().to_dict()["warmup_steps"] == 0
    assert (
        TrainingConfig(write_shard_checkpoints=False).to_dict()["write_shard_checkpoints"]
        is False
    )
    assert (
        TrainingConfig(carry_optimizer_state_across_shards=False).to_dict()[
            "carry_optimizer_state_across_shards"
        ]
        is False
    )
    assert TrainingConfig(warmup_steps=3).to_dict()["warmup_steps"] == 3


def test_learning_rate_warmup_scales_initial_steps() -> None:
    assert _learning_rate_for_step(1.0e-3, warmup_steps=0, optimizer_step=1) == 1.0e-3
    assert _learning_rate_for_step(1.0e-3, warmup_steps=4, optimizer_step=1) == 2.5e-4
    assert _learning_rate_for_step(1.0e-3, warmup_steps=4, optimizer_step=4) == 1.0e-3
    assert _learning_rate_for_step(1.0e-3, warmup_steps=4, optimizer_step=5) == 1.0e-3


def test_train_model_warmup_uses_scaled_learning_rate(tmp_path: Path) -> None:
    dataset = tiny_dataset(sample_count=1)
    warmup_model = PolicyValueNet(tiny_config())
    scaled_lr_model = PolicyValueNet(tiny_config())
    scaled_lr_model.update(warmup_model.parameters())

    train_model(
        dataset,
        tmp_path / "warmup",
        model=warmup_model,
        config=TrainingConfig(
            epochs=1,
            batch_size=1,
            learning_rate=4.0e-3,
            warmup_steps=4,
            validation_fraction=0.0,
        ),
    )
    train_model(
        dataset,
        tmp_path / "scaled-lr",
        model=scaled_lr_model,
        config=TrainingConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1.0e-3,
            validation_fraction=0.0,
        ),
    )

    for warmup_param, scaled_lr_param in zip(
        _parameter_arrays(warmup_model), _parameter_arrays(scaled_lr_model), strict=True
    ):
        np.testing.assert_allclose(warmup_param, scaled_lr_param)


def test_training_config_rejects_negative_warmup_steps() -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        TrainingConfig(warmup_steps=-1)


def test_train_model_rejects_empty_dataset(tmp_path: Path) -> None:
    dataset = tiny_dataset(sample_count=1)
    empty = SelfPlayDataset(
        positions=dataset.positions[:0],
        legal_masks=dataset.legal_masks[:0],
        mcts_policies=dataset.mcts_policies[:0],
        outcomes=dataset.outcomes[:0],
        metadata=SelfPlayMetadata.create(SelfPlayConfig(games=1, max_plies=0), sample_count=0),
        games=[],
    )

    with pytest.raises(ValueError, match="at least one self-play sample"):
        train_model(empty, tmp_path, model=PolicyValueNet(tiny_config()))


def test_train_model_rejects_illegal_policy_targets(tmp_path: Path) -> None:
    dataset = tiny_dataset(sample_count=1)
    illegal_action = int(np.flatnonzero(dataset.legal_masks[0] == 0.0)[0])
    corrupted_policy = np.zeros_like(dataset.mcts_policies)
    corrupted_policy[0, illegal_action] = 1.0
    corrupted = SelfPlayDataset(
        positions=dataset.positions,
        legal_masks=dataset.legal_masks,
        mcts_policies=corrupted_policy,
        outcomes=dataset.outcomes,
        metadata=dataset.metadata,
        games=dataset.games,
    )

    with pytest.raises(ValueError, match="only on legal actions"):
        train_model(corrupted, tmp_path, model=PolicyValueNet(tiny_config()))


def test_train_model_continues_checkpoint_step_metadata(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    dataset = tiny_dataset(sample_count=1)
    first = train_model(
        dataset,
        first_dir,
        model=PolicyValueNet(tiny_config()),
        config=TrainingConfig(epochs=1, batch_size=1, learning_rate=1.0e-3),
    )
    loaded = load_checkpoint(first.checkpoint_dir)

    second = train_model(
        dataset,
        second_dir,
        model=loaded.model,
        config=TrainingConfig(epochs=1, batch_size=1, learning_rate=1.0e-3),
        initial_step=loaded.metadata.training_step,
    )

    assert second.steps == 1
    assert second.final_training_step == 2
    assert load_checkpoint_metadata(second.checkpoint_dir).training_step == 2


def test_train_script_consumes_dataset_and_writes_checkpoint(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "train-output"
    save_self_play_dataset(tiny_dataset(sample_count=1), dataset_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--dataset",
            str(dataset_dir),
            "--output",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--learning-rate",
            "0.001",
            "--warmup",
            "3",
            "--residual-channels",
            "8",
            "--residual-blocks",
            "1",
            "--policy-channels",
            "2",
            "--value-channels",
            "1",
            "--value-hidden-dim",
            "8",
        ],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "training complete" in result.stdout
    assert "epoch 1:" in result.stdout
    assert "validation_loss=n/a" in result.stdout
    assert "steps=1" in result.stdout
    assert "epoch_metrics=" in result.stdout
    assert not (output_dir / "metrics.jsonl").exists()
    assert (output_dir / "checkpoint-final" / DEFAULT_WEIGHTS_FILENAME).is_file()
    assert load_checkpoint_metadata(output_dir / "checkpoint-final").training_step == 1
    training_data = json.loads((output_dir / "training.json").read_text())
    assert training_data["training_config"]["warmup_steps"] == 3
