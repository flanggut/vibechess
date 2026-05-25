import json
from pathlib import Path

import mlx.core as mx
import pytest

from tinychess.engine import Game
from tinychess.nn import (
    ACTION_SPACE_SIZE,
    ACTION_SPACE_VERSION,
    CHECKPOINT_METADATA_SCHEMA_VERSION,
    DEFAULT_METADATA_FILENAME,
    DEFAULT_WEIGHTS_FILENAME,
    ENCODER_VERSION,
    CheckpointMetadata,
    PolicyValueConfig,
    PolicyValueInference,
    PolicyValueNet,
    encode_game,
    legal_move_mask,
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
    tensor_shape,
)


def scalar(value: object) -> float:
    return float(value.item())  # type: ignore[attr-defined]


def tiny_config() -> PolicyValueConfig:
    return PolicyValueConfig(
        residual_channels=8,
        residual_blocks=1,
        policy_channels=2,
        value_channels=1,
        value_hidden_dim=8,
    )


def test_policy_value_model_forward_shapes_and_value_range() -> None:
    model = PolicyValueNet(tiny_config())
    output = model(encode_game(Game.new()))
    mx.eval(output.policy_logits, output.value)

    assert output.policy_logits.dtype == mx.float32
    assert output.value.dtype == mx.float32
    assert tensor_shape(output.policy_logits) == (1, ACTION_SPACE_SIZE)
    assert tensor_shape(output.value) == (1,)
    assert -1.0 <= scalar(output.value[0]) <= 1.0


def test_policy_value_model_accepts_batched_channel_first_inputs() -> None:
    model = PolicyValueNet(tiny_config())
    tensor = encode_game(Game.new())
    output = model(mx.stack([tensor, tensor]))
    mx.eval(output.policy_logits, output.value)

    assert tensor_shape(output.policy_logits) == (2, ACTION_SPACE_SIZE)
    assert tensor_shape(output.value) == (2,)


def test_policy_value_model_accepts_batched_channel_last_inputs() -> None:
    model = PolicyValueNet(tiny_config())
    tensor = encode_game(Game.new())
    channel_last = mx.transpose(tensor[None, :, :, :], (0, 2, 3, 1))
    output = model(channel_last)
    mx.eval(output.policy_logits, output.value)

    assert tensor_shape(output.policy_logits) == (1, ACTION_SPACE_SIZE)
    assert tensor_shape(output.value) == (1,)


def test_policy_value_model_rejects_wrong_input_shape() -> None:
    model = PolicyValueNet(tiny_config())

    with pytest.raises(ValueError, match="expected input shape"):
        model(mx.zeros((8, 8, 20), dtype=mx.float32))


def test_inference_wrapper_masks_policy_to_legal_moves() -> None:
    result = PolicyValueInference(PolicyValueNet(tiny_config())).predict(Game.new())
    mx.eval(result.policy, result.policy_logits, result.legal_mask)

    assert tensor_shape(result.policy_logits) == (ACTION_SPACE_SIZE,)
    assert tensor_shape(result.policy) == (ACTION_SPACE_SIZE,)
    assert result.legal_mask is not None
    assert tensor_shape(result.legal_mask) == (ACTION_SPACE_SIZE,)
    assert scalar(mx.sum(result.legal_mask)) == 20.0
    assert scalar(mx.sum(result.policy)) == pytest.approx(1.0)
    assert scalar(mx.sum(mx.where(result.legal_mask > 0, 0.0, result.policy))) == pytest.approx(0.0)
    assert -1.0 <= result.value <= 1.0


def test_inference_wrapper_unmasked_policy_normalizes_over_all_actions() -> None:
    game = Game.new()
    result = PolicyValueInference(PolicyValueNet(tiny_config())).predict(
        game,
        mask_legal_moves=False,
    )
    expected_legal_mask = legal_move_mask(game)
    mx.eval(result.policy, expected_legal_mask)

    assert result.legal_mask is None
    assert tensor_shape(result.policy) == (ACTION_SPACE_SIZE,)
    assert scalar(mx.sum(result.policy)) == pytest.approx(1.0)
    assert scalar(mx.sum(mx.where(expected_legal_mask > 0, 0.0, result.policy))) > 0.0


def test_inference_wrapper_returns_zero_policy_for_terminal_position() -> None:
    game = Game.from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    result = PolicyValueInference(PolicyValueNet(tiny_config())).predict(game)
    mx.eval(result.policy)

    assert result.legal_mask is not None
    assert scalar(mx.sum(result.legal_mask)) == 0.0
    assert scalar(mx.sum(result.policy)) == 0.0


def test_checkpoint_save_load_round_trips_weights_and_metadata(tmp_path: Path) -> None:
    config = tiny_config()
    model = PolicyValueNet(config)
    tensor = encode_game(Game.new())
    before = model(tensor)
    mx.eval(before.policy_logits, before.value)
    metadata = CheckpointMetadata.initial(config, training_step=7, notes="unit test")

    saved_metadata = save_checkpoint(model, tmp_path, metadata=metadata)
    loaded = load_checkpoint(tmp_path)
    after = loaded.model(tensor)
    mx.eval(after.policy_logits, after.value)

    assert saved_metadata == metadata
    assert loaded.metadata == metadata
    assert (tmp_path / DEFAULT_WEIGHTS_FILENAME).exists()
    assert (tmp_path / DEFAULT_METADATA_FILENAME).exists()
    assert bool(mx.allclose(before.policy_logits, after.policy_logits).item())
    assert bool(mx.allclose(before.value, after.value).item())


def test_checkpoint_metadata_sidecar_schema(tmp_path: Path) -> None:
    config = tiny_config()
    metadata = save_checkpoint(PolicyValueNet(config), tmp_path)
    data = json.loads((tmp_path / DEFAULT_METADATA_FILENAME).read_text())

    assert metadata.schema_version == CHECKPOINT_METADATA_SCHEMA_VERSION
    assert data["schema_version"] == CHECKPOINT_METADATA_SCHEMA_VERSION
    assert data["model_config"] == config.to_dict()
    assert data["action_space_version"] == ACTION_SPACE_VERSION
    assert data["encoder_version"] == ENCODER_VERSION
    assert data["training_step"] == 0
    assert data["optimizer_state_available"] is False
    assert load_checkpoint_metadata(tmp_path) == metadata


def test_checkpoint_rejects_mismatched_metadata_config(tmp_path: Path) -> None:
    model = PolicyValueNet(tiny_config())
    metadata = CheckpointMetadata.initial(PolicyValueConfig(residual_channels=16))

    with pytest.raises(ValueError, match="model_config"):
        save_checkpoint(model, tmp_path, metadata=metadata)


def test_metadata_integer_fields_reject_booleans() -> None:
    with pytest.raises(TypeError, match="residual_channels"):
        PolicyValueConfig.from_dict({"residual_channels": True})

    data = CheckpointMetadata.initial(tiny_config()).to_dict()
    data["training_step"] = False
    with pytest.raises(TypeError, match="training_step"):
        CheckpointMetadata.from_dict(data)
