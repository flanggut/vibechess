import json
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import numpy as np
import pytest

from tinychess.engine import Game, Move
from tinychess.nn import (
    ACTION_SPACE_SIZE,
    ACTION_SPACE_VERSION,
    CHECKPOINT_METADATA_SCHEMA_VERSION,
    DEFAULT_METADATA_FILENAME,
    DEFAULT_WEIGHTS_FILENAME,
    ENCODER_VERSION,
    TENSOR_SHAPE,
    CheckpointMetadata,
    PolicyValueConfig,
    PolicyValueInference,
    PolicyValueNet,
    PolicyValueOutput,
    encode_game,
    legal_action_indices,
    legal_move_mask,
    load_checkpoint,
    load_checkpoint_metadata,
    move_to_action_index,
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


class FixedOutputModel:
    def __init__(self, logits: Any, *, value: float = 0.125) -> None:
        self.logits = mx.array(logits, dtype=mx.float32)
        self.value = value

    def __call__(self, inputs: object) -> PolicyValueOutput:
        shape = tensor_shape(inputs)
        batch_size = 1 if shape == TENSOR_SHAPE else shape[0]
        return PolicyValueOutput(
            policy_logits=mx.broadcast_to(
                self.logits.reshape(1, ACTION_SPACE_SIZE),
                (batch_size, ACTION_SPACE_SIZE),
            ),
            value=mx.full((batch_size,), self.value, dtype=mx.float32),
        )


def fixed_inference(logits: Any, *, value: float = 0.125) -> PolicyValueInference:
    return PolicyValueInference(cast(PolicyValueNet, FixedOutputModel(logits, value=value)))


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


@pytest.mark.parametrize(
    "game",
    [
        Game.new(),
        Game.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
        Game.from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"),
    ],
)
def test_predict_with_legal_moves_matches_masked_full_softmax(game: Game) -> None:
    logits = np.linspace(-1.0, 1.0, ACTION_SPACE_SIZE, dtype=np.float32)
    inference = fixed_inference(logits)

    full = inference.predict(game, mask_legal_moves=True)
    compact = inference.predict_with_legal_moves(game, game.legal_moves)
    mx.eval(full.policy, compact.policy, compact.legal_mask, compact.legal_policy)

    assert tensor_shape(compact.policy_logits) == (ACTION_SPACE_SIZE,)
    assert tensor_shape(compact.policy) == (ACTION_SPACE_SIZE,)
    assert compact.legal_mask is not None
    assert tensor_shape(compact.legal_mask) == (ACTION_SPACE_SIZE,)
    assert compact.legal_moves == game.legal_moves
    assert compact.legal_action_indices == legal_action_indices(game, game.legal_moves)
    assert compact.legal_policy is not None
    assert tensor_shape(compact.legal_policy) == (len(game.legal_moves),)
    np.testing.assert_allclose(
        np.asarray(compact.policy, dtype=np.float32),
        np.asarray(full.policy, dtype=np.float32),
        rtol=1e-6,
        atol=1e-7,
    )


def test_predict_with_legal_moves_ignores_illegal_high_logits() -> None:
    game = Game.new()
    legal_move = Move.from_uci("e2e4")
    illegal_move = Move.from_uci("e2e5")
    logits = np.zeros((ACTION_SPACE_SIZE,), dtype=np.float32)
    logits[move_to_action_index(legal_move, game.board)] = 1.0
    logits[move_to_action_index(illegal_move, game.board)] = 1000.0
    inference = fixed_inference(logits)

    result = inference.predict_with_legal_moves(game, game.legal_moves)
    mx.eval(result.policy, result.legal_policy)

    assert illegal_move not in game.legal_moves
    assert scalar(result.policy[move_to_action_index(illegal_move, game.board)]) == 0.0
    assert scalar(mx.sum(result.policy)) == pytest.approx(1.0)
    assert scalar(result.policy[move_to_action_index(legal_move, game.board)]) > 0.0


def test_predict_with_legal_moves_returns_zero_policy_for_empty_legal_list() -> None:
    game = Game.from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    inference = fixed_inference(np.ones((ACTION_SPACE_SIZE,), dtype=np.float32))
    result = inference.predict_with_legal_moves(game, game.legal_moves)
    mx.eval(result.policy, result.legal_mask, result.legal_policy)

    assert result.legal_mask is not None
    assert result.legal_moves == ()
    assert result.legal_action_indices == ()
    assert result.legal_policy is not None
    assert tensor_shape(result.legal_policy) == (0,)
    assert scalar(mx.sum(result.legal_mask)) == 0.0
    assert scalar(mx.sum(result.policy)) == 0.0


def test_predict_batch_matches_repeated_single_game_inference() -> None:
    games = (
        Game.new(),
        Game.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
        Game.from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"),
    )
    logits = np.linspace(-1.0, 1.0, ACTION_SPACE_SIZE, dtype=np.float32)
    inference = fixed_inference(logits, value=-0.25)

    batch = inference.predict_batch(
        games,
        legal_moves=tuple(game.legal_moves for game in games),
    )
    mx.eval(batch.policy_logits, batch.policy, batch.legal_masks)

    assert tensor_shape(batch.policy_logits) == (len(games), ACTION_SPACE_SIZE)
    assert tensor_shape(batch.policy) == (len(games), ACTION_SPACE_SIZE)
    assert batch.legal_masks is not None
    assert tensor_shape(batch.legal_masks) == (len(games), ACTION_SPACE_SIZE)
    assert batch.values == pytest.approx((-0.25, -0.25, -0.25))
    assert batch.legal_moves == tuple(game.legal_moves for game in games)
    for index, game in enumerate(games):
        single = inference.predict(game, mask_legal_moves=True)
        row = batch.result_at(index)
        mx.eval(single.policy, row.policy, row.legal_mask, row.legal_policy)
        np.testing.assert_allclose(
            np.asarray(row.policy, dtype=np.float32),
            np.asarray(single.policy, dtype=np.float32),
            rtol=1e-6,
            atol=1e-7,
        )
        assert row.value == pytest.approx(single.value)
        assert row.legal_moves == game.legal_moves
        assert row.legal_action_indices == legal_action_indices(game, game.legal_moves)
        assert row.legal_policy is not None
        assert tensor_shape(row.legal_policy) == (len(game.legal_moves),)


def test_predict_batch_accepts_encoded_positions_with_legal_masks() -> None:
    games = (Game.new(), Game.from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"))
    encoded = mx.stack([encode_game(game) for game in games])
    masks = mx.stack([legal_move_mask(game) for game in games])
    inference = fixed_inference(np.arange(ACTION_SPACE_SIZE, dtype=np.float32) / 1000.0)

    batch = inference.predict_batch(encoded, legal_masks=masks)
    mx.eval(batch.policy, batch.legal_masks)

    assert tensor_shape(batch.policy_logits) == (2, ACTION_SPACE_SIZE)
    assert tensor_shape(batch.policy) == (2, ACTION_SPACE_SIZE)
    assert batch.legal_masks is not None
    np.testing.assert_array_equal(
        np.asarray(batch.legal_masks, dtype=np.float32),
        np.asarray(masks, dtype=np.float32),
    )
    assert np.allclose(np.asarray(mx.sum(batch.policy, axis=1), dtype=np.float32), 1.0)
    assert np.all(
        np.asarray(mx.where(batch.legal_masks > 0, 0.0, batch.policy), dtype=np.float32) == 0.0
    )


def test_predict_batch_rejects_empty_game_batch() -> None:
    inference = fixed_inference(np.ones((ACTION_SPACE_SIZE,), dtype=np.float32))

    with pytest.raises(ValueError, match="at least one position"):
        inference.predict_batch(())


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
