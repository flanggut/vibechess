"""Small MLX policy/value network and inference helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, TypeAlias, cast

import mlx.core as mx
import mlx.nn as _nn

from tinychess.engine.game import Game
from tinychess.engine.move import Move
from tinychess.nn.encode import (
    ACTION_SPACE_SIZE,
    ENCODER_CHANNELS,
    TENSOR_SHAPE,
    encode_game,
    legal_action_indices,
    legal_move_mask,
    tensor_shape,
    to_mlx,
)

MLXArray: TypeAlias = Any
nn: Any = _nn


@dataclass(frozen=True, slots=True)
class PolicyValueConfig:
    """Configuration for the tiny residual policy/value network."""

    input_channels: int = ENCODER_CHANNELS
    board_size: int = 8
    residual_channels: int = 32
    residual_blocks: int = 2
    policy_channels: int = 4
    value_channels: int = 2
    value_hidden_dim: int = 64
    action_space_size: int = ACTION_SPACE_SIZE

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serializable configuration dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> PolicyValueConfig:
        """Build a config from metadata loaded from JSON."""
        kwargs: dict[str, int] = {}
        for field_name in cls.__dataclass_fields__:
            value = values.get(field_name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"model config field {field_name!r} must be an integer")
                kwargs[field_name] = value
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class PolicyValueOutput:
    """Raw batched model output."""

    policy_logits: MLXArray
    value: MLXArray


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Single-position inference result with optional legal-move masking.

    ``legal_moves``, ``legal_action_indices``, and ``legal_policy`` are populated only
    by search-oriented helpers that receive a precomputed legal move tuple. Public
    ``predict()`` callers continue to receive the full 4672-action policy contract.
    """

    policy_logits: MLXArray
    policy: MLXArray
    value: float
    legal_mask: MLXArray | None = None
    legal_moves: tuple[Move, ...] | None = None
    legal_action_indices: tuple[int, ...] = ()
    legal_policy: MLXArray | None = None


class ResidualBlock(nn.Module):  # type: ignore[misc]
    """A minimal residual convolution block for 8x8 chess tensors."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def __call__(self, x: MLXArray) -> MLXArray:
        residual = x
        x = nn.relu(self.conv1(x))
        x = self.conv2(x)
        return nn.relu(x + residual)


class PolicyValueNet(nn.Module):  # type: ignore[misc]
    """Small configurable residual CNN with AlphaZero-style policy/value heads.

    Public inputs use the encoder's channel-first shape ``[20, 8, 8]`` or batched
    ``[N, 20, 8, 8]``. Internally MLX convolutions run channel-last.
    """

    def __init__(self, config: PolicyValueConfig | None = None) -> None:
        super().__init__()
        self.config = PolicyValueConfig() if config is None else config
        if self.config.board_size != 8:
            raise ValueError("PolicyValueNet currently supports only 8x8 boards")
        if self.config.input_channels != ENCODER_CHANNELS:
            raise ValueError(f"input_channels must be {ENCODER_CHANNELS}")
        if self.config.action_space_size != ACTION_SPACE_SIZE:
            raise ValueError(f"action_space_size must be {ACTION_SPACE_SIZE}")
        self.input_conv = nn.Conv2d(
            self.config.input_channels,
            self.config.residual_channels,
            kernel_size=3,
            padding=1,
        )
        self.residual_tower = [
            ResidualBlock(self.config.residual_channels)
            for _ in range(self.config.residual_blocks)
        ]
        self.policy_conv = nn.Conv2d(
            self.config.residual_channels,
            self.config.policy_channels,
            kernel_size=1,
        )
        self.policy_head = nn.Linear(
            self.config.board_size * self.config.board_size * self.config.policy_channels,
            self.config.action_space_size,
        )
        self.value_conv = nn.Conv2d(
            self.config.residual_channels,
            self.config.value_channels,
            kernel_size=1,
        )
        self.value_hidden = nn.Linear(
            self.config.board_size * self.config.board_size * self.config.value_channels,
            self.config.value_hidden_dim,
        )
        self.value_head = nn.Linear(self.config.value_hidden_dim, 1)

    def __call__(self, inputs: MLXArray) -> PolicyValueOutput:
        x = _prepare_batch(inputs)
        x = nn.relu(self.input_conv(x))
        for block in self.residual_tower:
            x = block(x)

        policy = nn.relu(self.policy_conv(x))
        policy = policy.reshape(policy.shape[0], -1)
        policy_logits = self.policy_head(policy)

        value = nn.relu(self.value_conv(x))
        value = value.reshape(value.shape[0], -1)
        value = nn.relu(self.value_hidden(value))
        value = mx.tanh(self.value_head(value)).reshape(-1)
        return PolicyValueOutput(policy_logits=policy_logits, value=value)


class PolicyValueInference:
    """Inference wrapper around :class:`PolicyValueNet` for encoded games."""

    def __init__(self, model: PolicyValueNet) -> None:
        self.model = model

    def predict(self, game: Game, *, mask_legal_moves: bool = True) -> InferenceResult:
        """Run model inference for one game position.

        Returns raw logits for all 4672 actions, a probability vector, and a
        scalar value from the side-to-move perspective in ``[-1, 1]``. When
        ``mask_legal_moves`` is true, illegal actions receive zero probability.
        Terminal/no-legal-move positions return an all-zero policy.
        """
        output = self.model(encode_game(game))
        logits = cast(MLXArray, output.policy_logits[0])
        value = float(output.value[0].item())
        if not mask_legal_moves:
            policy = mx.softmax(logits)
            mx.eval(policy)
            return InferenceResult(policy_logits=logits, policy=policy, value=value)

        mask = legal_move_mask(game)
        if float(mx.sum(mask).item()) == 0.0:
            policy = mx.zeros((ACTION_SPACE_SIZE,), dtype=mx.float32)
        else:
            masked_logits = mx.where(mask > 0, logits, mx.full(logits.shape, -1.0e9))
            policy = mx.softmax(masked_logits) * mask
            policy = policy / mx.sum(policy)
        mx.eval(policy)
        return InferenceResult(policy_logits=logits, policy=policy, value=value, legal_mask=mask)

    def predict_with_legal_moves(
        self,
        game: Game,
        legal_moves: tuple[Move, ...],
    ) -> InferenceResult:
        """Run inference using precomputed legal moves for compact legal priors.

        The returned ``policy`` and ``legal_mask`` keep the public 4672-action shape,
        but only the compact legal softmax vector is evaluated eagerly. This avoids
        recomputing legal moves and avoids synchronizing a full policy vector in
        search code that only needs priors for cached legal moves.
        """
        legal = tuple(legal_moves)
        output = self.model(encode_game(game))
        logits = cast(MLXArray, output.policy_logits[0])
        value = float(output.value[0].item())
        indices = legal_action_indices(game, legal)
        if not indices:
            legal_policy = mx.zeros((0,), dtype=mx.float32)
            policy = mx.zeros((ACTION_SPACE_SIZE,), dtype=mx.float32)
            mask = mx.zeros((ACTION_SPACE_SIZE,), dtype=mx.float32)
        else:
            index_array = mx.array(indices)
            legal_logits = logits[index_array]
            legal_policy = mx.softmax(legal_logits)
            policy = mx.zeros((ACTION_SPACE_SIZE,), dtype=mx.float32).at[index_array].add(
                legal_policy
            )
            mask = mx.zeros((ACTION_SPACE_SIZE,), dtype=mx.float32).at[index_array].add(1.0)
        mx.eval(legal_policy)
        return InferenceResult(
            policy_logits=logits,
            policy=policy,
            value=value,
            legal_mask=mask,
            legal_moves=legal,
            legal_action_indices=indices,
            legal_policy=legal_policy,
        )


def _prepare_batch(inputs: MLXArray) -> MLXArray:
    tensor = to_mlx(inputs)
    shape = tensor_shape(tensor)
    if shape == TENSOR_SHAPE:
        return mx.transpose(tensor[None, :, :, :], (0, 2, 3, 1))
    if len(shape) == 4 and shape[1:] == TENSOR_SHAPE:
        return mx.transpose(tensor, (0, 2, 3, 1))
    if len(shape) == 4 and shape[1:] == (8, 8, ENCODER_CHANNELS):
        return tensor
    raise ValueError(
        "expected input shape "
        f"{TENSOR_SHAPE}, [N, {TENSOR_SHAPE}], or "
        f"[N, 8, 8, {ENCODER_CHANNELS}], got {shape}"
    )
