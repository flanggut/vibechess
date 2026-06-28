"""Small MLX policy/value network architecture."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import mlx.core as mx
import mlx.nn as _nn

from vibechess.nn.encode import (
    ACTION_PLANES,
    ACTION_SPACE_SIZE,
    ENCODER_CHANNELS,
    TENSOR_SHAPE,
    tensor_shape,
    to_mlx,
)

if TYPE_CHECKING:
    from vibechess.nn.inference import (
        BatchInferenceResult,
        InferenceResult,
        LegalPolicyBatchResult,
        LegalPolicyResult,
        PolicyValueInference,
    )

MLXArray: TypeAlias = Any
nn: Any = _nn
MODEL_ARCHITECTURE_RESNET = "resnet"
MODEL_ARCHITECTURE_TRANSFORMER = "transformer"


_COMPAT_INFERENCE_EXPORTS = frozenset(
    {
        "BatchInferenceResult",
        "InferenceResult",
        "LegalPolicyBatchResult",
        "LegalPolicyResult",
        "PolicyValueInference",
    }
)

__all__ = [
    "MLXArray",
    "nn",
    "MODEL_ARCHITECTURE_RESNET",
    "MODEL_ARCHITECTURE_TRANSFORMER",
    "ModelConfig",
    "PolicyValueConfig",
    "TransformerPolicyValueConfig",
    "PolicyValueOutput",
    "ResidualBlock",
    "PolicyValueNet",
    "PolicyValueModel",
    "PolicyValueTransformerNet",
    "BatchInferenceResult",
    "InferenceResult",
    "LegalPolicyBatchResult",
    "LegalPolicyResult",
    "PolicyValueInference",
]


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
class TransformerPolicyValueConfig:
    """Configuration for a square-token Transformer policy/value network.

    Defaults stay below the parameter budget of ``data/checkpoints/strongest`` while
    shifting capacity from the Transformer trunk into the per-square policy head.
    The encoder and 64 * 73 policy action layout match ``PolicyValueNet``.
    """

    input_channels: int = ENCODER_CHANNELS
    board_size: int = 8
    model_dim: int = 224
    transformer_layers: int = 6
    attention_heads: int = 8
    mlp_dim: int = 536
    policy_hidden_dim: int = 3352
    value_hidden_dim: int = 256
    action_space_size: int = ACTION_SPACE_SIZE

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serializable configuration dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> TransformerPolicyValueConfig:
        """Build a config from metadata loaded from JSON."""
        kwargs: dict[str, int] = {}
        for field_name in cls.__dataclass_fields__:
            value = values.get(field_name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"model config field {field_name!r} must be an integer")
                kwargs[field_name] = value
        return cls(**kwargs)


ModelConfig: TypeAlias = PolicyValueConfig | TransformerPolicyValueConfig


@dataclass(frozen=True, slots=True)
class PolicyValueOutput:
    """Raw batched model output."""

    policy_logits: MLXArray
    value: MLXArray


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
            ResidualBlock(self.config.residual_channels) for _ in range(self.config.residual_blocks)
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


class PolicyValueTransformerNet(nn.Module):  # type: ignore[misc]
    """Square-token Transformer with AlphaZero-style policy/value heads.

    Public inputs use the encoder's channel-first shape ``[20, 8, 8]`` or batched
    ``[N, 20, 8, 8]``. Internally each square becomes one token ordered by the
    policy action layout: ``square * 73 + move_plane``.
    """

    def __init__(self, config: TransformerPolicyValueConfig | None = None) -> None:
        super().__init__()
        self.config = TransformerPolicyValueConfig() if config is None else config
        if self.config.board_size != 8:
            raise ValueError("PolicyValueTransformerNet currently supports only 8x8 boards")
        if self.config.input_channels != ENCODER_CHANNELS:
            raise ValueError(f"input_channels must be {ENCODER_CHANNELS}")
        if self.config.action_space_size != ACTION_SPACE_SIZE:
            raise ValueError(f"action_space_size must be {ACTION_SPACE_SIZE}")
        if self.config.model_dim % self.config.attention_heads != 0:
            raise ValueError("model_dim must be divisible by attention_heads")
        token_count = self.config.board_size * self.config.board_size
        self.token_projection = nn.Linear(self.config.input_channels, self.config.model_dim)
        self.square_embedding = (
            mx.random.normal((token_count, self.config.model_dim)) * self.config.model_dim**-0.5
        )
        self.transformer = nn.TransformerEncoder(
            num_layers=self.config.transformer_layers,
            dims=self.config.model_dim,
            num_heads=self.config.attention_heads,
            mlp_dims=self.config.mlp_dim,
            dropout=0.0,
            norm_first=True,
        )
        self.policy_hidden = nn.Linear(self.config.model_dim, self.config.policy_hidden_dim)
        self.policy_head = nn.Linear(self.config.policy_hidden_dim, ACTION_PLANES)
        self.value_hidden = nn.Linear(self.config.model_dim * 2, self.config.value_hidden_dim)
        self.value_head = nn.Linear(self.config.value_hidden_dim, 1)

    def __call__(self, inputs: MLXArray) -> PolicyValueOutput:
        x = _prepare_batch(inputs)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.config.board_size * self.config.board_size, -1)
        x = self.token_projection(x) + self.square_embedding
        x = self.transformer(x, None)

        policy = nn.relu(self.policy_hidden(x))
        policy_logits = self.policy_head(policy).reshape(batch_size, self.config.action_space_size)

        value = mx.concatenate([mx.mean(x, axis=1), mx.max(x, axis=1)], axis=1)
        value = nn.relu(self.value_hidden(value))
        value = mx.tanh(self.value_head(value)).reshape(-1)
        return PolicyValueOutput(policy_logits=policy_logits, value=value)


PolicyValueModel: TypeAlias = PolicyValueNet | PolicyValueTransformerNet


def __getattr__(name: str) -> object:
    """Lazily preserve historical inference exports from ``vibechess.nn.model``."""
    if name in _COMPAT_INFERENCE_EXPORTS:
        from vibechess.nn import inference

        value = getattr(inference, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
