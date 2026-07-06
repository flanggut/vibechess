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
MODEL_ARCHITECTURE_CHESSFORMER = "chessformer"


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
    "MODEL_ARCHITECTURE_CHESSFORMER",
    "ModelConfig",
    "PolicyValueConfig",
    "TransformerPolicyValueConfig",
    "ChessformerPolicyValueConfig",
    "PolicyValueOutput",
    "ResidualBlock",
    "PolicyValueNet",
    "PolicyValueModel",
    "PolicyValueTransformerNet",
    "GeometricAttentionBias",
    "ChessformerEncoderLayer",
    "ChessformerPolicyValueNet",
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

@dataclass(frozen=True, slots=True)
class ChessformerPolicyValueConfig:
    """Configuration for a Chessformer-style square-token policy/value network.

    The default keeps the same encoder, scalar value, and 64 * 73 action layout as
    the existing networks while replacing absolute square embeddings with
    Geometric Attention Bias. The budget stays below the current 3.8M-parameter
    target for small local checkpoints.
    """

    input_channels: int = ENCODER_CHANNELS
    board_size: int = 8
    model_dim: int = 224
    transformer_layers: int = 6
    attention_heads: int = 8
    mlp_dim: int = 536
    gab_dim1: int = 8
    gab_dim2: int = 32
    gab_dim3: int = 32
    gab_use_average_pool: int = 1
    policy_dim: int = 224
    promotion_hidden_dim: int = 64
    value_hidden_dim: int = 256
    action_space_size: int = ACTION_SPACE_SIZE

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serializable configuration dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> ChessformerPolicyValueConfig:
        """Build a config from metadata loaded from JSON."""
        kwargs: dict[str, int] = {}
        for field_name in cls.__dataclass_fields__:
            value = values.get(field_name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"model config field {field_name!r} must be an integer")
                kwargs[field_name] = value
        return cls(**kwargs)


ModelConfig: TypeAlias = (
    PolicyValueConfig | TransformerPolicyValueConfig | ChessformerPolicyValueConfig
)


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

class GeometricAttentionBias(nn.Module):  # type: ignore[misc]
    """Input-dependent additive attention bias for 64 chess-square tokens."""

    def __init__(
        self,
        *,
        token_count: int,
        model_dim: int,
        attention_heads: int,
        dim1: int,
        dim2: int,
        dim3: int,
        use_average_pool: int,
    ) -> None:
        super().__init__()
        self.token_count = token_count
        self.attention_heads = attention_heads
        self.dim3 = dim3
        self.use_average_pool = bool(use_average_pool)
        if self.use_average_pool:
            self.summary1 = None
            summary_dim = model_dim
        else:
            self.summary1 = nn.Linear(model_dim, dim1)
            summary_dim = token_count * dim1
        self.summary2 = nn.Linear(summary_dim, dim2)
        self.norm1 = nn.LayerNorm(dim2)
        self.summary3 = nn.Linear(dim2, attention_heads * dim3)
        self.norm2 = nn.LayerNorm(attention_heads * dim3)
        self.position_templates = (
            mx.random.normal((token_count * token_count, dim3)) * dim3**-0.5
        )

    def __call__(self, x: MLXArray) -> MLXArray:
        batch_size = x.shape[0]
        if self.use_average_pool:
            y = mx.mean(x, axis=1)
        else:
            y = self.summary1(x)
            y = y.reshape(batch_size, self.token_count * y.shape[-1])
        y = nn.gelu(self.summary2(y))
        y = self.norm1(y)
        y = nn.gelu(self.summary3(y))
        y = self.norm2(y).reshape(batch_size, self.attention_heads, self.dim3)
        bias = mx.einsum("bhd,td->bht", y, self.position_templates)
        return bias.reshape(batch_size, self.attention_heads, self.token_count, self.token_count)


class ChessformerEncoderLayer(nn.Module):  # type: ignore[misc]
    """Transformer encoder layer with per-layer Geometric Attention Bias."""

    def __init__(self, config: ChessformerPolicyValueConfig) -> None:
        super().__init__()
        token_count = config.board_size * config.board_size
        self.norm1 = nn.LayerNorm(config.model_dim)
        self.attention = nn.MultiHeadAttention(config.model_dim, config.attention_heads)
        self.gab = GeometricAttentionBias(
            token_count=token_count,
            model_dim=config.model_dim,
            attention_heads=config.attention_heads,
            dim1=config.gab_dim1,
            dim2=config.gab_dim2,
            dim3=config.gab_dim3,
            use_average_pool=config.gab_use_average_pool,
        )
        self.norm2 = nn.LayerNorm(config.model_dim)
        self.linear1 = nn.Linear(config.model_dim, config.mlp_dim)
        self.linear2 = nn.Linear(config.mlp_dim, config.model_dim)

    def __call__(self, x: MLXArray) -> MLXArray:
        y = self.norm1(x)
        y = self.attention(y, y, y, self.gab(y))
        x = x + y
        y = self.norm2(x)
        y = self.linear2(nn.gelu(self.linear1(y)))
        return x + y


class ChessformerPolicyValueNet(nn.Module):  # type: ignore[misc]
    """Chessformer-style GAB trunk with the existing AlphaZero action contract."""

    def __init__(self, config: ChessformerPolicyValueConfig | None = None) -> None:
        super().__init__()
        self.config = ChessformerPolicyValueConfig() if config is None else config
        if self.config.board_size != 8:
            raise ValueError("ChessformerPolicyValueNet currently supports only 8x8 boards")
        if self.config.input_channels != ENCODER_CHANNELS:
            raise ValueError(f"input_channels must be {ENCODER_CHANNELS}")
        if self.config.action_space_size != ACTION_SPACE_SIZE:
            raise ValueError(f"action_space_size must be {ACTION_SPACE_SIZE}")
        if self.config.model_dim % self.config.attention_heads != 0:
            raise ValueError("model_dim must be divisible by attention_heads")
        if self.config.gab_use_average_pool not in {0, 1}:
            raise ValueError("gab_use_average_pool must be 0 or 1")
        self.token_count = self.config.board_size * self.config.board_size
        self.token_projection = nn.Linear(self.config.input_channels, self.config.model_dim)
        self.layers = [
            ChessformerEncoderLayer(self.config) for _ in range(self.config.transformer_layers)
        ]
        self.policy_hidden = nn.Linear(self.config.model_dim, self.config.policy_dim)
        self.policy_head = nn.Linear(self.config.policy_dim, ACTION_PLANES)
        self.value_hidden = nn.Linear(self.config.model_dim * 2, self.config.value_hidden_dim)
        self.value_head = nn.Linear(self.config.value_hidden_dim, 1)

    def __call__(self, inputs: MLXArray) -> PolicyValueOutput:
        x = _prepare_batch(inputs)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.token_count, -1)
        x = self.token_projection(x)
        for layer in self.layers:
            x = layer(x)

        policy = nn.relu(self.policy_hidden(x))
        policy_logits = self.policy_head(policy).reshape(batch_size, self.config.action_space_size)

        value = mx.concatenate([mx.mean(x, axis=1), mx.max(x, axis=1)], axis=1)
        value = nn.relu(self.value_hidden(value))
        value = mx.tanh(self.value_head(value)).reshape(-1)
        return PolicyValueOutput(policy_logits=policy_logits, value=value)


PolicyValueModel: TypeAlias = PolicyValueNet | PolicyValueTransformerNet | ChessformerPolicyValueNet


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
