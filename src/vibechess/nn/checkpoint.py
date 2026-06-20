"""MLX checkpoint persistence for policy/value models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from vibechess import _jsonio
from vibechess.nn.encode import ACTION_SPACE_VERSION, ENCODER_VERSION
from vibechess.nn.model import PolicyValueConfig, PolicyValueNet

CHECKPOINT_METADATA_SCHEMA_VERSION = "vibechess-checkpoint-v1"
DEFAULT_WEIGHTS_FILENAME = "weights.safetensors"
DEFAULT_METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Sidecar metadata stored next to MLX model weights."""

    schema_version: str
    model_config: PolicyValueConfig
    action_space_version: str
    encoder_version: str
    training_step: int
    optimizer_state_available: bool
    notes: str | None = None

    @classmethod
    def initial(
        cls,
        model_config: PolicyValueConfig,
        *,
        training_step: int = 0,
        optimizer_state_available: bool = False,
        notes: str | None = None,
    ) -> CheckpointMetadata:
        """Return metadata for an inference-only or initial training checkpoint."""
        return cls(
            schema_version=CHECKPOINT_METADATA_SCHEMA_VERSION,
            model_config=model_config,
            action_space_version=ACTION_SPACE_VERSION,
            encoder_version=ENCODER_VERSION,
            training_step=training_step,
            optimizer_state_available=optimizer_state_available,
            notes=notes,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable metadata dictionary."""
        data: dict[str, object] = {
            "schema_version": self.schema_version,
            "model_config": self.model_config.to_dict(),
            "action_space_version": self.action_space_version,
            "encoder_version": self.encoder_version,
            "training_step": self.training_step,
            "optimizer_state_available": self.optimizer_state_available,
        }
        if self.notes is not None:
            data["notes"] = self.notes
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CheckpointMetadata:
        """Parse and validate checkpoint metadata."""
        schema_version = _expect_str(data, "schema_version")
        if schema_version != CHECKPOINT_METADATA_SCHEMA_VERSION:
            raise ValueError(f"unsupported checkpoint metadata schema: {schema_version}")
        action_space_version = _expect_str(data, "action_space_version")
        if action_space_version != ACTION_SPACE_VERSION:
            raise ValueError(f"unsupported action space version: {action_space_version}")
        encoder_version = _expect_str(data, "encoder_version")
        if encoder_version != ENCODER_VERSION:
            raise ValueError(f"unsupported encoder version: {encoder_version}")
        model_config_data = data.get("model_config")
        if not isinstance(model_config_data, dict):
            raise TypeError("checkpoint metadata field 'model_config' must be an object")
        notes = data.get("notes")
        if notes is not None and not isinstance(notes, str):
            raise TypeError("checkpoint metadata field 'notes' must be a string when present")
        return cls(
            schema_version=schema_version,
            model_config=PolicyValueConfig.from_dict(model_config_data),
            action_space_version=action_space_version,
            encoder_version=encoder_version,
            training_step=_expect_int(data, "training_step"),
            optimizer_state_available=_expect_bool(data, "optimizer_state_available"),
            notes=notes,
        )


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Model and metadata loaded from disk."""

    model: PolicyValueNet
    metadata: CheckpointMetadata


def save_checkpoint(
    model: PolicyValueNet,
    directory: str | Path,
    *,
    metadata: CheckpointMetadata | None = None,
) -> CheckpointMetadata:
    """Save model weights and JSON sidecar metadata into ``directory``."""
    checkpoint_dir = Path(directory)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resolved_metadata = metadata or CheckpointMetadata.initial(model.config)
    if resolved_metadata.model_config != model.config:
        raise ValueError("checkpoint metadata model_config must match model.config")

    weights_path = checkpoint_dir / DEFAULT_WEIGHTS_FILENAME
    metadata_path = checkpoint_dir / DEFAULT_METADATA_FILENAME
    model.save_weights(str(weights_path))
    metadata_path.write_text(json.dumps(resolved_metadata.to_dict(), indent=2) + "\n")
    return resolved_metadata


def load_checkpoint(directory: str | Path) -> LoadedCheckpoint:
    """Load model weights and metadata from ``directory``."""
    checkpoint_dir = Path(directory)
    metadata = load_checkpoint_metadata(checkpoint_dir)
    model = PolicyValueNet(metadata.model_config)
    model.load_weights(str(checkpoint_dir / DEFAULT_WEIGHTS_FILENAME))
    # Force lazy MLX loading before returning so missing/corrupt weights fail here.
    mx.eval(model.parameters())
    return LoadedCheckpoint(model=model, metadata=metadata)


def load_checkpoint_metadata(directory: str | Path) -> CheckpointMetadata:
    """Load only the JSON checkpoint sidecar metadata."""
    metadata_path = Path(directory) / DEFAULT_METADATA_FILENAME
    data = json.loads(metadata_path.read_text())
    if not isinstance(data, dict):
        raise TypeError("checkpoint metadata must be a JSON object")
    return CheckpointMetadata.from_dict(data)


_FIELD_LABEL = "checkpoint metadata field"


def _expect_str(data: dict[str, object], key: str) -> str:
    return _jsonio.expect_str(data, key, label=_FIELD_LABEL)


def _expect_int(data: dict[str, object], key: str) -> int:
    return _jsonio.expect_int(data, key, label=_FIELD_LABEL)


def _expect_bool(data: dict[str, object], key: str) -> bool:
    return _jsonio.expect_bool(data, key, label=_FIELD_LABEL)


__all__ = [
    "CHECKPOINT_METADATA_SCHEMA_VERSION",
    "DEFAULT_METADATA_FILENAME",
    "DEFAULT_WEIGHTS_FILENAME",
    "CheckpointMetadata",
    "LoadedCheckpoint",
    "load_checkpoint",
    "load_checkpoint_metadata",
    "save_checkpoint",
]
