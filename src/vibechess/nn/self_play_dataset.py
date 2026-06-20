"""Self-play dataset records, metadata, persistence, and validation."""

from __future__ import annotations

import json
import subprocess
import zipfile
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

import vibechess
from vibechess import _jsonio
from vibechess.engine.game import Game
from vibechess.engine.move import Move
from vibechess.engine.outcome import Outcome, OutcomeReason
from vibechess.engine.piece import Color
from vibechess.nn.encode import (
    ACTION_SPACE_SIZE,
    ACTION_SPACE_VERSION,
    ENCODER_VERSION,
    TENSOR_SHAPE,
    encode_game_np,
    legal_move_mask_from_legal_moves_np,
)
from vibechess.profiling import profile_scope

if TYPE_CHECKING:
    from vibechess.nn.self_play import SelfPlayConfig

SELF_PLAY_DATASET_SCHEMA_VERSION_V1 = "vibechess-selfplay-v1"
SELF_PLAY_DATASET_SCHEMA_VERSION = "vibechess-selfplay-v2"
POLICY_TARGET_FORMAT_DENSE = "dense_mcts_policies"
POLICY_TARGET_FORMAT_SPARSE_CSR = "sparse_csr"
DEFAULT_DATASET_FILENAME = "samples.npz"
TEMP_SHARD_DATASET_FILENAME = "samples.tmp.npz"
DEFAULT_METADATA_FILENAME = "metadata.json"
DEFAULT_GAMES_FILENAME = "games.jsonl"


@dataclass(frozen=True, slots=True)
class SparsePolicyTargets:
    """CSR-style sparse policy targets for fixed-size action-space rows."""

    offsets: npt.NDArray[np.int64]
    indices: npt.NDArray[np.int32]
    probabilities: npt.NDArray[np.float32]

    def __post_init__(self) -> None:
        offsets = np.asarray(self.offsets, dtype=np.int64)
        indices = np.asarray(self.indices, dtype=np.int32)
        probabilities = np.asarray(self.probabilities, dtype=np.float32)
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "probabilities", probabilities)
        _validate_sparse_storage_shapes(self)

    @property
    def sample_count(self) -> int:
        """Return the number of policy rows."""
        return int(self.offsets.shape[0] - 1)

    @classmethod
    def from_rows(
        cls,
        rows: list[tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]],
    ) -> SparsePolicyTargets:
        """Build sparse targets from per-row action indices and probabilities."""
        offsets = np.zeros((len(rows) + 1,), dtype=np.int64)
        row_indices: list[npt.NDArray[np.int32]] = []
        row_probabilities: list[npt.NDArray[np.float32]] = []
        total = 0
        for row_number, (indices, probabilities) in enumerate(rows):
            row_index_array = np.asarray(indices, dtype=np.int32)
            row_probability_array = np.asarray(probabilities, dtype=np.float32)
            if row_index_array.ndim != 1:
                raise ValueError("sparse policy row indices must be one-dimensional")
            if row_probability_array.ndim != 1:
                raise ValueError("sparse policy row probabilities must be one-dimensional")
            if row_index_array.shape[0] != row_probability_array.shape[0]:
                raise ValueError(
                    f"sparse policy row {row_number} index/probability length mismatch"
                )
            row_indices.append(row_index_array)
            row_probabilities.append(row_probability_array)
            total += int(row_index_array.shape[0])
            offsets[row_number + 1] = total
        if row_indices:
            indices_array = np.concatenate(row_indices).astype(np.int32, copy=False)
            probabilities_array = np.concatenate(row_probabilities).astype(np.float32, copy=False)
        else:
            indices_array = np.zeros((0,), dtype=np.int32)
            probabilities_array = np.zeros((0,), dtype=np.float32)
        return cls(offsets=offsets, indices=indices_array, probabilities=probabilities_array)

    @classmethod
    def from_dense(cls, policies: npt.NDArray[np.float32]) -> SparsePolicyTargets:
        """Convert dense ``[N, ACTION_SPACE_SIZE]`` policy targets to sparse rows."""
        dense = np.asarray(policies, dtype=np.float32)
        if dense.ndim != 2 or dense.shape[1] != ACTION_SPACE_SIZE:
            raise ValueError(f"mcts_policies shape mismatch: {dense.shape}")
        rows: list[tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]] = []
        for row in dense:
            indices = np.flatnonzero(row > 0.0).astype(np.int32, copy=False)
            rows.append((indices, row[indices].astype(np.float32, copy=False)))
        return cls.from_rows(rows)

    @classmethod
    def concatenate(cls, targets: list[SparsePolicyTargets]) -> SparsePolicyTargets:
        """Concatenate multiple sparse policy target matrices without densifying."""
        if not targets:
            return cls.from_rows([])
        total_rows = sum(target.sample_count for target in targets)
        offsets = np.zeros((total_rows + 1,), dtype=np.int64)
        indices: list[npt.NDArray[np.int32]] = []
        probabilities: list[npt.NDArray[np.float32]] = []
        row_cursor = 0
        nnz_cursor = 0
        for target in targets:
            row_count = target.sample_count
            row_lengths = np.diff(target.offsets)
            offsets[row_cursor + 1 : row_cursor + row_count + 1] = nnz_cursor + np.cumsum(
                row_lengths,
                dtype=np.int64,
            )
            row_cursor += row_count
            nnz_cursor += int(target.indices.shape[0])
            indices.append(target.indices)
            probabilities.append(target.probabilities)
        if indices:
            indices_array = np.concatenate(indices).astype(np.int32, copy=False)
            probabilities_array = np.concatenate(probabilities).astype(np.float32, copy=False)
        else:
            indices_array = np.zeros((0,), dtype=np.int32)
            probabilities_array = np.zeros((0,), dtype=np.float32)
        return cls(offsets=offsets, indices=indices_array, probabilities=probabilities_array)

    def row(self, index: int) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
        """Return sparse action indices and probabilities for one row."""
        if index < 0 or index >= self.sample_count:
            raise IndexError("sparse policy row index out of range")
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return self.indices[start:end], self.probabilities[start:end]

    def dense_rows(self, row_indices: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
        """Densify selected rows for batch training or compatibility checks."""
        requested = np.asarray(row_indices, dtype=np.int64)
        dense = np.zeros((requested.shape[0], ACTION_SPACE_SIZE), dtype=np.float32)
        for output_index, source_index in enumerate(requested):
            indices, probabilities = self.row(int(source_index))
            dense[output_index, indices] = probabilities
        return dense

    def to_dense(self) -> npt.NDArray[np.float32]:
        """Densify all sparse policy target rows."""
        return self.dense_rows(np.arange(self.sample_count, dtype=np.int64))


@dataclass(frozen=True, slots=True)
class SelfPlayGameRecord:
    """Game-level metadata for one generated self-play game."""

    game_index: int
    plies: int
    outcome_reason: str
    winner: str | None
    final_fen: str
    moves_uci: list[str]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable game record."""
        return {
            "game_index": self.game_index,
            "plies": self.plies,
            "outcome_reason": self.outcome_reason,
            "winner": self.winner,
            "final_fen": self.final_fen,
            "moves_uci": self.moves_uci,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SelfPlayGameRecord:
        """Parse a game record from JSON data."""
        moves = data.get("moves_uci")
        if not isinstance(moves, list) or not all(isinstance(move, str) for move in moves):
            raise TypeError("game record field 'moves_uci' must be a list of strings")
        winner = data.get("winner")
        if winner is not None and not isinstance(winner, str):
            raise TypeError("game record field 'winner' must be a string or null")
        return cls(
            game_index=_expect_int(data, "game_index"),
            plies=_expect_int(data, "plies"),
            outcome_reason=_expect_str(data, "outcome_reason"),
            winner=winner,
            final_fen=_expect_str(data, "final_fen"),
            moves_uci=moves,
        )


@dataclass(frozen=True, slots=True)
class SelfPlayMetadata:
    """Dataset-level metadata stored next to self-play tensor batches."""

    schema_version: str
    generated_at: str
    engine_version: str
    git_commit: str | None
    action_space_version: str
    encoder_version: str
    model_checkpoint_id: str | None
    generation_settings: dict[str, object]
    sample_count: int
    game_count: int
    policy_target_format: str = POLICY_TARGET_FORMAT_SPARSE_CSR

    @classmethod
    def create(
        cls,
        config: SelfPlayConfig,
        *,
        sample_count: int,
        batching_mode: str | None = None,
        inference_batch_size: int | None = None,
    ) -> SelfPlayMetadata:
        """Create metadata for a generated dataset."""
        return cls(
            schema_version=SELF_PLAY_DATASET_SCHEMA_VERSION,
            generated_at=datetime.now(UTC).isoformat(),
            engine_version=vibechess.__version__,
            git_commit=_git_commit(),
            action_space_version=ACTION_SPACE_VERSION,
            encoder_version=ENCODER_VERSION,
            model_checkpoint_id=config.model_checkpoint_id,
            generation_settings=config.to_dict(
                batching_mode=batching_mode,
                inference_batch_size=inference_batch_size,
            ),
            sample_count=sample_count,
            game_count=config.games,
        )

    @classmethod
    def create_from_settings(
        cls,
        generation_settings: dict[str, object],
        *,
        sample_count: int,
        game_count: int,
        model_checkpoint_id: str | None = None,
    ) -> SelfPlayMetadata:
        """Create metadata from already-materialized generation settings."""
        return cls(
            schema_version=SELF_PLAY_DATASET_SCHEMA_VERSION,
            generated_at=datetime.now(UTC).isoformat(),
            engine_version=vibechess.__version__,
            git_commit=_git_commit(),
            action_space_version=ACTION_SPACE_VERSION,
            encoder_version=ENCODER_VERSION,
            model_checkpoint_id=model_checkpoint_id,
            generation_settings=dict(generation_settings),
            sample_count=sample_count,
            game_count=game_count,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable metadata dictionary."""
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "engine_version": self.engine_version,
            "git_commit": self.git_commit,
            "action_space_version": self.action_space_version,
            "encoder_version": self.encoder_version,
            "model_checkpoint_id": self.model_checkpoint_id,
            "generation_settings": self.generation_settings,
            "sample_count": self.sample_count,
            "game_count": self.game_count,
            "policy_target_format": self.policy_target_format,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SelfPlayMetadata:
        """Parse and validate dataset metadata."""
        schema_version = _expect_str(data, "schema_version")
        if schema_version == SELF_PLAY_DATASET_SCHEMA_VERSION_V1:
            policy_target_format = data.get("policy_target_format", POLICY_TARGET_FORMAT_DENSE)
            if policy_target_format != POLICY_TARGET_FORMAT_DENSE:
                raise ValueError(f"unsupported v1 policy target format: {policy_target_format}")
        elif schema_version == SELF_PLAY_DATASET_SCHEMA_VERSION:
            policy_target_format = _expect_str(data, "policy_target_format")
            if policy_target_format != POLICY_TARGET_FORMAT_SPARSE_CSR:
                raise ValueError(f"unsupported v2 policy target format: {policy_target_format}")
        else:
            raise ValueError(f"unsupported self-play dataset schema: {schema_version}")
        action_space_version = _expect_str(data, "action_space_version")
        if action_space_version != ACTION_SPACE_VERSION:
            raise ValueError(f"unsupported action space version: {action_space_version}")
        encoder_version = _expect_str(data, "encoder_version")
        if encoder_version != ENCODER_VERSION:
            raise ValueError(f"unsupported encoder version: {encoder_version}")
        settings = data.get("generation_settings")
        if not isinstance(settings, dict):
            raise TypeError("metadata field 'generation_settings' must be an object")
        git_commit = data.get("git_commit")
        if git_commit is not None and not isinstance(git_commit, str):
            raise TypeError("metadata field 'git_commit' must be a string or null")
        model_checkpoint_id = data.get("model_checkpoint_id")
        if model_checkpoint_id is not None and not isinstance(model_checkpoint_id, str):
            raise TypeError("metadata field 'model_checkpoint_id' must be a string or null")
        return cls(
            schema_version=schema_version,
            generated_at=_expect_str(data, "generated_at"),
            engine_version=_expect_str(data, "engine_version"),
            git_commit=git_commit,
            action_space_version=action_space_version,
            encoder_version=encoder_version,
            model_checkpoint_id=model_checkpoint_id,
            generation_settings=dict(settings),
            sample_count=_expect_int(data, "sample_count"),
            game_count=_expect_int(data, "game_count"),
            policy_target_format=policy_target_format,
        )


@dataclass(frozen=True, slots=True, init=False)
class SelfPlayDataset:
    """In-memory self-play samples plus metadata."""

    positions: npt.NDArray[np.float32]
    legal_masks: npt.NDArray[np.float32]
    policy_targets: SparsePolicyTargets
    outcomes: npt.NDArray[np.float32]
    metadata: SelfPlayMetadata
    games: list[SelfPlayGameRecord]
    _mcts_policies_cache: npt.NDArray[np.float32] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        *,
        positions: npt.NDArray[np.float32],
        legal_masks: npt.NDArray[np.float32],
        outcomes: npt.NDArray[np.float32],
        metadata: SelfPlayMetadata,
        games: list[SelfPlayGameRecord],
        mcts_policies: npt.NDArray[np.float32] | None = None,
        policy_targets: SparsePolicyTargets | None = None,
    ) -> None:
        """Create a dataset from sparse targets or legacy dense MCTS policies."""
        if (mcts_policies is None) == (policy_targets is None):
            raise ValueError("provide exactly one of mcts_policies or policy_targets")
        dense_cache: npt.NDArray[np.float32] | None = None
        if policy_targets is None:
            if mcts_policies is None:  # pragma: no cover - narrowed above
                raise ValueError("mcts_policies are required")
            dense_cache = np.asarray(mcts_policies, dtype=np.float32)
            policy_targets = SparsePolicyTargets.from_dense(dense_cache)
        if metadata.sample_count != policy_targets.sample_count:
            raise ValueError("policy target sample count does not match dataset metadata")
        object.__setattr__(self, "positions", np.asarray(positions, dtype=np.float32))
        object.__setattr__(self, "legal_masks", np.asarray(legal_masks, dtype=np.float32))
        object.__setattr__(self, "policy_targets", policy_targets)
        object.__setattr__(self, "outcomes", np.asarray(outcomes, dtype=np.float32))
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "games", games)
        object.__setattr__(self, "_mcts_policies_cache", dense_cache)

    @property
    def mcts_policies(self) -> npt.NDArray[np.float32]:
        """Return dense MCTS policy targets for compatibility with legacy callers."""
        cached = self._mcts_policies_cache
        if cached is None:
            cached = self.policy_targets.to_dense()
            object.__setattr__(self, "_mcts_policies_cache", cached)
        return cached


@dataclass(frozen=True, slots=True)
class SelfPlayShardTensors:
    """Tensor payload loaded from one temporary self-play shard."""

    path: Path
    metadata: SelfPlayMetadata
    games: list[SelfPlayGameRecord]
    positions: npt.NDArray[np.float32]
    legal_masks: npt.NDArray[np.float32]
    policy_offsets: npt.NDArray[np.int64]
    policy_indices: npt.NDArray[np.int32]
    policy_probabilities: npt.NDArray[np.float32]
    outcomes: npt.NDArray[np.float32]

    @property
    def policy_targets(self) -> SparsePolicyTargets:
        return SparsePolicyTargets(
            offsets=self.policy_offsets,
            indices=self.policy_indices,
            probabilities=self.policy_probabilities,
        )


@dataclass(frozen=True, slots=True)
class SelfPlayShardManifest:
    """Metadata needed to stream one temporary shard into the final dataset."""

    path: Path
    start_game: int
    game_count: int
    sample_count: int
    policy_nnz: int
    metadata: SelfPlayMetadata
    games: list[SelfPlayGameRecord]

def merge_self_play_datasets(
    datasets: list[SelfPlayDataset],
    *,
    config: SelfPlayConfig | None = None,
    generation_settings_extra: dict[str, object] | None = None,
) -> SelfPlayDataset:
    """Merge self-play dataset shards into one dataset with contiguous game indexes."""
    with profile_scope("dataset.merge", shards=len(datasets)):
        return _merge_self_play_datasets_impl(
            datasets,
            config=config,
            generation_settings_extra=generation_settings_extra,
        )


def _merge_self_play_datasets_impl(
    datasets: list[SelfPlayDataset],
    *,
    config: SelfPlayConfig | None = None,
    generation_settings_extra: dict[str, object] | None = None,
) -> SelfPlayDataset:
    if not datasets:
        raise ValueError("at least one self-play dataset is required")

    first = datasets[0]
    model_checkpoint_id = first.metadata.model_checkpoint_id

    games: list[SelfPlayGameRecord] = []
    for dataset in datasets:
        _validate_dataset_counts(dataset)
        if dataset.metadata.schema_version not in {
            SELF_PLAY_DATASET_SCHEMA_VERSION_V1,
            SELF_PLAY_DATASET_SCHEMA_VERSION,
        }:
            schema = dataset.metadata.schema_version
            raise ValueError(f"unsupported self-play dataset schema: {schema}")
        if dataset.metadata.action_space_version != ACTION_SPACE_VERSION:
            action_space = dataset.metadata.action_space_version
            raise ValueError(f"unsupported action space version: {action_space}")
        if dataset.metadata.encoder_version != ENCODER_VERSION:
            raise ValueError(f"unsupported encoder version: {dataset.metadata.encoder_version}")
        if dataset.metadata.model_checkpoint_id != model_checkpoint_id:
            raise ValueError("cannot merge datasets from different model checkpoints")
        for record in dataset.games:
            games.append(replace(record, game_index=len(games)))

    positions = np.concatenate([dataset.positions for dataset in datasets], axis=0)
    legal_masks = np.concatenate([dataset.legal_masks for dataset in datasets], axis=0)
    policy_targets = SparsePolicyTargets.concatenate(
        [dataset.policy_targets for dataset in datasets]
    )
    outcomes = np.concatenate([dataset.outcomes for dataset in datasets], axis=0)

    if config is None:
        generation_settings: dict[str, object] = {
            "merged_from": len(datasets),
            "source_generation_settings": [
                dataset.metadata.generation_settings for dataset in datasets
            ],
            **(generation_settings_extra or {}),
        }
        metadata = SelfPlayMetadata.create_from_settings(
            generation_settings,
            sample_count=int(outcomes.shape[0]),
            game_count=len(games),
            model_checkpoint_id=model_checkpoint_id,
        )
    else:
        metadata_batching_settings = _metadata_batching_settings(first.metadata)
        if metadata_batching_settings is None:
            metadata = SelfPlayMetadata.create(config, sample_count=int(outcomes.shape[0]))
        else:
            batching_mode, inference_batch_size = metadata_batching_settings
            metadata = SelfPlayMetadata.create(
                config,
                sample_count=int(outcomes.shape[0]),
                batching_mode=batching_mode,
                inference_batch_size=inference_batch_size,
            )
        if generation_settings_extra:
            metadata = replace(
                metadata,
                generation_settings={
                    **metadata.generation_settings,
                    **generation_settings_extra,
                },
            )

    return SelfPlayDataset(
        positions=positions,
        legal_masks=legal_masks,
        policy_targets=policy_targets,
        outcomes=outcomes,
        metadata=metadata,
        games=games,
    )


def _metadata_batching_settings(metadata: SelfPlayMetadata) -> tuple[str, int] | None:
    settings = metadata.generation_settings
    batching_mode = settings.get("batching_mode")
    inference_batch_size = settings.get("inference_batch_size")
    if (
        not isinstance(batching_mode, str)
        or isinstance(inference_batch_size, bool)
        or not isinstance(inference_batch_size, int)
    ):
        return None
    return batching_mode, inference_batch_size


def save_self_play_dataset(dataset: SelfPlayDataset, directory: str | Path) -> None:
    """Write a self-play dataset as compressed NPZ tensors plus JSON/JSONL metadata."""
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = replace(
        dataset.metadata,
        schema_version=SELF_PLAY_DATASET_SCHEMA_VERSION,
        policy_target_format=POLICY_TARGET_FORMAT_SPARSE_CSR,
    )
    with profile_scope("dataset.save"):
        with profile_scope("dataset.save_npz_compressed"):
            np.savez_compressed(
                output_dir / DEFAULT_DATASET_FILENAME,
                positions=dataset.positions,
                legal_masks=dataset.legal_masks,
                policy_offsets=dataset.policy_targets.offsets,
                policy_indices=dataset.policy_targets.indices,
                policy_probabilities=dataset.policy_targets.probabilities,
                outcomes=dataset.outcomes,
            )
        with profile_scope("dataset.write_metadata"):
            (output_dir / DEFAULT_METADATA_FILENAME).write_text(
                json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n"
            )
        with profile_scope("dataset.write_games_jsonl"):
            (output_dir / DEFAULT_GAMES_FILENAME).write_text(
                "".join(
                    json.dumps(record.to_dict(), sort_keys=True) + "\n"
                    for record in dataset.games
                )
            )


def save_self_play_shard(dataset: SelfPlayDataset, directory: str | Path) -> None:
    """Write a temporary self-play shard using uncompressed NPZ tensors."""
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = replace(
        dataset.metadata,
        schema_version=SELF_PLAY_DATASET_SCHEMA_VERSION,
        policy_target_format=POLICY_TARGET_FORMAT_SPARSE_CSR,
    )
    with profile_scope("dataset.save_shard"):
        with profile_scope("dataset.save_shard_npz_uncompressed"):
            np.savez(
                output_dir / TEMP_SHARD_DATASET_FILENAME,
                positions=dataset.positions,
                legal_masks=dataset.legal_masks,
                policy_offsets=dataset.policy_targets.offsets,
                policy_indices=dataset.policy_targets.indices,
                policy_probabilities=dataset.policy_targets.probabilities,
                outcomes=dataset.outcomes,
            )
        _write_self_play_sidecars(output_dir, metadata, dataset.games)


def load_self_play_shard_manifest(
    directory: str | Path,
    *,
    start_game: int,
) -> SelfPlayShardManifest:
    """Read temporary shard metadata without constructing a full dataset."""
    input_dir = Path(directory)
    metadata, games = _read_self_play_sidecars(input_dir)
    policy_shape = _npz_member_shape(input_dir / TEMP_SHARD_DATASET_FILENAME, "policy_indices")
    policy_nnz = int(policy_shape[0]) if policy_shape else 0
    return SelfPlayShardManifest(
        path=input_dir,
        start_game=start_game,
        game_count=metadata.game_count,
        sample_count=metadata.sample_count,
        policy_nnz=policy_nnz,
        metadata=metadata,
        games=games,
    )


def load_self_play_shard_tensors(directory: str | Path) -> SelfPlayShardTensors:
    """Load and validate tensors from one temporary self-play shard."""
    input_dir = Path(directory)
    metadata, games = _read_self_play_sidecars(input_dir)
    with np.load(input_dir / TEMP_SHARD_DATASET_FILENAME) as tensors:
        positions = np.asarray(tensors["positions"], dtype=np.float32)
        legal_masks = np.asarray(tensors["legal_masks"], dtype=np.float32)
        policy_targets = SparsePolicyTargets(
            offsets=np.asarray(tensors["policy_offsets"], dtype=np.int64),
            indices=np.asarray(tensors["policy_indices"], dtype=np.int32),
            probabilities=np.asarray(tensors["policy_probabilities"], dtype=np.float32),
        )
        outcomes = np.asarray(tensors["outcomes"], dtype=np.float32)
    _validate_tensor_shapes(metadata, positions, legal_masks, policy_targets, outcomes)
    if len(games) != metadata.game_count:
        raise ValueError("game metadata count does not match dataset metadata")
    _validate_game_records(metadata, games, positions, legal_masks, policy_targets, outcomes)
    return SelfPlayShardTensors(
        path=input_dir,
        metadata=metadata,
        games=games,
        positions=positions,
        legal_masks=legal_masks,
        policy_offsets=policy_targets.offsets,
        policy_indices=policy_targets.indices,
        policy_probabilities=policy_targets.probabilities,
        outcomes=outcomes,
    )


def save_merged_self_play_shards(
    shards: list[SelfPlayShardManifest],
    output: str | Path,
    *,
    config: SelfPlayConfig,
    generation_settings_extra: dict[str, object] | None = None,
    compressed: bool = True,
) -> SelfPlayMetadata:
    """Stream temporary shards into one public self-play dataset output."""
    if not shards:
        raise ValueError("at least one self-play shard is required")

    ordered = sorted(shards, key=lambda shard: shard.start_game)
    first = ordered[0]
    model_checkpoint_id = first.metadata.model_checkpoint_id
    games: list[SelfPlayGameRecord] = []
    total_samples = 0
    total_policy_nnz = 0
    game_cursor = 0
    for shard in ordered:
        if shard.start_game != game_cursor:
            raise ValueError("self-play shard ranges must be contiguous from game zero")
        _validate_shard_manifest(shard, model_checkpoint_id)
        total_samples += shard.sample_count
        total_policy_nnz += shard.policy_nnz
        game_cursor += shard.game_count
        for record in shard.games:
            games.append(replace(record, game_index=len(games)))

    positions = np.empty((total_samples, *TENSOR_SHAPE), dtype=np.float32)
    legal_masks = np.empty((total_samples, ACTION_SPACE_SIZE), dtype=np.float32)
    outcomes = np.empty((total_samples,), dtype=np.float32)
    policy_offsets = np.empty((total_samples + 1,), dtype=np.int64)
    policy_indices = np.empty((total_policy_nnz,), dtype=np.int32)
    policy_probabilities = np.empty((total_policy_nnz,), dtype=np.float32)
    policy_offsets[0] = 0

    sample_cursor = 0
    nnz_cursor = 0
    with profile_scope("dataset.merge_shards_streamed", shards=len(ordered)):
        for shard in ordered:
            tensors = load_self_play_shard_tensors(shard.path)
            row_count = tensors.metadata.sample_count
            shard_nnz = int(tensors.policy_indices.shape[0])
            if row_count != shard.sample_count:
                raise ValueError("shard manifest sample count does not match tensors")
            if shard_nnz != shard.policy_nnz:
                raise ValueError("shard manifest policy nnz does not match tensors")
            next_sample = sample_cursor + row_count
            next_nnz = nnz_cursor + shard_nnz
            positions[sample_cursor:next_sample] = tensors.positions
            legal_masks[sample_cursor:next_sample] = tensors.legal_masks
            outcomes[sample_cursor:next_sample] = tensors.outcomes
            policy_offsets[sample_cursor + 1 : next_sample + 1] = (
                tensors.policy_offsets[1:] + nnz_cursor
            )
            policy_indices[nnz_cursor:next_nnz] = tensors.policy_indices
            policy_probabilities[nnz_cursor:next_nnz] = tensors.policy_probabilities
            sample_cursor = next_sample
            nnz_cursor = next_nnz

    _validate_merge_config(config, model_checkpoint_id, game_count=len(games))
    metadata = _merged_metadata(
        first.metadata,
        config=config,
        sample_count=total_samples,
        game_count=len(games),
        generation_settings_extra=generation_settings_extra,
    )
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with profile_scope("dataset.save"):
        with profile_scope(
            "dataset.save_npz_compressed" if compressed else "dataset.save_npz_uncompressed"
        ):
            save_npz = np.savez_compressed if compressed else np.savez
            save_npz(
                output_dir / DEFAULT_DATASET_FILENAME,
                positions=positions,
                legal_masks=legal_masks,
                policy_offsets=policy_offsets,
                policy_indices=policy_indices,
                policy_probabilities=policy_probabilities,
                outcomes=outcomes,
            )
        _write_self_play_sidecars(output_dir, metadata, games)
    return metadata

def load_self_play_dataset(directory: str | Path) -> SelfPlayDataset:
    """Load and validate a self-play dataset from disk."""
    input_dir = Path(directory)
    metadata_data = json.loads((input_dir / DEFAULT_METADATA_FILENAME).read_text())
    if not isinstance(metadata_data, dict):
        raise TypeError("self-play metadata must be a JSON object")
    metadata = SelfPlayMetadata.from_dict(metadata_data)
    dense_cache: npt.NDArray[np.float32] | None = None
    policy_targets: SparsePolicyTargets | None = None
    with np.load(input_dir / DEFAULT_DATASET_FILENAME) as tensors:
        positions = np.asarray(tensors["positions"], dtype=np.float32)
        legal_masks = np.asarray(tensors["legal_masks"], dtype=np.float32)
        outcomes = np.asarray(tensors["outcomes"], dtype=np.float32)
        if metadata.policy_target_format == POLICY_TARGET_FORMAT_DENSE:
            dense_cache = np.asarray(tensors["mcts_policies"], dtype=np.float32)
        else:
            policy_targets = SparsePolicyTargets(
                offsets=np.asarray(tensors["policy_offsets"], dtype=np.int64),
                indices=np.asarray(tensors["policy_indices"], dtype=np.int32),
                probabilities=np.asarray(tensors["policy_probabilities"], dtype=np.float32),
            )
    if dense_cache is not None:
        _validate_dense_policy_shape(metadata, dense_cache)
        _validate_dense_policy_targets(metadata, dense_cache, legal_masks)
        policy_targets = SparsePolicyTargets.from_dense(dense_cache)
    if policy_targets is None:  # pragma: no cover - narrowed by metadata format validation
        raise ValueError("policy targets were not loaded")
    _validate_tensor_shapes(metadata, positions, legal_masks, policy_targets, outcomes)
    games = [
        SelfPlayGameRecord.from_dict(record)
        for record in _read_jsonl(input_dir / DEFAULT_GAMES_FILENAME)
    ]
    if len(games) != metadata.game_count:
        raise ValueError("game metadata count does not match dataset metadata")
    _validate_game_records(metadata, games, positions, legal_masks, policy_targets, outcomes)
    if dense_cache is not None:
        return SelfPlayDataset(
            positions=positions,
            legal_masks=legal_masks,
            mcts_policies=dense_cache,
            outcomes=outcomes,
            metadata=metadata,
            games=games,
        )
    return SelfPlayDataset(
        positions=positions,
        legal_masks=legal_masks,
        policy_targets=policy_targets,
        outcomes=outcomes,
        metadata=metadata,
        games=games,
    )


def _validate_dataset_counts(dataset: SelfPlayDataset) -> None:
    expected = dataset.metadata.sample_count
    if dataset.positions.shape[0] != expected:
        raise ValueError("positions sample count does not match dataset metadata")
    if dataset.legal_masks.shape[0] != expected:
        raise ValueError("legal_masks sample count does not match dataset metadata")
    if dataset.policy_targets.sample_count != expected:
        raise ValueError("policy target sample count does not match dataset metadata")
    if dataset.outcomes.shape[0] != expected:
        raise ValueError("outcomes sample count does not match dataset metadata")
    if len(dataset.games) != dataset.metadata.game_count:
        raise ValueError("game count does not match dataset metadata")


def _outcome_values(game: Game, sides: list[Color]) -> list[float]:
    outcome = game.outcome
    if outcome is None or outcome.winner is None:
        return [0.0 for _side in sides]
    return [1.0 if outcome.winner is side else -1.0 for side in sides]


def _validate_tensor_shapes(
    metadata: SelfPlayMetadata,
    positions: npt.NDArray[np.float32],
    legal_masks: npt.NDArray[np.float32],
    policy_targets: SparsePolicyTargets,
    outcomes: npt.NDArray[np.float32],
) -> None:
    expected = metadata.sample_count
    if positions.shape != (expected, *TENSOR_SHAPE):
        raise ValueError(f"positions shape mismatch: {positions.shape}")
    if legal_masks.shape != (expected, ACTION_SPACE_SIZE):
        raise ValueError(f"legal_masks shape mismatch: {legal_masks.shape}")
    if policy_targets.sample_count != expected:
        raise ValueError("policy target sample count does not match dataset metadata")
    if outcomes.shape != (expected,):
        raise ValueError(f"outcomes shape mismatch: {outcomes.shape}")


def _validate_dense_policy_shape(
    metadata: SelfPlayMetadata,
    mcts_policies: npt.NDArray[np.float32],
) -> None:
    expected = metadata.sample_count
    if mcts_policies.shape != (expected, ACTION_SPACE_SIZE):
        raise ValueError(f"mcts_policies shape mismatch: {mcts_policies.shape}")


def _validate_dense_policy_targets(
    metadata: SelfPlayMetadata,
    mcts_policies: npt.NDArray[np.float32],
    legal_masks: npt.NDArray[np.float32],
) -> None:
    expected = metadata.sample_count
    if legal_masks.shape != (expected, ACTION_SPACE_SIZE):
        raise ValueError(f"legal_masks shape mismatch: {legal_masks.shape}")
    if not np.all(np.isfinite(mcts_policies)):
        raise ValueError("mcts_policies contain non-finite values")
    if np.any(mcts_policies < 0.0):
        raise ValueError("mcts_policies contain negative values")
    row_sums = mcts_policies.sum(axis=1)
    if not np.all(np.isclose(row_sums, 1.0)):
        raise ValueError("mcts_policies rows must sum to 1.0")
    if np.any((mcts_policies > 0.0) & (legal_masks <= 0.0)):
        raise ValueError("mcts_policies assign probability to illegal moves")


def _validate_sparse_storage_shapes(policy_targets: SparsePolicyTargets) -> None:
    if policy_targets.offsets.ndim != 1:
        raise ValueError("policy_offsets must be one-dimensional")
    if policy_targets.indices.ndim != 1:
        raise ValueError("policy_indices must be one-dimensional")
    if policy_targets.probabilities.ndim != 1:
        raise ValueError("policy_probabilities must be one-dimensional")
    if policy_targets.offsets.shape[0] < 1:
        raise ValueError("policy_offsets must contain at least one element")
    if policy_targets.offsets[0] != 0:
        raise ValueError("policy_offsets must start at zero")
    if np.any(np.diff(policy_targets.offsets) < 0):
        raise ValueError("policy_offsets must be monotonic")
    if policy_targets.indices.shape != policy_targets.probabilities.shape:
        raise ValueError("policy_indices and policy_probabilities length mismatch")
    if int(policy_targets.offsets[-1]) != int(policy_targets.indices.shape[0]):
        raise ValueError("policy_offsets final value must match policy index count")


def _validate_game_records(
    metadata: SelfPlayMetadata,
    games: list[SelfPlayGameRecord],
    positions: npt.NDArray[np.float32],
    legal_masks: npt.NDArray[np.float32],
    policy_targets: SparsePolicyTargets,
    outcomes: npt.NDArray[np.float32],
) -> None:
    sample_index = 0
    for expected_game_index, record in enumerate(games):
        if record.game_index != expected_game_index:
            raise ValueError("game_index values must be contiguous starting at 0")
        if record.plies != len(record.moves_uci):
            raise ValueError("game record plies must match moves_uci length")
        game = Game.new()
        sides: list[Color] = []
        for move_uci in record.moves_uci:
            if sample_index >= metadata.sample_count:
                raise ValueError("game records contain more plies than tensor samples")
            expected_position = encode_game_np(game)
            if not np.allclose(positions[sample_index], expected_position):
                raise ValueError("position tensor does not match replayed game state")
            legal = game.legal_moves
            expected_mask = legal_move_mask_from_legal_moves_np(game, legal)
            if not np.array_equal(legal_masks[sample_index], expected_mask):
                raise ValueError("legal mask does not match replayed game state")
            _validate_sparse_policy_row(policy_targets, sample_index, expected_mask)
            move = Move.from_uci(move_uci)
            if move not in legal:
                raise ValueError(f"illegal move in game record: {move_uci}")
            sides.append(game.board.side_to_move)
            game = game.play_known_legal(move)
            sample_index += 1
        if game.to_fen() != record.final_fen:
            raise ValueError("game record final_fen does not match replayed moves")
        _validate_recorded_outcome(record, game)
        expected_game = _game_with_recorded_outcome(record, game)
        expected_outcomes = np.asarray(_outcome_values(expected_game, sides))
        start = sample_index - record.plies
        if not np.allclose(outcomes[start:sample_index], expected_outcomes):
            raise ValueError("outcome targets do not match recorded game outcome")
    if sample_index != metadata.sample_count:
        raise ValueError("total game plies does not match metadata sample_count")


def _validate_sparse_policy_row(
    policy_targets: SparsePolicyTargets,
    row_index: int,
    legal_mask: npt.NDArray[np.float32],
) -> None:
    indices, probabilities = policy_targets.row(row_index)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("policy target contains non-finite values")
    if np.any(probabilities < 0.0):
        raise ValueError("policy target contains negative values")
    if np.any(indices < 0) or np.any(indices >= ACTION_SPACE_SIZE):
        raise ValueError("policy target contains out-of-range action indices")
    if np.unique(indices).shape[0] != indices.shape[0]:
        raise ValueError("policy target contains duplicate action indices")
    if not np.isclose(float(probabilities.sum()), 1.0):
        raise ValueError("policy target row must sum to 1.0")
    if indices.size and np.any(legal_mask[indices] <= 0.0):
        raise ValueError("policy target assigns probability to illegal moves")

def _merged_metadata(
    first_metadata: SelfPlayMetadata,
    *,
    config: SelfPlayConfig,
    sample_count: int,
    game_count: int,
    generation_settings_extra: dict[str, object] | None,
) -> SelfPlayMetadata:
    metadata_batching_settings = _metadata_batching_settings(first_metadata)
    if metadata_batching_settings is None:
        metadata = SelfPlayMetadata.create(config, sample_count=sample_count)
    else:
        batching_mode, inference_batch_size = metadata_batching_settings
        metadata = SelfPlayMetadata.create(
            config,
            sample_count=sample_count,
            batching_mode=batching_mode,
            inference_batch_size=inference_batch_size,
        )
    if metadata.model_checkpoint_id != first_metadata.model_checkpoint_id:
        metadata = replace(metadata, model_checkpoint_id=first_metadata.model_checkpoint_id)
    if metadata.game_count != game_count:
        metadata = replace(metadata, game_count=game_count)
    if generation_settings_extra:
        metadata = replace(
            metadata,
            generation_settings={
                **metadata.generation_settings,
                **generation_settings_extra,
            },
        )
    return metadata


def _validate_merge_config(
    config: SelfPlayConfig,
    model_checkpoint_id: str | None,
    *,
    game_count: int,
) -> None:
    if config.model_checkpoint_id != model_checkpoint_id:
        raise ValueError("merge config model checkpoint does not match shard metadata")
    if config.games != game_count:
        raise ValueError("merge config game count does not match shard ranges")


def _validate_shard_manifest(
    shard: SelfPlayShardManifest,
    model_checkpoint_id: str | None,
) -> None:
    if shard.metadata.schema_version != SELF_PLAY_DATASET_SCHEMA_VERSION:
        schema = shard.metadata.schema_version
        raise ValueError(f"unsupported self-play shard schema: {schema}")
    if shard.metadata.policy_target_format != POLICY_TARGET_FORMAT_SPARSE_CSR:
        raise ValueError("temporary self-play shards must use sparse CSR policy targets")
    if shard.metadata.action_space_version != ACTION_SPACE_VERSION:
        action_space = shard.metadata.action_space_version
        raise ValueError(f"unsupported action space version: {action_space}")
    if shard.metadata.encoder_version != ENCODER_VERSION:
        raise ValueError(f"unsupported encoder version: {shard.metadata.encoder_version}")
    if shard.metadata.model_checkpoint_id != model_checkpoint_id:
        raise ValueError("cannot merge shards from different model checkpoints")
    if shard.game_count != shard.metadata.game_count:
        raise ValueError("shard game count does not match metadata")
    if shard.sample_count != shard.metadata.sample_count:
        raise ValueError("shard sample count does not match metadata")
    if len(shard.games) != shard.game_count:
        raise ValueError("shard game records do not match metadata")


def _write_self_play_sidecars(
    output_dir: Path,
    metadata: SelfPlayMetadata,
    games: list[SelfPlayGameRecord],
) -> None:
    with profile_scope("dataset.write_metadata"):
        (output_dir / DEFAULT_METADATA_FILENAME).write_text(
            json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n"
        )
    with profile_scope("dataset.write_games_jsonl"):
        (output_dir / DEFAULT_GAMES_FILENAME).write_text(
            "".join(json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in games)
        )


def _read_self_play_sidecars(input_dir: Path) -> tuple[SelfPlayMetadata, list[SelfPlayGameRecord]]:
    metadata_data = json.loads((input_dir / DEFAULT_METADATA_FILENAME).read_text())
    if not isinstance(metadata_data, dict):
        raise TypeError("self-play metadata must be a JSON object")
    metadata = SelfPlayMetadata.from_dict(metadata_data)
    games = [
        SelfPlayGameRecord.from_dict(record)
        for record in _read_jsonl(input_dir / DEFAULT_GAMES_FILENAME)
    ]
    return metadata, games


def _npz_member_shape(path: Path, name: str) -> tuple[int, ...]:
    with zipfile.ZipFile(path) as archive:
        with archive.open(f"{name}.npy") as member:
            version = np.lib.format.read_magic(member)
            if version == (1, 0):
                shape, _fortran_order, _dtype = np.lib.format.read_array_header_1_0(member)
            elif version == (2, 0):
                shape, _fortran_order, _dtype = np.lib.format.read_array_header_2_0(member)
            else:
                raise ValueError(f"unsupported npy header version: {version}")
    return tuple(int(dimension) for dimension in shape)




def _validate_recorded_outcome(record: SelfPlayGameRecord, game: Game) -> None:
    recorded = _recorded_outcome(record)
    actual = game.outcome
    if actual is None and recorded.reason is not OutcomeReason.MAX_PLIES:
        raise ValueError("non-terminal replay must be recorded as max_plies")
    if actual is not None and actual != recorded:
        raise ValueError("game record outcome does not match replayed game outcome")


def _game_with_recorded_outcome(record: SelfPlayGameRecord, game: Game) -> Game:
    recorded = _recorded_outcome(record)
    if game.outcome == recorded:
        return game
    return game.with_forced_outcome(recorded)


def _recorded_outcome(record: SelfPlayGameRecord) -> Outcome:
    try:
        reason = OutcomeReason(record.outcome_reason)
    except ValueError as exc:
        raise ValueError(f"unsupported game outcome reason: {record.outcome_reason}") from exc
    winner = None
    if record.winner is not None:
        try:
            winner = Color(record.winner)
        except ValueError as exc:
            raise ValueError(f"unsupported game winner: {record.winner}") from exc
    return Outcome(reason=reason, winner=winner)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise TypeError("game metadata JSONL records must be objects")
        records.append(record)
    return records


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _expect_str(data: dict[str, object], key: str) -> str:
    return _jsonio.expect_str(data, key)


def _expect_int(data: dict[str, object], key: str) -> int:
    return _jsonio.expect_int(data, key)


__all__ = [
    "DEFAULT_DATASET_FILENAME",
    "DEFAULT_GAMES_FILENAME",
    "DEFAULT_METADATA_FILENAME",
    "POLICY_TARGET_FORMAT_DENSE",
    "POLICY_TARGET_FORMAT_SPARSE_CSR",
    "SELF_PLAY_DATASET_SCHEMA_VERSION",
    "SELF_PLAY_DATASET_SCHEMA_VERSION_V1",
    "SelfPlayDataset",
    "SelfPlayGameRecord",
    "SelfPlayMetadata",
    "SelfPlayShardManifest",
    "SelfPlayShardTensors",
    "SparsePolicyTargets",
    "TEMP_SHARD_DATASET_FILENAME",
    "load_self_play_dataset",
    "load_self_play_shard_manifest",
    "load_self_play_shard_tensors",
    "merge_self_play_datasets",
    "save_merged_self_play_shards",
    "save_self_play_dataset",
    "save_self_play_shard",
]
