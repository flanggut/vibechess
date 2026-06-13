from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import vibechess.nn.self_play as old_self_play
from vibechess.engine import Game, OutcomeReason
from vibechess.nn.encode import (
    ACTION_SPACE_SIZE,
    encode_game_np,
    legal_move_mask_from_legal_moves_np,
    move_to_action_index,
)
from vibechess.nn.self_play import SelfPlayConfig
from vibechess.nn.self_play_dataset import (
    DEFAULT_DATASET_FILENAME,
    DEFAULT_GAMES_FILENAME,
    DEFAULT_METADATA_FILENAME,
    POLICY_TARGET_FORMAT_DENSE,
    POLICY_TARGET_FORMAT_SPARSE_CSR,
    SELF_PLAY_DATASET_SCHEMA_VERSION,
    SELF_PLAY_DATASET_SCHEMA_VERSION_V1,
    SelfPlayDataset,
    SelfPlayGameRecord,
    SelfPlayMetadata,
    SparsePolicyTargets,
    load_self_play_dataset,
    merge_self_play_datasets,
    save_self_play_dataset,
)


def _one_move_dataset(*, checkpoint_id: str | None = None) -> SelfPlayDataset:
    game = Game.new()
    legal = game.legal_moves
    move = legal[0]
    policy = np.zeros((ACTION_SPACE_SIZE,), dtype=np.float32)
    policy[move_to_action_index(move, game.board)] = 1.0
    next_game = game.play_known_legal(move)
    metadata = SelfPlayMetadata.create(
        SelfPlayConfig(games=1, max_plies=1, model_checkpoint_id=checkpoint_id),
        sample_count=1,
    )
    return SelfPlayDataset(
        positions=np.asarray([encode_game_np(game)], dtype=np.float32),
        legal_masks=np.asarray(
            [legal_move_mask_from_legal_moves_np(game, legal)],
            dtype=np.float32,
        ),
        mcts_policies=np.asarray([policy], dtype=np.float32),
        outcomes=np.asarray([0.0], dtype=np.float32),
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


def test_sparse_policy_targets_round_trip_dense_rows_and_empty() -> None:
    dense = np.zeros((2, ACTION_SPACE_SIZE), dtype=np.float32)
    dense[0, [1, 3]] = [0.25, 0.75]
    dense[1, [7]] = [1.0]

    sparse = SparsePolicyTargets.from_dense(dense)

    np.testing.assert_array_equal(sparse.to_dense(), dense)
    np.testing.assert_array_equal(sparse.dense_rows(np.asarray([1, 0])), dense[[1, 0]])
    empty = SparsePolicyTargets.from_rows([])
    np.testing.assert_array_equal(empty.offsets, np.asarray([0], dtype=np.int64))
    assert empty.indices.shape == (0,)
    assert empty.probabilities.shape == (0,)


def test_self_play_dataset_accepts_sparse_targets_and_exposes_dense_compatibility() -> None:
    dense_dataset = _one_move_dataset()
    sparse = SparsePolicyTargets.from_dense(dense_dataset.mcts_policies)

    dataset = SelfPlayDataset(
        positions=dense_dataset.positions,
        legal_masks=dense_dataset.legal_masks,
        policy_targets=sparse,
        outcomes=dense_dataset.outcomes,
        metadata=dense_dataset.metadata,
        games=dense_dataset.games,
    )

    assert dataset.policy_targets is sparse
    np.testing.assert_array_equal(dataset.mcts_policies, dense_dataset.mcts_policies)


def test_metadata_parses_v1_dense_and_v2_sparse_formats() -> None:
    base = _one_move_dataset().metadata.to_dict()
    v1 = dict(base)
    v1["schema_version"] = SELF_PLAY_DATASET_SCHEMA_VERSION_V1
    v1.pop("policy_target_format", None)

    parsed_v1 = SelfPlayMetadata.from_dict(v1)
    parsed_v2 = SelfPlayMetadata.from_dict(base)

    assert parsed_v1.policy_target_format == POLICY_TARGET_FORMAT_DENSE
    assert parsed_v2.policy_target_format == POLICY_TARGET_FORMAT_SPARSE_CSR
    with pytest.raises(ValueError, match="unsupported v2 policy target format"):
        SelfPlayMetadata.from_dict({**base, "policy_target_format": "unknown"})


def test_self_play_dataset_module_saves_and_loads_round_trip(tmp_path: Path) -> None:
    dataset = _one_move_dataset(checkpoint_id="shared")

    save_self_play_dataset(dataset, tmp_path)
    loaded = load_self_play_dataset(tmp_path)

    assert loaded.metadata.schema_version == SELF_PLAY_DATASET_SCHEMA_VERSION
    assert loaded.metadata.policy_target_format == POLICY_TARGET_FORMAT_SPARSE_CSR
    assert loaded.metadata.model_checkpoint_id == "shared"
    with np.load(tmp_path / DEFAULT_DATASET_FILENAME) as tensors:
        assert "mcts_policies" not in tensors.files
        assert "policy_offsets" in tensors.files
        assert "policy_indices" in tensors.files
        assert "policy_probabilities" in tensors.files
    np.testing.assert_array_equal(loaded.positions, dataset.positions)
    np.testing.assert_array_equal(loaded.legal_masks, dataset.legal_masks)
    np.testing.assert_array_equal(loaded.mcts_policies, dataset.mcts_policies)
    np.testing.assert_array_equal(loaded.outcomes, dataset.outcomes)
    assert loaded.games == dataset.games


def test_load_self_play_dataset_accepts_legacy_v1_dense_npz(tmp_path: Path) -> None:
    dataset = _one_move_dataset(checkpoint_id="legacy")
    metadata = dataset.metadata.to_dict()
    metadata["schema_version"] = SELF_PLAY_DATASET_SCHEMA_VERSION_V1
    metadata.pop("policy_target_format", None)
    (tmp_path / DEFAULT_METADATA_FILENAME).write_text(json.dumps(metadata) + "\n")
    (tmp_path / DEFAULT_GAMES_FILENAME).write_text(
        "".join(json.dumps(record.to_dict()) + "\n" for record in dataset.games)
    )
    np.savez_compressed(
        tmp_path / DEFAULT_DATASET_FILENAME,
        positions=dataset.positions,
        legal_masks=dataset.legal_masks,
        mcts_policies=dataset.mcts_policies,
        outcomes=dataset.outcomes,
    )

    loaded = load_self_play_dataset(tmp_path)

    assert loaded.metadata.schema_version == SELF_PLAY_DATASET_SCHEMA_VERSION_V1
    assert loaded.metadata.policy_target_format == POLICY_TARGET_FORMAT_DENSE
    np.testing.assert_array_equal(loaded.mcts_policies, dataset.mcts_policies)
    np.testing.assert_array_equal(loaded.policy_targets.to_dense(), dataset.mcts_policies)


@pytest.mark.parametrize(
    ("invalid_value", "error"),
    [
        (-0.5, "negative values"),
        (np.nan, "non-finite values"),
    ],
)
def test_load_self_play_dataset_rejects_corrupt_legacy_v1_dense_npz(
    tmp_path: Path,
    invalid_value: float,
    error: str,
) -> None:
    dataset = _one_move_dataset(checkpoint_id="legacy")
    invalid_policy = dataset.mcts_policies.copy()
    illegal_action = int(np.flatnonzero(dataset.legal_masks[0] == 0.0)[0])
    invalid_policy[0, illegal_action] = invalid_value
    metadata = dataset.metadata.to_dict()
    metadata["schema_version"] = SELF_PLAY_DATASET_SCHEMA_VERSION_V1
    metadata.pop("policy_target_format", None)
    (tmp_path / DEFAULT_METADATA_FILENAME).write_text(json.dumps(metadata) + "\n")
    (tmp_path / DEFAULT_GAMES_FILENAME).write_text(
        "".join(json.dumps(record.to_dict()) + "\n" for record in dataset.games)
    )
    np.savez_compressed(
        tmp_path / DEFAULT_DATASET_FILENAME,
        positions=dataset.positions,
        legal_masks=dataset.legal_masks,
        mcts_policies=invalid_policy,
        outcomes=dataset.outcomes,
    )

    with pytest.raises(ValueError, match=error):
        load_self_play_dataset(tmp_path)


@pytest.mark.parametrize(
    ("indices", "probabilities", "error"),
    [
        (
            np.asarray([0, 0], dtype=np.int32),
            np.asarray([0.5, 0.5], dtype=np.float32),
            "duplicate action indices",
        ),
        (
            np.asarray([ACTION_SPACE_SIZE], dtype=np.int32),
            np.asarray([1.0], dtype=np.float32),
            "out-of-range action indices",
        ),
        (
            np.asarray([0], dtype=np.int32),
            np.asarray([-1.0], dtype=np.float32),
            "negative values",
        ),
        (
            np.asarray([0], dtype=np.int32),
            np.asarray([np.nan], dtype=np.float32),
            "non-finite values",
        ),
        (
            np.asarray([0], dtype=np.int32),
            np.asarray([0.5], dtype=np.float32),
            "sum to 1.0",
        ),
    ],
)
def test_load_self_play_dataset_rejects_corrupt_sparse_rows(
    tmp_path: Path,
    indices: np.ndarray,
    probabilities: np.ndarray,
    error: str,
) -> None:
    dataset = _one_move_dataset()
    save_self_play_dataset(dataset, tmp_path)
    legal_action = int(np.flatnonzero(dataset.legal_masks[0] > 0.0)[0])
    indices = np.where(indices == 0, legal_action, indices).astype(np.int32, copy=False)
    np.savez_compressed(
        tmp_path / DEFAULT_DATASET_FILENAME,
        positions=dataset.positions,
        legal_masks=dataset.legal_masks,
        policy_offsets=np.asarray([0, indices.shape[0]], dtype=np.int64),
        policy_indices=indices,
        policy_probabilities=probabilities,
        outcomes=dataset.outcomes,
    )

    with pytest.raises(ValueError, match=error):
        load_self_play_dataset(tmp_path)


def test_self_play_dataset_module_merges_and_reindexes_games() -> None:
    first = _one_move_dataset(checkpoint_id="shared")
    second = _one_move_dataset(checkpoint_id="shared")

    merged = merge_self_play_datasets([first, second])

    assert merged.metadata.schema_version == SELF_PLAY_DATASET_SCHEMA_VERSION
    assert merged.metadata.policy_target_format == POLICY_TARGET_FORMAT_SPARSE_CSR
    assert merged.metadata.sample_count == 2
    assert merged.metadata.game_count == 2
    assert merged.metadata.model_checkpoint_id == "shared"
    assert [record.game_index for record in merged.games] == [0, 1]
    np.testing.assert_array_equal(
        merged.mcts_policies,
        np.concatenate([first.mcts_policies, second.mcts_policies], axis=0),
    )


def test_old_self_play_dataset_imports_reexport_new_module_symbols() -> None:
    assert old_self_play.SelfPlayDataset is SelfPlayDataset
    assert old_self_play.SelfPlayGameRecord is SelfPlayGameRecord
    assert old_self_play.SelfPlayMetadata is SelfPlayMetadata
    assert old_self_play.load_self_play_dataset is load_self_play_dataset
    assert old_self_play.merge_self_play_datasets is merge_self_play_datasets
    assert old_self_play.save_self_play_dataset is save_self_play_dataset
