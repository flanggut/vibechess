from __future__ import annotations

from pathlib import Path

import numpy as np

import tinychess.nn.self_play as old_self_play
from tinychess.engine import Game, OutcomeReason
from tinychess.nn.encode import (
    ACTION_SPACE_SIZE,
    encode_game_np,
    legal_move_mask_from_legal_moves_np,
    move_to_action_index,
)
from tinychess.nn.self_play import SelfPlayConfig
from tinychess.nn.self_play_dataset import (
    SELF_PLAY_DATASET_SCHEMA_VERSION,
    SelfPlayDataset,
    SelfPlayGameRecord,
    SelfPlayMetadata,
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


def test_self_play_dataset_module_saves_and_loads_round_trip(tmp_path: Path) -> None:
    dataset = _one_move_dataset(checkpoint_id="shared")

    save_self_play_dataset(dataset, tmp_path)
    loaded = load_self_play_dataset(tmp_path)

    assert loaded.metadata.schema_version == SELF_PLAY_DATASET_SCHEMA_VERSION
    assert loaded.metadata.model_checkpoint_id == "shared"
    np.testing.assert_array_equal(loaded.positions, dataset.positions)
    np.testing.assert_array_equal(loaded.legal_masks, dataset.legal_masks)
    np.testing.assert_array_equal(loaded.mcts_policies, dataset.mcts_policies)
    np.testing.assert_array_equal(loaded.outcomes, dataset.outcomes)
    assert loaded.games == dataset.games


def test_self_play_dataset_module_merges_and_reindexes_games() -> None:
    first = _one_move_dataset(checkpoint_id="shared")
    second = _one_move_dataset(checkpoint_id="shared")

    merged = merge_self_play_datasets([first, second])

    assert merged.metadata.sample_count == 2
    assert merged.metadata.game_count == 2
    assert merged.metadata.model_checkpoint_id == "shared"
    assert [record.game_index for record in merged.games] == [0, 1]


def test_old_self_play_dataset_imports_reexport_new_module_symbols() -> None:
    assert old_self_play.SelfPlayDataset is SelfPlayDataset
    assert old_self_play.SelfPlayGameRecord is SelfPlayGameRecord
    assert old_self_play.SelfPlayMetadata is SelfPlayMetadata
    assert old_self_play.load_self_play_dataset is load_self_play_dataset
    assert old_self_play.merge_self_play_datasets is merge_self_play_datasets
    assert old_self_play.save_self_play_dataset is save_self_play_dataset
