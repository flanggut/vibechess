from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

import tinychess.nn.self_play as self_play
from tinychess.ai.neural_mcts import NeuralMCTSConfig
from tinychess.ai.search_config import MCTSConfig
from tinychess.engine import Game, Move, OutcomeReason
from tinychess.nn.checkpoint import save_checkpoint
from tinychess.nn.encode import (
    ACTION_SPACE_SIZE,
    ACTION_SPACE_VERSION,
    ENCODER_VERSION,
    TENSOR_SHAPE,
    legal_move_mask_from_legal_moves_np,
    move_to_action_index,
)
from tinychess.nn.model import (
    InferenceResult,
    LegalPolicyBatchResult,
    MLXArray,
    PolicyValueConfig,
    PolicyValueInference,
    PolicyValueNet,
)
from tinychess.nn.self_play import (
    BATCHING_MODE_CENTRAL_INFERENCE_QUEUE,
    BATCHING_MODE_SERIAL,
    DEFAULT_DATASET_FILENAME,
    DEFAULT_GAMES_FILENAME,
    DEFAULT_METADATA_FILENAME,
    DEFAULT_PROFILE_FILENAME,
    LABEL_SOURCE_CLASSICAL,
    LABEL_SOURCE_NEURAL,
    SELF_PLAY_DATASET_SCHEMA_VERSION,
    SelfPlayConfig,
    SelfPlayMetadata,
    generate_self_play_dataset,
    load_self_play_dataset,
    merge_self_play_datasets,
    save_self_play_dataset,
    self_play_profile,
)


@dataclass(slots=True)
class FakeInference:
    calls: int = 0

    def predict(self, game: Game, *, mask_legal_moves: bool = True) -> InferenceResult:
        assert mask_legal_moves is True
        self.calls += 1
        values = [0.0] * ACTION_SPACE_SIZE
        if game.legal_moves:
            values[move_to_action_index(game.legal_moves[0], game.board)] = 1.0
        policy = mx.array(values, dtype=mx.float32)
        mx.eval(policy)
        return InferenceResult(policy_logits=policy, policy=policy, value=0.0)


class CountingPolicyValueInference(PolicyValueInference):
    def __init__(self, model: PolicyValueNet) -> None:
        super().__init__(model)
        self.batch_calls = 0
        self.legal_batch_calls = 0
        self.legal_batch_sizes: list[int] = []

    def predict_batch(
        self,
        inputs: Any,
        *,
        legal_masks: Any | None = None,
        legal_moves: Any | None = None,
        mask_legal_moves: bool = True,
    ) -> Any:
        self.batch_calls += 1
        return super().predict_batch(
            inputs,
            legal_masks=legal_masks,
            legal_moves=legal_moves,
            mask_legal_moves=mask_legal_moves,
        )

    def predict_legal_batch(self, games: Any, legal_moves: Any) -> Any:
        self.legal_batch_calls += 1
        self.legal_batch_sizes.append(len(games))
        return super().predict_legal_batch(games, legal_moves)


class DeterministicPolicyValueInference(PolicyValueInference):
    """Policy/value inference with identical single-row and batched legal results."""

    def __init__(self) -> None:
        super().__init__(
            PolicyValueNet(
                PolicyValueConfig(
                    residual_channels=4,
                    residual_blocks=0,
                    policy_channels=1,
                    value_channels=1,
                    value_hidden_dim=4,
                )
            )
        )

    def predict_with_legal_moves(
        self,
        game: Game,
        legal_moves: tuple[Move, ...],
    ) -> InferenceResult:
        legal = tuple(legal_moves)
        legal_policy = self._legal_policy(legal)
        policy = mx.zeros((ACTION_SPACE_SIZE,), dtype=mx.float32)
        mx.eval(policy, legal_policy)
        return InferenceResult(
            policy_logits=policy,
            policy=policy,
            value=0.0,
            legal_moves=legal,
            legal_action_indices=tuple(move_to_action_index(move, game.board) for move in legal),
            legal_policy=legal_policy,
        )

    def predict_legal_batch(
        self,
        games: Sequence[Game],
        legal_moves: Sequence[Sequence[Move]],
    ) -> LegalPolicyBatchResult:
        games_tuple = tuple(games)
        legal_tuple = tuple(tuple(row) for row in legal_moves)
        policies = tuple(self._legal_policy(legal) for legal in legal_tuple)
        mx.eval(*policies)
        return LegalPolicyBatchResult(
            values=tuple(0.0 for _game in games_tuple),
            legal_moves=legal_tuple,
            legal_action_indices=tuple(
                tuple(move_to_action_index(move, game.board) for move in legal)
                for game, legal in zip(games_tuple, legal_tuple, strict=True)
            ),
            legal_policies=policies,
        )

    @staticmethod
    def _legal_policy(legal_moves: tuple[Move, ...]) -> MLXArray:
        if not legal_moves:
            return mx.zeros((0,), dtype=mx.float32)
        weights = [1.0 / len(legal_moves)] * len(legal_moves)
        return mx.array(weights, dtype=mx.float32)


def test_generate_self_play_dataset_completes_smoke_game() -> None:
    dataset = generate_self_play_dataset(
        FakeInference(),
        SelfPlayConfig(
            games=1,
            max_plies=4,
            mcts=NeuralMCTSConfig(simulations=2, temperature=0.0, seed=3),
            model_checkpoint_id="fake-checkpoint",
            seed=3,
        ),
    )

    assert dataset.metadata.schema_version == SELF_PLAY_DATASET_SCHEMA_VERSION
    assert dataset.metadata.action_space_version
    assert dataset.metadata.engine_version
    assert dataset.metadata.model_checkpoint_id == "fake-checkpoint"
    assert dataset.metadata.generation_settings["batching_mode"] == BATCHING_MODE_SERIAL
    assert dataset.metadata.generation_settings["inference_batch_size"] == 1
    mcts_settings = dataset.metadata.generation_settings["mcts"]
    assert isinstance(mcts_settings, dict)
    removed_field = "leaf" + "_parallelism"
    assert removed_field not in mcts_settings
    assert dataset.positions.shape == (4, *TENSOR_SHAPE)
    assert dataset.legal_masks.shape == (4, ACTION_SPACE_SIZE)
    assert dataset.mcts_policies.shape == (4, ACTION_SPACE_SIZE)
    assert dataset.outcomes.shape == (4,)
    assert np.all(dataset.legal_masks.sum(axis=1) > 0)
    assert np.allclose(dataset.mcts_policies.sum(axis=1), 1.0)
    assert np.all(dataset.outcomes == 0.0)
    assert dataset.games[0].plies == 4
    assert dataset.games[0].outcome_reason == OutcomeReason.MAX_PLIES.value
    assert len(dataset.games[0].moves_uci) == 4


def test_batched_neural_self_play_preserves_schema_and_legal_targets() -> None:
    inference = CountingPolicyValueInference(
        PolicyValueNet(
            PolicyValueConfig(
                residual_channels=4,
                residual_blocks=0,
                policy_channels=1,
                value_channels=1,
                value_hidden_dim=4,
            )
        )
    )

    dataset = generate_self_play_dataset(
        inference,
        SelfPlayConfig(
            games=2,
            max_plies=2,
            mcts=NeuralMCTSConfig(simulations=1, temperature=0.0, seed=17),
            seed=17,
            batch_size=2,
        ),
    )

    assert inference.batch_calls == 0
    assert inference.legal_batch_calls >= 1
    assert 2 in inference.legal_batch_sizes
    assert dataset.metadata.schema_version == SELF_PLAY_DATASET_SCHEMA_VERSION
    assert dataset.metadata.generation_settings["batch_size"] == 2
    assert (
        dataset.metadata.generation_settings["batching_mode"]
        == BATCHING_MODE_CENTRAL_INFERENCE_QUEUE
    )
    assert dataset.metadata.generation_settings["inference_batch_size"] == 2
    assert dataset.positions.shape == (4, *TENSOR_SHAPE)
    assert dataset.legal_masks.shape == (4, ACTION_SPACE_SIZE)
    assert dataset.mcts_policies.shape == (4, ACTION_SPACE_SIZE)
    assert dataset.outcomes.shape == (4,)
    assert [record.game_index for record in dataset.games] == [0, 1]
    assert [record.plies for record in dataset.games] == [2, 2]
    assert np.all(dataset.legal_masks.sum(axis=1) > 0)
    assert np.allclose(dataset.mcts_policies.sum(axis=1), 1.0)
    assert np.all(dataset.mcts_policies >= 0.0)
    assert np.all(dataset.mcts_policies <= dataset.legal_masks)


def test_batched_neural_self_play_uses_legal_batch_queue_after_root_expansion() -> None:
    inference = CountingPolicyValueInference(
        PolicyValueNet(
            PolicyValueConfig(
                residual_channels=4,
                residual_blocks=0,
                policy_channels=1,
                value_channels=1,
                value_hidden_dim=4,
            )
        )
    )

    dataset = generate_self_play_dataset(
        inference,
        SelfPlayConfig(
            games=2,
            max_plies=1,
            mcts=NeuralMCTSConfig(simulations=3, temperature=0.0, seed=19),
            seed=19,
            batch_size=2,
        ),
    )

    assert dataset.metadata.sample_count == 2
    assert inference.batch_calls == 0
    assert inference.legal_batch_calls > 1
    assert inference.legal_batch_sizes[0] == 2
    assert 2 in inference.legal_batch_sizes[1:]


def test_batched_neural_self_play_is_reproducible_for_fixed_seed() -> None:
    model = PolicyValueNet(
        PolicyValueConfig(
            residual_channels=4,
            residual_blocks=0,
            policy_channels=1,
            value_channels=1,
            value_hidden_dim=4,
        )
    )
    config = SelfPlayConfig(
        games=2,
        max_plies=2,
        mcts=NeuralMCTSConfig(simulations=1, temperature=0.0, seed=23),
        seed=23,
        batch_size=2,
    )

    first = generate_self_play_dataset(PolicyValueInference(model), config)
    second = generate_self_play_dataset(PolicyValueInference(model), config)

    np.testing.assert_array_equal(first.positions, second.positions)
    np.testing.assert_array_equal(first.legal_masks, second.legal_masks)
    np.testing.assert_array_equal(first.mcts_policies, second.mcts_policies)
    np.testing.assert_array_equal(first.outcomes, second.outcomes)
    assert [record.moves_uci for record in first.games] == [
        record.moves_uci for record in second.games
    ]


def test_central_neural_self_play_matches_serial_with_deterministic_inference() -> None:
    base_config = SelfPlayConfig(
        games=2,
        max_plies=2,
        mcts=NeuralMCTSConfig(simulations=3, temperature=0.0, seed=37),
        model_checkpoint_id="fake-compact-checkpoint",
        seed=37,
    )

    serial = generate_self_play_dataset(
        DeterministicPolicyValueInference(),
        base_config,
    )
    central = generate_self_play_dataset(
        DeterministicPolicyValueInference(),
        replace(base_config, batch_size=2),
    )

    np.testing.assert_array_equal(serial.positions, central.positions)
    np.testing.assert_array_equal(serial.legal_masks, central.legal_masks)
    np.testing.assert_array_equal(serial.mcts_policies, central.mcts_policies)
    np.testing.assert_array_equal(serial.outcomes, central.outcomes)
    assert [record.moves_uci for record in serial.games] == [
        record.moves_uci for record in central.games
    ]
    assert [record.final_fen for record in serial.games] == [
        record.final_fen for record in central.games
    ]


def test_self_play_profile_counts_serial_neural_search_categories() -> None:
    inference = PolicyValueInference(
        PolicyValueNet(
            PolicyValueConfig(
                residual_channels=4,
                residual_blocks=0,
                policy_channels=1,
                value_channels=1,
                value_hidden_dim=4,
            )
        )
    )
    with self_play_profile() as profile:
        dataset = generate_self_play_dataset(
            inference,
            SelfPlayConfig(
                games=1,
                max_plies=2,
                mcts=NeuralMCTSConfig(simulations=2, temperature=0.0, seed=31),
                seed=31,
            ),
        )

    assert dataset.metadata.sample_count == 2
    report = profile.to_dict()
    assert report["format_version"] == 2
    assert "zones" in report
    timers = report["timers"]
    assert isinstance(timers, dict)
    assert timers["game_legal_moves"]["calls"] > 0
    assert timers["determine_outcome"]["calls"] > 0
    assert timers["game_play_known_legal"]["calls"] >= 2
    assert timers["board_apply_move"]["calls"] >= timers["game_play_known_legal"]["calls"]
    assert timers["model_single"]["calls"] > 0
    assert timers["model_batch"]["calls"] == 0
    assert timers["search"]["calls"] == 2
    assert timers["search"]["completed_simulations"] == 4
    assert timers["search"]["materialized_nodes"] >= 2
    zones = report["zones"]
    assert zones["mcts.search"]["inclusive_seconds"] >= zones["mcts.search"]["exclusive_seconds"]
    assert "search_state.legal_moves" in zones
    assert "board.apply_move" in zones


def test_serial_recording_reuses_precomputed_legal_masks(monkeypatch: Any) -> None:
    recorded_legal_counts: list[int] = []

    def spy_legal_mask(game: Game, legal: tuple[Move, ...]) -> Any:
        recorded_legal_counts.append(len(legal))
        return legal_move_mask_from_legal_moves_np(game, legal)

    monkeypatch.setattr(self_play, "legal_move_mask_from_legal_moves_np", spy_legal_mask)

    dataset = generate_self_play_dataset(
        FakeInference(),
        SelfPlayConfig(
            games=1,
            max_plies=2,
            mcts=NeuralMCTSConfig(simulations=1, temperature=0.0, seed=29),
            seed=29,
        ),
    )

    start = Game.new()
    np.testing.assert_array_equal(
        dataset.legal_masks[0],
        legal_move_mask_from_legal_moves_np(start, start.legal_moves),
    )
    assert recorded_legal_counts == [20, 20]
    assert dataset.positions.shape == (2, *TENSOR_SHAPE)
    assert dataset.legal_masks.shape == (2, ACTION_SPACE_SIZE)
    assert np.allclose(dataset.mcts_policies.sum(axis=1), 1.0)


def test_generate_self_play_dataset_can_use_classical_mcts_labels() -> None:
    dataset = generate_self_play_dataset(
        None,
        SelfPlayConfig(
            games=1,
            max_plies=2,
            classical_mcts=MCTSConfig(simulations=2, max_rollout_plies=1, seed=13),
            label_source=LABEL_SOURCE_CLASSICAL,
            seed=13,
        ),
    )

    assert dataset.metadata.generation_settings["label_source"] == LABEL_SOURCE_CLASSICAL
    classical_settings = dataset.metadata.generation_settings["classical_mcts"]
    assert isinstance(classical_settings, dict)
    assert classical_settings["simulations"] == 2
    assert dataset.positions.shape == (2, *TENSOR_SHAPE)
    assert dataset.legal_masks.shape == (2, ACTION_SPACE_SIZE)
    assert dataset.mcts_policies.shape == (2, ACTION_SPACE_SIZE)
    assert np.all(dataset.legal_masks.sum(axis=1) > 0)
    assert np.allclose(dataset.mcts_policies.sum(axis=1), 1.0)
    assert np.all(dataset.mcts_policies <= dataset.legal_masks)
    assert dataset.games[0].plies == 2


def test_serial_fallbacks_with_batch_size_record_serial_metadata() -> None:
    classical = generate_self_play_dataset(
        None,
        SelfPlayConfig(
            games=2,
            max_plies=1,
            classical_mcts=MCTSConfig(simulations=1, max_rollout_plies=1, seed=41),
            label_source=LABEL_SOURCE_CLASSICAL,
            seed=41,
            batch_size=2,
        ),
    )
    custom_neural = generate_self_play_dataset(
        FakeInference(),
        SelfPlayConfig(
            games=2,
            max_plies=1,
            mcts=NeuralMCTSConfig(simulations=1, temperature=0.0, seed=43),
            seed=43,
            batch_size=2,
        ),
    )

    for dataset in (classical, custom_neural):
        assert dataset.metadata.generation_settings["batch_size"] == 2
        assert dataset.metadata.generation_settings["batching_mode"] == BATCHING_MODE_SERIAL
        assert dataset.metadata.generation_settings["inference_batch_size"] == 1


def test_neural_self_play_requires_inference() -> None:
    try:
        generate_self_play_dataset(
            None,
            SelfPlayConfig(games=1, max_plies=1, label_source=LABEL_SOURCE_NEURAL),
        )
    except ValueError as exc:
        assert "requires inference" in str(exc)
    else:  # pragma: no cover - defensive assertion path
        raise AssertionError("expected neural generation without inference to be rejected")


def test_self_play_dataset_writes_and_reads_versioned_files(tmp_path: Path) -> None:
    dataset = generate_self_play_dataset(
        FakeInference(),
        SelfPlayConfig(
            games=1,
            max_plies=2,
            mcts=NeuralMCTSConfig(simulations=1, temperature=0.0, seed=5),
            seed=5,
        ),
    )

    save_self_play_dataset(dataset, tmp_path)
    loaded = load_self_play_dataset(tmp_path)

    assert (tmp_path / DEFAULT_DATASET_FILENAME).is_file()
    assert (tmp_path / DEFAULT_METADATA_FILENAME).is_file()
    assert (tmp_path / DEFAULT_GAMES_FILENAME).is_file()
    np.testing.assert_array_equal(loaded.positions, dataset.positions)
    np.testing.assert_array_equal(loaded.legal_masks, dataset.legal_masks)
    np.testing.assert_array_equal(loaded.mcts_policies, dataset.mcts_policies)
    np.testing.assert_array_equal(loaded.outcomes, dataset.outcomes)
    assert loaded.metadata.schema_version == SELF_PLAY_DATASET_SCHEMA_VERSION
    assert loaded.metadata.sample_count == 2
    assert loaded.games[0].moves_uci == dataset.games[0].moves_uci


def test_policy_target_falls_back_to_selected_move_when_all_root_visits_are_zero() -> None:
    dataset = generate_self_play_dataset(
        FakeInference(),
        SelfPlayConfig(games=1, max_plies=1, mcts=NeuralMCTSConfig(simulations=1, seed=1)),
    )

    move = Move.from_uci(dataset.games[0].moves_uci[0])
    selected_index = move_to_action_index(move, Game.new().board)
    assert dataset.mcts_policies[0, selected_index] == 1.0
    assert dataset.mcts_policies[0].sum() == 1.0


def test_load_self_play_dataset_rejects_inconsistent_game_records(tmp_path: Path) -> None:
    dataset = generate_self_play_dataset(
        FakeInference(),
        SelfPlayConfig(games=1, max_plies=2, mcts=NeuralMCTSConfig(simulations=1, seed=7)),
    )
    save_self_play_dataset(dataset, tmp_path)
    record = dataset.games[0].to_dict()
    moves = dataset.games[0].moves_uci
    record["moves_uci"] = ["a1a2", *moves[1:]]
    (tmp_path / DEFAULT_GAMES_FILENAME).write_text(json.dumps(record) + "\n")

    try:
        load_self_play_dataset(tmp_path)
    except ValueError as exc:
        assert "illegal move" in str(exc) or "position tensor" in str(exc)
    else:  # pragma: no cover - defensive assertion path
        raise AssertionError("expected corrupted game record to be rejected")


def test_load_self_play_dataset_rejects_policy_on_illegal_action(tmp_path: Path) -> None:
    dataset = generate_self_play_dataset(
        FakeInference(),
        SelfPlayConfig(games=1, max_plies=1, mcts=NeuralMCTSConfig(simulations=1, seed=9)),
    )
    save_self_play_dataset(dataset, tmp_path)
    illegal_index = int(np.flatnonzero(dataset.legal_masks[0] == 0.0)[0])
    corrupted_policy = np.zeros_like(dataset.mcts_policies)
    corrupted_policy[0, illegal_index] = 1.0
    np.savez_compressed(
        tmp_path / DEFAULT_DATASET_FILENAME,
        positions=dataset.positions,
        legal_masks=dataset.legal_masks,
        mcts_policies=corrupted_policy,
        outcomes=dataset.outcomes,
    )

    try:
        load_self_play_dataset(tmp_path)
    except ValueError as exc:
        assert "illegal moves" in str(exc)
    else:  # pragma: no cover - defensive assertion path
        raise AssertionError("expected illegal policy target to be rejected")


def test_merge_self_play_datasets_reindexes_games_and_preserves_counts() -> None:
    first = generate_self_play_dataset(
        FakeInference(),
        SelfPlayConfig(
            games=1,
            max_plies=1,
            mcts=NeuralMCTSConfig(simulations=1, seed=1),
            model_checkpoint_id="shared",
            seed=1,
        ),
    )
    second = generate_self_play_dataset(
        FakeInference(),
        SelfPlayConfig(
            games=1,
            max_plies=2,
            mcts=NeuralMCTSConfig(simulations=1, seed=2),
            model_checkpoint_id="shared",
            seed=2,
        ),
    )

    merged = merge_self_play_datasets([first, second])

    assert merged.metadata.game_count == 2
    assert merged.metadata.sample_count == 3
    assert merged.metadata.model_checkpoint_id == "shared"
    assert [record.game_index for record in merged.games] == [0, 1]
    assert merged.positions.shape[0] == 3
    assert merged.legal_masks.shape[0] == 3
    assert merged.mcts_policies.shape[0] == 3
    assert merged.outcomes.shape[0] == 3


def test_merge_self_play_datasets_rejects_empty_input() -> None:
    try:
        merge_self_play_datasets([])
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:  # pragma: no cover - defensive assertion path
        raise AssertionError("expected empty merge input to be rejected")


def test_merge_self_play_datasets_rejects_mismatched_checkpoint_id() -> None:
    first = generate_self_play_dataset(
        FakeInference(),
        SelfPlayConfig(games=1, max_plies=1, model_checkpoint_id="first"),
    )
    second = generate_self_play_dataset(
        FakeInference(),
        SelfPlayConfig(games=1, max_plies=1, model_checkpoint_id="second"),
    )

    try:
        merge_self_play_datasets([first, second])
    except ValueError as exc:
        assert "different model checkpoints" in str(exc)
    else:  # pragma: no cover - defensive assertion path
        raise AssertionError("expected checkpoint mismatch to be rejected")


def test_merge_self_play_datasets_rejects_malformed_counts() -> None:
    dataset = generate_self_play_dataset(
        FakeInference(),
        SelfPlayConfig(games=1, max_plies=1, mcts=NeuralMCTSConfig(simulations=1, seed=11)),
    )
    malformed = replace(
        dataset,
        metadata=replace(dataset.metadata, sample_count=dataset.metadata.sample_count + 1),
    )

    try:
        merge_self_play_datasets([malformed])
    except ValueError as exc:
        assert "sample count" in str(exc)
    else:  # pragma: no cover - defensive assertion path
        raise AssertionError("expected malformed sample count to be rejected")


def test_self_play_script_creates_classical_mcts_dataset(tmp_path: Path) -> None:
    output = tmp_path / "classical-output"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/self_play.py",
            "--label-source",
            "classical",
            "--games",
            "1",
            "--max-plies",
            "1",
            "--simulations",
            "2",
            "--classical-max-rollout-plies",
            "1",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    dataset = load_self_play_dataset(output)
    assert dataset.metadata.generation_settings["label_source"] == LABEL_SOURCE_CLASSICAL
    assert dataset.metadata.sample_count == 1
    assert np.allclose(dataset.mcts_policies.sum(axis=1), 1.0)


def test_self_play_script_creates_documented_files(tmp_path: Path) -> None:
    output = tmp_path / "script-output"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/self_play.py",
            "--games",
            "1",
            "--max-plies",
            "1",
            "--simulations",
            "1",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "games=1" in result.stdout
    assert "samples=1" in result.stdout
    assert (output / DEFAULT_DATASET_FILENAME).is_file()
    assert (output / DEFAULT_METADATA_FILENAME).is_file()
    assert (output / DEFAULT_GAMES_FILENAME).is_file()


def test_self_play_script_accepts_batch_size(tmp_path: Path) -> None:
    output = tmp_path / "script-batch-output"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/self_play.py",
            "--games",
            "2",
            "--max-plies",
            "1",
            "--simulations",
            "1",
            "--batch-size",
            "2",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    dataset = load_self_play_dataset(output)
    assert dataset.metadata.generation_settings["batch_size"] == 2
    assert (
        dataset.metadata.generation_settings["batching_mode"]
        == BATCHING_MODE_CENTRAL_INFERENCE_QUEUE
    )
    assert dataset.metadata.generation_settings["inference_batch_size"] == 2
    assert dataset.metadata.game_count == 2
    assert dataset.metadata.sample_count == 2
    assert [record.game_index for record in dataset.games] == [0, 1]


def test_self_play_script_help_describes_central_batching() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/self_play.py", "--help"],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "central inference batch size" in result.stdout
    assert "independent neural self-play games/searches" in result.stdout
    assert "not within-tree" in result.stdout
    assert "leaf parallelism" in result.stdout


def test_self_play_script_rejects_removed_parallel_batch_flag(tmp_path: Path) -> None:
    output = tmp_path / "script-removed-parallel-batch-output"
    removed_flag = "--" + "leaf" + "-parallelism"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/self_play.py",
            "--games",
            "1",
            "--max-plies",
            "1",
            "--simulations",
            "3",
            removed_flag,
            "2",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert f"unrecognized arguments: {removed_flag}" in result.stderr


def test_historical_metadata_with_opaque_parallel_batch_setting_still_loads() -> None:
    historical_field = "leaf" + "_parallelism"
    metadata = SelfPlayMetadata.from_dict(
        {
            "schema_version": SELF_PLAY_DATASET_SCHEMA_VERSION,
            "generated_at": "2026-06-05T00:00:00+00:00",
            "engine_version": "0.1.0",
            "git_commit": None,
            "action_space_version": ACTION_SPACE_VERSION,
            "encoder_version": ENCODER_VERSION,
            "model_checkpoint_id": None,
            "generation_settings": {
                "mcts": {"simulations": 3, historical_field: 2}
            },
            "sample_count": 0,
            "game_count": 0,
        }
    )

    mcts_settings = metadata.generation_settings["mcts"]
    assert isinstance(mcts_settings, dict)
    assert mcts_settings[historical_field] == 2


def test_self_play_script_rejects_invalid_worker_count(tmp_path: Path) -> None:
    output = tmp_path / "invalid-workers"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/self_play.py",
            "--workers",
            "0",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "--workers must be at least 1" in result.stderr


def test_self_play_script_rejects_classical_checkpoint_id(tmp_path: Path) -> None:
    output = tmp_path / "classical-checkpoint-id"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/self_play.py",
            "--label-source",
            "classical",
            "--checkpoint-id",
            "unused-model",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "--checkpoint-id is only supported with --label-source neural" in result.stderr


def test_self_play_script_can_generate_classical_labels_in_parallel(tmp_path: Path) -> None:
    output = tmp_path / "classical-parallel-output"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/self_play.py",
            "--label-source",
            "classical",
            "--games",
            "2",
            "--max-plies",
            "1",
            "--simulations",
            "2",
            "--classical-max-rollout-plies",
            "1",
            "--workers",
            "2",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    dataset = load_self_play_dataset(output)
    assert dataset.metadata.generation_settings["label_source"] == LABEL_SOURCE_CLASSICAL
    assert dataset.metadata.game_count == 2
    assert dataset.metadata.sample_count == 2
    assert [record.game_index for record in dataset.games] == [0, 1]


def test_self_play_script_can_generate_in_parallel(tmp_path: Path) -> None:
    output = tmp_path / "parallel-output"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/self_play.py",
            "--games",
            "2",
            "--max-plies",
            "1",
            "--simulations",
            "1",
            "--workers",
            "2",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "games=2" in result.stdout
    assert "samples=2" in result.stdout
    dataset = load_self_play_dataset(output)
    assert [record.game_index for record in dataset.games] == [0, 1]
    parallel_settings = dataset.metadata.generation_settings["parallel"]
    assert isinstance(parallel_settings, dict)
    assert parallel_settings["workers"] == 2


def test_self_play_script_writes_profile_for_parallel_workers(tmp_path: Path) -> None:
    output = tmp_path / "profiled-parallel-output"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/self_play.py",
            "--games",
            "2",
            "--max-plies",
            "1",
            "--simulations",
            "1",
            "--workers",
            "2",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "TINYCHESS_SELF_PLAY_PROFILE": "1"},
    )

    assert result.returncode == 0, result.stderr
    profile_path = output / DEFAULT_PROFILE_FILENAME
    assert profile_path.is_file()
    profile = json.loads(profile_path.read_text())
    assert profile["scope"] == "self_play_generation"
    assert profile["format_version"] == 2
    assert profile["profile_level"] == "detailed"
    assert len(profile["worker_profiles"]) == 2
    for worker_profile in profile["worker_profiles"]:
        assert worker_profile["metadata"]["worker_id"] in {0, 1}
        assert worker_profile["metadata"]["games"] == 1
        assert "pid" in worker_profile["metadata"]
    assert "worker.pool_elapsed" in profile["stats"]["zones"]
    assert "dataset.merge" in profile["stats"]["zones"]
    timers = profile["stats"]["timers"]
    assert timers["search"]["completed_simulations"] == 2
    assert timers["model_single"]["calls"] == 2
    assert timers["game_legal_moves"]["calls"] > 0
    dataset = load_self_play_dataset(output)
    assert dataset.metadata.sample_count == 2


def test_self_play_script_parallel_workers_share_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    model_config = PolicyValueConfig(
        residual_channels=4,
        residual_blocks=0,
        policy_channels=1,
        value_channels=1,
        value_hidden_dim=4,
    )
    save_checkpoint(PolicyValueNet(model_config), checkpoint)
    output = tmp_path / "checkpoint-output"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/self_play.py",
            "--games",
            "2",
            "--max-plies",
            "1",
            "--simulations",
            "1",
            "--workers",
            "2",
            "--checkpoint",
            str(checkpoint),
            "--checkpoint-id",
            "shared-test-checkpoint",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    dataset = load_self_play_dataset(output)
    assert dataset.metadata.model_checkpoint_id == "shared-test-checkpoint"
    assert dataset.metadata.game_count == 2
    assert dataset.metadata.sample_count == 2
