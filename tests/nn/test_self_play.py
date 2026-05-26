from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from tinychess.ai.neural_mcts import NeuralMCTSConfig
from tinychess.engine import Game, Move, OutcomeReason
from tinychess.nn.encode import ACTION_SPACE_SIZE, TENSOR_SHAPE, move_to_action_index
from tinychess.nn.model import InferenceResult
from tinychess.nn.self_play import (
    DEFAULT_DATASET_FILENAME,
    DEFAULT_GAMES_FILENAME,
    DEFAULT_METADATA_FILENAME,
    SELF_PLAY_DATASET_SCHEMA_VERSION,
    SelfPlayConfig,
    generate_self_play_dataset,
    load_self_play_dataset,
    save_self_play_dataset,
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
